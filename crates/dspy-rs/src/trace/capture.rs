//! Task-local capture scope for the unified trace format.
//!
//! [`capture`] mirrors the shape of a task-local scope: only `Predict` calls on
//! the same task record spans; spawned subtasks do not inherit the scope; nested
//! scopes are exclusive (innermost wins). Zero-cost when inactive — one
//! task-local probe per `Predict` call.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::task_local;

use crate::trace::span::{
    CompId, JsonMap, ModelEntry, ModelId, PrefixEntry, PrefixId, Span, SpanError, SpanErrorKind,
    SpanEvent, SpanId, Trace, TraceMeta, request_hash,
};
use crate::utils::hash::stable_hash_debug;
use crate::{LMConfig, LmUsage, Message};

task_local! {
    static ACTIVE: TraceSink;
}

pub(crate) fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

fn mint_trace_id() -> String {
    use rand::Rng;
    let random: u64 = rand::thread_rng().r#gen();
    format!("{:016x}{:016x}", now_us(), random)
}

/// Shared handle to an in-progress trace.
#[derive(Clone)]
pub struct TraceSink(Arc<Mutex<SinkInner>>);

struct SinkInner {
    trace: Trace,
    /// Stack of open span ids — parent attribution for Predict-in-tool.
    open: Vec<SpanId>,
    /// Per-component next-seq counters, indexed by `CompId`.
    seqs: Vec<u32>,
    comp_index: HashMap<String, CompId>,
    /// Prefix/model intern maps: content-hash -> id.
    prefix_index: HashMap<u64, PrefixId>,
    model_index: HashMap<u64, ModelId>,
    /// Monotonic clock anchor for duration measurement.
    epoch: Instant,
}

impl TraceSink {
    fn new(mut meta: TraceMeta) -> Self {
        meta.v = 1;
        if meta.trace_id.is_empty() {
            meta.trace_id = mint_trace_id();
        }
        if meta.started_at_us == 0 {
            meta.started_at_us = now_us();
        }
        Self(Arc::new(Mutex::new(SinkInner {
            trace: Trace {
                meta,
                ..Trace::default()
            },
            open: Vec::new(),
            seqs: Vec::new(),
            comp_index: HashMap::new(),
            prefix_index: HashMap::new(),
            model_index: HashMap::new(),
            epoch: Instant::now(),
        })))
    }

    fn finish(self) -> Trace {
        match Arc::try_unwrap(self.0) {
            Ok(mutex) => mutex.into_inner().unwrap().trace,
            // Fallback: clone if an orphaned task still holds a reference.
            Err(arc) => arc.lock().unwrap().trace.clone(),
        }
    }

    fn begin(&self, req: SpanRequest<'_>) -> SpanGuard {
        let mut inner = self.0.lock().unwrap();

        let component = match inner.comp_index.get(req.component) {
            Some(&id) => id,
            None => {
                let id = CompId(inner.trace.components.len() as u32);
                inner.trace.components.push(req.component.to_string());
                inner.comp_index.insert(req.component.to_string(), id);
                inner.seqs.push(0);
                id
            }
        };
        let seq = inner.seqs[component.0 as usize];
        inner.seqs[component.0 as usize] += 1;

        let prefix = req.prefix.map(|messages| {
            let hash = stable_hash_debug(&messages);
            match inner.prefix_index.get(&hash) {
                Some(&id) => id,
                None => {
                    let id = PrefixId(inner.trace.prefixes.len() as u32);
                    inner.trace.prefixes.push(PrefixEntry {
                        messages: messages.to_vec(),
                    });
                    inner.prefix_index.insert(hash, id);
                    id
                }
            }
        });

        let entry = ModelEntry::from_config(req.model);
        let model = match inner.model_index.get(&entry.config_hash) {
            Some(&id) => id,
            None => {
                let id = ModelId(inner.trace.models.len() as u32);
                inner.model_index.insert(entry.config_hash, id);
                inner.trace.models.push(entry);
                id
            }
        };

        let id = SpanId(inner.trace.spans.len() as u32);
        let parent = inner.open.last().copied();

        inner.trace.spans.push(Span {
            id,
            component,
            seq,
            parent,
            prefix,
            suffix: req.suffix.to_vec(),
            input: req.input,
            model,
            request_hash: req.request_hash.unwrap_or(0),
            events: Vec::new(),
            raw_output: None,
            output: None,
            usage: LmUsage::default(),
            error: None,
            started_at_us: now_us(),
            duration_us: 0,
            complete: true,
        });
        inner.open.push(id);
        let started = inner.epoch.elapsed();

        SpanGuard {
            id,
            sink: self.clone(),
            started,
            done: false,
            hash_override: req.request_hash.is_some(),
        }
    }

    fn close(
        &self,
        id: SpanId,
        started: std::time::Duration,
        outcome: SpanOutcome,
        hash_override: bool,
    ) {
        let mut inner = self.0.lock().unwrap();
        let duration_us = inner.epoch.elapsed().saturating_sub(started).as_micros() as u64;
        inner.open.retain(|open| *open != id);

        // An explicit request_hash (holes) was stamped at open; everything
        // else hashes the redacted config + rendered prompt at close.
        if !hash_override {
            let config_hash = {
                let span = &inner.trace.spans[id.0 as usize];
                inner.trace.models[span.model.0 as usize].config_hash
            };
            let hash = {
                let span = &inner.trace.spans[id.0 as usize];
                let prefix = span
                    .prefix
                    .map(|p| inner.trace.prefixes[p.0 as usize].messages.as_slice())
                    .unwrap_or(&[]);
                request_hash(config_hash, prefix, &span.suffix)
            };
            inner.trace.spans[id.0 as usize].request_hash = hash;
        }

        let span = &mut inner.trace.spans[id.0 as usize];
        span.events = outcome.events;
        span.raw_output = outcome.raw_output;
        span.output = outcome.output;
        span.usage = outcome.usage;
        span.error = outcome.error;
        span.duration_us = duration_us;
    }
}

/// Runs `f` while recording every `Predict` call on this task into a [`Trace`].
///
/// Task-local scoping: spawned subtasks do NOT inherit the scope. Nested
/// `capture` scopes are exclusive — the innermost scope records, the outer one
/// records nothing for that region (this is how LM-as-judge metric calls stay
/// out of the execution trace).
pub async fn capture<F, Fut, R>(f: F) -> (R, Trace)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = R>,
{
    capture_with_meta(TraceMeta::default(), f).await
}

/// [`capture`] with caller-provided rollout metadata (input, candidate hash,
/// tags). Missing `trace_id`/`started_at_us` are minted at scope start.
pub async fn capture_with_meta<F, Fut, R>(meta: TraceMeta, f: F) -> (R, Trace)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = R>,
{
    let sink = TraceSink::new(meta);
    let result = ACTIVE.scope(sink.clone(), f()).await;
    (result, sink.finish())
}

/// Returns `true` if the current task is inside a [`capture`] scope.
pub fn is_capturing() -> bool {
    ACTIVE.try_with(|_| ()).is_ok()
}

/// Everything recorded eagerly at span open.
pub struct SpanRequest<'a> {
    /// Component name: `trace_name` / fx slot name / dotted path.
    pub component: &'a str,
    /// System + demo prefix messages, when the call rendered one.
    pub prefix: Option<&'a [Message]>,
    /// Live suffix of the rendered prompt (user turn / caller-owned history).
    pub suffix: &'a [Message],
    /// Typed input fields as JSON, when available.
    pub input: Option<JsonMap>,
    pub model: &'a LMConfig,
    /// Explicit `request_hash` preimage override. `None` (the norm) computes
    /// the hash at close from the redacted config + rendered prompt. Leaves
    /// with no prompt-shaped identity — holes (RFC 0003 §4.4: impl hash ++
    /// canonical input ++ caps) — pass their own hash so every span keys
    /// replay on a real preimage instead of a degenerate empty-prompt one.
    pub request_hash: Option<u64>,
}

/// Everything recorded lazily at span close.
pub struct SpanOutcome {
    pub events: Vec<SpanEvent>,
    pub raw_output: Option<String>,
    pub output: Option<JsonMap>,
    pub usage: LmUsage,
    pub error: Option<SpanError>,
}

/// Opens a span in the active capture scope. Returns `None` when no scope is
/// active — the caller does nothing else in that case (zero-overhead path).
///
/// Used by `Predict`; public so custom leaf modules can record spans too.
pub fn begin_span(req: SpanRequest<'_>) -> Option<SpanGuard> {
    let sink = ACTIVE.try_with(|sink| sink.clone()).ok()?;
    Some(sink.begin(req))
}

/// Owns a [`SpanId`] — event/close attribution travels with the guard, never
/// through "innermost open span" lookups, so interleaved `Predict`s on one task
/// (`futures::join!`) cannot cross-attribute. Dropping the guard without
/// [`finish`](SpanGuard::finish) marks the span `Cancelled`.
pub struct SpanGuard {
    id: SpanId,
    sink: TraceSink,
    started: std::time::Duration,
    done: bool,
    /// The span opened with an explicit `request_hash`; close must not
    /// overwrite it with the prompt-derived hash.
    hash_override: bool,
}

impl SpanGuard {
    pub fn id(&self) -> SpanId {
        self.id
    }

    pub fn finish(mut self, out: SpanOutcome) {
        self.done = true;
        self.sink
            .close(self.id, self.started, out, self.hash_override);
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.done {
            self.sink.close(
                self.id,
                self.started,
                SpanOutcome {
                    events: Vec::new(),
                    raw_output: None,
                    output: None,
                    usage: LmUsage::default(),
                    error: Some(SpanError {
                        kind: SpanErrorKind::Cancelled,
                        message: "scope ended while span was open".to_string(),
                    }),
                },
                self.hash_override,
            );
        }
    }
}
