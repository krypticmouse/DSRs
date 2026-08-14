//! Replay scope: serve `Predict` calls from a recorded [`Trace`] (RFC 0001 §4d/§4e).
//!
//! A trace is a complete set of canned LM responses. [`replay`] opens a
//! task-local scope symmetric to [`capture`](crate::trace::capture); every
//! `Predict` call inside it is intercepted **above** the LM: the call's
//! `request_hash` (same preimage the spans record — redacted model config ++
//! full rendered prompt) is compared against the next recorded span for that
//! component, and on a match the recorded parsed output is served without
//! constructing a client, calling a provider, or executing a tool — tool
//! effects are already baked into the recorded events.
//!
//! Two modes ([`ReplayMode`]):
//!
//! - **`Strict`** — every call must match its recording; any mismatch is a
//!   typed [`ReplayError`] surfaced as
//!   [`PredictError::Replay`](crate::PredictError::Replay). Temporal's
//!   determinism check; what fixtures and CI use:
//!
//!   ```ignore
//!   let (out, trace) = capture(|| pipeline(input.clone())).await;   // once, live
//!   std::fs::write("fixtures/qa.trace.jsonl", trace.to_jsonl()?)?;
//!
//!   // in tests, forever, zero API calls:
//!   let trace = Trace::from_jsonl(&std::fs::read_to_string("fixtures/qa.trace.jsonl")?)?;
//!   let (out, report) = trace::replay(&trace, ReplayMode::Strict, || pipeline(input)).await;
//!   assert_eq!(report.live, 0);
//!   ```
//!
//! - **`UntilDivergence`** — serve while hashes match; the first mismatching
//!   call *and every call after it* go to the live LM. Divergence is
//!   *detected*, never declared: a mutated parameter changes the rendered
//!   prompt (and therefore the hash) of exactly the calls it affects, so the
//!   unchanged prefix of a pipeline replays free and only the counterfactual
//!   suffix spends tokens (§4e):
//!
//!   ```ignore
//!   let mut params = fx::Params::new();
//!   params.set_instruction("refiner", mutated_instruction);
//!   let ((result, new_trace), report) = trace::replay(&base_trace, ReplayMode::UntilDivergence, || {
//!       capture(|| fx::with_params(params, pipeline(input)))
//!   }).await;
//!   // spans before the first "refiner" call: served free from base_trace
//!   // "refiner" and everything after: live LM calls only
//!   ```
//!
//! Once diverged, the session stays live even if a later call's hash happens
//! to match the recording again — downstream state may differ in ways the
//! prompt does not capture. Limitation (RFC §4e): served spans never re-run
//! tools, but post-divergence live calls do execute tools against the real
//! world; counterfactual replay of side-effectful harnesses needs a
//! resettable environment.
//!
//! Like `capture`, the scope is task-local: spawned subtasks do not inherit
//! it, and nested scopes are exclusive (innermost wins).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::task_local;

use crate::trace::span::{ModelEntry, Span, SpanId, Trace, request_hash};
use crate::{LMConfig, Message};

task_local! {
    static ACTIVE: ReplaySession;
}

/// How a replay scope treats a call that does not match its recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayMode {
    /// Every call must match: the next recorded span for `(component, seq)`
    /// must exist, be complete, and have an equal `request_hash`. Any mismatch
    /// is a typed error. For fixtures and CI.
    Strict,
    /// Serve recorded spans while hashes match; on the first mismatch, that
    /// call and ALL subsequent calls go to the live LM. For counterfactual
    /// replay.
    UntilDivergence,
}

/// Why a call could not be served from the recording.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ReplayError {
    /// The live request's hash differs from the recorded span's — the prompt
    /// or model config changed relative to the recording.
    #[error(
        "replay divergence at `{component}` seq {seq}: recorded span {expected_span:?} \
         has request_hash {expected_hash:#018x}, live call hashed {got_hash:#018x}"
    )]
    Divergence {
        component: String,
        seq: u32,
        /// The recorded span the live call was checked against.
        expected_span: SpanId,
        expected_hash: u64,
        got_hash: u64,
    },

    /// The recorded span is unusable: truncated/redacted (`complete == false`)
    /// or it has no parsed output (the recorded call failed).
    #[error(
        "recorded span {span:?} for `{component}` seq {seq} is incomplete \
         (truncated, redacted, or no parsed output) and cannot be replayed"
    )]
    Incomplete {
        component: String,
        seq: u32,
        span: SpanId,
    },

    /// The trace has no recorded span for `(component, seq)` — the live run
    /// makes more calls than the recording (or calls a component the
    /// recording never saw).
    #[error("no recorded span for `{component}` seq {seq}: the trace is exhausted")]
    Exhausted { component: String, seq: u32 },

    /// The recorded span matched but its stored output does not deserialize
    /// into the signature's output type (schema drift since recording).
    #[error(
        "recorded output of span {span:?} for `{component}` seq {seq} \
         does not fit the signature output type: {message}"
    )]
    OutputDecode {
        component: String,
        seq: u32,
        span: SpanId,
        message: String,
    },
}

/// What a replay scope did, returned by [`replay`] alongside the closure's
/// result.
#[derive(Clone, Debug, Default)]
pub struct ReplayReport {
    /// Calls served from the recording (zero provider calls).
    pub served: usize,
    /// Calls that went to the live LM (always 0 in `Strict` mode).
    pub live: usize,
    /// The recorded span at which divergence was first detected, when the
    /// mismatch could be attributed to one (hash mismatch / incomplete span).
    pub diverged_at: Option<SpanId>,
    /// The first mismatch, verbatim — the error `Strict` mode surfaced, or
    /// the reason `UntilDivergence` switched live.
    pub divergence: Option<ReplayError>,
}

/// Shared handle to an in-progress replay session.
#[derive(Clone)]
pub(crate) struct ReplaySession(Arc<Mutex<ReplayInner>>);

struct ReplayInner {
    trace: Trace,
    mode: ReplayMode,
    /// Per-component next-seq cursors, keyed by component name.
    cursors: HashMap<String, u32>,
    /// `UntilDivergence` only: once true, every call goes live.
    diverged: bool,
    report: ReplayReport,
}

/// What `Predict` should do with an intercepted call.
pub(crate) enum ReplayDirective {
    /// Serve this recorded span verbatim — no client, no provider, no tools.
    Serve(Box<Span>),
    /// Call the live LM (post-divergence `UntilDivergence`).
    Live,
    /// Refuse the call (`Strict`): surface the typed error.
    Refuse(ReplayError),
}

impl ReplaySession {
    fn new(trace: Trace, mode: ReplayMode) -> Self {
        Self(Arc::new(Mutex::new(ReplayInner {
            trace,
            mode,
            cursors: HashMap::new(),
            diverged: false,
            report: ReplayReport::default(),
        })))
    }

    fn finish(self) -> ReplayReport {
        match Arc::try_unwrap(self.0) {
            Ok(mutex) => mutex.into_inner().unwrap().report,
            // Fallback: clone if an orphaned task still holds a reference.
            Err(arc) => arc.lock().unwrap().report.clone(),
        }
    }

    /// Decides the fate of one `Predict` call: serve, go live, or refuse.
    fn decide(&self, component: &str, config: &LMConfig, chat: &[Message]) -> ReplayDirective {
        let mut inner = self.0.lock().unwrap();

        if inner.diverged {
            inner.report.live += 1;
            return ReplayDirective::Live;
        }

        let seq = inner.cursors.get(component).copied().unwrap_or(0);
        let recorded = inner.trace.component_id(component).and_then(|id| {
            inner
                .trace
                .spans
                .iter()
                .find(|span| span.component == id && span.seq == seq)
                .cloned()
        });
        let span = match recorded {
            Some(span) => span,
            None => {
                return inner.mismatch(
                    None,
                    ReplayError::Exhausted {
                        component: component.to_string(),
                        seq,
                    },
                );
            }
        };

        if !span.complete || span.output.is_none() {
            return inner.mismatch(
                Some(span.id),
                ReplayError::Incomplete {
                    component: component.to_string(),
                    seq,
                    span: span.id,
                },
            );
        }

        // The same preimage `TraceSink::close` hashes: redacted-config hash ++
        // full rendered prompt. The prefix/suffix split does not matter — the
        // hash streams prefix ++ suffix, and `chat` is exactly that.
        let got_hash = request_hash(ModelEntry::from_config(config).config_hash, &[], chat);
        if got_hash != span.request_hash {
            return inner.mismatch(
                Some(span.id),
                ReplayError::Divergence {
                    component: component.to_string(),
                    seq,
                    expected_span: span.id,
                    expected_hash: span.request_hash,
                    got_hash,
                },
            );
        }

        inner.cursors.insert(component.to_string(), seq + 1);
        inner.report.served += 1;
        ReplayDirective::Serve(Box::new(span))
    }
}

impl ReplayInner {
    /// Routes a mismatch by mode: `Strict` refuses, `UntilDivergence` flips
    /// the session live. Only the first mismatch is recorded in the report.
    fn mismatch(&mut self, at: Option<SpanId>, error: ReplayError) -> ReplayDirective {
        if self.report.divergence.is_none() {
            self.report.diverged_at = at;
            self.report.divergence = Some(error.clone());
        }
        match self.mode {
            ReplayMode::Strict => ReplayDirective::Refuse(error),
            ReplayMode::UntilDivergence => {
                self.diverged = true;
                self.report.live += 1;
                ReplayDirective::Live
            }
        }
    }
}

/// Runs `f` with `trace` as the canned-response source for every `Predict`
/// call on this task. Returns the closure's result and a [`ReplayReport`]
/// of what was served versus live.
///
/// Task-local scoping mirrors [`capture`](crate::trace::capture): spawned
/// subtasks do not inherit the scope; nesting is exclusive (innermost wins).
/// Compose with `capture` (replay outside, capture inside) to record the
/// counterfactual rollout while serving its unchanged prefix from `trace`.
pub async fn replay<F, Fut, R>(trace: &Trace, mode: ReplayMode, f: F) -> (R, ReplayReport)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = R>,
{
    let session = ReplaySession::new(trace.clone(), mode);
    let result = ACTIVE.scope(session.clone(), f()).await;
    (result, session.finish())
}

/// Returns `true` if the current task is inside a [`replay`] scope.
pub fn is_replaying() -> bool {
    ACTIVE.try_with(|_| ()).is_ok()
}

/// Consults the active replay scope, if any, about one `Predict` call.
/// `None` means no scope is active — proceed live, unconditionally.
///
/// Called by `Predict::call_and_parse_with_input` with the exact `Chat` it
/// would send and the config of the LM it would send it to.
pub(crate) fn intercept(
    component: &str,
    config: &LMConfig,
    chat: &[Message],
) -> Option<ReplayDirective> {
    let session = ACTIVE.try_with(|session| session.clone()).ok()?;
    Some(session.decide(component, config, chat))
}
