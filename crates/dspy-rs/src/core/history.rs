//! Bounded execution history, independent of response caching and trace capture.
use std::collections::VecDeque;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::trace::{JsonMap, SpanError, SpanErrorKind, SpanEvent};
use crate::{Chat, LMConfig, LMResponse, LmUsage, Message};

pub const DEFAULT_HISTORY_CAPACITY: usize = 100;

/// One logical LM invocation, including cache hits and failed calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub timestamp_us: u64,
    pub duration_us: u64,
    pub model: String,
    /// Generation settings with credentials removed and endpoint reduced to origin.
    pub config: LMConfig,
    pub messages: Vec<Message>,
    /// JSON rendering of the input messages, for compatibility with cache inspection.
    pub prompt: String,
    pub response: Option<Message>,
    pub raw_output: Option<String>,
    pub output: Option<JsonMap>,
    /// None when usage could not be recovered after a failed/cancelled call.
    pub usage: Option<LmUsage>,
    /// None when pricing is unavailable; token usage is not treated as cost.
    pub cost: Option<f64>,
    pub cache_hit: bool,
    pub module_path: Vec<String>,
    pub events: Vec<SpanEvent>,
    pub error: Option<SpanError>,
}

/// Shared live record. Parsed output is attached by the interpreter after the LM returns.
#[derive(Clone, Debug)]
pub struct HistoryRecord(Arc<Mutex<HistoryEntry>>);

impl HistoryRecord {
    pub fn snapshot(&self) -> HistoryEntry {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub(crate) fn parsed<T, E: std::fmt::Display>(&self, result: &Result<(JsonMap, T), E>) {
        let mut entry = self.0.lock().unwrap_or_else(|p| p.into_inner());
        match result {
            Ok((output, _)) => {
                entry.output = Some(output.clone());
                entry.error = None;
            }
            Err(error) => {
                entry.error = Some(SpanError {
                    kind: SpanErrorKind::Parse,
                    message: error.to_string(),
                })
            }
        }
    }
}

#[derive(Debug)]
struct Window {
    capacity: usize,
    entries: VecDeque<HistoryRecord>,
}

/// A bounded history store. Clones share the same store; snapshots are newest first.
#[derive(Clone, Debug)]
pub struct HistoryBuffer(Arc<Mutex<Window>>);

impl Default for HistoryBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_HISTORY_CAPACITY)
    }
}

impl HistoryBuffer {
    pub fn new(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(Window {
            capacity,
            entries: VecDeque::new(),
        })))
    }

    pub fn set_capacity(&self, capacity: usize) {
        let mut window = self.0.lock().unwrap_or_else(|p| p.into_inner());
        window.capacity = capacity;
        while window.entries.len() > capacity {
            window.entries.pop_front();
        }
    }

    pub fn clear(&self) {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .clear();
    }

    fn is_enabled(&self) -> bool {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).capacity != 0
    }

    pub fn inspect(&self, n: usize) -> Vec<HistoryEntry> {
        let records = self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>();
        records.iter().map(HistoryRecord::snapshot).collect()
    }

    fn push(&self, record: HistoryRecord) {
        let mut window = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if window.capacity == 0 {
            return;
        }
        window.entries.push_back(record);
        while window.entries.len() > window.capacity {
            window.entries.pop_front();
        }
    }

    /// Collects LM invocations made by this future and its nested scopes.
    /// Spawned tasks do not inherit the caller stack.
    pub async fn scope<F: std::future::Future>(
        &self,
        name: impl Into<String>,
        future: F,
    ) -> F::Output {
        let mut scopes = CALLERS.try_with(Clone::clone).unwrap_or_default();
        scopes.push(Caller {
            name: name.into(),
            buffer: self.clone(),
        });
        CALLERS.scope(scopes, future).await
    }
}

#[derive(Clone)]
struct Caller {
    name: String,
    buffer: HistoryBuffer,
}
tokio::task_local! { static CALLERS: Vec<Caller>; }
static GLOBAL: LazyLock<HistoryBuffer> = LazyLock::new(HistoryBuffer::default);

pub fn inspect_history(n: usize) -> Vec<HistoryEntry> {
    GLOBAL.inspect(n)
}
pub fn clear_history() {
    GLOBAL.clear();
}
pub fn set_history_capacity(capacity: usize) {
    GLOBAL.set_capacity(capacity);
}

pub(crate) struct HistoryCall {
    record: HistoryRecord,
    sinks: Vec<HistoryBuffer>,
    started: Instant,
    finished: bool,
}

impl HistoryCall {
    pub(crate) fn begin(
        config: &LMConfig,
        chat: &Chat,
        lm_history: &HistoryBuffer,
    ) -> Option<Self> {
        let enabled = lm_history.is_enabled()
            || GLOBAL.is_enabled()
            || CALLERS
                .try_with(|callers| callers.iter().any(|caller| caller.buffer.is_enabled()))
                .unwrap_or(false);
        if !enabled {
            return None;
        }
        let callers = CALLERS.try_with(Clone::clone).unwrap_or_default();
        let record = HistoryRecord(Arc::new(Mutex::new(HistoryEntry {
            id: crate::trace::capture::mint_trace_id(),
            timestamp_us: crate::trace::capture::now_us(),
            duration_us: 0,
            model: config.model.clone(),
            config: crate::trace::ModelEntry::from_config(config).config,
            messages: chat.messages.clone(),
            prompt: chat.to_json().to_string(),
            response: None,
            raw_output: None,
            output: None,
            usage: None,
            cost: None,
            cache_hit: false,
            module_path: callers.iter().map(|c| c.name.clone()).collect(),
            events: Vec::new(),
            error: None,
        })));
        let mut sinks = vec![lm_history.clone(), GLOBAL.clone()];
        for caller in callers {
            if !sinks
                .iter()
                .any(|sink| Arc::ptr_eq(&sink.0, &caller.buffer.0))
            {
                sinks.push(caller.buffer);
            }
        }
        Some(Self {
            record,
            sinks,
            started: Instant::now(),
            finished: false,
        })
    }

    pub(crate) fn finish(mut self, result: &mut anyhow::Result<LMResponse>) {
        {
            let mut entry = self.record.0.lock().unwrap_or_else(|p| p.into_inner());
            match result {
                Ok(response) => {
                    entry.response = response
                        .events
                        .iter()
                        .rev()
                        .find_map(|event| match event {
                            SpanEvent::Exchange { message, .. } => Some(message.clone()),
                            _ => None,
                        })
                        .or_else(|| Some(response.output.clone()));
                    entry.raw_output = entry.response.as_ref().map(Message::content);
                    entry.cache_hit = response.cache_hit;
                    entry.usage = Some(if response.cache_hit {
                        LmUsage::default()
                    } else {
                        response.usage
                    });
                    entry.events = response.events.clone();
                    response.history = Some(self.record.clone());
                }
                Err(error) => {
                    entry.error = Some(SpanError {
                        kind: SpanErrorKind::Lm,
                        message: error.to_string(),
                    })
                }
            }
            entry.duration_us = self.started.elapsed().as_micros() as u64;
        }
        self.finished = true;
        for sink in &self.sinks {
            sink.push(self.record.clone());
        }
    }
}

impl Drop for HistoryCall {
    fn drop(&mut self) {
        if !self.finished {
            {
                let mut entry = self.record.0.lock().unwrap_or_else(|p| p.into_inner());
                entry.duration_us = self.started.elapsed().as_micros() as u64;
                entry.error = Some(SpanError {
                    kind: SpanErrorKind::Cancelled,
                    message: "LM invocation ended before completion".into(),
                });
            }
            for sink in &self.sinks {
                sink.push(self.record.clone());
            }
        }
    }
}
