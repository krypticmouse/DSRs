//! RL rollout export (RFC 0001 §4f): the Agent Lightning / verifiers span
//! convention — one rollout as message lists + reward + per-subcall
//! transitions.
//!
//! A pure projection of the trace: because spans keep full [`Message`]
//! structure (tool-call blocks, reasoning blocks), the export needs no lossy
//! text munging. Each transition's `messages` is the exact rendered prompt
//! ([`Trace::prompt`]: prefix ++ suffix) and `completion` is everything the
//! policy emitted, rebuilt from the span's events
//! ([`Span::completion_messages`]): each `Exchange`'s assistant message
//! verbatim, `ToolRun`s as intervening tool-result context.
//!
//! ```ignore
//! let (result, mut trace) = capture(|| pipeline(input)).await;
//! trace.outcome = Some(TraceOutcome { eval: Some(metric_eval), ..Default::default() });
//! let rollout = trace.to_rl_rollout().expect("eval recorded");
//! writeln!(dataset, "{}", rollout.to_json_line()?)?;   // one JSONL record per rollout
//! ```

use std::collections::BTreeMap;

use anyhow::Result;
use serde::Serialize;

use crate::trace::span::{Span, Trace};
use crate::{LmUsage, Message};

/// One rollout: reward plus per-subcall transitions. Serializes to a single
/// JSON object — the JSONL record RL trainers consume.
#[derive(Debug, Serialize)]
pub struct RlRollout<'a> {
    pub trace_id: &'a str,
    /// The rollout-level reward: `trace.outcome.eval.score`.
    pub reward: f64,
    pub transitions: Vec<RlTransition<'a>>,
    /// Free-form run tags (`trace.meta.tags`).
    pub metadata: &'a BTreeMap<String, String>,
}

/// One policy subcall — a `Predict` invocation as (prompt messages, emitted
/// completion) with its span metadata.
#[derive(Debug, Serialize)]
pub struct RlTransition<'a> {
    /// The optimizable unit this subcall belongs to (`"drafter"`) — the
    /// per-agent credit assignment key.
    pub component: &'a str,
    /// 0-based invocation index of the component within the rollout.
    pub seq: u32,
    /// Full prompt as messages (prefix ++ suffix), provider-agnostic roles.
    pub messages: Vec<Message>,
    /// Everything the policy emitted for this subcall: each `Exchange`'s
    /// assistant message, with `ToolRun`s as intervening tool-result context.
    pub completion: Vec<Message>,
    pub usage: LmUsage,
    /// Model identifier from the span's interned config.
    pub model: &'a str,
}

impl RlRollout<'_> {
    /// Serializes to one JSONL record.
    pub fn to_json_line(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

impl Trace {
    /// Projects this rollout onto the RL export convention. `None` when no
    /// eval was recorded — a rollout without a reward is not trainable.
    ///
    /// Spans whose policy emitted nothing (provider failures, cancelled spans
    /// — no `Exchange` event) are omitted: they contribute no completion to
    /// train on. Parse-failure spans keep their transition: the emitted text
    /// exists even though it did not parse.
    pub fn to_rl_rollout(&self) -> Option<RlRollout<'_>> {
        let reward = self.outcome.as_ref()?.eval.as_ref()?.score;
        let transitions = self
            .spans
            .iter()
            .filter_map(|span| self.rl_transition(span))
            .collect();
        Some(RlRollout {
            trace_id: &self.meta.trace_id,
            reward,
            transitions,
            metadata: &self.meta.tags,
        })
    }

    fn rl_transition<'a>(&'a self, span: &'a Span) -> Option<RlTransition<'a>> {
        let completion = span.completion_messages();
        if completion.is_empty() {
            return None;
        }
        Some(RlTransition {
            component: self.component_name(span.component),
            seq: span.seq,
            messages: self.prompt(span),
            completion,
            usage: span.usage,
            model: &self.models[span.model.0 as usize].config.model,
        })
    }
}
