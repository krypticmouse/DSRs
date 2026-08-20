//! The unified trace format (RFC 0001): one [`Span`] per `Predict` invocation,
//! one [`Trace`] per rollout.
//!
//! Spans are addressed by the same name the params system uses (fx slot name or
//! facet dotted path), interned per-trace as a [`CompId`]. The wire form is
//! JSONL (see [`Trace::to_jsonl`]); the in-memory form always speaks interned ids.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::utils::hash::{StableHasher, stable_hash_debug};
use crate::{LMConfig, LmUsage, Message};

/// Index of a span within its trace. Dense, assigned in insertion order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpanId(pub u32);

/// Per-trace interned component index. The component *name* is the same string
/// the params system addresses: an fx slot name (`"drafter"`) or a facet dotted
/// path (`"pipeline.rater"`). `Trace::components` maps `CompId` -> name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompId(pub u32);

/// Per-trace interned prompt prefix (system message + demo turns). One entry per
/// distinct (component, candidate) configuration — demos serialize once per
/// trace, not once per span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrefixId(pub u32);

/// Per-trace interned model configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub u32);

pub type JsonMap = serde_json::Map<String, serde_json::Value>;

/// One `Predict` invocation: one rendered prompt in, one parsed output out,
/// with the tool loop's inner structure as ordered events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Span {
    pub id: SpanId,
    /// Who ran. Resolve to the name via [`Trace::component_name`].
    pub component: CompId,
    /// 0-based invocation index of this component within this rollout.
    /// `(component, seq)` is unique per trace — this is what makes loops
    /// addressable and what replay keys on.
    pub seq: u32,
    /// Enclosing span, if this Predict ran inside another span's tool
    /// execution (Predict-in-tool). Best-effort: set from the innermost open
    /// span at begin time; `None` at top level.
    pub parent: Option<SpanId>,

    // ---- request (recorded eagerly at span open) ----
    /// Interned system+demos prefix. `None` when the call had no prefix
    /// (e.g. a chat-level continuation on a caller-owned chat).
    pub prefix: Option<PrefixId>,
    /// The live suffix of the rendered prompt: the user turn (and, for
    /// multi-turn continuations, the full caller-provided history).
    /// `prefix + suffix` reconstructs the exact `Chat` sent.
    pub suffix: Vec<Message>,
    /// Signature input fields as JSON. `None` for continuations where no typed
    /// input exists.
    pub input: Option<JsonMap>,
    pub model: ModelId,
    /// Stable hash over (redacted model config ++ full rendered prompt),
    /// computed at span close. The replay key and determinism check.
    pub request_hash: u64,

    // ---- execution (filled at span close) ----
    /// Ordered inner events: provider round-trips and tool executions.
    /// Always ends with the final `Exchange` on success.
    pub events: Vec<SpanEvent>,
    /// Final assistant text, pre-parse.
    pub raw_output: Option<String>,
    /// Parsed signature output fields as JSON. `None` = call or parse failed.
    pub output: Option<JsonMap>,
    /// Aggregated across all exchanges.
    pub usage: LmUsage,
    pub error: Option<SpanError>,

    // ---- timing ----
    /// Microseconds since UNIX epoch.
    pub started_at_us: u64,
    pub duration_us: u64,

    /// False when any text field was truncated at serialization (§5.4) or the
    /// span was redacted. Replay refuses incomplete spans.
    #[serde(default = "default_true")]
    pub complete: bool,
}

fn default_true() -> bool {
    true
}

/// Ordered events inside a span. Tagged; readers skip unknown tags — this is
/// the streaming seam (RFC 0001 §6).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpanEvent {
    /// One provider round-trip: the full assistant message (text, tool-call
    /// blocks, reasoning blocks) and that round-trip's own usage.
    Exchange { message: Message, usage: LmUsage },
    /// One tool execution between exchanges.
    ToolRun {
        /// Provider tool-call id.
        id: String,
        name: String,
        args: serde_json::Value,
        /// Tool result text as fed back to the model.
        result: String,
        duration_us: u64,
        /// Tool-level failure that was reported back to the model as text.
        error: Option<String>,
    },
    /// Unknown tag from a newer writer; preserved as a placeholder on read,
    /// dropped from the canonical JSONL on re-serialize.
    #[doc(hidden)]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanError {
    pub kind: SpanErrorKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanErrorKind {
    /// Provider/network failure.
    Lm,
    /// Response received but output parsing failed. `raw_output` is still
    /// populated in this case — parse failures are prime reflection material.
    Parse,
    /// A tool execution aborted the loop.
    Tool,
    /// Scope ended while the span was open (task cancelled).
    Cancelled,
}

impl SpanErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lm => "lm",
            Self::Parse => "parse",
            Self::Tool => "tool",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Interned model configuration: [`LMConfig`] minus live state, minus secrets.
/// `api_key` is `#[serde(skip)]` on `LMConfig` already; `base_url` is
/// additionally reduced to origin (scheme+host+port) at intern time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub config: LMConfig,
    /// Stable hash of the redacted config; part of `request_hash`'s preimage.
    pub config_hash: u64,
}

impl ModelEntry {
    /// Builds the redacted, hashable entry from a live config.
    pub fn from_config(config: &LMConfig) -> Self {
        let mut config = config.clone();
        config.api_key = None;
        config.base_url = config.base_url.as_deref().map(url_origin);
        let config_hash = stable_hash_debug(&config);
        Self {
            config,
            config_hash,
        }
    }
}

/// Reduces a URL to `scheme://host[:port]`, stripping userinfo, path, and query
/// (vLLM tokens hide in query strings).
fn url_origin(url: &str) -> String {
    let (scheme, rest) = match url.split_once("://") {
        Some((scheme, rest)) => (scheme, rest),
        None => ("", url),
    };
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(rest);
    // Strip userinfo (`user:pass@host`).
    let host = authority.rsplit('@').next().unwrap_or(authority);
    if scheme.is_empty() {
        host.to_string()
    } else {
        format!("{scheme}://{host}")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixEntry {
    /// System message + demo user/assistant turns, exactly as the predictor
    /// rendered them.
    pub messages: Vec<Message>,
}

/// One rollout: ordered spans plus per-trace intern tables.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    pub meta: TraceMeta,
    /// `CompId` -> component name (fx name / dotted path).
    pub components: Vec<String>,
    /// RFC 0001 §1's reserved join column, parallel to `components`: the
    /// global [`ParamId`](crate::ir::ParamId)s of each component's slots in a
    /// program (per RFC 0002 §3.3). Empty until
    /// [`attach_program`](Trace::attach_program) fills it; a `None` entry is
    /// a component the program has no leaf for (static-lane harnesses leave
    /// the whole column empty). Additive — no format version bump.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param_ids: Vec<Option<Vec<crate::ir::ParamId>>>,
    pub models: Vec<ModelEntry>,
    pub prefixes: Vec<PrefixEntry>,
    /// Insertion order == start order == (for the sequential case) topological
    /// order — the same invariant the old Graph had.
    pub spans: Vec<Span>,
    /// Rollout-level result, filled after the scope closes / metric runs.
    pub outcome: Option<TraceOutcome>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TraceMeta {
    /// Format version. This RFC: 1.
    pub v: u32,
    /// Unique id string, minted at capture start.
    pub trace_id: String,
    pub started_at_us: u64,
    /// Hash of the candidate under evaluation, when the harness recorded one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_hash: Option<u64>,
    /// Rollout input (the example's input fields), when the harness entry
    /// point recorded it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<JsonMap>,
    /// Free-form run tags ("optimizer": "gepa", "gen": "3").
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TraceOutcome {
    /// Rollout output fields (the final prediction, serialized).
    pub output: Option<JsonMap>,
    pub error: Option<String>,
    /// Metric result, if a metric ran over this rollout.
    pub eval: Option<Eval>,
    pub duration_us: u64,
}

/// The metric result type: one score, optional textual feedback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Eval {
    pub score: f64,
    pub feedback: Option<String>,
}

impl Eval {
    /// Score-only result. Sufficient for COPRO and MIPROv2; GEPA requires
    /// [`with_feedback`](Eval::with_feedback).
    pub fn score(score: f64) -> Self {
        Self {
            score,
            feedback: None,
        }
    }

    /// Score plus textual feedback explaining *why* — GEPA's reflection input.
    pub fn with_feedback(score: f64, feedback: impl Into<String>) -> Self {
        Self {
            score,
            feedback: Some(feedback.into()),
        }
    }
}

impl Trace {
    pub fn component_id(&self, name: &str) -> Option<CompId> {
        self.components
            .iter()
            .position(|c| c == name)
            .map(|idx| CompId(idx as u32))
    }

    pub fn component_name(&self, id: CompId) -> &str {
        &self.components[id.0 as usize]
    }

    /// Every invocation of one component, in order — GEPA's `pred_trace`.
    pub fn for_component<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Span> + 'a {
        let id = self.component_id(name);
        self.spans
            .iter()
            .filter(move |span| Some(span.component) == id)
    }

    /// Reconstructs the exact rendered prompt of a span (prefix ++ suffix).
    pub fn prompt(&self, span: &Span) -> Vec<Message> {
        let prefix = span
            .prefix
            .map(|id| self.prefixes[id.0 as usize].messages.as_slice())
            .unwrap_or(&[]);
        let mut messages = Vec::with_capacity(prefix.len() + span.suffix.len());
        messages.extend(prefix.iter().cloned());
        messages.extend(span.suffix.iter().cloned());
        messages
    }

    /// Resolves a span's model config.
    pub fn model(&self, span: &Span) -> &LMConfig {
        &self.models[span.model.0 as usize].config
    }

    /// Spans that completed with parsed output.
    pub fn successes(&self) -> impl Iterator<Item = &'_ Span> + '_ {
        self.spans.iter().filter(|span| span.output.is_some())
    }

    /// Fills the [`param_ids`](Trace::param_ids) join column against a
    /// program (RFC 0002 §3.3): for each interned component whose name is one
    /// of the program's leaf names, the entry becomes that leaf's node-owned
    /// [`ParamId`](crate::ir::ParamId)s in id order — the same entities
    /// `Program::param_id("<leaf>.<slot>")` addresses. Components the program
    /// doesn't know (static-lane names, judge calls) get `None`.
    ///
    /// One addressing story: a span's `component` string == the leaf name ==
    /// the `ParamPath` prefix, so after attaching, spans join to optimizable
    /// slots without string surgery.
    pub fn attach_program(&mut self, program: &crate::ir::Program) {
        let mut by_leaf: std::collections::HashMap<&str, Vec<crate::ir::ParamId>> =
            std::collections::HashMap::new();
        for (id, slot) in program.params.iter() {
            if let crate::ir::ParamOwner::Node(node) = slot.owner
                && let Some(leaf) = program.leaf_name(node)
            {
                by_leaf.entry(leaf).or_default().push(id);
            }
        }
        self.param_ids = self
            .components
            .iter()
            .map(|name| by_leaf.get(name.as_str()).cloned())
            .collect();
    }

}

impl Span {
    /// Rebuilds everything the policy emitted for this span from its events:
    /// each `Exchange`'s assistant message verbatim, with consecutive
    /// `ToolRun`s batched into a single user-role tool-result turn — the same
    /// shape the live tool loop feeds back to the model.
    ///
    /// Appending this to [`Trace::prompt`] reconstructs the full conversation.
    /// Used by replay to rebuild the returned chat and by the RL export's
    /// `completion` field.
    pub fn completion_messages(&self) -> Vec<Message> {
        use rig::OneOrMany;
        use rig::message::UserContent;

        fn flush(out: &mut Vec<Message>, pending: &mut Vec<UserContent>) {
            if pending.is_empty() {
                return;
            }
            let contents = std::mem::take(pending);
            let rig_msg = rig::message::Message::User {
                content: OneOrMany::many(contents)
                    .expect("flush is only called with a non-empty batch"),
            };
            out.push(Message::from(rig_msg));
        }

        let mut out = Vec::new();
        let mut pending: Vec<UserContent> = Vec::new();
        for event in &self.events {
            match event {
                SpanEvent::Exchange { message, .. } => {
                    flush(&mut out, &mut pending);
                    out.push(message.clone());
                }
                SpanEvent::ToolRun { id, result, .. } => {
                    pending.push(UserContent::tool_result(
                        id.clone(),
                        OneOrMany::one(result.clone().into()),
                    ));
                }
                _ => {}
            }
        }
        flush(&mut out, &mut pending);
        out
    }
}

/// Computes a span's `request_hash`: stable hash over the redacted model config
/// hash and the full rendered prompt (prefix ++ suffix), streamed through the
/// messages' `Debug` representation.
pub(crate) fn request_hash(config_hash: u64, prefix: &[Message], suffix: &[Message]) -> u64 {
    use crate::utils::hash::HashWriter;
    use std::fmt::Write as _;
    use std::hash::Hasher as _;

    let mut hasher = StableHasher::new();
    hasher.write(&config_hash.to_le_bytes());
    let mut writer = HashWriter(&mut hasher);
    for message in prefix.iter().chain(suffix) {
        let _ = write!(writer, "{message:?}");
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_origin_strips_secrets_and_paths() {
        assert_eq!(
            url_origin("https://user:pass@vllm.internal:8000/v1?api_key=sk-123"),
            "https://vllm.internal:8000"
        );
        assert_eq!(url_origin("http://localhost:8080"), "http://localhost:8080");
        assert_eq!(url_origin("localhost:8080/v1"), "localhost:8080");
    }

    #[test]
    fn model_entry_redacts_api_key() {
        let config = LMConfig {
            api_key: Some("sk-secret".to_string()),
            base_url: Some("https://token@host/path".to_string()),
            ..LMConfig::default()
        };
        let entry = ModelEntry::from_config(&config);
        assert_eq!(entry.config.api_key, None);
        assert_eq!(entry.config.base_url.as_deref(), Some("https://host"));
    }

    #[test]
    fn request_hash_changes_with_prompt_and_config() {
        let prefix = vec![Message::system("sys")];
        let suffix = vec![Message::user("hi")];
        let base = request_hash(1, &prefix, &suffix);
        assert_eq!(base, request_hash(1, &prefix, &suffix));
        assert_ne!(base, request_hash(2, &prefix, &suffix));
        assert_ne!(base, request_hash(1, &prefix, &[Message::user("bye")]));
    }
}
