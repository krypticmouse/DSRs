use std::ops::Deref;

use indexmap::IndexMap;
use rig::message::ToolCall;

use crate::{Flag, LmUsage};

/// Per-field details from parsing an LM response.
///
/// Each output field gets a `FieldMeta` recording the raw text the LM produced for that
/// field, any flags raised during parsing, and the results of constraint checks.
#[derive(Debug, Clone)]
pub struct FieldMeta {
    /// The raw text the LM produced for this field, before coercion.
    pub raw_text: String,
    /// Flags raised during parsing (e.g. jsonish coercion warnings).
    pub flags: Vec<Flag>,
    /// Results of `#[check(...)]` and `#[assert(...)]` constraints on this field.
    pub checks: Vec<ConstraintResult>,
}

/// Outcome of evaluating a single constraint on a field value.
#[derive(Debug, Clone)]
pub struct ConstraintResult {
    /// The constraint's label (from `#[check("label", ...)]`).
    pub label: String,
    /// The constraint expression that was evaluated.
    pub expression: String,
    /// Whether the constraint passed.
    pub passed: bool,
}

/// Runtime bookkeeping from a single LM call — what happened, not what was asked.
///
/// Carried by [`Predicted`] alongside the typed output. None of this enters any prompt.
/// Token counts, the raw response text, tool invocations, and per-field parse details
/// all live here.
///
/// ```
/// use dspy_rs::CallMetadata;
///
/// let meta = CallMetadata::default();
/// assert_eq!(meta.lm_usage.total_tokens, 0);
/// assert!(!meta.has_failed_checks());
/// ```
#[derive(Debug, Clone)]
pub struct CallMetadata {
    /// The full text the LM returned, before any parsing.
    pub raw_response: String,
    /// Token usage for this call (prompt, completion, total).
    pub lm_usage: LmUsage,
    /// Tool calls the LM requested during this invocation.
    pub tool_calls: Vec<ToolCall>,
    /// Results from executing tool calls.
    pub tool_executions: Vec<String>,
    /// Trace node ID, if tracing is active.
    pub node_id: Option<usize>,
    /// Per-field parse details, keyed by field name.
    pub field_meta: IndexMap<String, FieldMeta>,
}

impl Default for CallMetadata {
    fn default() -> Self {
        Self {
            raw_response: String::new(),
            lm_usage: LmUsage::default(),
            tool_calls: Vec::new(),
            tool_executions: Vec::new(),
            node_id: None,
            field_meta: IndexMap::new(),
        }
    }
}

impl CallMetadata {
    pub fn new(
        raw_response: String,
        lm_usage: LmUsage,
        tool_calls: Vec<ToolCall>,
        tool_executions: Vec<String>,
        node_id: Option<usize>,
        field_meta: IndexMap<String, FieldMeta>,
    ) -> Self {
        Self {
            raw_response,
            lm_usage,
            tool_calls,
            tool_executions,
            node_id,
            field_meta,
        }
    }

    pub fn field_meta(&self) -> &IndexMap<String, FieldMeta> {
        &self.field_meta
    }

    pub fn field_flags(&self, field: &str) -> &[Flag] {
        self.field_meta
            .get(field)
            .map(|meta| meta.flags.as_slice())
            .unwrap_or(&[])
    }

    pub fn field_checks(&self, field: &str) -> &[ConstraintResult] {
        self.field_meta
            .get(field)
            .map(|meta| meta.checks.as_slice())
            .unwrap_or(&[])
    }

    pub fn field_raw(&self, field: &str) -> Option<&str> {
        self.field_meta
            .get(field)
            .map(|meta| meta.raw_text.as_str())
    }

    pub fn field_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.field_meta.keys().map(|name| name.as_str())
    }

    pub fn has_failed_checks(&self) -> bool {
        self.field_meta
            .values()
            .flat_map(|meta| &meta.checks)
            .any(|check| !check.passed)
    }
}

/// Typed output paired with call metadata from a module invocation.
///
/// Two channels of information come back from every [`Module::call`](crate::Module::call):
///
/// 1. **The output `O`** — fields the LM actually produced, shaped by the signature.
///    For `Predict<QA>`: `QAOutput { answer }`. For `ChainOfThought<QA>`:
///    `WithReasoning<QAOutput>` (reasoning is a real prompt field the LM generates).
///
/// 2. **[`CallMetadata`]** — runtime bookkeeping. Token counts, raw response text,
///    tool call records, per-field constraint results. Never enters any prompt.
///
/// `Predicted` derefs to `O`, so output fields are directly accessible: `result.answer`.
/// Metadata is separate: `result.metadata()`.
///
/// This distinction matters for module authors: if your module changes what the LM is
/// asked to produce (like adding `reasoning`), change `Output`. If it just selects or
/// transforms results (like `BestOfN` picking the best of N attempts), keep the same
/// `Output` — selection info is metadata, not a prompt field.
///
/// Note: [`CallMetadata`] is a fixed struct, not an extensible bag. There's currently no
/// mechanism for modules to attach custom metadata (e.g. "which attempt won"). Known
/// limitation.
///
/// ```
/// use dspy_rs::{Predicted, CallMetadata};
///
/// #[derive(Debug)]
/// struct QAOutput { answer: String }
///
/// let result = Predicted::new(
///     QAOutput { answer: "42".into() },
///     CallMetadata::default(),
/// );
/// assert_eq!(result.answer, "42");             // output field via Deref
/// let _usage = &result.metadata().lm_usage;    // runtime info, never in prompts
/// let (output, meta) = result.into_parts();    // decompose for ownership
/// assert_eq!(output.answer, "42");
/// ```
#[derive(Debug, Clone)]
pub struct Predicted<O> {
    output: O,
    metadata: CallMetadata,
}

impl<O> Predicted<O> {
    /// Creates a new `Predicted` from an output value and call metadata.
    pub fn new(output: O, metadata: CallMetadata) -> Self {
        Self { output, metadata }
    }

    /// Returns the call metadata (raw response, token usage, tool calls, field-level details).
    pub fn metadata(&self) -> &CallMetadata {
        &self.metadata
    }

    /// Unwraps the typed output, discarding metadata.
    pub fn into_inner(self) -> O {
        self.output
    }

    /// Splits into the typed output and call metadata.
    pub fn into_parts(self) -> (O, CallMetadata) {
        (self.output, self.metadata)
    }
}

impl<O> Deref for Predicted<O> {
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.output
    }
}
