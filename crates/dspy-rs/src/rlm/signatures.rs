#![cfg(feature = "rlm")]

//! Internal typed signatures for RLM operations.
//!
//! These signatures follow DSPy's "RLM is a Module composed of Predictors" pattern
//! where signatures are types, not runtime objects. Dynamic instructions can be
//! injected via `Predict::instruction_override` at runtime.

/// Action signature for asking the LLM "what should I run next?"
///
/// This replaces direct `lm.prompt()` calls in the RLM loop.
///
/// Note: `rlm = false` disables RLM input auto-injection so we can use
/// BAML-native types like `REPLHistory` without requiring PyO3 conversions.
#[derive(Clone, Debug, crate::Signature)]
#[signature(
    rlm = false,
    system_template = r#"
You are an expert Python programmer operating in a REPL environment.
Your goal is to use the provided variables to produce the required outputs.

Available tools:
- llm_query(prompt: str) -> str: Make an LLM call
- llm_query_batched(prompts: list[str]) -> list[str]: Batch LLM calls
- SUBMIT(**kwargs): Submit final outputs when complete

Rules:
1. Examine variables before processing
2. Use Python to explore and transform data
3. Call SUBMIT(field=value, ...) when ready
4. Each iteration should make progress

Your input fields are:
{% for field in input_fields -%}
{{ loop.index }}. `{{ field.name }}` ({{ field.type_name }}){% if field.description %}: {{ field.description }}{% endif %}
{% endfor %}
Your output fields are:
{% for field in output_fields -%}
{{ loop.index }}. `{{ field.name }}` ({{ field.type_name }}){% if field.description %}: {{ field.description }}{% endif %}
{% endfor %}
Respond with `[[ ## reasoning ## ]]` followed by your step-by-step thinking, then `[[ ## code ## ]]` followed by Python code in a ```python block, then `[[ ## completed ## ]]`.

{{ instruction }}
"#
)]
pub struct RlmActionSig {
    /// Metadata about variables available in REPL
    #[input]
    pub variables_info: String,

    /// Current iteration (e.g., '3/20')
    #[input]
    pub iteration: String,

    /// Previous REPL interactions
    #[input]
    pub repl_history: REPLHistory,

    /// Step-by-step reasoning about what to do next
    #[output]
    pub reasoning: String,

    /// Python code to execute in the REPL
    #[output]
    pub code: String,
}

use std::marker::PhantomData;

use crate::{FieldSpec, OutputFormatContent, Signature, TypeIR};
use super::history::REPLHistory;

// ============================================================================
// RlmExtractSig<S> - Extraction fallback signature
// ============================================================================

/// Input for the extraction fallback signature.
///
/// Used when RLM hits max iterations without SUBMIT. Asks the LLM to directly
/// extract final outputs from the REPL trajectory.
#[derive(Debug, Clone, crate::BamlType)]
pub struct RlmExtractInput {
    /// Metadata about variables available in REPL
    pub variables_info: String,

    /// Previous REPL interactions
    pub repl_history: REPLHistory,
}

/// Static input field specs for RlmExtractSig.
static EXTRACT_INPUT_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "variables_info",
        rust_name: "variables_info",
        description: "Metadata about variables available in REPL",
        type_ir: || TypeIR::string(),
        constraints: &[],
        format: None,
        render: None,
    },
    FieldSpec {
        name: "repl_history",
        rust_name: "repl_history",
        description: "Previous REPL interactions",
        type_ir: || <REPLHistory as crate::baml_bridge::BamlTypeInternal>::baml_type_ir(),
        constraints: &[],
        format: None,
        render: None,
    },
];

/// Extraction fallback signature - generic wrapper over user's signature S.
///
/// When RLM hits max iterations without a successful SUBMIT, this signature
/// asks the LLM to directly extract the final outputs from the REPL trajectory.
///
/// # Design
///
/// This signature is generic because output fields must forward to S's output fields.
/// The derive macro can't express `output_fields() -> S::output_fields()`, so we
/// implement Signature manually.
///
/// # Usage
///
/// ```ignore
/// let extract_result = self.extract.call(RlmExtractInput {
///     variables_info,
///     repl_history: history.render(),
/// }).await?;
/// // extract_result is S::Output directly!
/// ```
// FIXME(rlm-signatures): Consider whether this should use a custom user_template
// to guide the LLM on extraction format.
pub struct RlmExtractSig<S: Signature> {
    _marker: PhantomData<S>,
}

impl<S: Signature> Signature for RlmExtractSig<S> {
    type Input = RlmExtractInput;
    type Output = S::Output; // Delegate to S

    fn instruction() -> &'static str {
        "Extract the final outputs from the REPL trajectory. \
         Use the variables and history to determine the correct values for each output field."
    }

    fn input_fields() -> &'static [FieldSpec] {
        EXTRACT_INPUT_FIELDS
    }

    fn output_fields() -> &'static [FieldSpec] {
        S::output_fields() // Forward to S
    }

    fn output_format_content() -> &'static OutputFormatContent {
        S::output_format_content() // Forward to S
    }

    fn from_parts(input: Self::Input, output: Self::Output) -> Self {
        // We don't actually construct instances of this type - it's just a signature marker
        let _ = (input, output);
        Self {
            _marker: PhantomData,
        }
    }

    fn into_parts(self) -> (Self::Input, Self::Output) {
        // This signature type is only used as a type parameter, never constructed
        unreachable!("RlmExtractSig is a marker type and should not be instantiated")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Signature;

    #[test]
    fn rlm_action_sig_has_correct_input_fields() {
        let input_names: Vec<_> = RlmActionSig::input_fields()
            .iter()
            .map(|f| f.name)
            .collect();
        assert_eq!(input_names, vec!["variables_info", "iteration", "repl_history"]);
    }

    #[test]
    fn rlm_action_sig_has_correct_output_fields() {
        let output_names: Vec<_> = RlmActionSig::output_fields()
            .iter()
            .map(|f| f.name)
            .collect();
        assert_eq!(output_names, vec!["reasoning", "code"]);
    }

    #[test]
    fn rlm_action_sig_input_type_is_generated() {
        // Verify the derive macro generated the Input type correctly
        let _input = RlmActionSigInput {
            variables_info: "test vars".to_string(),
            repl_history: REPLHistory::new(),
            iteration: "1/10".to_string(),
        };
    }

    // ==================== RlmExtractSig Tests ====================

    // A simple test signature to use with RlmExtractSig
    #[derive(Clone, Debug, crate::Signature)]
    struct TestSig {
        #[input(desc = "The question")]
        pub question: String,
        #[output(desc = "The answer")]
        pub answer: String,
    }

    #[test]
    fn rlm_extract_sig_has_correct_input_fields() {
        let input_names: Vec<_> = <RlmExtractSig<TestSig>>::input_fields()
            .iter()
            .map(|f| f.name)
            .collect();
        assert_eq!(input_names, vec!["variables_info", "repl_history"]);
    }

    #[test]
    fn rlm_extract_sig_forwards_output_fields_to_inner_signature() {
        // RlmExtractSig<TestSig> should have TestSig's output fields
        let output_names: Vec<_> = <RlmExtractSig<TestSig>>::output_fields()
            .iter()
            .map(|f| f.name)
            .collect();
        assert_eq!(output_names, vec!["answer"]);
    }

    #[test]
    fn rlm_extract_sig_has_extraction_instruction() {
        let instruction = <RlmExtractSig<TestSig>>::instruction();
        assert!(instruction.contains("Extract"));
        assert!(instruction.contains("REPL trajectory"));
    }

    #[test]
    fn rlm_extract_input_can_be_constructed() {
        let _input = RlmExtractInput {
            variables_info: "test vars".to_string(),
            repl_history: REPLHistory::new(),
        };
    }
}
