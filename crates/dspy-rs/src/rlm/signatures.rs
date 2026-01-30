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
#[signature(rlm = false)]
pub struct RlmActionSig {
    /// Metadata about the variables available in the REPL.
    #[input]
    pub variables_info: String,

    /// Previous REPL code executions and their outputs.
    #[input]
    pub repl_history: REPLHistory,

    /// Current iteration number (1-indexed) out of max_iterations.
    #[input]
    pub iteration: String,

    /// Think step-by-step: what do you know? What remains? Plan your next action.
    #[output]
    pub reasoning: String,

    /// Python code to execute. Use markdown code block format: ```python\n<code>\n```.
    #[output]
    pub code: String,
}

use crate::{FieldSpec, OutputFormatContent, Signature, TypeIR};

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
        style: None,
        renderer: None,
        render_settings: None,
    },
    FieldSpec {
        name: "repl_history",
        rust_name: "repl_history",
        description: "Previous REPL interactions",
        type_ir: || <REPLHistory as crate::baml_bridge::BamlTypeInternal>::baml_type_ir(),
        constraints: &[],
        style: None,
        renderer: None,
        render_settings: None,
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
///     repl_history: history.clone(),
/// }).await?;
/// let (_, output) = extract_result.output.into_parts();
/// ```
pub struct RlmExtractSig<S: Signature> {
    input: RlmExtractInput,
    output: S::Output,
}

impl<S: Signature> Clone for RlmExtractSig<S>
where
    S::Output: Clone,
{
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            output: self.output.clone(),
        }
    }
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
        Self { input, output }
    }

    fn into_parts(self) -> (Self::Input, Self::Output) {
        (self.input, self.output)
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
        assert_eq!(input_names, vec!["variables_info", "repl_history", "iteration"]);
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
        /// The question.
        #[input]
        pub question: String,
        /// The answer.
        #[output]
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
