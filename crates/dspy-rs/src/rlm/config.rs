#![cfg(feature = "rlm")]

use indexmap::IndexMap;

use crate::{ConstraintKind, ConstraintResult, FieldMeta, Signature};

/// Configuration for TypedRlm execution.
#[derive(Debug, Clone)]
pub struct RlmConfig {
    /// Maximum REPL iterations before extraction fallback.
    pub max_iterations: usize,
    /// Maximum sub-LLM calls allowed.
    pub max_llm_calls: usize,
    /// Whether to attempt extraction on max iterations (vs error).
    pub enable_extraction_fallback: bool,
    /// Whether assertion failures are fatal (vs allowing retry).
    pub strict_assertions: bool,
    /// Maximum characters to include from REPL output.
    pub max_output_chars: usize,
    /// Maximum characters to include when rendering REPL history in prompts.
    pub max_history_output_chars: usize,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            max_llm_calls: 50,
            enable_extraction_fallback: true,
            strict_assertions: true,
            max_output_chars: 100_000,
            max_history_output_chars: 5_000,
        }
    }
}

/// Summary of constraint outcomes captured during execution.
#[derive(Debug, Default, Clone)]
pub struct ConstraintSummary {
    pub checks_passed: usize,
    pub checks_failed: usize,
    pub assertions_passed: usize,
}

impl ConstraintSummary {
    pub fn from_field_metas(field_metas: &IndexMap<String, FieldMeta>) -> Self {
        let mut summary = Self::default();
        for meta in field_metas.values() {
            for check in &meta.checks {
                if check.passed {
                    summary.checks_passed += 1;
                } else {
                    summary.checks_failed += 1;
                }
            }
        }
        summary
    }
}

/// Result of a TypedRlm execution.
#[derive(Debug, Clone)]
pub struct RlmResult<S: Signature> {
    /// The typed input (preserved from call).
    pub input: S::Input,
    /// The typed output.
    pub output: S::Output,
    /// Per-field metadata (flags, constraint checks).
    pub field_metas: IndexMap<String, FieldMeta>,
    /// Number of REPL iterations used.
    pub iterations: usize,
    /// Number of sub-LLM calls made.
    pub llm_calls: usize,
    /// Whether output was obtained via extraction fallback.
    pub extraction_fallback: bool,
    /// Summary of constraint results.
    pub constraint_summary: ConstraintSummary,
}

impl<S: Signature> RlmResult<S> {
    pub fn new(
        input: S::Input,
        output: S::Output,
        field_metas: IndexMap<String, FieldMeta>,
        iterations: usize,
        llm_calls: usize,
        extraction_fallback: bool,
    ) -> Self {
        let mut summary = ConstraintSummary::from_field_metas(&field_metas);
        summary.assertions_passed = count_assertions::<S>();
        Self {
            input,
            output,
            field_metas,
            iterations,
            llm_calls,
            extraction_fallback,
            constraint_summary: summary,
        }
    }

    /// Reconstruct the full signature struct if needed.
    pub fn to_signature(&self) -> S
    where
        S: Clone,
        S::Input: Clone,
        S::Output: Clone,
    {
        S::from_parts(self.input.clone(), self.output.clone())
    }

    /// Get all failed soft checks.
    pub fn failed_checks(&self) -> Vec<&ConstraintResult> {
        self.field_metas
            .values()
            .flat_map(|meta| &meta.checks)
            .filter(|check| !check.passed)
            .collect()
    }

    /// Whether any soft constraints failed.
    pub fn has_constraint_warnings(&self) -> bool {
        !self.failed_checks().is_empty()
    }

    /// Whether this was a fallback extraction.
    pub fn is_fallback(&self) -> bool {
        self.extraction_fallback
    }
}

fn count_assertions<S: Signature>() -> usize {
    S::output_fields()
        .iter()
        .map(|field| {
            field
                .constraints
                .iter()
                .filter(|constraint| constraint.kind == ConstraintKind::Assert)
                .count()
        })
        .sum()
}
