use crate::{BamlValue, RawExample};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Rich evaluation metric pairing a numerical score with textual feedback.
///
/// Used by [`GEPA`](crate::GEPA) to guide evolutionary instruction search. The
/// `feedback` string is appended to candidate instructions during mutation, so
/// it should explain *why* the score is what it is — not just restate the score.
///
/// Good feedback: "The answer correctly identifies the capital but misspells 'Canberra'"
/// Bad feedback: "Score: 0.5"
///
/// // TODO(vector-feedback): `score` should be `Vec<f32>` (or a named score vector)
/// // so metrics can express multi-dimensional quality (accuracy, fluency, brevity, etc.)
/// // and the Pareto frontier can operate on the full vector instead of a scalar collapse.
///
/// ```
/// use dspy_rs::FeedbackMetric;
///
/// let fb = FeedbackMetric::new(0.7, "Correct answer but verbose explanation");
/// assert_eq!(fb.score, 0.7);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeedbackMetric {
    /// Numerical score (typically 0.0 to 1.0, but can be any range)
    pub score: f32,

    /// Rich textual feedback explaining the score.
    pub feedback: String,

    /// Optional structured metadata for additional context.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl FeedbackMetric {
    pub fn new(score: f32, feedback: impl Into<String>) -> Self {
        Self {
            score,
            feedback: feedback.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(
        score: f32,
        feedback: impl Into<String>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            score,
            feedback: feedback.into(),
            metadata,
        }
    }

    pub fn add_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl Default for FeedbackMetric {
    fn default() -> Self {
        Self {
            score: 0.0,
            feedback: String::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Execution trace capturing inputs, outputs, feedback, and errors from a single run.
///
/// Used internally by optimizers to record what happened during evaluation. The
/// [`format_for_reflection`](ExecutionTrace::format_for_reflection) method produces a
/// human-readable summary suitable for including in LM prompts (e.g. for GEPA's
/// feedback-driven mutation).
///
/// Not related to the [`trace`](crate::trace) module's computation graph — this is
/// a flat record of one evaluation, not a DAG of LM calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub inputs: RawExample,
    pub outputs: Option<BamlValue>,
    pub feedback: Option<FeedbackMetric>,
    pub intermediate_steps: Vec<(String, serde_json::Value)>,
    pub errors: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionTrace {
    /// Creates a trace with just inputs and outputs, no feedback or errors.
    pub fn simple(inputs: RawExample, outputs: BamlValue) -> Self {
        Self {
            inputs,
            outputs: Some(outputs),
            feedback: None,
            intermediate_steps: Vec::new(),
            errors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn builder(inputs: RawExample) -> ExecutionTraceBuilder {
        ExecutionTraceBuilder::new(inputs)
    }

    pub fn with_feedback(mut self, feedback: FeedbackMetric) -> Self {
        self.feedback = Some(feedback);
        self
    }

    /// Returns `true` if the execution produced output and had no errors.
    pub fn is_successful(&self) -> bool {
        self.outputs.is_some() && self.errors.is_empty()
    }

    pub fn score(&self) -> Option<f32> {
        self.feedback.as_ref().map(|f| f.score)
    }

    /// Formats the trace as a human-readable string for LM prompt inclusion.
    ///
    /// Includes inputs, execution steps, outputs, errors, and feedback score.
    /// Suitable for appending to optimization prompts where the LM needs to
    /// understand what happened in a previous evaluation.
    pub fn format_for_reflection(&self) -> String {
        let mut result = String::new();

        result.push_str("Input:\n");
        result.push_str(&format!("{:?}\n\n", self.inputs));

        if !self.intermediate_steps.is_empty() {
            result.push_str("Execution Steps:\n");
            for (i, (step_name, output)) in self.intermediate_steps.iter().enumerate() {
                result.push_str(&format!("{}. {}: {:?}\n", i + 1, step_name, output));
            }
            result.push('\n');
        }

        if let Some(ref outputs) = self.outputs {
            result.push_str("Output:\n");
            result.push_str(&format!("{:?}\n\n", outputs));
        }

        if !self.errors.is_empty() {
            result.push_str("Errors:\n");
            for error in &self.errors {
                result.push_str(&format!("- {}\n", error));
            }
            result.push('\n');
        }

        if let Some(ref feedback) = self.feedback {
            result.push_str("Evaluation:\n");
            result.push_str(&format!("Score: {:.3}\n", feedback.score));
            result.push_str(&format!("Feedback: {}\n", feedback.feedback));
        }

        result
    }
}

pub struct ExecutionTraceBuilder {
    trace: ExecutionTrace,
}

impl ExecutionTraceBuilder {
    pub fn new(inputs: RawExample) -> Self {
        Self {
            trace: ExecutionTrace {
                inputs,
                outputs: None,
                feedback: None,
                intermediate_steps: Vec::new(),
                errors: Vec::new(),
                metadata: HashMap::new(),
            },
        }
    }

    pub fn outputs(mut self, outputs: BamlValue) -> Self {
        self.trace.outputs = Some(outputs);
        self
    }

    pub fn feedback(mut self, feedback: FeedbackMetric) -> Self {
        self.trace.feedback = Some(feedback);
        self
    }

    pub fn add_step(mut self, name: impl Into<String>, output: serde_json::Value) -> Self {
        self.trace.intermediate_steps.push((name.into(), output));
        self
    }

    pub fn add_error(mut self, error: impl Into<String>) -> Self {
        self.trace.errors.push(error.into());
        self
    }

    pub fn add_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.trace.metadata.insert(key.into(), value);
        self
    }

    pub fn build(self) -> ExecutionTrace {
        self.trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_feedback_metric_creation() {
        let feedback = FeedbackMetric::new(0.8, "Good result");
        assert_eq!(feedback.score, 0.8);
        assert_eq!(feedback.feedback, "Good result");
        assert!(feedback.metadata.is_empty());
    }

    #[test]
    fn test_feedback_metric_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("tokens".to_string(), json!(120));
        let feedback = FeedbackMetric::with_metadata(0.9, "Great", meta.clone());
        assert_eq!(feedback.score, 0.9);
        assert_eq!(feedback.feedback, "Great");
        assert_eq!(feedback.metadata, meta);
    }

    #[test]
    fn test_execution_trace_builder() {
        let inputs = RawExample::new(
            [("question".to_string(), json!("What is 2+2?"))].into(),
            vec!["question".to_string()],
            vec![],
        );

        let trace = ExecutionTrace::builder(inputs)
            .outputs(BamlValue::String("4".to_string()))
            .feedback(FeedbackMetric::new(1.0, "Correct"))
            .add_step("model_call", json!({"latency_ms": 42}))
            .build();

        assert!(trace.is_successful());
        assert_eq!(trace.score(), Some(1.0));
        assert_eq!(trace.intermediate_steps.len(), 1);
    }
}
