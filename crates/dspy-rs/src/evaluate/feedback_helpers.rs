/// Helper functions for creating rich feedback [`Eval`]s
///
/// This module provides utilities for common feedback patterns in different domains:
/// - Document retrieval (precision, recall, F1)
/// - Code generation (compilation, execution, testing)
/// - Multi-objective evaluation
/// - String similarity and classification
use super::Eval;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Retrieval Feedback Helpers
// ============================================================================

/// Create feedback for document retrieval tasks
///
/// # Arguments
/// * `retrieved` - Documents retrieved by the system
/// * `expected` - Expected/gold documents
/// * `context_docs` - Optional list of all available documents for context
///
/// # Example Feedback
/// ```text
/// Retrieved 3/5 correct documents (Precision: 0.6, Recall: 0.6, F1: 0.6)
/// Correctly retrieved: doc1, doc2, doc3
/// Missed: doc4, doc5
/// Incorrectly retrieved: doc6, doc7
/// ```
pub fn retrieval_feedback(
    retrieved: &[impl AsRef<str>],
    expected: &[impl AsRef<str>],
    context_docs: Option<&[impl AsRef<str>]>,
) -> Eval {
    let retrieved_set: HashSet<String> = retrieved.iter().map(|s| s.as_ref().to_string()).collect();

    let expected_set: HashSet<String> = expected.iter().map(|s| s.as_ref().to_string()).collect();

    let correct: Vec<String> = retrieved_set.intersection(&expected_set).cloned().collect();

    let missed: Vec<String> = expected_set.difference(&retrieved_set).cloned().collect();

    let incorrect: Vec<String> = retrieved_set.difference(&expected_set).cloned().collect();

    let precision = if retrieved.is_empty() {
        0.0
    } else {
        correct.len() as f64 / retrieved.len() as f64
    };

    let recall = if expected.is_empty() {
        1.0
    } else {
        correct.len() as f64 / expected.len() as f64
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    let mut feedback = format!(
        "Retrieved {}/{} correct documents (Precision: {:.3}, Recall: {:.3}, F1: {:.3})\n",
        correct.len(),
        expected.len(),
        precision,
        recall,
        f1
    );

    if !correct.is_empty() {
        feedback.push_str(&format!("Correctly retrieved: {}\n", correct.join(", ")));
    }

    if !missed.is_empty() {
        feedback.push_str(&format!("Missed: {}\n", missed.join(", ")));
    }

    if !incorrect.is_empty() {
        feedback.push_str(&format!(
            "Incorrectly retrieved: {}\n",
            incorrect.join(", ")
        ));
    }

    if let Some(docs) = context_docs {
        feedback.push_str(&format!("Total available documents: {}\n", docs.len()));
    }

    Eval::with_feedback(f1, feedback)
}

// ============================================================================
// Code Generation Feedback Helpers
// ============================================================================

/// Stage in code execution pipeline
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodeStage {
    Parse,
    Compile,
    Execute,
    Test,
}

impl std::fmt::Display for CodeStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeStage::Parse => write!(f, "Parse"),
            CodeStage::Compile => write!(f, "Compile"),
            CodeStage::Execute => write!(f, "Execute"),
            CodeStage::Test => write!(f, "Test"),
        }
    }
}

/// Result of a code stage
#[derive(Debug, Clone)]
pub enum StageResult {
    Success,
    Failure { error: String },
}

/// Create feedback for code generation pipelines
///
/// # Arguments
/// * `stages` - List of (stage, result) tuples showing pipeline progression
/// * `final_score` - Overall score (0.0 to 1.0)
///
/// # Example Feedback
/// ```text
/// Parse: Success
/// Compile: Success
/// Execute: RuntimeError: division by zero on line 10
/// ```
pub fn code_pipeline_feedback(stages: &[(CodeStage, StageResult)], final_score: f64) -> Eval {
    let mut feedback = String::new();

    for (stage, result) in stages {
        match result {
            StageResult::Success => {
                feedback.push_str(&format!("{}: Success\n", stage));
            }
            StageResult::Failure { error } => {
                feedback.push_str(&format!("{}: {}\n", stage, error));
                feedback.push_str(&format!("Failed at stage: {}\n", stage));
                break; // Stop at first failure
            }
        }
    }

    Eval::with_feedback(final_score, feedback)
}

// ============================================================================
// Multi-Objective Feedback Helpers
// ============================================================================

/// Create feedback for multi-objective optimization
///
/// # Arguments
/// * `objectives` - Map of objective name to (score, feedback) pairs
/// * `weights` - Optional weights for aggregating objectives
///
/// # Example Feedback
/// ```text
/// [Correctness] Score: 0.9 - Output matches expected format
/// [Latency] Score: 0.7 - Response took 450ms (target: <300ms)
/// [Privacy] Score: 1.0 - No PII detected in output
/// Overall: 0.87 (weighted average)
/// ```
pub fn multi_objective_feedback(
    objectives: &HashMap<String, (f64, String)>,
    weights: Option<&HashMap<String, f64>>,
) -> Eval {
    let mut feedback = String::new();

    let mut total_score = 0.0;
    let mut total_weight = 0.0;

    let mut objective_names: Vec<_> = objectives.keys().collect();
    objective_names.sort();

    for name in objective_names {
        if let Some((score, obj_feedback)) = objectives.get(name.as_str()) {
            let weight = weights
                .and_then(|w| w.get(name.as_str()))
                .copied()
                .unwrap_or(1.0);

            feedback.push_str(&format!(
                "[{}] Score: {:.3} - {}\n",
                name, score, obj_feedback
            ));

            total_score += score * weight;
            total_weight += weight;
        }
    }

    let aggregate_score = if total_weight > 0.0 {
        total_score / total_weight
    } else {
        0.0
    };

    feedback.push_str(&format!(
        "\nOverall: {:.3} (weighted average)",
        aggregate_score
    ));

    Eval::with_feedback(aggregate_score, feedback)
}

// ============================================================================
// String Similarity Feedback
// ============================================================================

/// Create feedback for string similarity tasks
///
/// Uses simple word-level comparison to provide actionable feedback
pub fn string_similarity_feedback(predicted: &str, expected: &str) -> Eval {
    let exact_match = predicted.trim() == expected.trim();

    if exact_match {
        return Eval::with_feedback(1.0, "Exact match");
    }

    let pred_lower = predicted.to_lowercase();
    let exp_lower = expected.to_lowercase();

    if pred_lower == exp_lower {
        return Eval::with_feedback(0.95, "Match ignoring case (minor formatting difference)");
    }

    // Word-level comparison
    let pred_words: HashSet<&str> = pred_lower.split_whitespace().collect();
    let exp_words: HashSet<&str> = exp_lower.split_whitespace().collect();

    let common_words: HashSet<_> = pred_words.intersection(&exp_words).collect();
    let missing_words: Vec<_> = exp_words.difference(&pred_words).collect();
    let extra_words: Vec<_> = pred_words.difference(&exp_words).collect();

    let recall = if !exp_words.is_empty() {
        common_words.len() as f64 / exp_words.len() as f64
    } else {
        1.0
    };

    let precision = if !pred_words.is_empty() {
        common_words.len() as f64 / pred_words.len() as f64
    } else {
        0.0
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    let mut feedback = format!("Partial match (F1: {:.3})\n", f1);
    feedback.push_str(&format!("Expected: \"{}\"\n", expected));
    feedback.push_str(&format!("Predicted: \"{}\"\n", predicted));

    if !missing_words.is_empty() {
        feedback.push_str(&format!(
            "Missing words: {}\n",
            missing_words
                .iter()
                .map(|w| format!("\"{}\"", w))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if !extra_words.is_empty() {
        feedback.push_str(&format!(
            "Extra words: {}\n",
            extra_words
                .iter()
                .map(|w| format!("\"{}\"", w))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    Eval::with_feedback(f1, feedback)
}

// ============================================================================
// Classification Feedback
// ============================================================================

/// Create feedback for classification tasks
pub fn classification_feedback(
    predicted_class: &str,
    expected_class: &str,
    confidence: Option<f64>,
) -> Eval {
    let correct = predicted_class == expected_class;
    let score = if correct { 1.0 } else { 0.0 };

    let mut feedback = if correct {
        format!("Correct classification: \"{}\"", predicted_class)
    } else {
        format!(
            "Incorrect classification\n  Expected: \"{}\"\n  Predicted: \"{}\"",
            expected_class, predicted_class
        )
    };

    if let Some(conf) = confidence {
        feedback.push_str(&format!("\n  Confidence: {:.3}", conf));
    }

    Eval::with_feedback(score, feedback)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feedback_text(eval: &Eval) -> &str {
        eval.feedback.as_deref().unwrap_or("")
    }

    #[test]
    fn test_retrieval_feedback_perfect() {
        let retrieved = vec!["doc1", "doc2", "doc3"];
        let expected = vec!["doc1", "doc2", "doc3"];

        let eval = retrieval_feedback(&retrieved, &expected, None::<&[&str]>);
        assert_eq!(eval.score, 1.0);
        assert!(feedback_text(&eval).contains("3/3"));
    }

    #[test]
    fn test_retrieval_feedback_partial() {
        let retrieved = vec!["doc1", "doc2", "doc4"];
        let expected = vec!["doc1", "doc2", "doc3"];

        let eval = retrieval_feedback(&retrieved, &expected, None::<&[&str]>);
        assert!(eval.score < 1.0 && eval.score > 0.0);
        assert!(feedback_text(&eval).contains("Missed: doc3"));
        assert!(feedback_text(&eval).contains("Incorrectly retrieved: doc4"));
    }

    #[test]
    fn test_code_pipeline_feedback() {
        let stages = vec![
            (CodeStage::Parse, StageResult::Success),
            (CodeStage::Compile, StageResult::Success),
            (
                CodeStage::Execute,
                StageResult::Failure {
                    error: "Division by zero".to_string(),
                },
            ),
        ];

        let eval = code_pipeline_feedback(&stages, 0.6);
        assert!(feedback_text(&eval).contains("Parse"));
        assert!(feedback_text(&eval).contains("Compile"));
        assert!(feedback_text(&eval).contains("Execute"));
        assert_eq!(eval.score, 0.6);
    }

    #[test]
    fn test_multi_objective_feedback() {
        let mut objectives = HashMap::new();
        objectives.insert("accuracy".to_string(), (0.9, "Good accuracy".to_string()));
        objectives.insert("latency".to_string(), (0.7, "Slow response".to_string()));

        let eval = multi_objective_feedback(&objectives, None);
        assert!(feedback_text(&eval).contains("[accuracy]"));
        assert!(feedback_text(&eval).contains("[latency]"));
        assert!((eval.score - 0.8).abs() < 0.01); // Average of 0.9 and 0.7
    }

    #[test]
    fn test_string_similarity_exact() {
        let eval = string_similarity_feedback("hello world", "hello world");
        assert_eq!(eval.score, 1.0);
    }

    #[test]
    fn test_string_similarity_case() {
        let eval = string_similarity_feedback("Hello World", "hello world");
        assert_eq!(eval.score, 0.95);
    }

    #[test]
    fn test_classification_feedback() {
        let eval = classification_feedback("positive", "positive", Some(0.95));
        assert_eq!(eval.score, 1.0);
        assert!(feedback_text(&eval).contains("Correct"));

        let eval = classification_feedback("negative", "positive", Some(0.85));
        assert_eq!(eval.score, 0.0);
        assert!(feedback_text(&eval).contains("Incorrect"));
    }
}
