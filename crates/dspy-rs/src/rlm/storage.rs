#![cfg(feature = "rlm")]

//! Storage types for serializing RLM execution results.
//!
//! This module provides [`StorableRlmResult`] which captures all relevant
//! execution data in a serializable format suitable for storage and later analysis.

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

use super::config::{ConstraintSummary, RlmResult};
use super::history::REPLHistory;
use crate::{ConstraintResult, Signature};

/// A serializable representation of an RLM execution result.
///
/// Unlike [`RlmResult`], this type stores input/output as JSON values
/// and omits non-serializable fields like `flags` from field metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorableRlmResult {
    /// Unique identifier for this execution.
    pub id: Uuid,
    /// When the execution started.
    pub created_at: DateTime<Utc>,
    /// The input as a JSON value.
    pub input_json: Value,
    /// The output as a JSON value.
    pub output_json: Value,
    /// Full REPL trajectory including reasoning, code, and outputs.
    pub trajectory: REPLHistory,
    /// Per-field metadata (raw text and constraint checks, without flags).
    pub field_metas: IndexMap<String, StorableFieldMeta>,
    /// Number of REPL iterations used.
    pub iterations: usize,
    /// Number of sub-LLM calls made.
    pub llm_calls: usize,
    /// Whether output was obtained via extraction fallback.
    pub extraction_fallback: bool,
    /// Summary of constraint results.
    pub constraint_summary: ConstraintSummary,
    /// User-provided metadata for tracking experiments, task IDs, etc.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Serializable field metadata without internal flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorableFieldMeta {
    /// The raw text extracted for this field.
    pub raw_text: String,
    /// Constraint check results for this field.
    pub checks: Vec<ConstraintResult>,
}

impl StorableRlmResult {
    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl<S: Signature> RlmResult<S> {
    /// Convert this result to a storable format.
    ///
    /// This serializes the typed input/output to JSON and converts field metadata
    /// to the storable format (dropping non-serializable flags).
    pub fn to_storable(&self) -> Result<StorableRlmResult, serde_json::Error>
    where
        S::Input: Serialize,
        S::Output: Serialize,
    {
        self.to_storable_with_metadata(HashMap::new())
    }

    /// Convert this result to a storable format with custom metadata.
    ///
    /// The metadata map can contain arbitrary JSON values for tracking
    /// experiments, task IDs, or other user-defined data.
    pub fn to_storable_with_metadata(
        &self,
        metadata: HashMap<String, Value>,
    ) -> Result<StorableRlmResult, serde_json::Error>
    where
        S::Input: Serialize,
        S::Output: Serialize,
    {
        let input_json = serde_json::to_value(&self.input)?;
        let output_json = serde_json::to_value(&self.output)?;

        let field_metas = self
            .field_metas
            .iter()
            .map(|(name, meta)| {
                (
                    name.clone(),
                    StorableFieldMeta {
                        raw_text: meta.raw_text.clone(),
                        checks: meta.checks.clone(),
                    },
                )
            })
            .collect();

        Ok(StorableRlmResult {
            id: self.trajectory.id,
            created_at: self.trajectory.created_at,
            input_json,
            output_json,
            trajectory: self.trajectory.clone(),
            field_metas,
            iterations: self.iterations,
            llm_calls: self.llm_calls,
            extraction_fallback: self.extraction_fallback,
            constraint_summary: self.constraint_summary.clone(),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storable_field_meta_serialization_roundtrip() {
        let meta = StorableFieldMeta {
            raw_text: "test value".to_string(),
            checks: vec![ConstraintResult {
                label: "non_empty".to_string(),
                expression: "len(this) > 0".to_string(),
                passed: true,
            }],
        };

        let json = serde_json::to_string(&meta).expect("serialize");
        let restored: StorableFieldMeta = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.raw_text, "test value");
        assert_eq!(restored.checks.len(), 1);
        assert!(restored.checks[0].passed);
    }

    #[test]
    fn storable_rlm_result_serialization_roundtrip() {
        let mut field_metas = IndexMap::new();
        field_metas.insert(
            "summary".to_string(),
            StorableFieldMeta {
                raw_text: "A brief summary".to_string(),
                checks: vec![],
            },
        );

        let mut metadata = HashMap::new();
        metadata.insert("task_id".to_string(), serde_json::json!("test-001"));

        let result = StorableRlmResult {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            input_json: serde_json::json!({"text": "input text"}),
            output_json: serde_json::json!({"summary": "output summary"}),
            trajectory: REPLHistory::new(),
            field_metas,
            iterations: 3,
            llm_calls: 5,
            extraction_fallback: false,
            constraint_summary: ConstraintSummary::default(),
            metadata,
        };

        let json = result.to_json_pretty().expect("serialize");
        let restored = StorableRlmResult::from_json(&json).expect("deserialize");

        assert_eq!(restored.id, result.id);
        assert_eq!(restored.iterations, 3);
        assert_eq!(restored.llm_calls, 5);
        assert!(!restored.extraction_fallback);
        assert_eq!(
            restored.metadata.get("task_id"),
            Some(&serde_json::json!("test-001"))
        );
        assert_eq!(restored.field_metas.len(), 1);
    }

    #[test]
    fn repl_history_serialization_preserves_id_and_timestamps() {
        let history = REPLHistory::new()
            .append("x = 1".to_string(), "".to_string())
            .append_with_reasoning(
                "calculate".to_string(),
                "y = x + 1".to_string(),
                "".to_string(),
            );

        let original_id = history.id;
        let original_created_at = history.created_at;

        let json = serde_json::to_string(&history).expect("serialize");
        let restored: REPLHistory = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.id, original_id);
        assert_eq!(restored.created_at, original_created_at);
        assert_eq!(restored.entries.len(), 2);
        assert_eq!(restored.entries[0].code, "x = 1");
        assert_eq!(restored.entries[1].reasoning, "calculate");
    }
}
