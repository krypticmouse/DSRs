use indexmap::IndexMap;
use rig::message::ToolCall;

use crate::{Flag, LmUsage};

pub struct CallResult<O> {
    pub output: O,
    pub raw_response: String,
    pub lm_usage: LmUsage,
    pub tool_calls: Vec<ToolCall>,
    pub tool_executions: Vec<String>,
    pub node_id: Option<usize>,
    fields: IndexMap<String, FieldMeta>,
}

#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub raw_text: String,
    pub flags: Vec<Flag>,
    pub checks: Vec<ConstraintResult>,
}

#[derive(Debug, Clone)]
pub struct ConstraintResult {
    pub label: String,
    pub expression: String,
    pub passed: bool,
}

impl<O> CallResult<O> {
    pub fn new(
        output: O,
        raw_response: String,
        lm_usage: LmUsage,
        tool_calls: Vec<ToolCall>,
        tool_executions: Vec<String>,
        node_id: Option<usize>,
        fields: IndexMap<String, FieldMeta>,
    ) -> Self {
        Self {
            output,
            raw_response,
            lm_usage,
            tool_calls,
            tool_executions,
            node_id,
            fields,
        }
    }

    pub fn field_flags(&self, field: &str) -> &[Flag] {
        self.fields
            .get(field)
            .map(|meta| meta.flags.as_slice())
            .unwrap_or(&[])
    }

    pub fn field_checks(&self, field: &str) -> &[ConstraintResult] {
        self.fields
            .get(field)
            .map(|meta| meta.checks.as_slice())
            .unwrap_or(&[])
    }

    pub fn field_raw(&self, field: &str) -> Option<&str> {
        self.fields.get(field).map(|meta| meta.raw_text.as_str())
    }

    pub fn field_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.fields.keys().map(|name| name.as_str())
    }

    pub fn field_metas(&self) -> &IndexMap<String, FieldMeta> {
        &self.fields
    }

    pub fn has_failed_checks(&self) -> bool {
        self.fields
            .values()
            .flat_map(|meta| &meta.checks)
            .any(|check| !check.passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_result_accessors() {
        let mut fields = IndexMap::new();
        fields.insert(
            "answer".to_string(),
            FieldMeta {
                raw_text: "42".to_string(),
                flags: Vec::new(),
                checks: vec![ConstraintResult {
                    label: "non_empty".to_string(),
                    expression: "this.len() > 0".to_string(),
                    passed: false,
                }],
            },
        );

        let result = CallResult::new(
            "ok",
            "raw".to_string(),
            LmUsage::default(),
            Vec::new(),
            Vec::new(),
            None,
            fields,
        );

        assert_eq!(result.field_raw("answer"), Some("42"));
        assert!(result.field_flags("missing").is_empty());
        assert!(result.has_failed_checks());
        let names: Vec<_> = result.field_names().collect();
        assert_eq!(names, vec!["answer"]);
    }
}
