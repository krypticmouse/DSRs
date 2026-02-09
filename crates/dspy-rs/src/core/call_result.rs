use crate::LmUsage;

use super::{CallMetadata, CallOutcome, CallOutcomeError, ConstraintResult, FieldMeta};

#[deprecated(
    since = "0.7.4",
    note = "Use CallOutcome<O> as the primary typed call surface"
)]
pub struct CallResult<O> {
    pub output: O,
    pub raw_response: String,
    pub lm_usage: LmUsage,
    pub tool_calls: Vec<rig::message::ToolCall>,
    pub tool_executions: Vec<String>,
    pub node_id: Option<usize>,
    fields: indexmap::IndexMap<String, FieldMeta>,
}

#[allow(deprecated)]
impl<O> CallResult<O> {
    pub fn new(
        output: O,
        raw_response: String,
        lm_usage: LmUsage,
        tool_calls: Vec<rig::message::ToolCall>,
        tool_executions: Vec<String>,
        node_id: Option<usize>,
        fields: indexmap::IndexMap<String, FieldMeta>,
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

    pub fn field_flags(&self, field: &str) -> &[crate::Flag] {
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
        self.fields.keys().map(|name: &String| name.as_str())
    }

    pub fn has_failed_checks(&self) -> bool {
        self.fields
            .values()
            .flat_map(|meta| &meta.checks)
            .any(|check| !check.passed)
    }

    pub fn into_outcome(self) -> CallOutcome<O> {
        CallOutcome::ok(
            self.output,
            CallMetadata::new(
                self.raw_response,
                self.lm_usage,
                self.tool_calls,
                self.tool_executions,
                self.node_id,
                self.fields,
            ),
        )
    }
}

#[allow(deprecated)]
impl<O> From<CallResult<O>> for CallOutcome<O> {
    fn from(value: CallResult<O>) -> Self {
        value.into_outcome()
    }
}

#[allow(deprecated)]
impl<O> TryFrom<CallOutcome<O>> for CallResult<O> {
    type Error = CallOutcomeError;

    fn try_from(value: CallOutcome<O>) -> Result<Self, Self::Error> {
        let metadata = value.metadata().clone();
        let output = value.into_result()?;
        Ok(Self::new(
            output,
            metadata.raw_response,
            metadata.lm_usage,
            metadata.tool_calls,
            metadata.tool_executions,
            metadata.node_id,
            metadata.field_meta,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_result_accessors() {
        let mut fields = indexmap::IndexMap::new();
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

        #[allow(deprecated)]
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
