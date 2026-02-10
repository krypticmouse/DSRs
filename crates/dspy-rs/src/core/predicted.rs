use std::ops::Deref;

use indexmap::IndexMap;
use rig::message::ToolCall;

use crate::{Flag, LmUsage};

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

#[derive(Debug, Clone)]
pub struct CallMetadata {
    pub raw_response: String,
    pub lm_usage: LmUsage,
    pub tool_calls: Vec<ToolCall>,
    pub tool_executions: Vec<String>,
    pub node_id: Option<usize>,
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
        self.field_meta.get(field).map(|meta| meta.raw_text.as_str())
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

#[derive(Debug, Clone)]
pub struct Predicted<O> {
    output: O,
    metadata: CallMetadata,
}

impl<O> Predicted<O> {
    pub fn new(output: O, metadata: CallMetadata) -> Self {
        Self { output, metadata }
    }

    pub fn metadata(&self) -> &CallMetadata {
        &self.metadata
    }

    pub fn into_inner(self) -> O {
        self.output
    }

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
