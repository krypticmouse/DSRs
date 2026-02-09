use std::ops::{Deref, DerefMut};

use bamltype::baml_types::BamlValue;
use indexmap::IndexMap;
use rig::message::ToolCall;

use crate::{ConversionError, Flag, LmError, LmUsage, ParseError, PredictError};

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

#[derive(Debug, thiserror::Error)]
pub enum CallOutcomeErrorKind {
    #[error("LM call failed")]
    Lm(#[source] LmError),

    #[error("failed to parse LLM response")]
    Parse(#[source] ParseError),

    #[error("failed to convert parsed value to output type")]
    Conversion(#[source] ConversionError, BamlValue),
}

#[derive(Debug, thiserror::Error)]
#[error("call outcome failed: {kind}")]
pub struct CallOutcomeError {
    pub metadata: CallMetadata,
    pub kind: CallOutcomeErrorKind,
}

impl CallOutcomeError {
    pub fn into_predict_error(self) -> PredictError {
        match self.kind {
            CallOutcomeErrorKind::Lm(source) => PredictError::Lm { source },
            CallOutcomeErrorKind::Parse(source) => PredictError::Parse {
                source,
                raw_response: self.metadata.raw_response,
                lm_usage: self.metadata.lm_usage,
            },
            CallOutcomeErrorKind::Conversion(source, parsed) => PredictError::Conversion {
                source,
                parsed,
            },
        }
    }
}

pub struct CallOutcome<O> {
    metadata: CallMetadata,
    result: Result<O, CallOutcomeErrorKind>,
}

impl<O> CallOutcome<O> {
    pub fn ok(output: O, metadata: CallMetadata) -> Self {
        Self {
            metadata,
            result: Ok(output),
        }
    }

    pub fn err(kind: CallOutcomeErrorKind, metadata: CallMetadata) -> Self {
        Self {
            metadata,
            result: Err(kind),
        }
    }

    pub fn metadata(&self) -> &CallMetadata {
        &self.metadata
    }

    pub fn into_result(self) -> Result<O, CallOutcomeError> {
        match self.result {
            Ok(output) => Ok(output),
            Err(kind) => Err(CallOutcomeError {
                metadata: self.metadata,
                kind,
            }),
        }
    }

    pub fn try_into_result(self) -> Result<O, CallOutcomeError> {
        self.into_result()
    }

    pub fn into_parts(self) -> (Result<O, CallOutcomeErrorKind>, CallMetadata) {
        (self.result, self.metadata)
    }

    pub fn result(&self) -> &Result<O, CallOutcomeErrorKind> {
        &self.result
    }
}

impl<O> Deref for CallOutcome<O> {
    type Target = Result<O, CallOutcomeErrorKind>;

    fn deref(&self) -> &Self::Target {
        &self.result
    }
}

impl<O> DerefMut for CallOutcome<O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.result
    }
}

#[cfg(feature = "nightly-try")]
impl<O> std::ops::Try for CallOutcome<O> {
    type Output = O;
    type Residual = CallOutcome<std::convert::Infallible>;

    fn from_output(output: Self::Output) -> Self {
        Self::ok(output, CallMetadata::default())
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self.into_parts() {
            (Ok(value), _) => std::ops::ControlFlow::Continue(value),
            (Err(err), metadata) => {
                std::ops::ControlFlow::Break(CallOutcome::err(err, metadata))
            }
        }
    }
}

#[cfg(feature = "nightly-try")]
impl<O> std::ops::FromResidual<CallOutcome<std::convert::Infallible>> for CallOutcome<O> {
    fn from_residual(residual: CallOutcome<std::convert::Infallible>) -> Self {
        let (result, metadata) = residual.into_parts();
        let err = match result {
            Ok(value) => match value {},
            Err(err) => err,
        };
        CallOutcome::err(err, metadata)
    }
}
