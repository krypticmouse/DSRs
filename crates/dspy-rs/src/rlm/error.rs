#![cfg(feature = "rlm")]

use crate::{BamlConvertError, BamlValue, LmError, ParseError, PredictError};
use pyo3::PyErr;

#[derive(Debug, thiserror::Error)]
pub enum RlmError {
    #[error("LLM call failed")]
    LlmError {
        #[source]
        source: LmError,
    },

    #[error("assertion '{label}' failed: {expression}")]
    AssertionFailed { label: String, expression: String },

    #[error("failed to convert SUBMIT output")]
    ConversionError {
        #[source]
        source: BamlConvertError,
        value: BamlValue,
    },

    #[error("failed to parse extraction fallback response")]
    ExtractionFailed {
        #[source]
        source: ParseError,
        raw_response: String,
    },

    #[error("predictor failed during {stage}")]
    PredictError {
        stage: &'static str,
        #[source]
        source: PredictError,
    },

    #[error("max iterations ({max}) reached without SUBMIT")]
    MaxIterationsReached { max: usize },

    #[error("max LLM calls ({max}) exceeded")]
    MaxLlmCallsExceeded { max: usize },

    #[error("python error: {message}")]
    PythonError { message: String },

    #[error("tokio runtime unavailable: {message}")]
    RuntimeUnavailable { message: String },

    #[error("configuration error: {message}")]
    ConfigurationError { message: String },
}

impl From<PyErr> for RlmError {
    fn from(err: PyErr) -> Self {
        RlmError::PythonError {
            message: err.to_string(),
        }
    }
}
