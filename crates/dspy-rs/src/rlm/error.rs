#![cfg(feature = "rlm")]

use crate::{BamlConvertError, BamlValue, LmError, ParseError};

#[derive(Debug, thiserror::Error)]
pub enum RlmError {
    #[error("LLM call failed")]
    Lm {
        #[source]
        source: LmError,
    },

    #[error("submission assertion failed: {label} ({expression})")]
    SubmitAssertion { label: String, expression: String },

    #[error("failed to convert SUBMIT output")]
    Conversion {
        #[source]
        source: BamlConvertError,
        value: BamlValue,
    },

    #[error("failed to parse extraction fallback response")]
    ExtractionParse {
        #[source]
        source: ParseError,
        raw_response: String,
    },

    #[error("max iterations ({max}) reached without SUBMIT")]
    MaxIterations { max: usize },

    #[error("tokio runtime unavailable: {message}")]
    RuntimeUnavailable { message: String },

    #[error("python setup failed: {message}")]
    PythonSetup { message: String },
}
