use std::{error::Error as StdError, time::Duration};

use crate::{BamlConvertError, BamlValue, LmUsage};

#[derive(Debug)]
pub struct JsonishError(pub(crate) anyhow::Error);

impl std::fmt::Display for JsonishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl StdError for JsonishError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

impl From<anyhow::Error> for JsonishError {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ErrorClass {
    BadRequest,
    NotFound,
    Forbidden,
    Temporary,
    BadResponse,
    Internal,
}

#[derive(Debug, thiserror::Error)]
pub enum PredictError {
    #[error("LLM call failed")]
    Lm {
        #[source]
        source: LmError,
    },

    #[error("failed to parse LLM response")]
    Parse {
        #[source]
        source: ParseError,
        raw_response: String,
        lm_usage: LmUsage,
    },

    #[error("failed to convert parsed value to output type")]
    Conversion {
        #[source]
        source: ConversionError,
        parsed: BamlValue,
    },
}

impl PredictError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Lm { source } => source.class(),
            Self::Parse { .. } => ErrorClass::BadResponse,
            Self::Conversion { .. } => ErrorClass::Internal,
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Lm { source } => source.is_retryable(),
            Self::Parse { .. } => true,
            Self::Conversion { .. } => false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("field `{field}` not found in response")]
    MissingField { field: String, raw_response: String },

    #[error("could not extract field `{field}` from response")]
    ExtractionFailed {
        field: String,
        raw_response: String,
        reason: String,
    },

    #[error("field `{field}` could not be parsed as {expected_type}")]
    CoercionFailed {
        field: String,
        expected_type: String,
        raw_text: String,
        #[source]
        source: JsonishError,
    },

    #[error("assertion `{label}` failed on field `{field}`")]
    AssertFailed {
        field: String,
        label: String,
        expression: String,
        value: BamlValue,
    },

    #[error("{} field(s) failed to parse", errors.len())]
    Multiple {
        errors: Vec<ParseError>,
        partial: Option<BamlValue>,
    },
}

impl ParseError {
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::MissingField { field, .. } => Some(field),
            Self::ExtractionFailed { field, .. } => Some(field),
            Self::CoercionFailed { field, .. } => Some(field),
            Self::AssertFailed { field, .. } => Some(field),
            Self::Multiple { .. } => None,
        }
    }

    pub fn fields(&self) -> Vec<&str> {
        match self {
            Self::Multiple { errors, .. } => errors.iter().filter_map(|e| e.field()).collect(),
            other => other.field().into_iter().collect(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: String,
    },

    #[error("missing required field `{field}` in class `{class}`")]
    MissingField { class: String, field: String },

    #[error("enum `{enum_name}` has no variant `{got}`")]
    UnknownVariant {
        enum_name: String,
        got: String,
        valid_variants: Vec<String>,
    },
}

impl From<BamlConvertError> for ConversionError {
    fn from(error: BamlConvertError) -> Self {
        ConversionError::TypeMismatch {
            expected: error.expected,
            actual: error.got,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LmError {
    #[error("could not reach {endpoint}")]
    Network {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },

    #[error("rate limited by provider")]
    RateLimit { retry_after: Option<Duration> },

    #[error("invalid response from provider: HTTP {status}")]
    InvalidResponse { status: u16, body: String },

    #[error("request timed out after {after:?}")]
    Timeout { after: Duration },

    #[error("provider error from {provider}: {message}")]
    Provider {
        provider: String,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
}

impl LmError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Network { .. } => ErrorClass::Temporary,
            Self::RateLimit { .. } => ErrorClass::Temporary,
            Self::InvalidResponse { status, .. } if *status >= 500 => ErrorClass::Temporary,
            Self::InvalidResponse { .. } => ErrorClass::BadRequest,
            Self::Timeout { .. } => ErrorClass::Temporary,
            Self::Provider { .. } => ErrorClass::Internal,
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::RateLimit { .. } => true,
            Self::Timeout { .. } => true,
            Self::InvalidResponse { status, .. } => *status >= 500,
            Self::Provider { .. } => false,
        }
    }
}
