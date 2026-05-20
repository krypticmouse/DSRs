use std::{error::Error as StdError, time::Duration};

use crate::{BamlConvertError, BamlValue, LmUsage};

/// Error from the jsonish coercion layer when LM output can't be parsed as a typed value.
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

/// Coarse error classification for retry and routing logic.
///
/// Use [`PredictError::class`] to get this. `Temporary` errors are generally retryable;
/// `BadResponse` suggests a prompt-engineering problem; `Internal` means a code bug.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ErrorClass {
    /// The request itself was malformed.
    BadRequest,
    /// The requested resource doesn't exist.
    NotFound,
    /// Access denied by the provider.
    Forbidden,
    /// Transient failure (network, rate limit, timeout, server 5xx) — retry may help.
    Temporary,
    /// The LM responded, but the output couldn't be parsed — prompt-engineering problem.
    BadResponse,
    /// A bug in the calling code or an unexpected provider response.
    Internal,
}

/// Failure from a [`Module::call`](crate::Module::call) invocation.
///
/// A call can fail at three stages, and which stage tells you what to do about it:
///
/// 1. **[`Lm`](PredictError::Lm)** — couldn't reach the LM or it errored. Network,
///    rate limit, timeout. Generally retryable.
/// 2. **[`Parse`](PredictError::Parse)** — the LM responded, but we couldn't extract
///    the expected fields from its output. Prompt-engineering problem. Retryable (the
///    LM might produce different output). Includes the raw response for debugging.
/// 3. **[`Conversion`](PredictError::Conversion)** — we parsed a valid `BamlValue`
///    from the response, but it doesn't fit the Rust output type. Code bug or schema
///    mismatch. **Not retryable** — the same parsed value will fail the same way.
///
/// Use [`is_retryable`](PredictError::is_retryable) for retry logic.
/// Use [`class`](PredictError::class) for coarse [`ErrorClass`] bucketing.
#[derive(Debug, thiserror::Error)]
pub enum PredictError {
    /// The LM provider failed before returning a response.
    #[error("LLM call failed")]
    Lm {
        #[source]
        source: LmError,
    },

    /// The LM responded, but the output couldn't be parsed into the expected fields.
    ///
    /// `raw_response` contains the full LM output for debugging. `lm_usage` records
    /// tokens consumed (you still pay for failed parses).
    #[error("failed to parse LLM response")]
    Parse {
        #[source]
        source: ParseError,
        raw_response: String,
        lm_usage: LmUsage,
    },

    /// The response parsed into a `BamlValue` but doesn't match the typed output struct.
    ///
    /// "Understood the LM, but the value doesn't fit the Rust type." Usually a code bug
    /// or schema mismatch — not something retrying will fix.
    #[error("failed to convert parsed value to output type")]
    Conversion {
        #[source]
        source: ConversionError,
        /// The successfully parsed `BamlValue` that failed type conversion.
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

/// The LM response couldn't be parsed into the expected output fields.
///
/// Each variant corresponds to a stage in the parse pipeline:
/// section extraction → jsonish coercion → constraint checking.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// An expected `[[ ## field ## ]]` section marker was not found in the response.
    #[error("field `{field}` not found in response")]
    MissingField { field: String, raw_response: String },

    /// The section marker was found, but the content couldn't be extracted.
    #[error("could not extract field `{field}` from response")]
    ExtractionFailed {
        field: String,
        raw_response: String,
        reason: String,
    },

    /// The field text was extracted but couldn't be coerced to the expected type
    /// (e.g. `"maybe"` for a `bool` field).
    #[error("field `{field}` could not be parsed as {expected_type}")]
    CoercionFailed {
        field: String,
        expected_type: String,
        raw_text: String,
        #[source]
        source: JsonishError,
    },

    /// A `#[assert(...)]` constraint failed on a successfully parsed field value.
    #[error("assertion `{label}` failed on field `{field}`")]
    AssertFailed {
        field: String,
        label: String,
        expression: String,
        value: BamlValue,
    },

    /// Multiple fields failed to parse. Contains all individual errors.
    #[error("{} field(s) failed to parse", errors.len())]
    Multiple {
        errors: Vec<ParseError>,
        /// Partially parsed output (fields that did succeed), if any.
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

/// A parsed `BamlValue` doesn't match the expected Rust output type.
///
/// This is distinct from [`ParseError`]: `ParseError` means "couldn't understand the LM text",
/// `ConversionError` means "understood it, but it doesn't fit the typed output struct."
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// Expected one BamlValue variant, got another (e.g. expected String, got Int).
    #[error("expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: String,
    },

    /// A required struct field is missing from the parsed map.
    #[error("missing required field `{field}` in class `{class}`")]
    MissingField { class: String, field: String },

    /// The parsed string doesn't match any variant of the target enum.
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

/// The LM provider failed before returning a usable response.
///
/// All variants except [`Provider`](LmError::Provider) are retryable.
/// Use [`is_retryable`](LmError::is_retryable) for retry logic.
#[derive(Debug, thiserror::Error)]
pub enum LmError {
    /// Could not reach the provider endpoint (DNS, connection refused, etc.).
    #[error("could not reach {endpoint}")]
    Network {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },

    /// The provider returned a rate limit response (HTTP 429).
    #[error("rate limited by provider")]
    RateLimit { retry_after: Option<Duration> },

    /// The provider returned an unexpected HTTP status.
    #[error("invalid response from provider: HTTP {status}")]
    InvalidResponse { status: u16, body: String },

    /// The request exceeded the configured timeout.
    #[error("request timed out after {after:?}")]
    Timeout { after: Duration },

    /// A provider-specific error that doesn't fit the other categories.
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
