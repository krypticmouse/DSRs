use std::error::Error as StdError;

use crate::LmUsage;

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
///    rate limit, timeout. Reported as a provider error; the rig client owns
///    transport-level retries.
/// 2. **[`Parse`](PredictError::Parse)** — the LM responded, but we couldn't extract
///    the expected fields from its output. Prompt-engineering problem. Retryable (the
///    LM might produce different output). Includes the raw response for debugging.
/// 3. **[`Conversion`](PredictError::Conversion)** — we parsed a valid `serde_json::Value`
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

    /// The response parsed into a `serde_json::Value` but doesn't match the typed output struct.
    ///
    /// "Understood the LM, but the value doesn't fit the Rust type." Usually a code bug
    /// or schema mismatch — not something retrying will fix.
    #[error("failed to convert parsed value to output type")]
    Conversion {
        #[source]
        source: ConversionError,
        /// The successfully parsed `serde_json::Value` that failed type conversion.
        parsed: serde_json::Value,
    },

    /// A strict [`replay`](crate::trace::replay) scope refused the call: the live
    /// request diverged from its recording (or the recorded span is unusable).
    ///
    /// The fixture and the code disagree — re-record the trace or fix the drift.
    /// **Not retryable**: the same request diverges the same way.
    #[error("strict replay refused the call")]
    Replay {
        #[source]
        source: crate::trace::ReplayError,
    },

    /// Candidate parameters could not be bound to the predictor they address.
    ///
    /// Raised before any LM call when a named parameter slot (an ambient
    /// [`fx::Params`](crate::fx::Params) entry or a saved state) doesn't fit
    /// the predictor's signature — a harness/optimizer configuration bug, not
    /// an LM failure. **Not retryable** — the same params fail the same way.
    #[error("params for predictor `{name}` don't fit its signature")]
    Params {
        /// The predictor name the params were addressed to.
        name: String,
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl PredictError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Lm { source } => source.class(),
            Self::Parse { .. } => ErrorClass::BadResponse,
            Self::Conversion { .. } => ErrorClass::Internal,
            Self::Replay { .. } => ErrorClass::Internal,
            Self::Params { .. } => ErrorClass::Internal,
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Lm { source } => source.is_retryable(),
            Self::Parse { .. } => true,
            Self::Conversion { .. } => false,
            Self::Replay { .. } => false,
            Self::Params { .. } => false,
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
        value: serde_json::Value,
    },

    /// Multiple fields failed to parse. Contains all individual errors.
    #[error("{} field(s) failed to parse", errors.len())]
    Multiple {
        errors: Vec<ParseError>,
        /// Partially parsed output (fields that did succeed), if any.
        partial: Option<serde_json::Value>,
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

/// A parsed `serde_json::Value` doesn't match the expected Rust output type.
///
/// This is distinct from [`ParseError`]: `ParseError` means "couldn't understand the LM text",
/// `ConversionError` means "understood it, but it doesn't fit the typed output struct."
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// Expected one serde_json::Value variant, got another (e.g. expected String, got Int).
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

/// The LM provider failed before returning a usable response.
///
/// Everything the provider stack reports arrives as [`Provider`](LmError::Provider):
/// the provider name plus its error message and source. Not retryable — the
/// underlying rig client owns transport-level retry behavior.
#[derive(Debug, thiserror::Error)]
pub enum LmError {
    /// A provider-reported error.
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
        ErrorClass::Internal
    }

    pub fn is_retryable(&self) -> bool {
        false
    }
}
