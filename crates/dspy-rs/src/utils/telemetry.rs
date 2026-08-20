use std::sync::OnceLock;
use thiserror::Error;
use tracing_subscriber::EnvFilter;

const DEFAULT_PRETTY_FILTER: &str = "dspy_rs=debug";
static TRACING_INITIALIZED: OnceLock<()> = OnceLock::new();

#[derive(Debug, Error)]
pub enum TelemetryInitError {
    #[error("invalid tracing filter directive `{directive}`: {source}")]
    InvalidFilter {
        directive: String,
        source: tracing_subscriber::filter::ParseError,
    },
    #[error("failed to install tracing subscriber: {0}")]
    SetGlobalDefault(#[from] tracing::subscriber::SetGlobalDefaultError),
}

/// Installs process-global, pretty tracing output for DSRs.
///
/// Behavior:
/// - Uses `RUST_LOG` when present.
/// - Falls back to `dspy_rs=debug` when `RUST_LOG` is unset/invalid.
/// - Is idempotent: repeated calls are no-ops after the first one, including
///   concurrent calls (initialization is claimed atomically up front, so a
///   racing loser returns `Ok` instead of a spurious `SetGlobalDefault` error).
pub fn init_tracing() -> Result<(), TelemetryInitError> {
    // Claim initialization before doing any work: exactly one caller wins the
    // `set`, every other (possibly concurrent) caller no-ops with Ok.
    if TRACING_INITIALIZED.set(()).is_err() {
        return Ok(());
    }

    let filter = resolve_filter()?;
    let subscriber = tracing_subscriber::fmt()
        .pretty()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

fn resolve_filter() -> Result<EnvFilter, TelemetryInitError> {
    match EnvFilter::try_from_default_env() {
        Ok(filter) => Ok(filter),
        Err(_) => EnvFilter::try_new(DEFAULT_PRETTY_FILTER).map_err(|source| {
            TelemetryInitError::InvalidFilter {
                directive: DEFAULT_PRETTY_FILTER.to_string(),
                source,
            }
        }),
    }
}

pub fn truncate(value: &str, max_chars: usize) -> &str {
    if value.chars().count() <= max_chars {
        value
    } else {
        let cutoff = value
            .char_indices()
            .nth(max_chars)
            .map(|(idx, _)| idx)
            .unwrap_or(value.len());
        &value[..cutoff]
    }
}
