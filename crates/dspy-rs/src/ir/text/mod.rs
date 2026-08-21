//! The `.dsrs` text format (RFC 0002 §4, stage IR-5).
//!
//! The text form is the **only** wire form of a program (RFC 0002 §5): a
//! `.dsrs` artifact is written by humans, LLMs, and `bake`, and the canonical
//! printed text (minus the `lineage` block) is the preimage of
//! [`Program::compute_hash`].
//!
//! - [`Program::from_dsrs`] — parse text into a validated [`Program`].
//! - [`Program::to_dsrs`] — deterministic canonical print
//!   (`parse(print(p))` prints identically; `print(parse(t))` is the
//!   canonical form of `t`). Ordering rules are documented in [`print`].
//! - [`Program::load_dsrs`] / [`Program::save_dsrs`] — file convenience;
//!   `.dsrs` artifacts are UTF-8 text only, non-text files are rejected.

use std::path::Path;

use crate::ir::graph::Program;

pub(crate) mod parse;
pub(crate) mod print;

/// A parse failure with the source position and what was expected — designed
/// to be actionable feedback for a model regenerating the program.
///
/// Re-exported from `dsrs-syntax`, the shared home of the `.dsrs` lexer and
/// structural grammar (also used by `include_program!` at macro expansion).
pub use dsrs_syntax::ParseError;

/// Failure loading or saving a `.dsrs` artifact file.
#[derive(Debug, thiserror::Error)]
pub enum DsrsFileError {
    #[error("failed to read `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// `.dsrs` artifacts are text; binary content is rejected outright.
    #[error("`{path}` is not a UTF-8 text artifact (`.dsrs` files are text-only)")]
    NotText { path: String },
    #[error("{path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: ParseError,
    },
}

impl Program {
    /// Parses `.dsrs` text into a validated program. The returned program has
    /// its `program_hash` sealed from the canonical text.
    pub fn from_dsrs(src: &str) -> Result<Program, ParseError> {
        parse::parse_program(src)
    }

    /// Reads and parses a `.dsrs` text artifact. Non-UTF-8 (binary) files are
    /// rejected — the text format is the only wire form of a program.
    pub fn load_dsrs(path: impl AsRef<Path>) -> Result<Program, DsrsFileError> {
        let path = path.as_ref();
        let display = path.display().to_string();
        let bytes = std::fs::read(path).map_err(|source| DsrsFileError::Io {
            path: display.clone(),
            source,
        })?;
        let text = String::from_utf8(bytes).map_err(|_| DsrsFileError::NotText {
            path: display.clone(),
        })?;
        Self::from_dsrs(&text).map_err(|source| DsrsFileError::Parse {
            path: display,
            source,
        })
    }

    /// Writes the canonical `.dsrs` text form to `path`.
    pub fn save_dsrs(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_dsrs())
    }
}
