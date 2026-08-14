//! `dsrs fmt`: canonical-print a `.dsrs` artifact.
//!
//! The canonical form is `Program::to_dsrs` — the same text that seals
//! `program_hash` (minus lineage) and that `bake` writes. Formatting is
//! therefore parse→print, never token shuffling: an artifact that doesn't
//! parse doesn't format.

use std::path::Path;

use anyhow::Context;
use dspy_rs::ir::Program;

/// Outcome of [`fmt_file`].
#[derive(Debug, PartialEq, Eq)]
pub enum FmtOutcome {
    /// `--write`: the file already was canonical; nothing touched.
    Unchanged,
    /// `--write`: the file was rewritten in canonical form.
    Rewrote,
    /// No `--write`: the canonical text, for stdout.
    Canonical(String),
}

/// Parses `path` and canonically prints it. With `write`, rewrites the file
/// in place (only when the bytes differ); otherwise returns the text.
pub fn fmt_file(path: impl AsRef<Path>, write: bool) -> anyhow::Result<FmtOutcome> {
    let path = path.as_ref();
    let program = Program::load_dsrs(path)?;
    let canonical = program.to_dsrs();
    if !write {
        return Ok(FmtOutcome::Canonical(canonical));
    }
    let original = std::fs::read_to_string(path)
        .with_context(|| format!("failed to re-read `{}`", path.display()))?;
    if original == canonical {
        return Ok(FmtOutcome::Unchanged);
    }
    std::fs::write(path, canonical)
        .with_context(|| format!("failed to write `{}`", path.display()))?;
    Ok(FmtOutcome::Rewrote)
}
