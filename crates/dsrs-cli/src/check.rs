//! `dsrs check`: parse + validate a `.dsrs` artifact.
//!
//! The error path is the product: `.dsrs` programs are written by humans and
//! LLMs, and [`DsrsFileError`] already carries the artifact path plus the
//! parser's `line N, column M: expected …` message — this command just puts
//! that on stderr with a non-zero exit code, making it a regeneration signal
//! for a model loop and a pre-commit gate for a human.

use std::fmt;
use std::path::Path;

use dspy_rs::ir::{DsrsFileError, Program};

/// What `dsrs check` reports on success.
#[derive(Debug)]
pub struct CheckReport {
    pub name: String,
    pub program_hash: u64,
    pub nodes: usize,
    pub sigs: usize,
    pub models: usize,
    pub tools: usize,
    pub caps: Vec<String>,
}

impl fmt::Display for CheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ok: program `{}` ({:016x}) — {} nodes, {} sigs, {} models, {} tools",
            self.name, self.program_hash, self.nodes, self.sigs, self.models, self.tools
        )?;
        if !self.caps.is_empty() {
            write!(f, ", caps {{ {} }}", self.caps.join(" "))?;
        }
        Ok(())
    }
}

/// Parses and validates the artifact at `path`. `Program::load_dsrs` runs the
/// full pipeline: lex/parse (with positions), lowering, `Program::validate`,
/// hash sealing — exactly what `Interpreter::load` would accept.
pub fn check_file(path: impl AsRef<Path>) -> Result<CheckReport, DsrsFileError> {
    let program = Program::load_dsrs(path)?;
    Ok(CheckReport {
        name: program.meta.name.to_string(),
        program_hash: program.meta.program_hash,
        nodes: program.nodes.len(),
        sigs: program.sigs.len(),
        models: program.models.len(),
        tools: program.tools.len(),
        caps: program.caps.iter().map(str::to_string).collect(),
    })
}
