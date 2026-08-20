//! Shared syntax layer for the `.dsrs` text format (RFC 0002 §4).
//!
//! This crate is the single home of the `.dsrs` **lexer** ([`lex`]) and the
//! **structural grammar** check ([`check`]). It exists so both consumers of
//! the grammar read from one source of truth:
//!
//! - `dspy-rs` — the full parser (`Program::from_dsrs`) pulls tokens from
//!   [`lex`] and lowers them through its program builder; types, dataflow,
//!   and every other semantic rule live there.
//! - `dsrs_macros` — `include_program!` validates artifacts at macro
//!   expansion via [`check`], the syntax-only structural pass. The macro
//!   crate cannot depend on `dspy-rs` (which depends on it), so this leaf
//!   crate is what breaks the cycle.
//!
//! Deliberately a leaf: no dependency on dspy-rs, dsrs_macros, facet, or rig
//! — proc-macro crates can depend on it without cycles. Grammar changes are
//! made **here once** and both frontends pick them up.

pub mod lex;
mod structure;

pub use structure::check;

/// A parse failure with the source position and what was expected — designed
/// to be actionable feedback for a model regenerating the program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1-based source line.
    pub line: u32,
    /// 1-based source column (bytes).
    pub col: u32,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, column {}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for ParseError {}

impl ParseError {
    /// Positions `message` at `span`.
    pub fn at(span: lex::Span, message: impl Into<String>) -> Self {
        Self {
            line: span.line,
            col: span.col,
            message: message.into(),
        }
    }
}
