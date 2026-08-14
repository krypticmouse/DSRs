//! The `dsrs` command line (RFC 0002 IR-7, §6.2): tooling over `.dsrs`
//! program artifacts.
//!
//! - [`check`] — `dsrs check program.dsrs`: parse + validate, LLM-friendly
//!   errors (line/column, what was expected), exit code.
//! - [`fmt`] — `dsrs fmt program.dsrs [--write]`: canonical print
//!   (`Program::to_dsrs`), the format's one true form.
//! - [`serve`] — `dsrs serve program.dsrs`: the serving host. Loads the
//!   program (plus an optional named-form overlay), binds models from the
//!   environment through [`dspy_rs::ir::RuntimeEnv`], and exposes the program
//!   over HTTP.
//!
//! Everything is a plain library function so tests drive the exact code the
//! binary runs — the binary in `main.rs` is argument parsing plus process
//! exit codes, nothing else.

pub mod check;
pub mod fmt;
pub mod serve;
