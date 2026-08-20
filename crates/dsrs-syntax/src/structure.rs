//! Build-time **syntax-only** validation of `.dsrs` artifacts (RFC 0002 §6.1).
//!
//! # Why this layer exists (the layering, honestly)
//!
//! `include_program!` wants to validate the artifact at macro expansion, the
//! sqlx/prost way. Full validation lives in `dspy-rs` (`Program::from_dsrs`
//! lowers through `ProgramBuilder` and runs `Program::validate`), and
//! `dspy-rs` depends on the proc-macro crate — so the macro cannot call the
//! full parser without a dependency cycle. This module is the shared
//! **structural grammar** over the one shared lexer ([`crate::lex`]): the
//! macro gets real syntax errors with positions, and the full parser and this
//! checker can never disagree about tokens because there is only one
//! tokenizer.
//!
//! # What it checks (and what it deliberately does not)
//!
//! Checked at build time — the classic authoring slips, with line/column:
//! - the `dsrs 1` pragma (and format-major rejection),
//! - `program <name>`,
//! - top-level keyword vocabulary (`caps model sig class enum tool lineage
//!   main`; unknown top-level keywords are rejected, per RFC 0002 §5 semver),
//! - the shape of each declaration header (`model x = "…"`,
//!   `tool x "…" caps [ … ] { … } js ``` … ````, `main: Sig = seq { … }`),
//! - balanced `{ } ( ) [ ]` with fence-aware raw regions,
//! - lexical validity: JSON strings, numbers, `js` code fences, comments,
//! - exactly one `main`, last, with nothing after it.
//!
//! **Not** checked here (first use / `cargo test` catches these via the full
//! parser): types, signature/field validity, dataflow ordering, capability
//! subsets, model references, demo row shapes — anything semantic. Inside a
//! balanced block this checker is deliberately *more permissive* than the
//! full parser: it must never reject an artifact `Program::from_dsrs`
//! accepts.

use crate::ParseError;
use crate::lex::{Lexed, Lexer, Span, Tok};

/// Checks `.dsrs` source for structural syntax validity. See the module docs
/// for the exact contract: this is syntax-only, and strictly more permissive
/// than `dspy_rs::ir::Program::from_dsrs`.
pub fn check(src: &str) -> Result<(), ParseError> {
    Checker::new(src)?.file()
}

struct Checker<'a> {
    lx: Lexer<'a>,
    cur: Lexed,
}

impl<'a> Checker<'a> {
    fn new(src: &'a str) -> Result<Self, ParseError> {
        let mut lx = Lexer::new(src);
        let cur = lx.next_token()?;
        Ok(Self { lx, cur })
    }

    fn bump(&mut self) -> Result<Lexed, ParseError> {
        let cur = std::mem::replace(&mut self.cur, self.lx.next_token()?);
        Ok(cur)
    }

    fn err(&self, message: impl Into<String>) -> ParseError {
        ParseError::at(self.cur.span, message)
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(&self.cur.tok, Tok::Ident(word) if word == kw)
    }

    fn expect_kw(&mut self, kw: &str, context: &str) -> Result<(), ParseError> {
        if self.at_kw(kw) {
            self.bump()?;
            Ok(())
        } else {
            Err(self.err(format!(
                "expected `{kw}` {context}, found {}",
                self.cur.tok.describe()
            )))
        }
    }

    fn expect_ident(&mut self, context: &str) -> Result<(), ParseError> {
        match &self.cur.tok {
            Tok::Ident(_) => {
                self.bump()?;
                Ok(())
            }
            other => Err(self.err(format!(
                "expected a name {context}, found {}",
                other.describe()
            ))),
        }
    }

    fn expect_str(&mut self, context: &str) -> Result<(), ParseError> {
        match &self.cur.tok {
            Tok::Str(_) => {
                self.bump()?;
                Ok(())
            }
            other => Err(self.err(format!(
                "expected a string {context}, found {}",
                other.describe()
            ))),
        }
    }

    fn expect_tok(&mut self, tok: Tok, context: &str) -> Result<(), ParseError> {
        if self.cur.tok == tok {
            self.bump()?;
            Ok(())
        } else {
            Err(self.err(format!(
                "expected {} {context}, found {}",
                tok.describe(),
                self.cur.tok.describe()
            )))
        }
    }

    /// Skips a raw fence region; the current token must be the fence opener.
    fn skip_fence(&mut self) -> Result<(), ParseError> {
        debug_assert_eq!(self.cur.tok, Tok::Fence);
        let (_, end) = self.lx.scan_code_fence(self.cur.start)?;
        let span = self.lx.span_at(end);
        self.lx.seek(end, span);
        self.cur = self.lx.next_token()?;
        Ok(())
    }

    /// Consumes a balanced `{ … }` / `[ … ]` / `( … )` region, fence-aware.
    /// The current token must be the opening delimiter. Content is not
    /// inspected — everything semantic is the full parser's job.
    fn skip_balanced(&mut self, context: &str) -> Result<(), ParseError> {
        let open = match self.cur.tok {
            Tok::LBrace | Tok::LBracket | Tok::LParen => self.cur.tok.clone(),
            _ => {
                return Err(self.err(format!(
                    "expected `{{` {context}, found {}",
                    self.cur.tok.describe()
                )));
            }
        };
        let open_span = self.cur.span;
        self.bump()?;
        let mut stack: Vec<(Tok, Span)> = vec![(open, open_span)];
        loop {
            match &self.cur.tok {
                Tok::LBrace | Tok::LBracket | Tok::LParen => {
                    stack.push((self.cur.tok.clone(), self.cur.span));
                    self.bump()?;
                }
                Tok::RBrace | Tok::RBracket | Tok::RParen => {
                    let (top, top_span) = stack.pop().expect("stack non-empty in loop");
                    let expected = match top {
                        Tok::LBrace => Tok::RBrace,
                        Tok::LBracket => Tok::RBracket,
                        Tok::LParen => Tok::RParen,
                        _ => unreachable!("only open delimiters are pushed"),
                    };
                    if self.cur.tok != expected {
                        return Err(self.err(format!(
                            "mismatched delimiter: found {} but {} opened at line {}, column {} \
                             expects {}",
                            self.cur.tok.describe(),
                            top.describe(),
                            top_span.line,
                            top_span.col,
                            expected.describe()
                        )));
                    }
                    self.bump()?;
                    if stack.is_empty() {
                        return Ok(());
                    }
                }
                Tok::Fence => self.skip_fence()?,
                Tok::Eof => {
                    let (top, top_span) = stack.last().expect("stack non-empty in loop");
                    return Err(self.err(format!(
                        "unexpected end of file: {} opened at line {}, column {} is never closed",
                        top.describe(),
                        top_span.line,
                        top_span.col
                    )));
                }
                _ => {
                    self.bump()?;
                }
            }
        }
    }

    fn file(mut self) -> Result<(), ParseError> {
        // dsrs 1
        self.expect_kw("dsrs", "at the start of the file (`dsrs 1`)")?;
        match &self.cur.tok {
            Tok::Num(raw) => {
                let major: Option<u32> = raw.parse().ok();
                if major != Some(1) {
                    return Err(self.err(format!(
                        "unsupported format major `{raw}`: this parser reads `dsrs 1`"
                    )));
                }
                self.bump()?;
            }
            other => {
                return Err(self.err(format!(
                    "expected an integer after `dsrs` (the format major), found {}",
                    other.describe()
                )));
            }
        }

        // program <name>
        self.expect_kw("program", "after the `dsrs 1` pragma")?;
        self.expect_ident("after `program`")?;

        // Top-level declarations until `main`.
        loop {
            match &self.cur.tok {
                Tok::Ident(word) => match word.as_str() {
                    "caps" => {
                        self.bump()?;
                        self.skip_balanced("after `caps`")?;
                    }
                    "model" => {
                        self.bump()?;
                        self.expect_ident("after `model`")?;
                        self.expect_tok(Tok::Eq, "after the model name")?;
                        self.expect_str("after `=` (the provider model string)")?;
                        if self.cur.tok == Tok::LBrace {
                            self.skip_balanced("to open the model options")?;
                        }
                    }
                    "class" => {
                        self.bump()?;
                        self.expect_ident("after `class`")?;
                        self.skip_balanced("to open the class body")?;
                    }
                    "enum" => {
                        self.bump()?;
                        self.expect_ident("after `enum`")?;
                        self.skip_balanced("to open the enum body")?;
                    }
                    "sig" => {
                        self.bump()?;
                        self.expect_ident("after `sig`")?;
                        self.skip_balanced("to open the signature body")?;
                    }
                    "tool" => {
                        self.bump()?;
                        self.expect_ident("after `tool`")?;
                        self.expect_str("after the tool name (the tool description)")?;
                        if self.at_kw("caps") {
                            self.bump()?;
                            if self.cur.tok != Tok::LBracket {
                                return Err(self.err(format!(
                                    "expected `[` after `caps`, found {}",
                                    self.cur.tok.describe()
                                )));
                            }
                            self.skip_balanced("after `caps`")?;
                        }
                        self.skip_balanced("to open the tool interface")?;
                        if self.at_kw("js") {
                            self.bump()?;
                            if self.cur.tok != Tok::Fence {
                                return Err(self.err(format!(
                                    "expected a ``` code fence after `js`, found {}",
                                    self.cur.tok.describe()
                                )));
                            }
                            self.skip_fence()?;
                        }
                    }
                    "lineage" => {
                        self.bump()?;
                        self.skip_balanced("after `lineage`")?;
                    }
                    "main" => break,
                    other => {
                        return Err(self.err(format!(
                            "unknown top-level keyword `{other}`: expected one of `caps`, \
                             `model`, `sig`, `class`, `enum`, `tool`, `lineage`, `main`"
                        )));
                    }
                },
                Tok::Eof => {
                    return Err(self.err(
                        "unexpected end of file: every program ends with `main: <Sig> = seq { ... }`",
                    ));
                }
                other => {
                    return Err(self.err(format!(
                        "expected a top-level declaration keyword, found {}",
                        other.describe()
                    )));
                }
            }
        }

        // main: <Sig> = seq { … }
        self.expect_kw("main", "")?;
        self.expect_tok(Tok::Colon, "after `main`")?;
        self.expect_ident("after `main:` (the program signature name)")?;
        self.expect_tok(Tok::Eq, "after the main signature name")?;
        if !self.at_kw("seq") {
            return Err(self.err(format!(
                "the main expression must be `seq {{ ... }}`, found {}",
                self.cur.tok.describe()
            )));
        }
        self.bump()?;
        self.skip_balanced("to open the main `seq` body")?;

        if self.cur.tok != Tok::Eof {
            return Err(self.err(format!(
                "expected end of file after `main`, found {}",
                self.cur.tok.describe()
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::check;
    use crate::ParseError;

    const MINI: &str = r#"
dsrs 1
program mini

model m = "openai:gpt-4o-mini"

sig Main {
  in question: string
  out answer: string
}

main: Main = seq {
  a = predict Main (question = $.question)
  out { answer = a.answer }
}
"#;

    fn check_err(src: &str) -> ParseError {
        check(src).expect_err("expected a syntax error")
    }

    #[test]
    fn accepts_a_minimal_program() {
        check(MINI).expect("minimal program is syntactically valid");
    }

    #[test]
    fn rejects_missing_pragma() {
        let err = check_err("program x\nmain: M = seq { }");
        assert_eq!((err.line, err.col), (1, 1));
        assert!(err.message.contains("expected `dsrs`"), "{}", err.message);
    }

    #[test]
    fn rejects_future_format_major() {
        let err = check_err("dsrs 2\nprogram x\nmain: M = seq { }");
        assert_eq!((err.line, err.col), (1, 6));
        assert!(
            err.message.contains("unsupported format major `2`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_unknown_top_level_keyword() {
        let err = check_err("dsrs 1\nprogram x\nwidget y { }\nmain: M = seq { }");
        assert_eq!((err.line, err.col), (3, 1));
        assert!(
            err.message.contains("unknown top-level keyword `widget`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_unbalanced_brace_with_open_position() {
        let err = check_err("dsrs 1\nprogram x\nsig Main {\n  in q: string\n");
        assert!(
            err.message.contains("`{` opened at line 3"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_mismatched_delimiters() {
        let err = check_err("dsrs 1\nprogram x\nsig Main { in q: (string] }\nmain: Main = seq { }");
        assert!(err.message.contains("mismatched delimiter"), "{}", err.message);
    }

    #[test]
    fn rejects_unterminated_string() {
        let err = check_err("dsrs 1\nprogram x\nsig Main {\n  \"oops\n}");
        assert_eq!(err.line, 4);
        assert!(
            err.message.contains("unterminated string literal"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_unterminated_code_fence() {
        let src = "dsrs 1\nprogram x\ntool t \"d\" { in a: string out b: string } js```\ncode\n";
        let err = check_err(src);
        assert!(
            err.message.contains("unterminated code block"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_missing_main() {
        let err = check_err("dsrs 1\nprogram x\nsig Main { in q: string out a: string }\n");
        assert!(
            err.message.contains("every program ends with `main:"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_trailing_content_after_main() {
        let err = check_err(&format!("{MINI}\nsig Extra {{ in q: string }}\n"));
        assert!(
            err.message.contains("expected end of file after `main`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn rejects_non_seq_main() {
        let err = check_err("dsrs 1\nprogram x\nmain: M = predict M (q = $.q)");
        assert!(
            err.message.contains("the main expression must be `seq"),
            "{}",
            err.message
        );
    }

    #[test]
    fn fences_inside_main_bodies_are_raw() {
        // A hole body containing unbalanced braces and stray quotes inside
        // the fence must not confuse the delimiter tracker.
        let src = r#"
dsrs 1
program h

sig Main {
  in q: string
  out a: string
}

main: Main = seq {
  x = hole Main (q = $.q) caps [] js```
(inp) => ({ a: "}}}" + inp.q })  // " {{{
```
  out { a = x.a }
}
"#;
        check(src).expect("fenced code is scanned raw");
    }

    #[test]
    fn json_demo_blocks_are_token_transparent() {
        let src = r#"
dsrs 1
program d

sig Main {
  in q: string
  out a: string
}

main: Main = seq {
  x = predict Main (q = $.q) { demos [{"input":{"q":"slashes // not comments"},"output":{"a":"y"}}] }
  out { a = x.a }
}
"#;
        check(src).expect("JSON literals lex as balanced tokens");
    }
}
