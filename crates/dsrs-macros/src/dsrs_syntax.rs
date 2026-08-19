//! Build-time **syntax-only** validation of `.dsrs` artifacts (RFC 0002 §6.1).
//!
//! # Why this module exists (the layering, honestly)
//!
//! `include_program!` wants to validate the artifact at macro expansion, the
//! sqlx/prost way. Full validation lives in `dspy-rs` (`Program::from_dsrs`
//! lowers through `ProgramBuilder` and runs `Program::validate`), and
//! `dspy-rs` depends on this proc-macro crate — depending on it back would be
//! a dependency cycle. Splitting the parser into a shared syntax crate is the
//! long-term clean answer; until that split, this module implements a
//! **standalone structural grammar check against the same surface grammar**
//! (`docs/dsrs-format.md`, mirror of `dspy-rs/src/ir/text/{lex,parse}.rs` —
//! keep in sync when the grammar changes).
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
//! real parser: it must never reject an artifact `Program::from_dsrs`
//! accepts.

/// A syntax failure with a 1-based source position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SyntaxError {
    pub line: u32,
    pub col: u32,
    pub message: String,
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, column {}: {}", self.line, self.col, self.message)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    line: u32,
    col: u32,
}

impl SyntaxError {
    fn at(span: Span, message: impl Into<String>) -> Self {
        Self {
            line: span.line,
            col: span.col,
            message: message.into(),
        }
    }
}

/// Token vocabulary — mirror of `dspy-rs/src/ir/text/lex.rs`.
#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Str,
    Num(String),
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Eq,
    Dot,
    Pipe,
    Question,
    At,
    Dollar,
    Caret,
    Colon,
    ColonColon,
    Arrow,
    Lt,
    Gt,
    /// A run of three-or-more backticks; the checker scans the raw code
    /// region from this token's start offset.
    Fence,
    Eof,
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Str => "a string".to_string(),
            Tok::Num(n) => format!("number `{n}`"),
            Tok::LBrace => "`{`".to_string(),
            Tok::RBrace => "`}`".to_string(),
            Tok::LParen => "`(`".to_string(),
            Tok::RParen => "`)`".to_string(),
            Tok::LBracket => "`[`".to_string(),
            Tok::RBracket => "`]`".to_string(),
            Tok::Comma => "`,`".to_string(),
            Tok::Eq => "`=`".to_string(),
            Tok::Dot => "`.`".to_string(),
            Tok::Pipe => "`|`".to_string(),
            Tok::Question => "`?`".to_string(),
            Tok::At => "`@`".to_string(),
            Tok::Dollar => "`$`".to_string(),
            Tok::Caret => "`^`".to_string(),
            Tok::Colon => "`:`".to_string(),
            Tok::ColonColon => "`::`".to_string(),
            Tok::Arrow => "`->`".to_string(),
            Tok::Lt => "`<`".to_string(),
            Tok::Gt => "`>`".to_string(),
            Tok::Fence => "a ``` code fence".to_string(),
            Tok::Eof => "end of file".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct Lexed {
    tok: Tok,
    span: Span,
    /// Byte offset of the token's first byte (fence raw scans restart here).
    start: usize,
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }

    fn bump(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn peek_byte(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek_byte() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => {
                    self.bump();
                }
                Some(b'/') if self.bytes.get(self.pos + 1) == Some(&b'/') => {
                    while let Some(b) = self.peek_byte() {
                        if b == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => break,
            }
        }
    }

    /// Repositions the cursor after a raw fence scan.
    fn seek(&mut self, pos: usize) {
        let mut line = 1u32;
        let mut col = 1u32;
        for &b in &self.bytes[..pos.min(self.bytes.len())] {
            if b == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        self.pos = pos;
        self.line = line;
        self.col = col;
    }

    fn next_token(&mut self) -> Result<Lexed, SyntaxError> {
        self.skip_trivia();
        let span = self.span();
        let start = self.pos;
        let Some(b) = self.peek_byte() else {
            return Ok(Lexed {
                tok: Tok::Eof,
                span,
                start,
            });
        };

        macro_rules! single {
            ($tok:expr) => {{
                self.bump();
                $tok
            }};
        }

        let tok = match b {
            b'{' => single!(Tok::LBrace),
            b'}' => single!(Tok::RBrace),
            b'(' => single!(Tok::LParen),
            b')' => single!(Tok::RParen),
            b'[' => single!(Tok::LBracket),
            b']' => single!(Tok::RBracket),
            b',' => single!(Tok::Comma),
            b'=' => single!(Tok::Eq),
            b'.' => single!(Tok::Dot),
            b'|' => single!(Tok::Pipe),
            b'?' => single!(Tok::Question),
            b'@' => single!(Tok::At),
            b'$' => single!(Tok::Dollar),
            b'^' => single!(Tok::Caret),
            b'<' => single!(Tok::Lt),
            b'>' => single!(Tok::Gt),
            b':' => {
                self.bump();
                if self.peek_byte() == Some(b':') {
                    self.bump();
                    Tok::ColonColon
                } else {
                    Tok::Colon
                }
            }
            b'-' if self.bytes.get(self.pos + 1) == Some(&b'>') => {
                self.bump();
                self.bump();
                Tok::Arrow
            }
            b'"' => {
                self.lex_string(span)?;
                Tok::Str
            }
            b'-' | b'0'..=b'9' => {
                let n = self.lex_number(span)?;
                Tok::Num(n)
            }
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                let ident_start = self.pos;
                while let Some(b) = self.peek_byte() {
                    if b.is_ascii_alphanumeric() || b == b'_' {
                        self.bump();
                    } else {
                        break;
                    }
                }
                Tok::Ident(self.src[ident_start..self.pos].to_string())
            }
            b'`' => {
                let mut ticks = 0usize;
                while self.peek_byte() == Some(b'`') {
                    ticks += 1;
                    self.bump();
                }
                if ticks < 3 {
                    return Err(SyntaxError::at(
                        span,
                        "unexpected ` — code fences are three or more backticks and only follow `js`",
                    ));
                }
                Tok::Fence
            }
            other => {
                return Err(SyntaxError::at(
                    span,
                    format!("unexpected character `{}`", char::from(other)),
                ));
            }
        };

        Ok(Lexed { tok, span, start })
    }

    /// Lexes a JSON string starting at the current `"` and validates its
    /// escapes with serde_json — exactly the real lexer's rule.
    fn lex_string(&mut self, span: Span) -> Result<(), SyntaxError> {
        let start = self.pos;
        self.bump(); // opening quote
        loop {
            match self.peek_byte() {
                None => {
                    return Err(SyntaxError::at(span, "unterminated string literal"));
                }
                Some(b'\\') => {
                    self.bump();
                    if self.bump().is_none() {
                        return Err(SyntaxError::at(span, "unterminated string literal"));
                    }
                }
                Some(b'"') => {
                    self.bump();
                    break;
                }
                Some(b'\n') => {
                    return Err(SyntaxError::at(
                        span,
                        "unterminated string literal (strings are JSON strings; escape newlines as \\n)",
                    ));
                }
                Some(_) => {
                    self.bump();
                }
            }
        }
        let raw = &self.src[start..self.pos];
        serde_json::from_str::<String>(raw)
            .map(|_| ())
            .map_err(|e| SyntaxError::at(span, format!("invalid string literal: {e}")))
    }

    fn lex_number(&mut self, span: Span) -> Result<String, SyntaxError> {
        let start = self.pos;
        if self.peek_byte() == Some(b'-') {
            self.bump();
        }
        let mut saw_digit = false;
        while let Some(b) = self.peek_byte() {
            match b {
                b'0'..=b'9' => {
                    saw_digit = true;
                    self.bump();
                }
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    self.bump();
                }
                _ => break,
            }
        }
        if !saw_digit {
            return Err(SyntaxError::at(span, "malformed number"));
        }
        let raw = &self.src[start..self.pos];
        if raw.parse::<f64>().is_err() {
            return Err(SyntaxError::at(span, format!("malformed number `{raw}`")));
        }
        Ok(raw.to_string())
    }

    /// Scans a raw fenced code block starting at byte `from` (the opening
    /// backtick run). Returns the byte offset one past the closing fence
    /// line. Mirrors `Lexer::scan_code_fence` in the real lexer.
    fn scan_code_fence(&self, from: usize) -> Result<usize, SyntaxError> {
        let span = {
            let mut probe = Lexer::new(self.src);
            probe.seek(from);
            probe.span()
        };
        let bytes = self.bytes;
        let mut i = from;
        let mut ticks = 0usize;
        while bytes.get(i) == Some(&b'`') {
            ticks += 1;
            i += 1;
        }
        if ticks < 3 {
            return Err(SyntaxError::at(
                span,
                "expected a code fence of at least three backticks (```) after `js`",
            ));
        }
        if bytes.get(i) == Some(&b'\r') {
            i += 1;
        }
        if bytes.get(i) != Some(&b'\n') {
            return Err(SyntaxError::at(
                span,
                "the opening code fence must be followed by a newline",
            ));
        }
        i += 1;
        let fence = vec![b'`'; ticks];
        let mut line_start = i;
        loop {
            let line_end = bytes[line_start..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|off| line_start + off);
            let (content_end, next_line) = match line_end {
                Some(e) => (e, e + 1),
                None => (bytes.len(), bytes.len()),
            };
            let mut trimmed_end = content_end;
            if trimmed_end > line_start && bytes[trimmed_end - 1] == b'\r' {
                trimmed_end -= 1;
            }
            if &bytes[line_start..trimmed_end] == fence.as_slice() {
                return Ok(next_line);
            }
            if line_end.is_none() {
                return Err(SyntaxError::at(
                    span,
                    format!(
                        "unterminated code block: no closing fence of {ticks} backticks on its own line"
                    ),
                ));
            }
            line_start = next_line;
        }
    }
}

// ---------------------------------------------------------------------------
// Structural checker
// ---------------------------------------------------------------------------

/// Checks `.dsrs` source for structural syntax validity. See the module docs
/// for the exact contract: this is syntax-only, and strictly more permissive
/// than `dspy_rs::ir::Program::from_dsrs`.
pub(crate) fn check(src: &str) -> Result<(), SyntaxError> {
    Checker::new(src)?.file()
}

struct Checker<'a> {
    lx: Lexer<'a>,
    cur: Lexed,
}

impl<'a> Checker<'a> {
    fn new(src: &'a str) -> Result<Self, SyntaxError> {
        let mut lx = Lexer::new(src);
        let cur = lx.next_token()?;
        Ok(Self { lx, cur })
    }

    fn bump(&mut self) -> Result<Lexed, SyntaxError> {
        let cur = std::mem::replace(&mut self.cur, self.lx.next_token()?);
        Ok(cur)
    }

    fn err(&self, message: impl Into<String>) -> SyntaxError {
        SyntaxError::at(self.cur.span, message)
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(&self.cur.tok, Tok::Ident(word) if word == kw)
    }

    fn expect_kw(&mut self, kw: &str, context: &str) -> Result<(), SyntaxError> {
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

    fn expect_ident(&mut self, context: &str) -> Result<(), SyntaxError> {
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

    fn expect_str(&mut self, context: &str) -> Result<(), SyntaxError> {
        match &self.cur.tok {
            Tok::Str => {
                self.bump()?;
                Ok(())
            }
            other => Err(self.err(format!(
                "expected a string {context}, found {}",
                other.describe()
            ))),
        }
    }

    fn expect_tok(&mut self, tok: Tok, context: &str) -> Result<(), SyntaxError> {
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
    fn skip_fence(&mut self) -> Result<(), SyntaxError> {
        debug_assert_eq!(self.cur.tok, Tok::Fence);
        let end = self.lx.scan_code_fence(self.cur.start)?;
        self.lx.seek(end);
        self.cur = self.lx.next_token()?;
        Ok(())
    }

    /// Consumes a balanced `{ … }` / `[ … ]` / `( … )` region, fence-aware.
    /// The current token must be the opening delimiter. Content is not
    /// inspected — everything semantic is the full parser's job.
    fn skip_balanced(&mut self, context: &str) -> Result<(), SyntaxError> {
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

    fn file(mut self) -> Result<(), SyntaxError> {
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
    use super::{SyntaxError, check};

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

    fn check_err(src: &str) -> SyntaxError {
        check(src).expect_err("expected a syntax error")
    }

    #[test]
    fn accepts_a_minimal_program() {
        check(MINI).expect("minimal program is syntactically valid");
    }

    // The load-bearing property: everything the full parser accepts, this
    // checker accepts. Run over the golden fixtures maintained next to the
    // real parser (in-repo paths; not part of the published package).
    #[test]
    fn parity_accepts_everything_the_full_parser_accepts() {
        let fixtures = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../dspy-rs/tests/fixtures"
        );
        let mut seen = 0usize;
        for entry in std::fs::read_dir(fixtures).expect("fixtures dir readable") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("dsrs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("fixture readable");
            check(&src).unwrap_or_else(|e| {
                panic!("syntax checker rejected {}: {e}", path.display())
            });
            seen += 1;
        }
        assert!(seen >= 3, "expected the golden .dsrs fixtures, found {seen}");
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
