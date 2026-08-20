//! Lexer for the `.dsrs` text format (RFC 0002 §4).
//!
//! Hand-rolled, position-tracking, and *lazy*: the parser pulls one token at a
//! time so that raw regions — ```` ``` ````-fenced code blocks and raw JSON
//! values — can be scanned in raw mode from an exact byte offset without the
//! tokenizer having speculated into them.
//!
//! Keywords are not distinguished here: they surface as [`Tok::Ident`] and the
//! parser dispatches on the string (the RFC grammar is keyword-led, so one
//! token of lookahead suffices).

use crate::ParseError;

/// A source position, 1-based.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    /// Identifier or keyword (the parser decides).
    Ident(String),
    /// JSON string literal, unescaped.
    Str(String),
    /// Numeric literal, raw text (parsed on demand as i64/u64/f32/f64).
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
    /// A run of three-or-more backticks (a code fence opener). The parser
    /// never advances *past* this token with `next_token`; it switches to
    /// [`Lexer::scan_code_fence`] at the token's start offset.
    Fence,
    Eof,
}

impl Tok {
    /// Human-readable token description for error messages.
    pub fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Str(_) => "a string".to_string(),
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

/// One lexed token with its source position and byte extent.
#[derive(Clone, Debug)]
pub struct Lexed {
    pub tok: Tok,
    pub span: Span,
    /// Byte offset of the first byte of the token (raw-mode scans restart
    /// here).
    pub start: usize,
}

pub struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
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
            // Column counting is byte-based; good enough for pointing at a spot.
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

    /// Repositions the cursor (used by the parser after raw-mode scans).
    pub fn seek(&mut self, pos: usize, span: Span) {
        self.pos = pos;
        self.line = span.line;
        self.col = span.col;
    }

    /// Line/col of an arbitrary byte offset (computed by rescanning; raw-mode
    /// scans are rare, so this stays off every hot path).
    pub fn span_at(&self, pos: usize) -> Span {
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
        Span { line, col }
    }

    pub fn next_token(&mut self) -> Result<Lexed, ParseError> {
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

        let tok = match b {
            b'{' => {
                self.bump();
                Tok::LBrace
            }
            b'}' => {
                self.bump();
                Tok::RBrace
            }
            b'(' => {
                self.bump();
                Tok::LParen
            }
            b')' => {
                self.bump();
                Tok::RParen
            }
            b'[' => {
                self.bump();
                Tok::LBracket
            }
            b']' => {
                self.bump();
                Tok::RBracket
            }
            b',' => {
                self.bump();
                Tok::Comma
            }
            b'=' => {
                self.bump();
                Tok::Eq
            }
            b'.' => {
                self.bump();
                Tok::Dot
            }
            b'|' => {
                self.bump();
                Tok::Pipe
            }
            b'?' => {
                self.bump();
                Tok::Question
            }
            b'@' => {
                self.bump();
                Tok::At
            }
            b'$' => {
                self.bump();
                Tok::Dollar
            }
            b'^' => {
                self.bump();
                Tok::Caret
            }
            b'<' => {
                self.bump();
                Tok::Lt
            }
            b'>' => {
                self.bump();
                Tok::Gt
            }
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
                let s = self.lex_string(span)?;
                Tok::Str(s)
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
                    return Err(ParseError::at(
                        span,
                        "unexpected ` — code fences are three or more backticks and only follow `js`",
                    ));
                }
                Tok::Fence
            }
            other => {
                return Err(ParseError::at(
                    span,
                    format!("unexpected character `{}`", char::from(other)),
                ));
            }
        };

        Ok(Lexed { tok, span, start })
    }

    /// Lexes a JSON string starting at the current `"`.
    fn lex_string(&mut self, span: Span) -> Result<String, ParseError> {
        let start = self.pos;
        self.bump(); // opening quote
        loop {
            match self.peek_byte() {
                None => {
                    return Err(ParseError::at(span, "unterminated string literal"));
                }
                Some(b'\\') => {
                    self.bump();
                    if self.bump().is_none() {
                        return Err(ParseError::at(span, "unterminated string literal"));
                    }
                }
                Some(b'"') => {
                    self.bump();
                    break;
                }
                Some(b'\n') => {
                    return Err(ParseError::at(
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
            .map_err(|e| ParseError::at(span, format!("invalid string literal: {e}")))
    }

    /// Lexes a JSON-style number starting at the current byte.
    fn lex_number(&mut self, span: Span) -> Result<String, ParseError> {
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
                    // Accept liberally; validated below by parsing.
                    self.bump();
                }
                _ => break,
            }
        }
        if !saw_digit {
            return Err(ParseError::at(span, "malformed number"));
        }
        let raw = &self.src[start..self.pos];
        if raw.parse::<f64>().is_err() {
            return Err(ParseError::at(span, format!("malformed number `{raw}`")));
        }
        Ok(raw.to_string())
    }

    /// Scans a raw fenced code block starting at byte `from` (which must point
    /// at the opening backtick run). Returns `(source, end_pos)`.
    ///
    /// Rules (canonical printer mirrors them): the opening fence is `k >= 3`
    /// backticks followed by a newline; the source is every byte up to (not
    /// including) the newline that precedes a line consisting of exactly `k`
    /// backticks and nothing else.
    pub fn scan_code_fence(&self, from: usize) -> Result<(String, usize), ParseError> {
        let span = self.span_at(from);
        let bytes = self.bytes;
        let mut i = from;
        let mut ticks = 0usize;
        while bytes.get(i) == Some(&b'`') {
            ticks += 1;
            i += 1;
        }
        if ticks < 3 {
            return Err(ParseError::at(
                span,
                "expected a code fence of at least three backticks (```) after `js`",
            ));
        }
        if bytes.get(i) == Some(&b'\r') {
            i += 1;
        }
        if bytes.get(i) != Some(&b'\n') {
            return Err(ParseError::at(
                span,
                "the opening code fence must be followed by a newline",
            ));
        }
        i += 1;
        let body_start = i;
        // Find a line that is exactly `ticks` backticks.
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
                // Source ends before the newline that precedes the fence line.
                let mut src_end = line_start;
                if src_end > body_start && bytes[src_end - 1] == b'\n' {
                    src_end -= 1;
                    if src_end > body_start && bytes[src_end - 1] == b'\r' {
                        src_end -= 1;
                    }
                }
                let source = self.src[body_start..src_end].to_string();
                return Ok((source, next_line));
            }
            if line_end.is_none() {
                return Err(ParseError::at(
                    span,
                    format!(
                        "unterminated code block: no closing fence of {ticks} backticks on its own line"
                    ),
                ));
            }
            line_start = next_line;
        }
    }

    /// Scans one raw JSON value starting at byte `from`. Returns the parsed
    /// value and the byte offset one past its end.
    pub fn scan_json(&self, from: usize) -> Result<(serde_json::Value, usize), ParseError> {
        let span = self.span_at(from);
        let rest = &self.src[from..];
        let mut de = serde_json::Deserializer::from_str(rest).into_iter::<serde_json::Value>();
        match de.next() {
            Some(Ok(value)) => Ok((value, from + de.byte_offset())),
            Some(Err(e)) => Err(ParseError::at(span, format!("invalid JSON: {e}"))),
            None => Err(ParseError::at(span, "expected a JSON value")),
        }
    }
}
