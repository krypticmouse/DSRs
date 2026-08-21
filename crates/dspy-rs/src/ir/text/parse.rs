//! Recursive-descent parser for the `.dsrs` text format (RFC 0002 §4).
//!
//! Hand-rolled LL(1)-after-keyword-dispatch, no parser generator. Every error
//! carries a line/column and says what was expected — these programs are
//! model-generated, and parse errors are the model's feedback signal.
//!
//! The parser reuses the Rust builder frontend ([`ProgramBuilder`] /
//! [`NodeSpec`]) for lowering, so both frontends construct the same runtime
//! [`Program`] by construction (RFC 0002 §4.4). A parallel *shadow tree*
//! records source spans so that post-lowering validation errors
//! ([`ValidateError`]) can still be reported with a position.

use std::collections::{HashMap, HashSet};

use crate::LMConfig;
use crate::ir::builder::{self, BuildError, NodeSpec, Port, ProgramBuilder};
use crate::ir::graph::{ModelId, NodeBudget, Program, SigId, ToolId};
use crate::ir::params::{ContextPolicy, DemoRow};
use crate::ir::sig::{ConstraintDef, FieldDef, RenderSpec, SignatureDef};
use crate::ir::validate::ValidateError;
use crate::typesys::{ClassDef, EnumDef, EnumValueDef, FieldType, TypeTable};

use super::ParseError;
use dsrs_syntax::lex::{Lexed, Lexer, Span, Tok};

/// Words that cannot be used as node/sig/tool/model/class/enum names.
const RESERVED: &[&str] = &[
    "dsrs", "program", "caps", "model", "sig", "class", "enum", "tool", "lineage", "main", "in",
    "out", "predict", "cot", "agent", "hole", "seq", "fork", "join", "route", "retry", "refine",
    "loop", "else", "js", "demos", "string", "int", "float", "bool", "map", "true", "false",
    "null", "while", "carry",
];

const EXPR_KEYWORDS: &[&str] = &[
    "predict", "cot", "agent", "hole", "seq", "fork", "route", "retry", "refine", "loop",
];

pub(crate) fn parse_program(src: &str) -> Result<Program, ParseError> {
    Parser::new(src)?.file()
}

// ---------------------------------------------------------------------------
// Shadow tree: spans for post-lowering error mapping
// ---------------------------------------------------------------------------

/// Mirrors the [`NodeSpec`] tree with source spans. Children are ordered
/// exactly as the builder lowers them (post-order id assignment), so node
/// entity displays (`n3`) can be recomputed and mapped back to positions.
struct Shadow {
    /// The leaf name for leaf nodes (`None` for containers — their entity
    /// display is the error handle).
    leaf: Option<String>,
    span: Span,
    children: Vec<Shadow>,
    /// Binding destinations declared on this node (args/out/join/carry), with
    /// the span of each destination identifier.
    binds: Vec<(String, Span)>,
}

impl Shadow {
    fn leaf(name: &str, span: Span) -> Self {
        Shadow {
            leaf: Some(name.to_string()),
            span,
            children: Vec::new(),
            binds: Vec::new(),
        }
    }

    fn container(span: Span) -> Self {
        Shadow {
            leaf: None,
            span,
            children: Vec::new(),
            binds: Vec::new(),
        }
    }
}

/// Span lookup tables keyed the way [`ValidateError`] names things.
struct SpanMaps {
    /// `at` handle (leaf name or `nK` entity display) → span.
    at: HashMap<String, Span>,
    /// `(at, dst_field)` → span of the binding destination.
    field: HashMap<(String, String), Span>,
}

fn build_span_maps(root: &Shadow) -> SpanMaps {
    fn walk(shadow: &Shadow, counter: &mut usize, maps: &mut SpanMaps) {
        for child in &shadow.children {
            walk(child, counter, maps);
        }
        let id = *counter;
        *counter += 1;
        let at = match &shadow.leaf {
            Some(name) => name.clone(),
            None => format!("n{id}"),
        };
        maps.at.insert(at.clone(), shadow.span);
        for (field, span) in &shadow.binds {
            maps.field.insert((at.clone(), field.clone()), *span);
        }
    }
    let mut maps = SpanMaps {
        at: HashMap::new(),
        field: HashMap::new(),
    };
    let mut counter = 0usize;
    walk(root, &mut counter, &mut maps);
    maps
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// A parsed-but-unregistered top declaration that owns a signature. Kept in
/// text order so signature arena ids follow declaration order.
enum SigItem {
    Sig {
        def: SignatureDef,
        span: Span,
    },
    Tool {
        name: String,
        desc: String,
        def: SignatureDef,
        caps: Vec<(String, Span)>,
        js: Option<String>,
        span: Span,
    },
}

struct Parser<'a> {
    lx: Lexer<'a>,
    cur: Lexed,

    builder: Option<ProgramBuilder>,
    program_caps: HashSet<String>,
    models: HashMap<String, ModelId>,
    sig_items: Vec<SigItem>,
    sigs: HashMap<String, SigId>,
    tools: HashMap<String, ToolId>,
    types: TypeTable,
    lineage: Option<crate::ir::graph::Lineage>,

    /// Node (leaf/step) names registered so far, with their spans.
    node_names: HashMap<String, Span>,
    /// Capability requirements of holes/tools, checked against the program
    /// ceiling after the whole file is parsed (decl order is free).
    deferred_caps: Vec<(String, Span, String)>,
    /// First span at which each class/enum token is referenced.
    type_spans: HashMap<String, Span>,
    /// First span at which each node name is referenced through a port.
    ref_spans: HashMap<String, Span>,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Result<Self, ParseError> {
        let mut lx = Lexer::new(src);
        let cur = lx.next_token()?;
        Ok(Self {
            lx,
            cur,
            builder: None,
            program_caps: HashSet::new(),
            models: HashMap::new(),
            sig_items: Vec::new(),
            sigs: HashMap::new(),
            tools: HashMap::new(),
            types: TypeTable::default(),
            lineage: None,
            node_names: HashMap::new(),
            deferred_caps: Vec::new(),
            type_spans: HashMap::new(),
            ref_spans: HashMap::new(),
        })
    }

    // -- token plumbing -----------------------------------------------------

    fn bump(&mut self) -> Result<Lexed, ParseError> {
        let next = self.lx.next_token()?;
        Ok(std::mem::replace(&mut self.cur, next))
    }

    fn err(&self, message: impl Into<String>) -> ParseError {
        ParseError::at(self.cur.span, message)
    }

    fn expect_tok(&mut self, tok: Tok, context: &str) -> Result<Lexed, ParseError> {
        if self.cur.tok == tok {
            self.bump()
        } else {
            Err(self.err(format!(
                "expected {} {context}, found {}",
                tok.describe(),
                self.cur.tok.describe()
            )))
        }
    }

    fn expect_ident(&mut self, context: &str) -> Result<(String, Span), ParseError> {
        match &self.cur.tok {
            Tok::Ident(name) => {
                let name = name.clone();
                let span = self.cur.span;
                self.bump()?;
                Ok((name, span))
            }
            other => Err(self.err(format!(
                "expected an identifier {context}, found {}",
                other.describe()
            ))),
        }
    }

    fn expect_name(&mut self, context: &str) -> Result<(String, Span), ParseError> {
        let (name, span) = self.expect_ident(context)?;
        if RESERVED.contains(&name.as_str()) {
            return Err(ParseError::at(
                span,
                format!("`{name}` is a reserved keyword and cannot be used as a name {context}"),
            ));
        }
        Ok((name, span))
    }

    fn expect_str(&mut self, context: &str) -> Result<(String, Span), ParseError> {
        match &self.cur.tok {
            Tok::Str(s) => {
                let s = s.clone();
                let span = self.cur.span;
                self.bump()?;
                Ok((s, span))
            }
            other => Err(self.err(format!(
                "expected a string {context}, found {}",
                other.describe()
            ))),
        }
    }

    fn expect_int<T: TryFrom<i64>>(&mut self, context: &str) -> Result<(T, Span), ParseError> {
        match &self.cur.tok {
            Tok::Num(raw) => {
                let span = self.cur.span;
                let value = raw.parse::<i64>().ok().and_then(|v| T::try_from(v).ok());
                match value {
                    Some(v) => {
                        self.bump()?;
                        Ok((v, span))
                    }
                    None => Err(ParseError::at(
                        span,
                        format!("`{raw}` is not a valid integer {context}"),
                    )),
                }
            }
            other => Err(self.err(format!(
                "expected an integer {context}, found {}",
                other.describe()
            ))),
        }
    }

    fn expect_bool(&mut self, context: &str) -> Result<bool, ParseError> {
        match &self.cur.tok {
            Tok::Ident(word) if word == "true" => {
                self.bump()?;
                Ok(true)
            }
            Tok::Ident(word) if word == "false" => {
                self.bump()?;
                Ok(false)
            }
            other => Err(self.err(format!(
                "expected `true` or `false` {context}, found {}",
                other.describe()
            ))),
        }
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(&self.cur.tok, Tok::Ident(word) if word == kw)
    }

    fn eat_kw(&mut self, kw: &str) -> Result<bool, ParseError> {
        if self.at_kw(kw) {
            self.bump()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect_kw(&mut self, kw: &str, context: &str) -> Result<Span, ParseError> {
        if self.at_kw(kw) {
            Ok(self.bump()?.span)
        } else {
            Err(self.err(format!(
                "expected `{kw}` {context}, found {}",
                self.cur.tok.describe()
            )))
        }
    }

    /// Re-syncs the token stream after a raw-mode scan ending at byte `end`.
    fn resync(&mut self, end: usize) -> Result<(), ParseError> {
        let span = self.lx.span_at(end);
        self.lx.seek(end, span);
        self.cur = self.lx.next_token()?;
        Ok(())
    }

    /// Scans one raw JSON value starting at the current token.
    fn raw_json(&mut self) -> Result<(serde_json::Value, Span), ParseError> {
        let span = self.cur.span;
        let (value, end) = self.lx.scan_json(self.cur.start)?;
        self.resync(end)?;
        Ok((value, span))
    }

    /// Scans a fenced code block; the current token must be the fence opener.
    fn raw_code(&mut self, context: &str) -> Result<String, ParseError> {
        if self.cur.tok != Tok::Fence {
            return Err(self.err(format!(
                "expected a ``` code fence {context}, found {}",
                self.cur.tok.describe()
            )));
        }
        let (source, end) = self.lx.scan_code_fence(self.cur.start)?;
        self.resync(end)?;
        Ok(source)
    }

    // -- file ---------------------------------------------------------------

    fn file(mut self) -> Result<Program, ParseError> {
        self.expect_kw("dsrs", "at the start of the file (`dsrs 1`)")?;
        let (format, format_span) = self.expect_int::<u32>("after `dsrs` (the format major)")?;
        if format != 1 {
            return Err(ParseError::at(
                format_span,
                format!("unsupported format major `{format}`: this parser reads `dsrs 1`"),
            ));
        }
        self.expect_kw("program", "after the `dsrs 1` pragma")?;
        let (program_name, _) = self.expect_name("after `program`")?;
        self.builder = Some(ProgramBuilder::new(&program_name));

        // Top-level declarations until `main`.
        loop {
            match &self.cur.tok {
                Tok::Ident(word) => match word.as_str() {
                    "caps" => self.caps_decl()?,
                    "model" => self.model_decl()?,
                    "sig" => self.sig_decl()?,
                    "class" => self.class_decl()?,
                    "enum" => self.enum_decl()?,
                    "tool" => self.tool_decl()?,
                    "lineage" => self.lineage_decl()?,
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

        // Register signatures and tools in declaration order (class/enum
        // tokens are resolvable now that every declaration has been seen).
        self.register_sig_items()?;

        // main: <Sig> = expr
        self.expect_kw("main", "")?;
        self.expect_tok(Tok::Colon, "after `main`")?;
        let (main_sig_name, main_sig_span) =
            self.expect_ident("after `main:` (the program signature name)")?;
        let main_sig = *self.sigs.get(&main_sig_name).ok_or_else(|| {
            ParseError::at(
                main_sig_span,
                format!(
                    "unknown sig `{main_sig_name}`: declare it with `sig {main_sig_name} {{ ... }}`"
                ),
            )
        })?;
        self.expect_tok(Tok::Eq, "after the main signature name")?;
        if !self.at_kw("seq") {
            return Err(self.err(format!(
                "the main expression must be `seq {{ ... }}`, found {}",
                self.cur.tok.describe()
            )));
        }
        let main_span = self.cur.span;
        let (root, root_shadow) = self.expr(None, main_span)?;

        if self.cur.tok != Tok::Eof {
            return Err(self.err(format!(
                "expected end of file after `main`, found {}",
                self.cur.tok.describe()
            )));
        }

        // Capability declarations: every hole/tool cap must be inside the
        // program ceiling (checked here with positions; `validate()` enforces
        // the same rule structurally).
        for (cap, span, owner) in &self.deferred_caps {
            if !self.program_caps.contains(cap) {
                return Err(ParseError::at(
                    *span,
                    format!(
                        "{owner} requires capability `{cap}`, which the program does not declare: \
                         add `{cap}` to the top-level `caps {{ ... }}` block"
                    ),
                ));
            }
        }

        let maps = build_span_maps(&root_shadow);
        let builder = self.builder.take().expect("builder present");
        let mut program = builder
            .main(main_sig, root)
            .map_err(|err| self.map_build_error(err, &maps, main_span))?;
        program.meta.lineage = self.lineage.take();
        Ok(program)
    }

    // -- top-level declarations --------------------------------------------

    fn caps_decl(&mut self) -> Result<(), ParseError> {
        self.bump()?; // caps
        self.expect_tok(Tok::LBrace, "after `caps`")?;
        while self.cur.tok != Tok::RBrace {
            let (cap, _) = self.cap("inside `caps { ... }`")?;
            self.program_caps.insert(cap.clone());
            self.builder.as_mut().expect("builder").cap(&cap);
        }
        self.bump()?; // }
        Ok(())
    }

    /// One capability name: `IDENT (":" IDENT)*`.
    fn cap(&mut self, context: &str) -> Result<(String, Span), ParseError> {
        let (mut cap, span) = self.expect_ident(context)?;
        while self.cur.tok == Tok::Colon {
            self.bump()?;
            let (part, _) = self.expect_ident("after `:` in a capability name")?;
            cap.push(':');
            cap.push_str(&part);
        }
        Ok((cap, span))
    }

    fn model_decl(&mut self) -> Result<(), ParseError> {
        self.bump()?; // model
        let (name, span) = self.expect_name("after `model`")?;
        if self.models.contains_key(&name) {
            return Err(ParseError::at(
                span,
                format!("duplicate model name `{name}`"),
            ));
        }
        self.expect_tok(Tok::Eq, "after the model name")?;
        let (model_str, _) = self.expect_str("after `=` (the provider model string)")?;
        let mut config = LMConfig {
            model: model_str,
            ..LMConfig::default()
        };
        if self.cur.tok == Tok::LBrace {
            self.bump()?;
            while self.cur.tok != Tok::RBrace {
                let (key, key_span) = self.expect_ident("as a model option key")?;
                match key.as_str() {
                    "base_url" => config.base_url = Some(self.expect_str("after `base_url`")?.0),
                    "temperature" => match &self.cur.tok {
                        Tok::Num(raw) => {
                            config.temperature = raw.parse::<f32>().map_err(|_| {
                                self.err(format!("`{raw}` is not a valid float for `temperature`"))
                            })?;
                            self.bump()?;
                        }
                        other => {
                            return Err(self.err(format!(
                                "expected a number after `temperature`, found {}",
                                other.describe()
                            )));
                        }
                    },
                    "max_tokens" => config.max_tokens = self.expect_int("after `max_tokens`")?.0,
                    "max_tool_iterations" => {
                        config.max_tool_iterations =
                            self.expect_int("after `max_tool_iterations`")?.0
                    }
                    "max_retries" => config.max_retries = self.expect_int("after `max_retries`")?.0,
                    "retry_base_delay_ms" => {
                        config.retry_base_delay_ms =
                            self.expect_int("after `retry_base_delay_ms`")?.0
                    }
                    "cache" => config.cache = self.expect_bool("after `cache`")?,
                    other => {
                        return Err(ParseError::at(
                            key_span,
                            format!(
                                "unknown model option `{other}`: expected `base_url`, \
                                 `temperature`, `max_tokens`, `max_tool_iterations`, \
                                 `max_retries`, `retry_base_delay_ms`, or `cache`"
                            ),
                        ));
                    }
                }
            }
            self.bump()?; // }
        }
        let id = self.builder.as_mut().expect("builder").model(&name, config);
        self.models.insert(name, id);
        Ok(())
    }

    fn sig_decl(&mut self) -> Result<(), ParseError> {
        self.bump()?; // sig
        let (name, span) = self.expect_name("after `sig`")?;
        if self.sig_items.iter().any(|item| match item {
            SigItem::Sig { def, .. } => *def.name == name,
            _ => false,
        }) {
            return Err(ParseError::at(span, format!("duplicate sig name `{name}`")));
        }
        self.expect_tok(Tok::LBrace, "after the sig name")?;
        let mut sig = SignatureDef::build(&name);
        if let Tok::Str(instruction) = &self.cur.tok {
            let instruction = instruction.clone();
            self.bump()?;
            sig = sig.instruction(&instruction);
        }
        let mut any_field = false;
        while self.cur.tok != Tok::RBrace {
            let (field, is_input) = self.sig_field()?;
            sig = if is_input {
                sig.input_full(field)
            } else {
                sig.output_full(field)
            };
            any_field = true;
        }
        if !any_field {
            return Err(self.err(format!(
                "sig `{name}` has no fields: declare at least one `in` and one `out` field"
            )));
        }
        self.bump()?; // }
        let def = sig
            .finish()
            .map_err(|e| ParseError::at(span, format!("invalid sig `{name}`: {e}")))?;
        self.sig_items.push(SigItem::Sig { def, span });
        Ok(())
    }

    /// One `in`/`out` field with its metadata. Returns `(field, is_input)`.
    fn sig_field(&mut self) -> Result<(FieldDef, bool), ParseError> {
        let is_input = match &self.cur.tok {
            Tok::Ident(word) if word == "in" => true,
            Tok::Ident(word) if word == "out" => false,
            other => {
                return Err(self.err(format!(
                    "expected `in` or `out` to declare a field, found {}",
                    other.describe()
                )));
            }
        };
        self.bump()?;
        let (name, _) = self.expect_name("as the field name")?;
        self.expect_tok(Tok::Colon, "after the field name")?;
        let ty = self.type_expr()?;
        let mut field = FieldDef::new(&name, ty);
        // Field metadata, any order; canonical print order is docs, alias,
        // constraints, format/jinja.
        loop {
            match &self.cur.tok {
                Tok::Str(docs) => {
                    let docs = docs.clone();
                    self.bump()?;
                    field = field.with_docs(&docs);
                }
                Tok::Ident(word) if word == "alias" => {
                    self.bump()?;
                    let (alias, _) = self.expect_str("after `alias`")?;
                    field = field.aliased(&alias);
                }
                Tok::Ident(word) if word == "check" || word == "assert" => {
                    let is_check = word == "check";
                    self.bump()?;
                    let (expr, label) = self.constraint_args(is_check)?;
                    field = field.with_constraint(if is_check {
                        ConstraintDef::check(&label, &expr)
                    } else if label.is_empty() {
                        ConstraintDef::assert(&expr)
                    } else {
                        ConstraintDef {
                            kind: crate::core::ConstraintKind::Assert,
                            label: label.into(),
                            expr: expr.into(),
                        }
                    });
                }
                Tok::Ident(word) if word == "format" => {
                    self.bump()?;
                    let (value, _) = self.expect_str("after `format`")?;
                    field = field.with_render(RenderSpec::Format(value.into()));
                }
                Tok::Ident(word) if word == "jinja" => {
                    self.bump()?;
                    let (template, _) = self.expect_str("after `jinja`")?;
                    field = field.with_render(RenderSpec::Jinja(template.into()));
                }
                _ => break,
            }
        }
        Ok((field, is_input))
    }

    /// `check("expr", "label")` / `assert("expr")` / `assert("expr", "label")`.
    fn constraint_args(&mut self, is_check: bool) -> Result<(String, String), ParseError> {
        self.expect_tok(Tok::LParen, "after `check`/`assert`")?;
        let (expr, _) = self.expect_str("as the constraint expression")?;
        let label = if self.cur.tok == Tok::Comma {
            self.bump()?;
            self.expect_str("as the constraint label")?.0
        } else if is_check {
            return Err(self.err("`check` requires a label: write check(\"<expr>\", \"<label>\")"));
        } else {
            String::new()
        };
        self.expect_tok(Tok::RParen, "to close the constraint")?;
        Ok((expr, label))
    }

    fn class_decl(&mut self) -> Result<(), ParseError> {
        self.bump()?; // class
        let (token, span) = self.qualified_name("after `class`")?;
        if self.types.classes.contains_key(&token) {
            return Err(ParseError::at(
                span,
                format!("duplicate class name `{token}`"),
            ));
        }
        let rendered = if self.eat_kw("alias")? {
            self.expect_str("after `alias`")?.0
        } else {
            token.clone()
        };
        self.expect_tok(Tok::LBrace, "after the class name")?;
        let docs = match &self.cur.tok {
            Tok::Str(docs) => {
                let docs = docs.clone();
                self.bump()?;
                Some(docs)
            }
            _ => None,
        };
        let mut fields = Vec::new();
        while self.cur.tok != Tok::RBrace {
            let (name, _) = self.expect_name("as a class field name")?;
            self.expect_tok(Tok::Colon, "after the class field name")?;
            let ty = self.type_expr()?;
            let mut rendered_name = name.clone();
            let mut field_docs = None;
            let mut constraints = Vec::new();
            loop {
                match &self.cur.tok {
                    Tok::Str(docs) => {
                        field_docs = Some(docs.clone());
                        self.bump()?;
                    }
                    Tok::Ident(word) if word == "alias" => {
                        self.bump()?;
                        rendered_name = self.expect_str("after `alias`")?.0;
                    }
                    Tok::Ident(word) if word == "check" || word == "assert" => {
                        let is_check = word == "check";
                        self.bump()?;
                        let (expr, label) = self.constraint_args(is_check)?;
                        constraints.push(crate::typesys::Constraint {
                            level: if is_check {
                                crate::typesys::ConstraintKind::Check
                            } else {
                                crate::typesys::ConstraintKind::Assert
                            },
                            label: (!label.is_empty()).then_some(label),
                            expression: expr,
                        });
                    }
                    _ => break,
                }
            }
            fields.push(crate::typesys::FieldDef {
                name,
                rendered_name,
                field_type: ty,
                docs: field_docs,
                constraints,
            });
        }
        self.bump()?; // }
        if fields.is_empty() {
            return Err(ParseError::at(
                span,
                format!("class `{token}` has no fields"),
            ));
        }
        self.types.classes.insert(
            token.clone(),
            ClassDef {
                internal_name: token,
                rendered_name: rendered,
                docs,
                fields,
                constraints: Vec::new(),
            },
        );
        Ok(())
    }

    fn enum_decl(&mut self) -> Result<(), ParseError> {
        self.bump()?; // enum
        let (token, span) = self.qualified_name("after `enum`")?;
        if self.types.enums.contains_key(&token) {
            return Err(ParseError::at(
                span,
                format!("duplicate enum name `{token}`"),
            ));
        }
        let rendered = if self.eat_kw("alias")? {
            self.expect_str("after `alias`")?.0
        } else {
            token.clone()
        };
        self.expect_tok(Tok::LBrace, "after the enum name")?;
        let docs = match &self.cur.tok {
            Tok::Str(docs) => {
                let docs = docs.clone();
                self.bump()?;
                Some(docs)
            }
            _ => None,
        };
        let mut values = Vec::new();
        while self.cur.tok != Tok::RBrace {
            let (name, _) = self.expect_name("as an enum value")?;
            let rendered_name = if self.eat_kw("alias")? {
                self.expect_str("after `alias`")?.0
            } else {
                name.clone()
            };
            let value_docs = match &self.cur.tok {
                Tok::Str(docs) => {
                    let docs = docs.clone();
                    self.bump()?;
                    Some(docs)
                }
                _ => None,
            };
            values.push(EnumValueDef {
                name,
                rendered_name,
                docs: value_docs,
            });
        }
        self.bump()?; // }
        if values.is_empty() {
            return Err(ParseError::at(span, format!("enum `{token}` is empty")));
        }
        self.types.enums.insert(
            token.clone(),
            EnumDef {
                internal_name: token,
                rendered_name: rendered,
                docs,
                values,
            },
        );
        Ok(())
    }

    fn tool_decl(&mut self) -> Result<(), ParseError> {
        self.bump()?; // tool
        let (name, span) = self.expect_name("after `tool`")?;
        if self.sig_items.iter().any(|item| match item {
            SigItem::Tool { name: existing, .. } => *existing == name,
            _ => false,
        }) {
            return Err(ParseError::at(
                span,
                format!("duplicate tool name `{name}`"),
            ));
        }
        let (desc, _) = self.expect_str("after the tool name (the tool description)")?;
        let mut caps = Vec::new();
        if self.at_kw("caps") {
            self.bump()?;
            self.expect_tok(Tok::LBracket, "after `caps`")?;
            while self.cur.tok != Tok::RBracket {
                let (cap, cap_span) = self.cap("inside `caps [ ... ]`")?;
                self.deferred_caps
                    .push((cap.clone(), cap_span, format!("tool `{name}`")));
                caps.push((cap, cap_span));
            }
            self.bump()?; // ]
        }
        self.expect_tok(Tok::LBrace, "to open the tool interface")?;
        let mut sig = SignatureDef::build(&name);
        while self.cur.tok != Tok::RBrace {
            let (field, is_input) = self.sig_field()?;
            sig = if is_input {
                sig.input_full(field)
            } else {
                sig.output_full(field)
            };
        }
        self.bump()?; // }
        let js = if self.at_kw("js") {
            self.bump()?;
            Some(self.raw_code("after `js`")?)
        } else {
            None
        };
        let def = sig
            .finish()
            .map_err(|e| ParseError::at(span, format!("invalid tool `{name}` interface: {e}")))?;
        self.sig_items.push(SigItem::Tool {
            name,
            desc,
            def,
            caps,
            js,
            span,
        });
        Ok(())
    }

    fn lineage_decl(&mut self) -> Result<(), ParseError> {
        self.bump()?; // lineage
        self.expect_tok(Tok::LBrace, "after `lineage`")?;
        let mut optimizer = String::new();
        let mut trainset = String::new();
        let mut budget = String::new();
        let mut parent = None;
        let mut overlay = None;
        let mut date = String::new();
        while self.cur.tok != Tok::RBrace {
            let (key, key_span) = self.expect_ident("as a lineage key")?;
            let (value, _) = self.expect_str("as the lineage value")?;
            match key.as_str() {
                "optimizer" => optimizer = value,
                "trainset" => trainset = value,
                "budget" => budget = value,
                "parent" => parent = Some(value.into_boxed_str()),
                "overlay" => overlay = Some(value.into_boxed_str()),
                "date" => date = value,
                other => {
                    return Err(ParseError::at(
                        key_span,
                        format!(
                            "unknown lineage key `{other}`: expected `optimizer`, `trainset`, \
                             `budget`, `parent`, `overlay`, or `date`"
                        ),
                    ));
                }
            }
        }
        self.bump()?; // }
        self.lineage = Some(crate::ir::graph::Lineage {
            optimizer: optimizer.into(),
            trainset: trainset.into(),
            budget: budget.into(),
            parent,
            overlay,
            date: date.into(),
        });
        Ok(())
    }

    /// Resolves class↔enum tokens and registers signatures/tools with the
    /// builder in declaration order.
    fn register_sig_items(&mut self) -> Result<(), ParseError> {
        let enums: HashSet<String> = self.types.enums.keys().cloned().collect();
        let builder = self.builder.as_mut().expect("builder");
        builder.add_types(&self.types);
        for item in std::mem::take(&mut self.sig_items) {
            match item {
                SigItem::Sig { mut def, span } => {
                    fixup_sig(&mut def, &enums);
                    let name = def.name.to_string();
                    let id = builder.sig(def);
                    self.sigs.insert(name, id);
                    let _ = span;
                }
                SigItem::Tool {
                    name,
                    desc,
                    mut def,
                    caps,
                    js,
                    span,
                } => {
                    fixup_sig(&mut def, &enums);
                    let sig_id = builder.sig(def);
                    let cap_strs: Vec<&str> = caps.iter().map(|(c, _)| c.as_str()).collect();
                    let id = match js {
                        None => builder.host_tool(&name, &desc, sig_id, &cap_strs),
                        Some(js) => builder.sandboxed_tool(&name, &desc, sig_id, &cap_strs, &js),
                    };
                    self.tools.insert(name, id);
                    let _ = span;
                }
            }
        }
        Ok(())
    }

    // -- types --------------------------------------------------------------

    /// `IDENT ("::" IDENT)*` — class/enum tokens may be path-qualified (the
    /// static lane names user types by module path).
    fn qualified_name(&mut self, context: &str) -> Result<(String, Span), ParseError> {
        let (mut name, span) = self.expect_name(context)?;
        while self.cur.tok == Tok::ColonColon {
            self.bump()?;
            let (part, _) = self.expect_ident("after `::`")?;
            name.push_str("::");
            name.push_str(&part);
        }
        Ok((name, span))
    }

    fn type_expr(&mut self) -> Result<FieldType, ParseError> {
        let mut units = vec![self.type_unit()?];
        while self.cur.tok == Tok::Pipe {
            self.bump()?;
            units.push(self.type_unit()?);
        }
        Ok(if units.len() == 1 {
            units.pop().unwrap()
        } else {
            FieldType::Union(units)
        })
    }

    fn type_unit(&mut self) -> Result<FieldType, ParseError> {
        let mut ty = self.type_prim()?;
        loop {
            match &self.cur.tok {
                Tok::LBracket => {
                    self.bump()?;
                    self.expect_tok(Tok::RBracket, "to close `[]`")?;
                    ty = FieldType::List(Box::new(ty));
                }
                Tok::Question => {
                    self.bump()?;
                    ty = FieldType::optional(ty);
                }
                _ => break,
            }
        }
        Ok(ty)
    }

    fn type_prim(&mut self) -> Result<FieldType, ParseError> {
        match &self.cur.tok {
            Tok::Ident(word) => match word.as_str() {
                "string" => {
                    self.bump()?;
                    Ok(FieldType::String)
                }
                "int" => {
                    self.bump()?;
                    Ok(FieldType::Int)
                }
                "float" => {
                    self.bump()?;
                    Ok(FieldType::Float)
                }
                "bool" => {
                    self.bump()?;
                    Ok(FieldType::Bool)
                }
                "map" => {
                    self.bump()?;
                    self.expect_tok(Tok::Lt, "after `map`")?;
                    let value = self.type_expr()?;
                    self.expect_tok(Tok::Gt, "to close `map<...>`")?;
                    Ok(FieldType::Map(Box::new(FieldType::String), Box::new(value)))
                }
                _ => {
                    let (token, span) = self.qualified_name("as a type")?;
                    self.type_spans.entry(token.clone()).or_insert(span);
                    // Provisionally a class; re-tagged as Enum during
                    // registration when the token names an enum.
                    Ok(FieldType::Class(token))
                }
            },
            Tok::Str(value) => {
                let value = value.clone();
                self.bump()?;
                Ok(FieldType::Literal(value))
            }
            Tok::LParen => {
                self.bump()?;
                let ty = self.type_expr()?;
                self.expect_tok(Tok::RParen, "to close the type group")?;
                Ok(ty)
            }
            other => Err(self.err(format!(
                "expected a type (string, int, float, bool, map<...>, a class/enum name, or a \
                 \"literal\"), found {}",
                other.describe()
            ))),
        }
    }

    // -- expressions --------------------------------------------------------

    /// Parses one expression. `name` is the binding name from the enclosing
    /// step/target position; leaves require it, containers register it.
    fn expr(
        &mut self,
        name: Option<(String, Span)>,
        kw_span: Span,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        let keyword = match &self.cur.tok {
            Tok::Ident(word) => word.clone(),
            other => {
                return Err(self.err(format!(
                    "expected an expression keyword ({}), found {}",
                    EXPR_KEYWORDS.join(", "),
                    other.describe()
                )));
            }
        };
        match keyword.as_str() {
            "predict" | "cot" | "agent" => self.lm_leaf(&keyword, name),
            "hole" => self.hole(name),
            "seq" => self.seq(name, kw_span),
            "fork" => self.fork(name, kw_span),
            "route" => self.route(name, kw_span),
            "retry" => self.retry(name, kw_span),
            "refine" => self.refine(name, kw_span),
            "loop" => self.loop_(name, kw_span),
            other => Err(self.err(format!(
                "unknown expression keyword `{other}`: expected one of {}",
                EXPR_KEYWORDS.join(", ")
            ))),
        }
    }

    /// A step or target position: `name = expr` (leaves and named containers)
    /// or a bare container expression.
    fn target(&mut self) -> Result<(NodeSpec, Shadow), ParseError> {
        match &self.cur.tok {
            Tok::Ident(word) if EXPR_KEYWORDS.contains(&word.as_str()) => {
                let kw_span = self.cur.span;
                let (spec, shadow) = self.expr(None, kw_span)?;
                Ok((spec, shadow))
            }
            Tok::Ident(_) => {
                let (name, span) = self.expect_name("as the node name")?;
                self.register_name(&name, span)?;
                self.expect_tok(Tok::Eq, "after the node name")?;
                let kw_span = self.cur.span;
                self.expr(Some((name, span)), kw_span)
            }
            other => Err(self.err(format!(
                "expected `name = <expr>` or a container expression, found {}",
                other.describe()
            ))),
        }
    }

    fn register_name(&mut self, name: &str, span: Span) -> Result<(), ParseError> {
        if let Some(_previous) = self.node_names.get(name) {
            return Err(ParseError::at(
                span,
                format!("duplicate name `{name}`: node and step names are program-unique"),
            ));
        }
        self.node_names.insert(name.to_string(), span);
        Ok(())
    }

    fn require_leaf_name(
        &self,
        name: Option<(String, Span)>,
        keyword: &str,
    ) -> Result<(String, Span), ParseError> {
        name.ok_or_else(|| {
            self.err(format!(
                "a `{keyword}` node needs a name here: write `name = {keyword} ...`"
            ))
        })
    }

    fn resolve_sig(&mut self) -> Result<SigId, ParseError> {
        let (name, span) = self.expect_ident("as the signature name")?;
        self.sigs.get(&name).copied().ok_or_else(|| {
            ParseError::at(
                span,
                format!("unknown sig `{name}`: declare it with `sig {name} {{ ... }}`"),
            )
        })
    }

    fn modelref(&mut self) -> Result<Option<ModelId>, ParseError> {
        if self.cur.tok != Tok::At {
            return Ok(None);
        }
        self.bump()?;
        let (name, span) = self.expect_ident("after `@` (a declared model name)")?;
        match self.models.get(&name) {
            Some(id) => Ok(Some(*id)),
            None => Err(ParseError::at(
                span,
                format!("unknown model `@{name}`: declare it with `model {name} = \"...\"`"),
            )),
        }
    }

    /// `predict` / `cot` / `agent` leaves.
    fn lm_leaf(
        &mut self,
        keyword: &str,
        name: Option<(String, Span)>,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // keyword
        let (name, name_span) = self.require_leaf_name(name, keyword)?;
        let sig = self.resolve_sig()?;
        let model = self.modelref()?;
        let mut shadow = Shadow::leaf(&name, name_span);
        let mut spec = match keyword {
            "predict" => builder::predict(&name, sig),
            "cot" => builder::cot(&name, sig),
            _ => builder::agent(&name, sig),
        };
        if let Some(model) = model {
            spec = spec.model(model);
        }
        spec = self.args(spec, &mut shadow)?;
        if keyword == "agent" {
            spec = self.agent_opts(spec)?;
        } else if self.cur.tok == Tok::LBrace {
            self.bump()?;
            while self.cur.tok != Tok::RBrace {
                let (key, key_span) = self.expect_ident("as a predict option")?;
                match key.as_str() {
                    "instruction" => {
                        let (text, _) = self.expect_str("after `instruction`")?;
                        spec = spec.instruction(&text);
                    }
                    "demos" => spec = spec.demos(self.demos_value()?),
                    "render" => {
                        let (mode, span) = self.expect_str("after `render`")?;
                        let mode = crate::ir::params::RenderMode::from_str_opt(&mode)
                            .ok_or_else(|| {
                                ParseError::at(
                                    span,
                                    format!(
                                        "unknown render mode `{mode}`: expected \
                                         \"markers\" or \"bare\""
                                    ),
                                )
                            })?;
                        spec = spec.render(mode);
                    }
                    other => {
                        return Err(ParseError::at(
                            key_span,
                            format!(
                                "unknown option `{other}` in a `{keyword}` block: expected \
                                 `instruction`, `demos`, or `render`"
                            ),
                        ));
                    }
                }
            }
            self.bump()?; // }
        }
        Ok((spec, shadow))
    }

    fn demos_value(&mut self) -> Result<Vec<DemoRow>, ParseError> {
        let (value, span) = self.raw_json()?;
        serde_json::from_value::<Vec<DemoRow>>(value).map_err(|e| {
            ParseError::at(
                span,
                format!(
                    "invalid demos array: {e} (expected \
                     [{{\"input\": {{...}}, \"output\": {{...}}}}, ...])"
                ),
            )
        })
    }

    fn agent_opts(&mut self, mut spec: NodeSpec) -> Result<NodeSpec, ParseError> {
        self.expect_tok(
            Tok::LBrace,
            "to open the agent options block (it may be empty: `{ }`)",
        )?;
        while self.cur.tok != Tok::RBrace {
            let (key, key_span) = self.expect_ident("as an agent option")?;
            match key.as_str() {
                "tools" => spec = spec.tools(self.tool_list("tools")?),
                "tool_set" => spec = spec.tool_set(self.tool_list("tool_set")?),
                "stop_tools" => spec = spec.stop_tools(self.tool_list("stop_tools")?),
                "max_turns" => {
                    let (turns, span) = self.expect_int::<u32>("after `max_turns`")?;
                    if turns == 0 {
                        return Err(ParseError::at(span, "`max_turns` must be at least 1"));
                    }
                    spec = spec.max_turns(turns);
                }
                "until_parse" => spec = spec.until_parse(self.expect_bool("after `until_parse`")?),
                "budget" => {
                    let mut budget = NodeBudget::default();
                    self.expect_tok(Tok::LBrace, "after `budget`")?;
                    while self.cur.tok != Tok::RBrace {
                        let (bkey, bspan) = self.expect_ident("as a budget key")?;
                        match bkey.as_str() {
                            "calls" => {
                                budget.max_lm_calls = Some(self.expect_int("after `calls`")?.0)
                            }
                            "tokens" => {
                                budget.max_tokens = Some(self.expect_int("after `tokens`")?.0)
                            }
                            "deadline_ms" => {
                                budget.deadline_ms = Some(self.expect_int("after `deadline_ms`")?.0)
                            }
                            "on_exhausted" => {
                                let (policy, pspan) = self.expect_ident("after `on_exhausted`")?;
                                budget.on_exhausted = match policy.as_str() {
                                    "fail" => crate::ir::graph::BudgetPolicy::Fail,
                                    "finalize" => crate::ir::graph::BudgetPolicy::Finalize,
                                    other => {
                                        return Err(ParseError::at(
                                            pspan,
                                            format!(
                                                "unknown budget policy `{other}`: expected \
                                                 `fail` or `finalize`"
                                            ),
                                        ));
                                    }
                                };
                            }
                            other => {
                                return Err(ParseError::at(
                                    bspan,
                                    format!(
                                        "unknown budget key `{other}`: expected `calls`, \
                                         `tokens`, `deadline_ms`, or `on_exhausted`"
                                    ),
                                ));
                            }
                        }
                    }
                    self.bump()?; // }
                    spec = spec.budget(budget);
                }
                "context" => {
                    let mut policy = ContextPolicy::default();
                    self.expect_tok(Tok::LBrace, "after `context`")?;
                    while self.cur.tok != Tok::RBrace {
                        let (ckey, cspan) = self.expect_ident("as a context key")?;
                        match ckey.as_str() {
                            "max_history_turns" => {
                                policy.max_history_turns =
                                    Some(self.expect_int("after `max_history_turns`")?.0)
                            }
                            "tool_result_max_bytes" => {
                                policy.tool_result_max_bytes =
                                    Some(self.expect_int("after `tool_result_max_bytes`")?.0)
                            }
                            "playbook" => {
                                policy.playbook = Some(self.expect_str("after `playbook`")?.0)
                            }
                            other => {
                                return Err(ParseError::at(
                                    cspan,
                                    format!(
                                        "unknown context key `{other}`: expected \
                                         `max_history_turns`, `tool_result_max_bytes`, or \
                                         `playbook`"
                                    ),
                                ));
                            }
                        }
                    }
                    self.bump()?; // }
                    spec = spec.context(policy);
                }
                "instruction" => {
                    let (text, _) = self.expect_str("after `instruction`")?;
                    spec = spec.instruction(&text);
                }
                "demos" => spec = spec.demos(self.demos_value()?),
                other => {
                    return Err(ParseError::at(
                        key_span,
                        format!(
                            "unknown agent option `{other}`: expected `tools`, `tool_set`, \
                             `stop_tools`, `max_turns`, `until_parse`, `budget`, `context`, \
                             `instruction`, or `demos`"
                        ),
                    ));
                }
            }
        }
        self.bump()?; // }
        Ok(spec)
    }

    fn tool_list(&mut self, context: &str) -> Result<Vec<ToolId>, ParseError> {
        self.expect_tok(Tok::LBracket, &format!("after `{context}`"))?;
        let mut ids = Vec::new();
        while self.cur.tok != Tok::RBracket {
            let (name, span) = self.expect_ident("as a tool name")?;
            let id = self.tools.get(&name).copied().ok_or_else(|| {
                ParseError::at(
                    span,
                    format!(
                        "unknown tool `{name}`: declare it with `tool {name} \"...\" {{ ... }}`"
                    ),
                )
            })?;
            ids.push(id);
        }
        self.bump()?; // ]
        Ok(ids)
    }

    fn hole(&mut self, name: Option<(String, Span)>) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // hole
        let (name, name_span) = self.require_leaf_name(name, "hole")?;
        let sig = self.resolve_sig()?;
        let mut shadow = Shadow::leaf(&name, name_span);
        let mut binds: Vec<(String, Port)> = Vec::new();
        if self.cur.tok == Tok::LParen {
            self.bump()?;
            while self.cur.tok != Tok::RParen {
                let (field, port, field_span) = self.bind()?;
                shadow.binds.push((field.clone(), field_span));
                binds.push((field, port));
                if self.cur.tok == Tok::Comma {
                    self.bump()?;
                } else {
                    break;
                }
            }
            self.expect_tok(Tok::RParen, "to close the argument list")?;
        }
        self.expect_kw(
            "caps",
            "after the hole arguments (every hole declares `caps [ ... ]`)",
        )?;
        self.expect_tok(Tok::LBracket, "after `caps`")?;
        let mut caps = Vec::new();
        while self.cur.tok != Tok::RBracket {
            let (cap, cap_span) = self.cap("inside `caps [ ... ]`")?;
            self.deferred_caps
                .push((cap.clone(), cap_span, format!("hole `{name}`")));
            caps.push(cap);
        }
        self.bump()?; // ]
        let cap_refs: Vec<&str> = caps.iter().map(String::as_str).collect();
        // `js <fence>` (sandboxed) or `extern "<hash>"` (host-bound, RFC 0003).
        let mut spec = if self.at_kw("extern") {
            self.bump()?; // extern
            let (text, span) = self.expect_str("after `extern` (the host implementation hash)")?;
            let hash = u64::from_str_radix(&text, 16).map_err(|_| {
                ParseError::at(
                    span,
                    format!("`extern` hash must be 16 hex digits, got `{text}`"),
                )
            })?;
            builder::extern_hole(&name, sig, hash, &cap_refs)
        } else {
            self.expect_kw("js", "after the hole capability list (or `extern \"<hash>\"`)")?;
            let code = self.raw_code("after `js`")?;
            builder::hole(&name, sig, &code, &cap_refs)
        };
        for (field, port) in binds {
            spec = spec.bind(&field, port);
        }
        Ok((spec, shadow))
    }

    /// Argument list on a predict/cot/agent leaf.
    fn args(&mut self, mut spec: NodeSpec, shadow: &mut Shadow) -> Result<NodeSpec, ParseError> {
        if self.cur.tok != Tok::LParen {
            return Ok(spec);
        }
        self.bump()?;
        while self.cur.tok != Tok::RParen {
            let (field, port, field_span) = self.bind()?;
            shadow.binds.push((field.clone(), field_span));
            spec = spec.bind(&field, port);
            if self.cur.tok == Tok::Comma {
                self.bump()?;
            } else {
                break;
            }
        }
        self.expect_tok(Tok::RParen, "to close the argument list")?;
        Ok(spec)
    }

    /// One `field = port` binding.
    fn bind(&mut self) -> Result<(String, Port, Span), ParseError> {
        let (field, span) = self.expect_name("as the destination field")?;
        self.expect_tok(Tok::Eq, "after the destination field")?;
        let port = self.port()?;
        Ok((field, port, span))
    }

    fn bindmap(&mut self, context: &str) -> Result<Vec<(String, Port, Span)>, ParseError> {
        self.expect_tok(Tok::LBrace, &format!("to open the `{context}` bindings"))?;
        let mut binds = Vec::new();
        while self.cur.tok != Tok::RBrace {
            binds.push(self.bind()?);
            if self.cur.tok == Tok::Comma {
                self.bump()?;
            } else {
                break;
            }
        }
        self.expect_tok(Tok::RBrace, &format!("to close the `{context}` bindings"))?;
        Ok(binds)
    }

    fn port(&mut self) -> Result<Port, ParseError> {
        match &self.cur.tok {
            Tok::Dollar => {
                self.bump()?;
                self.expect_tok(Tok::Dot, "after `$` (scope inputs are `$.field`)")?;
                let (field, _) = self.expect_name("after `$.`")?;
                Ok(builder::input(&field))
            }
            Tok::Caret => {
                self.bump()?;
                let (field, _) = self.expect_name("after `^` (a loop-carried value)")?;
                Ok(builder::carried(&field))
            }
            Tok::Str(value) => {
                let value = value.clone();
                self.bump()?;
                Ok(builder::lit(value))
            }
            Tok::Num(raw) => {
                let value = if let Ok(int) = raw.parse::<i64>() {
                    serde_json::Value::from(int)
                } else {
                    serde_json::Value::from(
                        raw.parse::<f64>()
                            .map_err(|_| self.err(format!("`{raw}` is not a valid number")))?,
                    )
                };
                self.bump()?;
                Ok(builder::lit(value))
            }
            Tok::LBracket | Tok::LBrace => {
                let (value, _) = self.raw_json()?;
                Ok(builder::lit(value))
            }
            Tok::Ident(word) if word == "true" => {
                self.bump()?;
                Ok(builder::lit(true))
            }
            Tok::Ident(word) if word == "false" => {
                self.bump()?;
                Ok(builder::lit(false))
            }
            Tok::Ident(word) if word == "null" => {
                self.bump()?;
                Ok(builder::lit(serde_json::Value::Null))
            }
            Tok::Ident(_) => {
                let (node, span) = self.expect_ident("as a node name")?;
                self.expect_tok(
                    Tok::Dot,
                    "after the node name (node outputs are `node.field`)",
                )?;
                let (field, _) = self.expect_name("after `.`")?;
                self.ref_spans.entry(node.clone()).or_insert(span);
                if !self.node_names.contains_key(&node) {
                    return Err(ParseError::at(
                        span,
                        format!(
                            "`{node}.{field}` references unknown node `{node}`: ports may only \
                             reference nodes named earlier in the program"
                        ),
                    ));
                }
                Ok(builder::out(node.as_str(), &field))
            }
            other => Err(self.err(format!(
                "expected a port (`$.field`, `node.field`, `^field`, or a JSON literal), \
                 found {}",
                other.describe()
            ))),
        }
    }

    // -- containers ---------------------------------------------------------

    fn seq(
        &mut self,
        name: Option<(String, Span)>,
        kw_span: Span,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // seq
        self.expect_tok(Tok::LBrace, "after `seq`")?;
        let mut shadow = Shadow::container(kw_span);
        let mut children = Vec::new();
        let mut outs: Vec<(String, Port, Span)> = Vec::new();
        while self.cur.tok != Tok::RBrace {
            if self.at_kw("out") {
                self.bump()?;
                outs.extend(self.bindmap("out")?);
            } else {
                let (child, child_shadow) = self.target()?;
                children.push(child);
                shadow.children.push(child_shadow);
            }
        }
        self.bump()?; // }
        let mut spec = builder::seq(children);
        for (field, port, span) in outs {
            shadow.binds.push((field.clone(), span));
            spec = spec.out(&field, port);
        }
        if let Some((name, _)) = name {
            spec = spec.named(&name);
        }
        Ok((spec, shadow))
    }

    fn fork(
        &mut self,
        name: Option<(String, Span)>,
        kw_span: Span,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // fork
        self.expect_tok(Tok::LBrace, "after `fork`")?;
        let mut shadow = Shadow::container(kw_span);
        let mut branches = Vec::new();
        while self.cur.tok != Tok::RBrace {
            let (branch, branch_shadow) = self.target()?;
            branches.push(branch);
            shadow.children.push(branch_shadow);
        }
        self.bump()?; // }
        self.expect_kw("join", "after the fork branches")?;
        let mut spec = builder::fork(branches);
        for (field, port, span) in self.bindmap("join")? {
            shadow.binds.push((field.clone(), span));
            spec = spec.join(&field, port);
        }
        if let Some((name, _)) = name {
            spec = spec.named(&name);
        }
        Ok((spec, shadow))
    }

    fn route(
        &mut self,
        name: Option<(String, Span)>,
        kw_span: Span,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // route
        let on = self.port()?;
        self.expect_tok(Tok::LBrace, "after the route discriminant port")?;
        let mut shadow = Shadow::container(kw_span);
        let mut spec = builder::route(on);
        let mut default_shadow: Option<Shadow> = None;
        while self.cur.tok != Tok::RBrace {
            if self.at_kw("else") {
                self.bump()?;
                self.expect_tok(Tok::Arrow, "after `else`")?;
                let (target, target_shadow) = self.target()?;
                spec = spec.default_arm(target);
                default_shadow = Some(target_shadow);
            } else {
                let (variant, _) = self.expect_name("as a route arm variant")?;
                self.expect_tok(Tok::Arrow, "after the arm variant")?;
                let (target, target_shadow) = self.target()?;
                spec = spec.arm(&variant, target);
                shadow.children.push(target_shadow);
            }
        }
        self.bump()?; // }
        if let Some(default) = default_shadow {
            // Builder lowering order: arms, then default, then the route node.
            shadow.children.push(default);
        }
        if let Some((name, _)) = name {
            spec = spec.named(&name);
        }
        Ok((spec, shadow))
    }

    fn retry(
        &mut self,
        name: Option<(String, Span)>,
        kw_span: Span,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // retry
        self.expect_tok(Tok::LParen, "after `retry`")?;
        let mut attempts: Option<u32> = None;
        let mut backoff_ms: u32 = 0;
        let mut feedback = false;
        while self.cur.tok != Tok::RParen {
            let (key, key_span) = self.expect_ident("as a retry option")?;
            match key.as_str() {
                "attempts" => {
                    let (value, span) = self.expect_int::<u32>("after `attempts`")?;
                    if value == 0 {
                        return Err(ParseError::at(span, "`attempts` must be at least 1"));
                    }
                    attempts = Some(value);
                }
                "backoff_ms" => backoff_ms = self.expect_int("after `backoff_ms`")?.0,
                "feedback" => feedback = self.expect_bool("after `feedback`")?,
                other => {
                    return Err(ParseError::at(
                        key_span,
                        format!(
                            "unknown retry option `{other}`: expected `attempts`, \
                             `backoff_ms`, or `feedback`"
                        ),
                    ));
                }
            }
        }
        self.bump()?; // )
        let attempts =
            attempts.ok_or_else(|| ParseError::at(kw_span, "retry requires `attempts <n>`"))?;
        let (child, child_shadow) = self.target()?;
        let mut shadow = Shadow::container(kw_span);
        shadow.children.push(child_shadow);
        let spec = builder::retry(child, attempts)
            .backoff_ms(backoff_ms)
            .feedback(feedback);
        let spec = match name {
            Some((name, _)) => spec.named(&name),
            None => spec,
        };
        Ok((spec, shadow))
    }

    fn refine(
        &mut self,
        name: Option<(String, Span)>,
        kw_span: Span,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // refine
        self.expect_tok(Tok::LParen, "after `refine`")?;
        let mut threshold: f64 = 1.0;
        let mut max_rounds: u32 = 2;
        let mut feedback_field: Option<String> = None;
        while self.cur.tok != Tok::RParen {
            let (key, key_span) = self.expect_ident("as a refine option")?;
            match key.as_str() {
                "threshold" => match &self.cur.tok {
                    Tok::Num(raw) => {
                        threshold = raw.parse::<f64>().map_err(|_| {
                            self.err(format!("`{raw}` is not a valid number for `threshold`"))
                        })?;
                        self.bump()?;
                    }
                    other => {
                        return Err(self.err(format!(
                            "expected a number after `threshold`, found {}",
                            other.describe()
                        )));
                    }
                },
                "max_rounds" => {
                    let (value, span) = self.expect_int::<u32>("after `max_rounds`")?;
                    if value == 0 {
                        return Err(ParseError::at(span, "`max_rounds` must be at least 1"));
                    }
                    max_rounds = value;
                }
                "feedback_field" => {
                    feedback_field = Some(self.expect_name("after `feedback_field`")?.0)
                }
                other => {
                    return Err(ParseError::at(
                        key_span,
                        format!(
                            "unknown refine option `{other}`: expected `threshold`, \
                             `max_rounds`, or `feedback_field`"
                        ),
                    ));
                }
            }
        }
        self.bump()?; // )
        let feedback_field = feedback_field.ok_or_else(|| {
            ParseError::at(
                kw_span,
                "refine requires `feedback_field <input>` (the child input that receives judge \
                 feedback)",
            )
        })?;
        self.expect_tok(Tok::LBrace, "after the refine options")?;
        self.expect_kw("body", "to open the refine body")?;
        self.expect_tok(Tok::Eq, "after `body`")?;
        let (child, child_shadow) = self.target()?;
        self.expect_kw("judge", "after the refine body")?;
        self.expect_tok(Tok::Eq, "after `judge`")?;
        let (judge, judge_shadow) = self.target()?;
        self.expect_tok(Tok::RBrace, "to close the refine block")?;
        let mut shadow = Shadow::container(kw_span);
        shadow.children.push(child_shadow);
        shadow.children.push(judge_shadow);
        let spec = builder::refine(child, judge, &feedback_field)
            .threshold(threshold)
            .max_rounds(max_rounds);
        let spec = match name {
            Some((name, _)) => spec.named(&name),
            None => spec,
        };
        Ok((spec, shadow))
    }

    fn loop_(
        &mut self,
        name: Option<(String, Span)>,
        kw_span: Span,
    ) -> Result<(NodeSpec, Shadow), ParseError> {
        self.bump()?; // loop
        self.expect_tok(Tok::LParen, "after `loop`")?;
        let mut max_iters: Option<u32> = None;
        while self.cur.tok != Tok::RParen {
            let (key, key_span) = self.expect_ident("as a loop option")?;
            match key.as_str() {
                "max_iters" => {
                    let (value, span) = self.expect_int::<u32>("after `max_iters`")?;
                    if value == 0 {
                        return Err(ParseError::at(span, "`max_iters` must be at least 1"));
                    }
                    max_iters = Some(value);
                }
                other => {
                    return Err(ParseError::at(
                        key_span,
                        format!("unknown loop option `{other}`: expected `max_iters`"),
                    ));
                }
            }
        }
        self.bump()?; // )
        let max_iters = max_iters.ok_or_else(|| {
            ParseError::at(kw_span, "loop requires `max_iters <n>` (loops are bounded)")
        })?;
        self.expect_tok(Tok::LBrace, "after the loop options")?;

        let mut body_shadow = Shadow::container(kw_span);
        let mut children = Vec::new();
        let mut body_outs: Vec<(String, Port, Span)> = Vec::new();
        let mut while_: Option<Port> = None;
        let mut carry: Vec<(String, Port, Span)> = Vec::new();
        let mut outs: Vec<(String, Port, Span)> = Vec::new();
        while self.cur.tok != Tok::RBrace {
            if self.at_kw("out") {
                self.bump()?;
                body_outs.extend(self.bindmap("out")?);
            } else if self.at_kw("while") {
                self.bump()?;
                while_ = Some(self.port()?);
            } else if self.at_kw("carry") {
                self.bump()?;
                carry.extend(self.bindmap("carry")?);
            } else if self.at_kw("join") {
                self.bump()?;
                outs.extend(self.bindmap("join")?);
            } else {
                let (child, child_shadow) = self.target()?;
                children.push(child);
                body_shadow.children.push(child_shadow);
            }
        }
        self.bump()?; // }

        let mut body = builder::seq(children);
        for (field, port, span) in body_outs {
            body_shadow.binds.push((field.clone(), span));
            body = body.out(&field, port);
        }
        let mut shadow = Shadow::container(kw_span);
        shadow.children.push(body_shadow);
        let mut spec = builder::loop_(body, max_iters);
        if let Some(port) = while_ {
            spec = spec.while_(port);
        }
        for (field, port, span) in carry {
            shadow.binds.push((field.clone(), span));
            spec = spec.carry(&field, port);
        }
        for (field, port, span) in outs {
            shadow.binds.push((field.clone(), span));
            spec = spec.out(&field, port);
        }
        let spec = match name {
            Some((name, _)) => spec.named(&name),
            None => spec,
        };
        Ok((spec, shadow))
    }

    // -- error mapping ------------------------------------------------------

    /// Maps a post-lowering [`BuildError`] onto a source position using the
    /// shadow-tree span tables.
    fn map_build_error(&self, err: BuildError, maps: &SpanMaps, fallback: Span) -> ParseError {
        let (span, message) = match &err {
            BuildError::UnknownNode { name } => (
                self.ref_spans.get(name).copied(),
                format!("`{name}` is referenced but never defined"),
            ),
            BuildError::MissingModel { at } => (
                maps.at.get(at).copied(),
                format!(
                    "`{at}` has no model: add `@<model>` (the program declares more than one \
                     model, so the reference must be explicit)"
                ),
            ),
            BuildError::DuplicateStepName { name } => (
                self.node_names.get(name).copied(),
                format!("duplicate step name `{name}`"),
            ),
            BuildError::Invalid(v) => {
                let (at, field) = validate_error_handles(v);
                let span = field
                    .and_then(|(at, field)| {
                        maps.field
                            .get(&(at.to_string(), field.to_string()))
                            .copied()
                    })
                    .or_else(|| at.and_then(|at| maps.at.get(at).copied()))
                    .or_else(|| self.extra_validate_span(v, maps));
                (span, v.to_string())
            }
        };
        ParseError::at(span.unwrap_or(fallback), message)
    }

    /// Fallback spans for validate errors that name types or program outputs
    /// rather than nodes.
    fn extra_validate_span(&self, v: &ValidateError, maps: &SpanMaps) -> Option<Span> {
        match v {
            ValidateError::UnknownTypeToken { token, .. } => self.type_spans.get(token).copied(),
            ValidateError::ProgramOutputMissing { field } => maps
                .field
                .iter()
                .find(|((_, f), _)| f == field)
                .map(|(_, span)| *span),
            ValidateError::DuplicateLeafName { name } => self.node_names.get(name).copied(),
            _ => None,
        }
    }
}

/// `(at, (at, field))` handles a [`ValidateError`] names things by.
fn validate_error_handles(v: &ValidateError) -> (Option<&str>, Option<(&str, &str)>) {
    use ValidateError as E;
    match v {
        E::UnboundInput { at, .. }
        | E::IdOutOfRange { at, .. }
        | E::NodeReused { at }
        | E::UnknownOutField { at, .. }
        | E::NodeNotVisible { at, .. }
        | E::RouteNotEnum { at, .. }
        | E::RouteUnknownVariant { at, .. }
        | E::RouteArmMismatch { at, .. }
        | E::RouteUncovered { at, .. }
        | E::CapsExceedProgram { at, .. }
        | E::ParamKindMismatch { at, .. }
        | E::ParamOwnerMismatch { at, .. }
        | E::StopToolNotDeclared { at }
        | E::ToolSetUndeclared { at, .. }
        | E::ToolSetDuplicate { at, .. }
        | E::RefineJudgeNotLeaf { at }
        | E::RefineJudgeInterface { at }
        | E::WhileNotBool { at, .. } => (Some(at), None),
        E::DuplicateBinding { at, field }
        | E::UnknownBindingDst { at, field }
        | E::UnknownScopeInput { at, field }
        | E::CarriedOutsideLoop { at, field }
        | E::BindingTypeMismatch { at, field, .. }
        | E::LiteralTypeMismatch { at, field, .. }
        | E::RefineFeedbackField { at, field }
        | E::CarryNotScopeInput { at, field } => (Some(at), Some((at, field))),
        _ => (None, None),
    }
}

/// Re-tags provisional `Class` tokens as `Enum` where the token names a
/// declared enum (references parse before declarations are complete).
fn fixup_sig(def: &mut SignatureDef, enums: &HashSet<String>) {
    for field in def.inputs.iter_mut().chain(def.outputs.iter_mut()) {
        fixup_type(&mut field.ty, enums);
    }
}

fn fixup_type(ty: &mut FieldType, enums: &HashSet<String>) {
    match ty {
        FieldType::Class(token) if enums.contains(token.as_str()) => {
            *ty = FieldType::Enum(std::mem::take(token));
        }
        FieldType::List(inner) | FieldType::Optional(inner) => fixup_type(inner, enums),
        FieldType::Map(key, value) => {
            fixup_type(key, enums);
            fixup_type(value, enums);
        }
        FieldType::Union(items) => items.iter_mut().for_each(|i| fixup_type(i, enums)),
        _ => {}
    }
}
