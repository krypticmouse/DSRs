//! Value-level signatures (RFC 0002 §1) — [`SignatureDef`] and friends.
//!
//! A [`SignatureDef`] is a self-contained, owned runtime value: name, instruction,
//! typed input/output fields, constraints, render hints. It is constructible with
//! zero context — no macros, no interner, no `'static` anywhere — and serde-derivable,
//! so loaded programs can carry their signatures as plain data.
//!
//! # Ownership story (the leaked-cache collapse)
//!
//! Before IR-1, four process-global `Box::leak` caches existed: the
//! `SignatureSchema::of` `TypeId` map, the blanket `Schema::output_schema` `TypeId`
//! map, the `internal_name_for_shape` string intern, and the per-derive `OnceLock`
//! fast path emitted by `#[derive(Signature)]`. Leaking once per *compile-time* type
//! is a static allocation in disguise; the failure mode was dynamic loading through
//! the same caches — an unbounded leak per load. They collapse to:
//!
//! - **Static lane:** `StaticSigCache` — the *one* deliberate leak, bounded by the
//!   closed set of compiled `Signature` types. One entry per type holds the
//!   [`SignatureDef`], its [`TypeTable`], and the legacy [`SignatureSchema`]
//!   façade. `Signature::schema()` keeps returning `&'static` — no API break.
//! - **Dynamic lane:** a loaded program owns its `SignatureDef`s and `TypeTable`
//!   outright and drops them with the program. Nothing constructed at runtime ever
//!   touches a global cache or leaks.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use serde::{Deserialize, Serialize};

use crate::core::{ConstraintKind, FieldSchema, InputRenderSpec, Signature, SignatureSchema};
use crate::typesys::{FieldType, TypeTable};

/// A signature as an owned value: what the derive macro knows at compile time,
/// available at runtime with no `'static` requirement (RFC 0002 §1.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignatureDef {
    pub name: Box<str>,
    /// The operative instruction. For derive-built defs this is the doc comment.
    pub instruction: Box<str>,
    pub inputs: Box<[FieldDef]>,
    pub outputs: Box<[FieldDef]>,
}

/// One input or output field of a [`SignatureDef`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    /// Canonical (Rust) name; the key in input/output `JsonMap`s.
    pub name: Box<str>,
    /// LM-facing name (`alias` or same as `name`).
    pub lm_name: Box<str>,
    pub ty: FieldType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub docs: Option<Box<str>>,
    #[serde(default, skip_serializing_if = "<[_]>::is_empty")]
    pub constraints: Box<[ConstraintDef]>,
    #[serde(default)]
    pub render: RenderSpec,
}

impl FieldDef {
    /// A plain field: `lm_name == name`, no docs, no constraints, default rendering.
    pub fn new(name: &str, ty: FieldType) -> Self {
        Self {
            name: name.into(),
            lm_name: name.into(),
            ty,
            docs: None,
            constraints: Box::new([]),
            render: RenderSpec::Default,
        }
    }

    /// Sets the LM-facing alias.
    pub fn aliased(mut self, lm_name: &str) -> Self {
        self.lm_name = lm_name.into();
        self
    }

    /// Sets the field docs (the derive's doc comment / `desc = "..."`).
    pub fn with_docs(mut self, docs: &str) -> Self {
        self.docs = Some(docs.into());
        self
    }

    /// Appends a constraint.
    pub fn with_constraint(mut self, constraint: ConstraintDef) -> Self {
        let mut constraints = self.constraints.into_vec();
        constraints.push(constraint);
        self.constraints = constraints.into_boxed_slice();
        self
    }

    /// Sets the input render policy.
    pub fn with_render(mut self, render: RenderSpec) -> Self {
        self.render = render;
        self
    }
}

/// Owned runtime form of [`ConstraintSpec`](crate::ConstraintSpec) (which stays
/// `&'static` for the derive).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintDef {
    pub kind: ConstraintKind,
    pub label: Box<str>,
    pub expr: Box<str>,
}

impl ConstraintDef {
    /// A soft `#[check(expr, label = ...)]` constraint. Checks require a label.
    pub fn check(label: &str, expr: &str) -> Self {
        Self {
            kind: ConstraintKind::Check,
            label: label.into(),
            expr: expr.into(),
        }
    }

    /// A hard `#[assert(expr)]` constraint.
    pub fn assert(expr: &str) -> Self {
        Self {
            kind: ConstraintKind::Assert,
            label: "".into(),
            expr: expr.into(),
        }
    }
}

/// Owned runtime form of [`InputRenderSpec`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderSpec {
    #[default]
    Default,
    /// `#[format("json" | "yaml" | "toon")]`.
    Format(Box<str>),
    /// `#[render(jinja = "...")]`.
    Jinja(Box<str>),
}

/// Validation failure from [`SignatureBuilder::finish`]. What the derive macro
/// rejects, this rejects — same closed subset in both lanes (RFC 0002 §1.3).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SigError {
    #[error("signature `{name}` requires at least one input field")]
    EmptyInputs { name: String },
    #[error("signature `{name}` requires at least one output field")]
    EmptyOutputs { name: String },
    #[error("duplicate {side} field `{name}`")]
    DuplicateName { side: &'static str, name: String },
    #[error(
        "duplicate {side} field name `{lm_name}` after aliasing; conflicts between `{previous}` and `{current}`"
    )]
    DuplicateLmName {
        side: &'static str,
        lm_name: String,
        previous: String,
        current: String,
    },
    #[error(
        "unsupported format value `{value}` on field `{field}`; use \"json\", \"yaml\", or \"toon\""
    )]
    InvalidFormat { field: String, value: String },
    #[error("invalid Jinja syntax in render template on field `{field}`")]
    InvalidJinja { field: String },
    #[error("check constraint on field `{field}` requires a label")]
    CheckMissingLabel { field: String },
    #[error("invalid constraint expression `{expr}` on field `{field}`: {message}")]
    InvalidConstraintExpr {
        field: String,
        expr: String,
        message: String,
    },
    #[error(
        "map keys must be String in Signature fields (field `{field}`); hint: use HashMap<String, V> or BTreeMap<String, V>"
    )]
    NonStringMapKey { field: String },
}

/// Structural mismatch reported by [`SignatureDef::matches`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("signature mismatch: {0}")]
pub struct SigMismatch(pub String);

impl SignatureDef {
    /// Starts a builder for a runtime-constructed signature.
    pub fn build(name: &str) -> SignatureBuilder {
        SignatureBuilder {
            name: name.to_string(),
            instruction: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Static → value lane bridge: the owned equivalent of `S::schema()`,
    /// converted once per type and cached in the crate's single `StaticSigCache`.
    pub fn of<S: Signature>() -> &'static SignatureDef {
        &static_sig_entry::<S>().def
    }

    /// The class/enum registry backing [`of::<S>()`](SignatureDef::of) — the
    /// definitions reachable from `S`'s output fields, from the same cache entry.
    pub fn types_of<S: Signature>() -> &'static TypeTable {
        &static_sig_entry::<S>().types
    }

    /// Structural equality against a static signature: same field names,
    /// `FieldType`s, and aliases, in order (instruction text excluded — it's a
    /// parameter; docs/constraints/render excluded — they don't change the shape).
    pub fn matches<S: Signature>(&self) -> Result<(), SigMismatch> {
        let expected = Self::of::<S>();
        match_side("input", &self.inputs, &expected.inputs)?;
        match_side("output", &self.outputs, &expected.outputs)
    }

    /// Prepends output fields (e.g. `reasoning: string`) — the value-lane
    /// `Augmented<S, Reasoning>`. Pure function; `self` is untouched.
    pub fn augmented_with(&self, prefix_outputs: &[FieldDef]) -> SignatureDef {
        let mut outputs = Vec::with_capacity(prefix_outputs.len() + self.outputs.len());
        outputs.extend(prefix_outputs.iter().cloned());
        outputs.extend(self.outputs.iter().cloned());
        SignatureDef {
            name: self.name.clone(),
            instruction: self.instruction.clone(),
            inputs: self.inputs.clone(),
            outputs: outputs.into_boxed_slice(),
        }
    }
}

fn match_side(side: &str, got: &[FieldDef], expected: &[FieldDef]) -> Result<(), SigMismatch> {
    if got.len() != expected.len() {
        return Err(SigMismatch(format!(
            "{side} field count differs: {} vs {}",
            got.len(),
            expected.len()
        )));
    }
    for (g, e) in got.iter().zip(expected) {
        if g.name != e.name {
            return Err(SigMismatch(format!(
                "{side} field name differs: `{}` vs `{}`",
                g.name, e.name
            )));
        }
        if g.lm_name != e.lm_name {
            return Err(SigMismatch(format!(
                "{side} field `{}` alias differs: `{}` vs `{}`",
                g.name, g.lm_name, e.lm_name
            )));
        }
        if g.ty != e.ty {
            return Err(SigMismatch(format!(
                "{side} field `{}` type differs: {:?} vs {:?}",
                g.name, g.ty, e.ty
            )));
        }
    }
    Ok(())
}

/// Builder for [`SignatureDef`]; validation happens at [`finish`](SignatureBuilder::finish).
pub struct SignatureBuilder {
    name: String,
    instruction: String,
    inputs: Vec<FieldDef>,
    outputs: Vec<FieldDef>,
}

impl SignatureBuilder {
    pub fn instruction(mut self, text: &str) -> Self {
        self.instruction = text.to_string();
        self
    }

    pub fn input(self, name: &str, ty: FieldType) -> Self {
        self.input_full(FieldDef::new(name, ty))
    }

    pub fn output(self, name: &str, ty: FieldType) -> Self {
        self.output_full(FieldDef::new(name, ty))
    }

    pub fn input_full(mut self, f: FieldDef) -> Self {
        self.inputs.push(f);
        self
    }

    pub fn output_full(mut self, f: FieldDef) -> Self {
        self.outputs.push(f);
        self
    }

    /// Validates and finishes. Errors mirror what `#[derive(Signature)]` rejects:
    /// empty side, duplicate (aliased) field names, invalid `format` value,
    /// invalid Jinja template, check without a label, malformed constraint
    /// expression, non-string map keys. Class/Enum token resolution happens
    /// against a [`TypeTable`] at program build (RFC 0002 §1.1) — deferred here.
    pub fn finish(self) -> Result<SignatureDef, SigError> {
        if self.inputs.is_empty() {
            return Err(SigError::EmptyInputs { name: self.name });
        }
        if self.outputs.is_empty() {
            return Err(SigError::EmptyOutputs { name: self.name });
        }
        validate_side("input", &self.inputs)?;
        validate_side("output", &self.outputs)?;
        Ok(SignatureDef {
            name: self.name.into_boxed_str(),
            instruction: self.instruction.into_boxed_str(),
            inputs: self.inputs.into_boxed_slice(),
            outputs: self.outputs.into_boxed_slice(),
        })
    }
}

fn validate_side(side: &'static str, fields: &[FieldDef]) -> Result<(), SigError> {
    let mut seen_names: HashMap<&str, ()> = HashMap::new();
    let mut seen_lm: HashMap<&str, &str> = HashMap::new();
    for field in fields {
        if seen_names.insert(&field.name, ()).is_some() {
            return Err(SigError::DuplicateName {
                side,
                name: field.name.to_string(),
            });
        }
        if let Some(previous) = seen_lm.insert(&field.lm_name, &field.name) {
            return Err(SigError::DuplicateLmName {
                side,
                lm_name: field.lm_name.to_string(),
                previous: previous.to_string(),
                current: field.name.to_string(),
            });
        }
        validate_field(field)?;
    }
    Ok(())
}

fn validate_field(field: &FieldDef) -> Result<(), SigError> {
    match &field.render {
        RenderSpec::Default => {}
        RenderSpec::Format(value) => {
            if !matches!(
                value.to_ascii_lowercase().as_str(),
                "json" | "yaml" | "toon"
            ) {
                return Err(SigError::InvalidFormat {
                    field: field.name.to_string(),
                    value: value.to_string(),
                });
            }
        }
        RenderSpec::Jinja(template) => {
            let mut env = minijinja::Environment::new();
            if env.add_template("__input_field__", template).is_err() {
                return Err(SigError::InvalidJinja {
                    field: field.name.to_string(),
                });
            }
        }
    }

    for constraint in &field.constraints {
        if constraint.kind == ConstraintKind::Check && constraint.label.is_empty() {
            return Err(SigError::CheckMissingLabel {
                field: field.name.to_string(),
            });
        }
        let env = minijinja::Environment::new();
        if let Err(err) = env.compile_expression(&constraint.expr) {
            return Err(SigError::InvalidConstraintExpr {
                field: field.name.to_string(),
                expr: constraint.expr.to_string(),
                message: err.to_string(),
            });
        }
    }

    if has_non_string_map_key(&field.ty) {
        return Err(SigError::NonStringMapKey {
            field: field.name.to_string(),
        });
    }

    Ok(())
}

fn has_non_string_map_key(ty: &FieldType) -> bool {
    match ty {
        FieldType::Map(key, value) => {
            !matches!(**key, FieldType::String) || has_non_string_map_key(value)
        }
        FieldType::List(inner) | FieldType::Optional(inner) => has_non_string_map_key(inner),
        FieldType::Union(items) => items.iter().any(has_non_string_map_key),
        _ => false,
    }
}

// --- StaticSigCache -------------------------------------------------------

/// One entry per compiled `Signature` type: the value-level def, its type
/// registry, and the legacy schema façade — all views of the same signature.
pub(crate) struct StaticSigEntry {
    pub(crate) def: SignatureDef,
    pub(crate) types: TypeTable,
    pub(crate) schema: SignatureSchema,
}

/// `StaticSigCache` (RFC 0002 §1.2): the single static-lane cache, `TypeId` →
/// leaked entry. Deliberately leaked — bounded by the closed set of signature
/// types compiled into the binary, so it is a static allocation in disguise.
/// Dynamically loaded signatures never touch this: loaded programs own their
/// arenas and leak nothing.
static STATIC_SIG_CACHE: LazyLock<RwLock<HashMap<TypeId, &'static StaticSigEntry>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub(crate) fn static_sig_entry<S: Signature>() -> &'static StaticSigEntry {
    {
        let guard = STATIC_SIG_CACHE.read().expect("static sig cache poisoned");
        if let Some(entry) = guard.get(&TypeId::of::<S>()) {
            return entry;
        }
    }

    let schema = SignatureSchema::build::<S>().unwrap_or_else(|err| {
        panic!(
            "failed to build SignatureSchema for `{}`: {err}",
            std::any::type_name::<S>()
        )
    });
    let def = SignatureDef {
        name: short_type_name(std::any::type_name::<S>()).into_boxed_str(),
        instruction: S::instruction().into(),
        inputs: schema.input_fields().iter().map(field_def_of).collect(),
        outputs: schema.output_fields().iter().map(field_def_of).collect(),
    };
    let types = schema.output_schema().types.clone();
    let leaked: &'static StaticSigEntry =
        Box::leak(Box::new(StaticSigEntry { def, types, schema }));

    let mut guard = STATIC_SIG_CACHE.write().expect("static sig cache poisoned");
    guard.entry(TypeId::of::<S>()).or_insert(leaked)
}

/// The legacy façade accessor behind [`SignatureSchema::of`](crate::SignatureSchema::of).
pub(crate) fn static_schema<S: Signature>() -> &'static SignatureSchema {
    &static_sig_entry::<S>().schema
}

/// `FieldSchema` (facet-derived, `'static`) → owned `FieldDef`. Flattened fields
/// keep their *leaf* name — the key they serialize flat under, and therefore the
/// key in value-lane `JsonMap`s.
fn field_def_of(field: &FieldSchema) -> FieldDef {
    let leaf = field.path().iter().last().unwrap_or(field.lm_name);
    FieldDef {
        name: leaf.into(),
        lm_name: field.lm_name.into(),
        ty: field.type_ir.clone(),
        docs: if field.docs.is_empty() {
            None
        } else {
            Some(field.docs.as_str().into())
        },
        constraints: field
            .constraints
            .iter()
            .map(|spec| ConstraintDef {
                kind: spec.kind,
                label: spec.label.into(),
                expr: spec.expression.into(),
            })
            .collect(),
        render: match field.input_render {
            InputRenderSpec::Default => RenderSpec::Default,
            InputRenderSpec::Format(value) => RenderSpec::Format(value.into()),
            InputRenderSpec::Jinja(template) => RenderSpec::Jinja(template.into()),
        },
    }
}

/// `"a::b::C<d::E, f::G>"` → `"C<E, G>"`: strips module paths from every path
/// segment while preserving generic structure.
fn short_type_name(full: &str) -> String {
    let mut out = String::with_capacity(full.len());
    let mut seg_start = 0;
    for (i, ch) in full.char_indices() {
        match ch {
            ':' => seg_start = i + 1,
            '<' | '>' | ',' | ' ' | '(' | ')' | '[' | ']' | ';' | '&' => {
                out.push_str(&full[seg_start..i]);
                out.push(ch);
                seg_start = i + 1;
            }
            _ => {}
        }
    }
    out.push_str(&full[seg_start..]);
    out
}
