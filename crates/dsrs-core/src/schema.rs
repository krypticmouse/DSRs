use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bamltype::baml_types::BamlValue;
use bamltype::baml_types::TypeIR;
use bamltype::build_type_ir_from_shape;
use bamltype::facet::{Def, Field, Shape, Type, UserType};
use bamltype::internal_baml_jinja::types::OutputFormatContent;

use crate::{Constraint, ConstraintKind, ConstraintSpec, Signature};

/// Dotted path to a field within a signature, accounting for `#[flatten]` nesting.
///
/// A field `answer` at the top level has path `["answer"]`. A field `reasoning` inside
/// a flattened `WithReasoning` wrapper has path `["inner", "reasoning"]` (or however the
/// flatten tree is structured). Used by the adapter for path-aware parsing and by
/// [`SignatureSchema`] for field lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldPath {
    parts: Vec<&'static str>,
}

impl FieldPath {
    pub fn new(parts: impl IntoIterator<Item = &'static str>) -> Self {
        Self {
            parts: parts.into_iter().collect(),
        }
    }

    pub fn push(&mut self, part: &'static str) {
        self.parts.push(part);
    }

    pub fn iter(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.parts.iter().copied()
    }

    pub fn display(&self) -> String {
        self.parts.join(".")
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

/// Static metadata for a single signature field, emitted by `#[derive(Signature)]`.
///
/// Carries the Rust field name, optional LM-facing alias, constraint specs, and
/// input render hints. Fed into [`SignatureSchema`] construction alongside Facet shape data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRenderSpec {
    /// Default behavior: strings are raw, non-strings are rendered as JSON.
    Default,
    /// Explicit format hint (`#[format("json" | "yaml" | "toon")]`).
    Format(&'static str),
    /// Custom Jinja template (`#[render(jinja = "...")]`).
    Jinja(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct FieldMetadataSpec {
    /// The Rust field name as written in the signature struct.
    pub rust_name: &'static str,
    /// Optional alias for the LM prompt (e.g. `#[rename = "query"]` on a `question` field).
    pub alias: Option<&'static str>,
    /// Constraint specs from `#[check(...)]` and `#[assert(...)]` attributes.
    pub constraints: &'static [ConstraintSpec],
    /// Input rendering policy for this field.
    pub input_render: InputRenderSpec,
}

/// Complete schema for a single field in a signature, combining Facet shape data with metadata.
///
/// Used by the adapter for prompt formatting and response parsing, and by the optimizer
/// for field inspection.
#[derive(Debug, Clone)]
pub struct FieldSchema {
    /// The field name shown to the LM (may differ from Rust name via aliasing).
    pub lm_name: &'static str,
    /// The dotted Rust path (e.g. `"inner.reasoning"` for flattened fields).
    pub rust_name: String,
    /// Documentation extracted from the field's doc comment.
    pub docs: String,
    /// Type representation used for edge validation and output format generation.
    pub type_ir: TypeIR,
    /// The Facet shape of this field's type.
    pub shape: &'static Shape,
    /// Path through the flatten tree to reach this field.
    pub path: FieldPath,
    /// Constraints declared on this field.
    pub constraints: &'static [ConstraintSpec],
    /// Input rendering policy.
    pub input_render: InputRenderSpec,
}

impl FieldSchema {
    pub fn path(&self) -> &FieldPath {
        &self.path
    }

    pub fn shape(&self) -> &'static Shape {
        self.shape
    }
}

/// Cached field-level schema for a [`Signature`], built from Facet shapes.
///
/// The shared backbone of the system. Every path that needs to know about a signature's
/// fields reads from here — the adapter formatting prompts, the graph validating edges,
/// optimizers inspecting structure. Built once per `Signature` type (keyed by `TypeId`),
/// leaked into `'static`, never mutated after init.
///
/// Contains the flattened list of input and output fields with their LM-facing names,
/// Rust paths (accounting for `#[flatten]`), type info, docs, and constraints. Derived
/// from Facet shape metadata at runtime, not from macro-emitted static arrays — Facet
/// is the single source of truth for type structure.
///
/// Access via [`SignatureSchema::of::<S>()`](SignatureSchema::of) or [`Signature::schema()`].
#[derive(Debug, Clone)]
pub struct SignatureSchema {
    instruction: &'static str,
    input_fields: Box<[FieldSchema]>,
    output_fields: Box<[FieldSchema]>,
    output_format: Arc<OutputFormatContent>,
}

impl SignatureSchema {
    /// Returns the cached schema for signature `S`, building it on first access.
    ///
    /// # Panics
    ///
    /// Panics if the schema can't be built (e.g. the input/output shapes aren't structs).
    pub fn of<S: Signature>() -> &'static Self {
        static CACHE: OnceLock<Mutex<HashMap<TypeId, &'static SignatureSchema>>> = OnceLock::new();

        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let guard = cache.lock().expect("schema cache lock poisoned");
            if let Some(schema) = guard.get(&TypeId::of::<S>()) {
                return schema;
            }
        }

        let built = Self::build::<S>().unwrap_or_else(|err| {
            panic!(
                "failed to build SignatureSchema for `{}`: {err}",
                std::any::type_name::<S>()
            )
        });
        let leaked = Box::leak(Box::new(built));

        let mut guard = cache.lock().expect("schema cache lock poisoned");
        guard.entry(TypeId::of::<S>()).or_insert(leaked)
    }

    fn build<S: Signature>() -> Result<Self, String> {
        let mut input_fields = collect_fields(
            "input",
            S::input_shape(),
            S::input_field_metadata(),
            S::instruction(),
        )?;
        let mut output_fields = collect_fields(
            "output",
            S::output_shape(),
            S::output_field_metadata(),
            S::instruction(),
        )?;

        ensure_unique_lm_names("input", &input_fields)?;
        ensure_unique_lm_names("output", &output_fields)?;

        // Keep declaration order deterministic.
        input_fields.shrink_to_fit();
        output_fields.shrink_to_fit();

        Ok(Self {
            instruction: S::instruction(),
            input_fields: input_fields.into_boxed_slice(),
            output_fields: output_fields.into_boxed_slice(),
            output_format: Arc::new(<S::Output as crate::BamlType>::baml_output_format().clone()),
        })
    }

    pub fn instruction(&self) -> &'static str {
        self.instruction
    }

    pub fn input_fields(&self) -> &[FieldSchema] {
        &self.input_fields
    }

    pub fn output_fields(&self) -> &[FieldSchema] {
        &self.output_fields
    }

    pub fn output_format(&self) -> &OutputFormatContent {
        &self.output_format
    }

    pub fn navigate_field<'a>(
        &self,
        path: &FieldPath,
        root: &'a BamlValue,
    ) -> Option<&'a BamlValue> {
        let mut current = root;
        for part in path.iter() {
            current = match current {
                BamlValue::Class(_, map) | BamlValue::Map(map) => map.get(part)?,
                _ => return None,
            };
        }
        Some(current)
    }

    pub fn field_by_rust<'a>(&'a self, rust_name: &str) -> Option<&'a FieldSchema> {
        self.input_fields()
            .iter()
            .chain(self.output_fields().iter())
            .find(|field| field.rust_name == rust_name)
    }

    pub fn input_field_by_rust<'a>(&'a self, rust_name: &str) -> Option<&'a FieldSchema> {
        self.input_fields()
            .iter()
            .find(|field| field.rust_name == rust_name)
    }

    pub fn output_field_by_rust<'a>(&'a self, rust_name: &str) -> Option<&'a FieldSchema> {
        self.output_fields()
            .iter()
            .find(|field| field.rust_name == rust_name)
    }

    pub fn with_fields(
        &self,
        input_fields: Vec<FieldSchema>,
        output_fields: Vec<FieldSchema>,
    ) -> Self {
        Self {
            instruction: self.instruction,
            input_fields: input_fields.into_boxed_slice(),
            output_fields: output_fields.into_boxed_slice(),
            output_format: Arc::clone(&self.output_format),
        }
    }

    pub fn field_paths(&self) -> impl Iterator<Item = &FieldPath> {
        self.input_fields
            .iter()
            .chain(self.output_fields.iter())
            .map(|field| &field.path)
    }
}

fn collect_fields(
    side: &'static str,
    root_shape: &'static Shape,
    metadata: &'static [FieldMetadataSpec],
    instruction: &'static str,
) -> Result<Vec<FieldSchema>, String> {
    let struct_type = match &root_shape.ty {
        Type::User(UserType::Struct(struct_type)) => struct_type,
        _ => {
            return Err(format!(
                "{side} shape for instruction `{instruction}` must be a struct; got `{}`",
                root_shape.type_identifier
            ));
        }
    };

    let mut metadata_by_name: HashMap<&'static str, &'static FieldMetadataSpec> = HashMap::new();
    for item in metadata {
        metadata_by_name.insert(item.rust_name, item);
    }

    let mut fields = Vec::new();
    for field in struct_type.fields.iter() {
        if field.should_skip_deserializing() {
            continue;
        }
        let path = FieldPath::new([field.name]);
        let field_meta = metadata_by_name.get(field.name).copied();
        emit_field(field, path, field_meta, &metadata_by_name, &mut fields)?;
    }

    Ok(fields)
}

fn emit_field(
    field: &'static Field,
    path: FieldPath,
    inherited: Option<&FieldMetadataSpec>,
    metadata_by_name: &HashMap<&'static str, &'static FieldMetadataSpec>,
    out: &mut Vec<FieldSchema>,
) -> Result<(), String> {
    if field.should_skip_deserializing() {
        return Ok(());
    }

    if field.is_flattened() {
        let shape = flatten_target(field.shape());
        let struct_type = match &shape.ty {
            Type::User(UserType::Struct(struct_type)) => struct_type,
            _ => {
                return Err(format!(
                    "flattened field `{}` points to non-struct shape `{}`",
                    path.display(),
                    shape.type_identifier
                ));
            }
        };

        for nested in struct_type.fields.iter() {
            if nested.should_skip_deserializing() {
                continue;
            }
            let mut nested_path = path.clone();
            nested_path.push(nested.name);
            let nested_meta = metadata_by_name.get(nested.name).copied().or(inherited);
            emit_field(nested, nested_path, nested_meta, metadata_by_name, out)?;
        }

        return Ok(());
    }

    let mut type_ir = build_type_ir_from_shape(field.shape());
    let constraints = inherited.map(|meta| meta.constraints).unwrap_or(&[]);
    if !constraints.is_empty() {
        type_ir
            .meta_mut()
            .constraints
            .extend(constraints.iter().map(to_baml_constraint));
    }

    let docs = doc_lines(field.doc);
    let lm_name = inherited
        .and_then(|meta| meta.alias)
        .unwrap_or_else(|| field.effective_name());
    let input_render = inherited
        .map(|meta| meta.input_render)
        .unwrap_or(InputRenderSpec::Default);

    out.push(FieldSchema {
        lm_name,
        rust_name: path.display(),
        docs,
        type_ir,
        shape: field.shape(),
        path,
        constraints,
        input_render,
    });

    Ok(())
}

fn flatten_target(mut shape: &'static Shape) -> &'static Shape {
    loop {
        match &shape.def {
            Def::Option(option_def) => shape = option_def.t,
            Def::Pointer(pointer_def) => {
                if let Some(inner) = pointer_def.pointee {
                    shape = inner;
                } else {
                    return shape;
                }
            }
            _ => return shape,
        }
    }
}

fn doc_lines(lines: &'static [&'static str]) -> String {
    lines
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn to_baml_constraint(constraint: &ConstraintSpec) -> Constraint {
    match constraint.kind {
        ConstraintKind::Check => Constraint::new_check(constraint.label, constraint.expression),
        ConstraintKind::Assert => Constraint::new_assert(constraint.label, constraint.expression),
    }
}

fn ensure_unique_lm_names(side: &'static str, fields: &[FieldSchema]) -> Result<(), String> {
    let mut by_alias: HashMap<&str, &FieldSchema> = HashMap::new();
    for field in fields {
        if let Some(previous) = by_alias.insert(field.lm_name, field) {
            return Err(format!(
                "{side} field alias collision for `{}` between `{}` and `{}`",
                field.lm_name,
                previous.path.display(),
                field.path.display()
            ));
        }
    }
    Ok(())
}
