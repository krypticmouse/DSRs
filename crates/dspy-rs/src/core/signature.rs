use std::collections::HashSet;

use crate::baml_bridge::baml_types::ir_type::UnionTypeViewGeneric;
use crate::baml_bridge::baml_types::{LiteralValue, StreamingMode};
use crate::baml_bridge::internal_baml_jinja::types::{Class, Enum};
use crate::baml_bridge::prompt::{PromptValue, RenderResult, RenderSession};
use crate::{Example, OutputFormatContent, RenderOptions, TypeIR};
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

mod compiled;
pub use compiled::*;

// ============================================================================
// Field specification (from derive macro)
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct FieldSpec {
    pub name: &'static str,
    pub rust_name: &'static str,
    pub description: &'static str,
    pub type_ir: fn() -> TypeIR,
    pub constraints: &'static [ConstraintSpec],
    pub style: Option<&'static str>,
    pub renderer: Option<FieldRendererSpec>,
    pub render_settings: Option<FieldRenderSettings>,
}

#[derive(Debug, Clone, Copy)]
pub enum FieldRendererSpec {
    Jinja { template: &'static str },
    Func {
        f: fn(&PromptValue, &RenderSession) -> RenderResult,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct FieldRenderSettings {
    pub max_string_chars: Option<usize>,
    pub max_list_items: Option<usize>,
    pub max_map_entries: Option<usize>,
    pub max_depth: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct ConstraintSpec {
    pub kind: ConstraintKind,
    pub label: &'static str,
    pub expression: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    Check,
    Assert,
}

// ============================================================================
// Structured type metadata (for templates)
// ============================================================================

/// Structured type metadata that templates can traverse.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SigTypeMeta {
    /// Accepts any type (Top)
    Any,
    /// Primitive type: string, int, float, bool, null, image, audio, etc.
    Primitive { name: String },
    /// Literal value: "foo", 42, true
    Literal { value: String },
    /// List of items
    List { item: Box<SigTypeMeta> },
    /// Map from key to value
    Map {
        key: Box<SigTypeMeta>,
        value: Box<SigTypeMeta>,
    },
    /// Tuple of items
    Tuple { items: Vec<SigTypeMeta> },
    /// Union of types, possibly nullable
    Union {
        options: Vec<SigTypeMeta>,
        nullable: bool,
    },
    /// Enum with named values
    Enum {
        name: String,
        dynamic: bool,
        values: Vec<SigEnumValue>,
    },
    /// Class with named fields
    Class {
        name: String,
        dynamic: bool,
        recursive: bool,
        fields: Vec<SigClassField>,
    },
    /// Reference to a type (cycle breaker)
    Ref { name: String },
    /// Unsupported type (Arrow, etc.)
    Other { description: String },
}

/// Enum value metadata.
#[derive(Debug, Clone, Serialize)]
pub struct SigEnumValue {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Class field metadata.
#[derive(Debug, Clone, Serialize)]
pub struct SigClassField {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub r#type: SigTypeMeta,
}

/// Metadata about a single signature field (input or output).
#[derive(Debug, Clone, Serialize)]
pub struct SigFieldMeta {
    /// Name as seen by the LLM (rendered name).
    pub llm_name: &'static str,
    /// Rust field name (for lookup in inputs map).
    pub rust_name: &'static str,
    /// Field description from the signature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<&'static str>,
    /// Simplified type name for prompt templates.
    pub type_name: String,
    /// Schema string for output fields (JSON/YAML format description).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    /// Structured type metadata (for nested render introspection).
    pub r#type: SigTypeMeta,
}

/// Metadata about a signature for prompt templates.
#[derive(Debug, Clone, Serialize)]
pub struct SigMeta {
    pub inputs: Vec<SigFieldMeta>,
    pub outputs: Vec<SigFieldMeta>,
}

// ============================================================================
// Type metadata builder
// ============================================================================

/// Tracks visited types during traversal to detect cycles.
#[derive(Default)]
struct VisitedTypes {
    classes: HashSet<String>,
    aliases: HashSet<String>,
}

impl SigMeta {
    /// Build from a Signature type using the provided OutputFormatContent.
    ///
    /// This should be called with the registry-built format that contains
    /// both input and output types.
    pub fn from_format<S: Signature>(format: &OutputFormatContent) -> Self {
        let mut visited = VisitedTypes::default();

        let inputs = S::input_fields()
            .iter()
            .map(|field| {
                let ty = (field.type_ir)();
                SigFieldMeta {
                    llm_name: field.name,
                    rust_name: field.rust_name,
                    description: if field.description.is_empty() {
                        None
                    } else {
                        Some(field.description)
                    },
                    type_name: simplify_type_name(&ty),
                    schema: None,
                    r#type: build_type_meta(format, &ty, &mut visited),
                }
            })
            .collect();

        let outputs = S::output_fields()
            .iter()
            .map(|field| {
                let ty = (field.type_ir)();
                SigFieldMeta {
                    llm_name: field.name,
                    rust_name: field.rust_name,
                    description: if field.description.is_empty() {
                        None
                    } else {
                        Some(field.description)
                    },
                    type_name: simplify_type_name(&ty),
                    schema: Some(build_field_schema(format, &ty)),
                    r#type: build_type_meta(format, &ty, &mut visited),
                }
            })
            .collect();

        Self { inputs, outputs }
    }
}

/// Build structured type metadata from a TypeIR.
fn build_type_meta(
    format: &OutputFormatContent,
    ty: &TypeIR,
    visited: &mut VisitedTypes,
) -> SigTypeMeta {
    match ty {
        TypeIR::Top(_) => SigTypeMeta::Any,

        TypeIR::Primitive(type_value, _) => SigTypeMeta::Primitive {
            name: type_value.to_string(),
        },

        TypeIR::Literal(lit, _) => SigTypeMeta::Literal {
            value: literal_to_string(lit),
        },

        TypeIR::List(inner, _) => SigTypeMeta::List {
            item: Box::new(build_type_meta(format, inner, visited)),
        },

        TypeIR::Map(key, value, _) => SigTypeMeta::Map {
            key: Box::new(build_type_meta(format, key, visited)),
            value: Box::new(build_type_meta(format, value, visited)),
        },

        TypeIR::Tuple(items, _) => SigTypeMeta::Tuple {
            items: items
                .iter()
                .map(|item| build_type_meta(format, item, visited))
                .collect(),
        },

        TypeIR::Union(union, _) => {
            let view = union.view();
            match view {
                UnionTypeViewGeneric::Null => SigTypeMeta::Primitive {
                    name: "null".to_string(),
                },
                UnionTypeViewGeneric::Optional(inner) => SigTypeMeta::Union {
                    options: vec![build_type_meta(format, inner, visited)],
                    nullable: true,
                },
                UnionTypeViewGeneric::OneOf(types) => SigTypeMeta::Union {
                    options: types
                        .into_iter()
                        .map(|t| build_type_meta(format, t, visited))
                        .collect(),
                    nullable: false,
                },
                UnionTypeViewGeneric::OneOfOptional(types) => SigTypeMeta::Union {
                    options: types
                        .into_iter()
                        .map(|t| build_type_meta(format, t, visited))
                        .collect(),
                    nullable: true,
                },
            }
        }

        TypeIR::Enum { name, dynamic, .. } => {
            let values = format
                .enums
                .get(name)
                .map(build_enum_values)
                .unwrap_or_default();

            // Use rendered name from enum definition if available, otherwise simplify
            let display_name = format
                .enums
                .get(name)
                .map(|e| e.name.rendered_name().to_string())
                .unwrap_or_else(|| simplify_name(name));

            SigTypeMeta::Enum {
                name: display_name,
                dynamic: *dynamic,
                values,
            }
        }

        TypeIR::Class { name, dynamic, .. } => {
            // Check if this is a recursive class
            let recursive = format.recursive_classes.contains(name);

            // Cycle detection - use simplified name as key
            let key = simplify_name(name);
            if visited.classes.contains(&key) {
                return SigTypeMeta::Ref { name: key };
            }

            // Look up class (always NonStreaming for dspy-rs)
            let class = lookup_class(format, name);

            match class {
                Some(cls) => {
                    let display_name = cls.name.rendered_name().to_string();
                    visited.classes.insert(key.clone());

                    let fields = cls
                        .fields
                        .iter()
                        .map(|(field_name, field_ty, desc, _)| SigClassField {
                            name: field_name.rendered_name().to_string(),
                            description: desc.clone(),
                            r#type: build_type_meta(format, field_ty, visited),
                        })
                        .collect();

                    visited.classes.remove(&key);

                    SigTypeMeta::Class {
                        name: display_name,
                        dynamic: *dynamic,
                        recursive,
                        fields,
                    }
                }
                None if *dynamic => SigTypeMeta::Class {
                    name: simplify_name(name),
                    dynamic: true,
                    recursive: false,
                    fields: vec![],
                },
                None => SigTypeMeta::Ref {
                    name: simplify_name(name),
                },
            }
        }

        TypeIR::RecursiveTypeAlias { name, .. } => {
            let key = simplify_name(name);

            // Cycle detection for aliases
            if visited.aliases.contains(&key) {
                return SigTypeMeta::Ref { name: key };
            }

            if let Some(target) = format.structural_recursive_aliases.get(name) {
                visited.aliases.insert(key.clone());
                let result = build_type_meta(format, target, visited);
                visited.aliases.remove(&key);
                result
            } else {
                SigTypeMeta::Ref { name: key }
            }
        }

        TypeIR::Arrow(_, _) => SigTypeMeta::Other {
            description: "function".to_string(),
        },
    }
}

/// Look up a class by name (always uses NonStreaming, with fallback to Streaming).
fn lookup_class<'a>(format: &'a OutputFormatContent, name: &str) -> Option<&'a Class> {
    let name_owned = name.to_string();
    format
        .classes
        .get(&(name_owned.clone(), StreamingMode::NonStreaming))
        .or_else(|| format.classes.get(&(name_owned, StreamingMode::Streaming)))
}

/// Build enum values from an Enum definition.
fn build_enum_values(e: &Enum) -> Vec<SigEnumValue> {
    e.values
        .iter()
        .map(|(name, description)| {
            let real = name.real_name().to_string();
            let rendered = name.rendered_name().to_string();
            SigEnumValue {
                name: rendered.clone(),
                alias: if real != rendered { Some(real) } else { None },
                description: description.clone(),
            }
        })
        .collect()
}

/// Convert a LiteralValue to a properly escaped display string.
fn literal_to_string(lit: &LiteralValue) -> String {
    match lit {
        LiteralValue::String(s) => serde_json::to_string(s).unwrap_or_else(|_| format!("{s:?}")),
        LiteralValue::Int(i) => i.to_string(),
        LiteralValue::Bool(b) => b.to_string(),
    }
}

/// Simplify a fully-qualified name to just the last segment.
fn simplify_name(name: &str) -> String {
    name.rsplit("::").next().unwrap_or(name).to_string()
}

/// Build a simplified type name string from TypeIR for display in templates.
fn simplify_type_name(type_ir: &TypeIR) -> String {
    let raw = type_ir.diagnostic_repr().to_string();
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut token = String::new();
            for next in chars.by_ref() {
                if next == '`' {
                    break;
                }
                token.push(next);
            }
            let simplified = token.rsplit("::").next().unwrap_or(&token);
            result.push_str(simplified);
        } else {
            result.push(ch);
        }
    }

    result
        .replace("class ", "")
        .replace("enum ", "")
        .replace(" | ", " or ")
        .trim()
        .to_string()
}

/// Build a schema string for an output field using BAML's renderer.
fn build_field_schema(output_format: &OutputFormatContent, type_ir: &TypeIR) -> String {
    let field_format = OutputFormatContent {
        enums: output_format.enums.clone(),
        classes: output_format.classes.clone(),
        recursive_classes: output_format.recursive_classes.clone(),
        structural_recursive_aliases: output_format.structural_recursive_aliases.clone(),
        target: type_ir.clone(),
    };

    field_format
        .render(RenderOptions::default().with_prefix(None))
        .ok()
        .flatten()
        .unwrap_or_else(|| type_ir.diagnostic_repr().to_string())
}

// ============================================================================
// Traits
// ============================================================================

pub trait MetaSignature: Send + Sync {
    fn demos(&self) -> Vec<Example>;
    fn set_demos(&mut self, demos: Vec<Example>) -> Result<()>;
    fn instruction(&self) -> String;
    fn input_fields(&self) -> Value;
    fn output_fields(&self) -> Value;

    fn update_instruction(&mut self, instruction: String) -> Result<()>;
    fn append(&mut self, name: &str, value: Value) -> Result<()>;
}

pub trait Signature: Send + Sync + 'static {
    type Input: baml_bridge::BamlType + Send + Sync;
    type Output: baml_bridge::BamlType + Send + Sync;

    fn instruction() -> &'static str;
    fn input_fields() -> &'static [FieldSpec];
    fn output_fields() -> &'static [FieldSpec];
    fn output_format_content() -> &'static OutputFormatContent;

    fn from_parts(input: Self::Input, output: Self::Output) -> Self;
    fn into_parts(self) -> (Self::Input, Self::Output);
}
