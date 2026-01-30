use crate::baml_bridge::prompt::{PromptValue, RenderResult, RenderSession};
use crate::{Example, OutputFormatContent, RenderOptions, TypeIR};
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

mod compiled;
pub use compiled::*;

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

/// Metadata about a single signature field (input or output).
#[derive(Debug, Clone, Serialize)]
pub struct SigFieldMeta {
    /// Name as seen by the LLM (rendered name).
    pub llm_name: &'static str,
    /// Rust field name (for lookup in inputs map).
    pub rust_name: &'static str,
    /// Simplified type name for prompt templates.
    pub type_name: String,
    /// Schema string for output fields (JSON/YAML format description).
    pub schema: Option<String>,
}

/// Metadata about a signature for prompt templates.
#[derive(Debug, Clone, Serialize)]
pub struct SigMeta {
    pub inputs: Vec<SigFieldMeta>,
    pub outputs: Vec<SigFieldMeta>,
}

impl SigMeta {
    /// Build from a Signature type's metadata.
    pub fn from_signature<S: Signature>() -> Self {
        let output_format = S::output_format_content();

        let inputs = S::input_fields()
            .iter()
            .map(|field| {
                let ty = (field.type_ir)();
                SigFieldMeta {
                    llm_name: field.name,
                    rust_name: field.rust_name,
                    type_name: simplify_type_name(&ty),
                    schema: None,
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
                    type_name: simplify_type_name(&ty),
                    schema: Some(build_field_schema(output_format, &ty)),
                }
            })
            .collect();

        Self { inputs, outputs }
    }
}

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
