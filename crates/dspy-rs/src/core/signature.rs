use crate::baml_bridge::prompt::{PromptValue, RenderResult, RenderSession};
use crate::{Example, OutputFormatContent, TypeIR};
use anyhow::Result;
use serde_json::Value;

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
