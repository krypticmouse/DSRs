//! Prompt rendering world for typed prompt infrastructure.

use std::sync::Arc;

use baml_types::{StreamingMode, TypeIR};
use indexmap::{IndexMap, IndexSet};
use internal_baml_jinja::types::{Class, Enum};
use minijinja::{Environment, UndefinedBehavior};

use super::jinja::register_prompt_filters;
use super::renderer::{RenderError, RenderSettings, RendererDb, RendererDbSeed};
use super::value::{default_union_resolver, UnionResolver};
use internal_baml_jinja::types::OutputFormatContent;

/// Type database extracted from OutputFormatContent.
#[derive(Debug, Clone)]
pub struct TypeDb {
    pub enums: Arc<IndexMap<String, Enum>>,
    pub classes: Arc<IndexMap<(String, StreamingMode), Class>>,
    pub structural_recursive_aliases: Arc<IndexMap<String, TypeIR>>,
    pub recursive_classes: Arc<IndexSet<String>>,
}

impl TypeDb {
    /// Look up a class by name and streaming mode.
    pub fn find_class(&self, name: &str, mode: StreamingMode) -> Option<&Class> {
        self.classes.get(&(name.to_string(), mode))
    }

    /// Get a field's type from a class.
    pub fn class_field_type(
        &self,
        name: &str,
        mode: StreamingMode,
        field: &str,
    ) -> Option<TypeIR> {
        let class = self.find_class(name, mode)?;
        class
            .fields
            .iter()
            .find(|(field_name, _, _, _)| {
                field_name.real_name() == field || field_name.rendered_name() == field
            })
            .map(|(_, r#type, _, _)| r#type.clone())
    }

    /// Resolve a recursive type alias.
    pub fn resolve_recursive_alias(&self, name: &str) -> Option<&TypeIR> {
        self.structural_recursive_aliases.get(name)
    }

    /// Check if a class is recursive.
    pub fn is_recursive(&self, name: &str) -> bool {
        self.recursive_classes.contains(name)
    }

    /// Look up an enum by name.
    pub fn find_enum(&self, name: &str) -> Option<&Enum> {
        self.enums.get(name)
    }
}

#[derive(Debug, Clone)]
pub struct PromptWorld {
    pub types: TypeDb,
    pub renderers: RendererDb,
    pub jinja: Environment<'static>,
    pub settings: RenderSettings,
    pub union_resolver: UnionResolver,
}

impl PromptWorld {
    #[allow(clippy::result_large_err)]
    pub fn from_registry(
        output_format: OutputFormatContent,
        renderer_seed: RendererDbSeed,
        settings: RenderSettings,
    ) -> Result<Self, RenderError> {
        let types = TypeDb {
            enums: output_format.enums.clone(),
            classes: output_format.classes.clone(),
            structural_recursive_aliases: output_format.structural_recursive_aliases.clone(),
            recursive_classes: output_format.recursive_classes.clone(),
        };

        let mut jinja = crate::jsonish::jinja_helpers::get_env();
        jinja.set_undefined_behavior(UndefinedBehavior::Strict);
        register_prompt_filters(&mut jinja);

        let renderers = RendererDb::compile_from_seed(renderer_seed, &mut jinja)?;

        Ok(Self {
            types,
            renderers,
            jinja,
            settings,
            union_resolver: default_union_resolver,
        })
    }
}
