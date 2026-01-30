//! Prompt rendering world for typed prompt infrastructure.

use std::sync::Arc;

use baml_types::{StreamingMode, TypeIR};
use indexmap::{IndexMap, IndexSet};
use internal_baml_jinja::types::{Class, Enum};

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
pub struct PromptWorld;
