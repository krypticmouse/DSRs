//! Type description traits for RLM-compatible types.
//!
//! This module provides traits and structs for describing Rust types
//! to language models in a way that enables better prompt generation
//! and type introspection.
//!
//! # Overview
//!
//! The core trait [`RlmDescribe`] enables types to describe themselves:
//! - Their type name and structure
//! - Their fields and properties
//! - Whether they are iterable or indexable
//! - Runtime value descriptions for prompt generation
//!
//! # Example
//!
//! ```ignore
//! use rlm_core::describe::{RlmDescribe, RlmFieldDesc, RlmTypeInfo};
//!
//! struct User {
//!     name: String,
//!     age: u32,
//! }
//!
//! impl RlmDescribe for User {
//!     fn type_name() -> &'static str { "User" }
//!
//!     fn fields() -> Vec<RlmFieldDesc> {
//!         vec![
//!             RlmFieldDesc::new("name", "String").with_desc("The user's name"),
//!             RlmFieldDesc::new("age", "u32").with_desc("The user's age in years"),
//!         ]
//!     }
//!
//!     fn describe_value(&self) -> String {
//!         format!("User {{ name: {}, age: {} }}", self.name, self.age)
//!     }
//! }
//! ```

use std::collections::HashMap;

/// Description of a struct field for prompt generation.
///
/// Contains metadata about a field that can be used to generate
/// descriptive prompts for language models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlmFieldDesc {
    /// The field name as it appears in Rust.
    pub name: &'static str,
    /// The type name of the field.
    pub type_name: &'static str,
    /// Optional human-readable description.
    pub description: Option<&'static str>,
    /// Whether this field is optional (`Option<T>`).
    pub is_optional: bool,
    /// Whether this field contains a collection (`Vec<T>`, etc.).
    pub is_collection: bool,
}

impl RlmFieldDesc {
    /// Create a new field description with required fields.
    pub const fn new(name: &'static str, type_name: &'static str) -> Self {
        Self {
            name,
            type_name,
            description: None,
            is_optional: false,
            is_collection: false,
        }
    }

    /// Add a human-readable description.
    pub const fn with_desc(mut self, desc: &'static str) -> Self {
        self.description = Some(desc);
        self
    }

    /// Mark this field as optional.
    pub const fn optional(mut self) -> Self {
        self.is_optional = true;
        self
    }

    /// Mark this field as a collection.
    pub const fn collection(mut self) -> Self {
        self.is_collection = true;
        self
    }
}

/// Description of a computed property for prompt generation.
///
/// Computed properties are derived values that don't correspond directly
/// to struct fields but are exposed to Python/prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlmPropertyDesc {
    /// The property name as exposed in Python.
    pub name: &'static str,
    /// The return type name.
    pub type_name: &'static str,
    /// Optional human-readable description.
    pub description: Option<&'static str>,
}

impl RlmPropertyDesc {
    /// Create a new property description.
    pub const fn new(name: &'static str, type_name: &'static str) -> Self {
        Self {
            name,
            type_name,
            description: None,
        }
    }

    /// Add a human-readable description.
    pub const fn with_desc(mut self, desc: &'static str) -> Self {
        self.description = Some(desc);
        self
    }
}

/// Core trait for describing types to language models.
///
/// This trait enables types to provide rich metadata about their structure
/// and contents, which can be used for:
/// - Generating descriptive prompts
/// - Schema discovery in REPLs
/// - Type-aware serialization
///
/// # Deriving
///
/// This trait can be derived using `#[derive(RlmDescribe)]` from the
/// `rlm-derive` crate, which will automatically implement all methods
/// based on struct field attributes.
pub trait RlmDescribe {
    /// Returns the type name as it should appear in prompts.
    ///
    /// This is typically the Rust struct name, but can be customized
    /// for Python-facing names.
    fn type_name() -> &'static str;

    /// Returns descriptions of all struct fields.
    ///
    /// Default implementation returns an empty vec for types without fields.
    fn fields() -> Vec<RlmFieldDesc> {
        Vec::new()
    }

    /// Returns descriptions of computed properties.
    ///
    /// Computed properties are methods that return derived values,
    /// like filtered collections or aggregations.
    fn properties() -> Vec<RlmPropertyDesc> {
        Vec::new()
    }

    /// Returns true if this type supports iteration (has `__iter__`).
    ///
    /// Types that implement this should also provide `describe_items()`.
    fn is_iterable() -> bool {
        false
    }

    /// Returns true if this type supports indexing (has `__getitem__`).
    fn is_indexable() -> bool {
        false
    }

    /// Describes a concrete value for inclusion in prompts.
    ///
    /// This method generates a string representation suitable for
    /// showing to a language model as an example or context.
    fn describe_value(&self) -> String;

    /// Returns the static type description for prompts.
    ///
    /// This is used when describing the type itself (not a concrete value),
    /// such as in function signatures or schema documentation.
    fn describe_type() -> String {
        let mut parts = vec![format!("type {}", Self::type_name())];

        let fields = Self::fields();
        if !fields.is_empty() {
            parts.push("fields:".to_string());
            for field in &fields {
                let mut field_desc = format!("  - {}: {}", field.name, field.type_name);
                if field.is_optional {
                    field_desc.push_str(" (optional)");
                }
                if field.is_collection {
                    field_desc.push_str(" (collection)");
                }
                if let Some(desc) = field.description {
                    field_desc.push_str(&format!(" - {}", desc));
                }
                parts.push(field_desc);
            }
        }

        let properties = Self::properties();
        if !properties.is_empty() {
            parts.push("properties:".to_string());
            for prop in &properties {
                let mut prop_desc = format!("  - {}: {}", prop.name, prop.type_name);
                if let Some(desc) = prop.description {
                    prop_desc.push_str(&format!(" - {}", desc));
                }
                parts.push(prop_desc);
            }
        }

        if Self::is_iterable() {
            parts.push("(iterable)".to_string());
        }
        if Self::is_indexable() {
            parts.push("(indexable)".to_string());
        }

        parts.join("\n")
    }
}

/// Helper trait for extracting type information at compile time.
///
/// This trait provides a way to get type metadata without requiring
/// a value instance. It's primarily used by derive macros.
pub trait RlmTypeInfo {
    /// The simple type name (e.g., "String", "`Vec<User>`").
    const TYPE_NAME: &'static str;

    /// Whether this is an optional type (`Option<T>`).
    const IS_OPTIONAL: bool = false;

    /// Whether this is a collection type (`Vec<T>`, etc.).
    const IS_COLLECTION: bool = false;

    /// Whether the inner type implements RlmDescribe.
    const IS_DESCRIBABLE: bool = false;
}

// ============================================================================
// Standard library implementations
// ============================================================================

impl RlmTypeInfo for String {
    const TYPE_NAME: &'static str = "String";
}

impl RlmTypeInfo for &str {
    const TYPE_NAME: &'static str = "str";
}

impl RlmTypeInfo for bool {
    const TYPE_NAME: &'static str = "bool";
}

impl RlmTypeInfo for i8 {
    const TYPE_NAME: &'static str = "i8";
}

impl RlmTypeInfo for i16 {
    const TYPE_NAME: &'static str = "i16";
}

impl RlmTypeInfo for i32 {
    const TYPE_NAME: &'static str = "i32";
}

impl RlmTypeInfo for i64 {
    const TYPE_NAME: &'static str = "i64";
}

impl RlmTypeInfo for i128 {
    const TYPE_NAME: &'static str = "i128";
}

impl RlmTypeInfo for isize {
    const TYPE_NAME: &'static str = "isize";
}

impl RlmTypeInfo for u8 {
    const TYPE_NAME: &'static str = "u8";
}

impl RlmTypeInfo for u16 {
    const TYPE_NAME: &'static str = "u16";
}

impl RlmTypeInfo for u32 {
    const TYPE_NAME: &'static str = "u32";
}

impl RlmTypeInfo for u64 {
    const TYPE_NAME: &'static str = "u64";
}

impl RlmTypeInfo for u128 {
    const TYPE_NAME: &'static str = "u128";
}

impl RlmTypeInfo for usize {
    const TYPE_NAME: &'static str = "usize";
}

impl RlmTypeInfo for f32 {
    const TYPE_NAME: &'static str = "f32";
}

impl RlmTypeInfo for f64 {
    const TYPE_NAME: &'static str = "f64";
}

// ============================================================================
// Container implementations
// ============================================================================

impl<T: RlmTypeInfo> RlmTypeInfo for Vec<T> {
    const TYPE_NAME: &'static str = "Vec";
    const IS_COLLECTION: bool = true;
    const IS_DESCRIBABLE: bool = T::IS_DESCRIBABLE;
}

impl<T: RlmDescribe> RlmDescribe for Vec<T> {
    fn type_name() -> &'static str {
        "Vec"
    }

    fn is_iterable() -> bool {
        true
    }

    fn is_indexable() -> bool {
        true
    }

    fn describe_value(&self) -> String {
        if self.is_empty() {
            return format!("Vec<{}> (empty)", T::type_name());
        }

        let item_descriptions: Vec<String> = self.iter().map(|item| item.describe_value()).collect();

        if self.len() <= 3 {
            format!(
                "Vec<{}> with {} items: [{}]",
                T::type_name(),
                self.len(),
                item_descriptions.join(", ")
            )
        } else {
            format!(
                "Vec<{}> with {} items: [{}, {}, ... and {} more]",
                T::type_name(),
                self.len(),
                item_descriptions[0],
                item_descriptions[1],
                self.len() - 2
            )
        }
    }

    fn describe_type() -> String {
        format!(
            "Vec<{}> - a collection of {} items (iterable, indexable)",
            T::type_name(),
            T::type_name()
        )
    }
}

impl<T: RlmTypeInfo> RlmTypeInfo for Option<T> {
    const TYPE_NAME: &'static str = "Option";
    const IS_OPTIONAL: bool = true;
    const IS_DESCRIBABLE: bool = T::IS_DESCRIBABLE;
}

impl<T: RlmDescribe> RlmDescribe for Option<T> {
    fn type_name() -> &'static str {
        "Option"
    }

    fn describe_value(&self) -> String {
        match self {
            Some(value) => format!("Some({})", value.describe_value()),
            None => "None".to_string(),
        }
    }

    fn describe_type() -> String {
        format!(
            "Option<{}> - an optional {} value",
            T::type_name(),
            T::type_name()
        )
    }
}

// ============================================================================
// Primitive RlmDescribe implementations
// ============================================================================

impl RlmDescribe for String {
    fn type_name() -> &'static str {
        "String"
    }

    fn describe_value(&self) -> String {
        if self.len() > 100 {
            format!("\"{}...\" ({} chars)", &self[..97], self.len())
        } else {
            format!("\"{}\"", self)
        }
    }

    fn describe_type() -> String {
        "String - a text value".to_string()
    }
}

impl RlmDescribe for bool {
    fn type_name() -> &'static str {
        "bool"
    }

    fn describe_value(&self) -> String {
        self.to_string()
    }

    fn describe_type() -> String {
        "bool - true or false".to_string()
    }
}

macro_rules! impl_rlm_describe_for_int {
    ($($ty:ty),*) => {
        $(
            impl RlmDescribe for $ty {
                fn type_name() -> &'static str {
                    stringify!($ty)
                }

                fn describe_value(&self) -> String {
                    self.to_string()
                }

                fn describe_type() -> String {
                    format!("{} - an integer value", stringify!($ty))
                }
            }
        )*
    };
}

impl_rlm_describe_for_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! impl_rlm_describe_for_float {
    ($($ty:ty),*) => {
        $(
            impl RlmDescribe for $ty {
                fn type_name() -> &'static str {
                    stringify!($ty)
                }

                fn describe_value(&self) -> String {
                    format!("{:.6}", self)
                }

                fn describe_type() -> String {
                    format!("{} - a floating-point number", stringify!($ty))
                }
            }
        )*
    };
}

impl_rlm_describe_for_float!(f32, f64);

// ============================================================================
// HashMap implementation
// ============================================================================

impl<K: RlmTypeInfo, V: RlmTypeInfo> RlmTypeInfo for HashMap<K, V> {
    const TYPE_NAME: &'static str = "HashMap";
    const IS_COLLECTION: bool = true;
}

impl<K: RlmDescribe + std::fmt::Debug, V: RlmDescribe> RlmDescribe for HashMap<K, V> {
    fn type_name() -> &'static str {
        "HashMap"
    }

    fn is_iterable() -> bool {
        true
    }

    fn is_indexable() -> bool {
        true
    }

    fn describe_value(&self) -> String {
        if self.is_empty() {
            return format!("HashMap<{}, {}> (empty)", K::type_name(), V::type_name());
        }

        let entries: Vec<String> = self
            .iter()
            .take(3)
            .map(|(k, v)| format!("{:?}: {}", k, v.describe_value()))
            .collect();

        if self.len() <= 3 {
            format!(
                "HashMap<{}, {}> with {} entries: {{{}}}",
                K::type_name(),
                V::type_name(),
                self.len(),
                entries.join(", ")
            )
        } else {
            format!(
                "HashMap<{}, {}> with {} entries: {{{}, ... and {} more}}",
                K::type_name(),
                V::type_name(),
                self.len(),
                entries.join(", "),
                self.len() - 3
            )
        }
    }

    fn describe_type() -> String {
        format!(
            "HashMap<{}, {}> - a key-value mapping (iterable, indexable by key)",
            K::type_name(),
            V::type_name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_desc_builder() {
        let field = RlmFieldDesc::new("name", "String")
            .with_desc("The user's name")
            .optional();

        assert_eq!(field.name, "name");
        assert_eq!(field.type_name, "String");
        assert_eq!(field.description, Some("The user's name"));
        assert!(field.is_optional);
        assert!(!field.is_collection);
    }

    #[test]
    fn test_property_desc_builder() {
        let prop = RlmPropertyDesc::new("full_name", "String").with_desc("First and last name");

        assert_eq!(prop.name, "full_name");
        assert_eq!(prop.type_name, "String");
        assert_eq!(prop.description, Some("First and last name"));
    }

    #[test]
    fn test_string_describe() {
        let s = "Hello, world!".to_string();
        assert_eq!(s.describe_value(), "\"Hello, world!\"");
        assert_eq!(String::type_name(), "String");
    }

    #[test]
    fn test_string_describe_truncation() {
        let long_string = "a".repeat(150);
        let desc = long_string.describe_value();
        assert!(desc.contains("..."));
        assert!(desc.contains("150 chars"));
    }

    #[test]
    fn test_vec_describe_empty() {
        let v: Vec<String> = vec![];
        assert_eq!(v.describe_value(), "Vec<String> (empty)");
    }

    #[test]
    fn test_vec_describe_few_items() {
        let v = vec!["a".to_string(), "b".to_string()];
        let desc = v.describe_value();
        assert!(desc.contains("2 items"));
        assert!(desc.contains("\"a\""));
        assert!(desc.contains("\"b\""));
    }

    #[test]
    fn test_vec_describe_many_items() {
        let v = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let desc = v.describe_value();
        assert!(desc.contains("5 items"));
        assert!(desc.contains("and 3 more"));
    }

    #[test]
    fn test_option_describe() {
        let some: Option<String> = Some("hello".to_string());
        let none: Option<String> = None;

        assert!(some.describe_value().contains("Some"));
        assert!(some.describe_value().contains("hello"));
        assert_eq!(none.describe_value(), "None");
    }

    #[test]
    fn test_type_info_primitives() {
        assert_eq!(String::TYPE_NAME, "String");
        assert_eq!(bool::TYPE_NAME, "bool");
        assert_eq!(i32::TYPE_NAME, "i32");
        assert_eq!(f64::TYPE_NAME, "f64");
    }

    #[test]
    fn test_type_info_containers() {
        assert!(Vec::<String>::IS_COLLECTION);
        assert!(Option::<String>::IS_OPTIONAL);
    }

    #[test]
    fn test_integer_describe() {
        let n: i32 = 42;
        assert_eq!(n.describe_value(), "42");
        assert_eq!(i32::type_name(), "i32");
    }

    #[test]
    fn test_float_describe() {
        let f: f64 = 3.14159;
        let desc = f.describe_value();
        assert!(desc.starts_with("3.14"));
    }

    #[test]
    fn test_bool_describe() {
        assert_eq!(true.describe_value(), "true");
        assert_eq!(false.describe_value(), "false");
    }
}
