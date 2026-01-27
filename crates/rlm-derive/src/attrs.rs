//! Attribute parsing for RLM derive macros using darling.
//!
//! This module defines the attribute structs for parsing `#[rlm(...)]` attributes
//! at both the container (struct) and field levels.

use darling::{FromDeriveInput, FromField, FromMeta};
use syn::{Ident, Type};

/// Metadata for computed (non-struct-field) properties exposed to Python.
///
/// These are documented in `__rlm_schema__` but don't correspond to struct fields.
/// Useful for derived/computed values like `session.all_steps` or filtered collections.
///
/// # Example
///
/// ```ignore
/// #[rlm_type]
/// #[rlm(property(name = "all_steps", desc = "Flattened list of all steps across trajectories"))]
/// pub struct Session {
///     pub trajectories: Vec<Trajectory>,
/// }
/// ```
#[derive(Debug, Clone, FromMeta)]
pub struct RlmPropertyAttrs {
    /// Property name as exposed in Python.
    pub name: String,
    /// Human-readable description for REPL discovery.
    #[darling(default)]
    pub desc: Option<String>,
}

/// Container-level attributes for `#[derive(RlmType)]`.
///
/// These control struct-wide behavior for Python integration.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(rlm), supports(struct_named))]
pub struct RlmTypeAttrs {
    /// The struct identifier.
    pub ident: Ident,
    /// Generics on the struct.
    pub generics: syn::Generics,
    /// The struct's data (fields).
    pub data: darling::ast::Data<(), RlmFieldAttrs>,

    /// Custom `__repr__` format string.
    /// Placeholders like `{field_name}` are substituted.
    #[darling(default)]
    pub repr: Option<String>,

    /// Field name to use for `__iter__` / `__len__`.
    /// The field must be iterable (Vec, slice, etc.).
    #[darling(default, rename = "iter")]
    pub iter_field: Option<String>,

    /// Field name to use for `__getitem__`.
    /// The field must be indexable.
    #[darling(default, rename = "index")]
    pub index_field: Option<String>,

    /// Override the Python class name (defaults to Rust struct name).
    #[darling(default)]
    pub pyclass_name: Option<String>,

    /// Computed properties to document in schema.
    #[darling(default, multiple)]
    pub property: Vec<RlmPropertyAttrs>,
}

/// Field-level attributes for `#[derive(RlmType)]`.
///
/// These control per-field behavior for getters, schema, and filtering.
#[derive(Debug, Clone, FromField)]
#[darling(attributes(rlm))]
pub struct RlmFieldAttrs {
    /// The field identifier (None for tuple structs, but we only support named).
    pub ident: Option<Ident>,
    /// The field's type.
    pub ty: Type,

    /// Human-readable description for REPL discovery.
    #[darling(default)]
    pub desc: Option<String>,

    /// Generate a filtered property with this name.
    /// Requires `filter_value` to also be set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[rlm(filter_property = "tool_calls", filter_value = "ToolCall", filter_field = "steps")]
    /// pub steps: Vec<Step>,
    /// ```
    ///
    /// Generates `tool_calls` property returning only steps where variant is ToolCall.
    #[darling(default)]
    pub filter_property: Option<String>,

    /// The variant or value to filter on.
    #[darling(default)]
    pub filter_value: Option<String>,

    /// The field on the enum/struct to check for filtering.
    /// If not specified, assumes discriminant/variant matching.
    #[darling(default)]
    pub filter_field: Option<String>,

    /// Flatten a nested collection into a computed property.
    /// Value is the property name to generate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[rlm(flatten_property = "all_steps", flatten_parent = "trajectories")]
    /// pub trajectories: Vec<Trajectory>,
    /// ```
    ///
    /// Generates `all_steps` that chains all trajectory.steps together.
    #[darling(default)]
    pub flatten_property: Option<String>,

    /// The parent collection to flatten from.
    #[darling(default)]
    pub flatten_parent: Option<String>,

    /// Skip generating a Python getter for this field.
    #[darling(default)]
    pub skip_python: bool,

    /// Skip including this field in the schema output.
    #[darling(default)]
    pub skip_schema: bool,
}

impl RlmTypeAttrs {
    /// Returns the Python class name (either explicit or derived from ident).
    pub fn python_class_name(&self) -> String {
        self.pyclass_name
            .clone()
            .unwrap_or_else(|| self.ident.to_string())
    }

    /// Returns an iterator over the struct fields.
    ///
    /// # Panics
    ///
    /// Panics if the data is not a struct (should be impossible due to darling supports).
    pub fn fields(&self) -> impl Iterator<Item = &RlmFieldAttrs> {
        match &self.data {
            darling::ast::Data::Struct(fields) => fields.iter(),
            _ => unreachable!("RlmTypeAttrs only supports named structs"),
        }
    }

    /// Find the field marked for iteration, if any.
    pub fn iter_field_attrs(&self) -> Option<&RlmFieldAttrs> {
        let iter_name = self.iter_field.as_ref()?;
        self.fields()
            .find(|f| f.ident.as_ref().map(|i| i.to_string()).as_ref() == Some(iter_name))
    }

    /// Find the field marked for indexing, if any.
    pub fn index_field_attrs(&self) -> Option<&RlmFieldAttrs> {
        let index_name = self.index_field.as_ref()?;
        self.fields()
            .find(|f| f.ident.as_ref().map(|i| i.to_string()).as_ref() == Some(index_name))
    }
}

impl RlmFieldAttrs {
    /// Returns the field name as a string.
    ///
    /// # Panics
    ///
    /// Panics if the field has no identifier (tuple struct field).
    pub fn name(&self) -> String {
        self.ident
            .as_ref()
            .expect("RlmFieldAttrs requires named fields")
            .to_string()
    }

    /// Check if this field has filter configuration.
    pub fn has_filter(&self) -> bool {
        self.filter_property.is_some() && self.filter_value.is_some()
    }

    /// Check if this field has flatten configuration.
    pub fn has_flatten(&self) -> bool {
        self.flatten_property.is_some()
    }

    /// Returns true if this field should have a Python getter generated.
    pub fn should_generate_getter(&self) -> bool {
        !self.skip_python
    }

    /// Returns true if this field should be included in schema.
    pub fn should_include_in_schema(&self) -> bool {
        !self.skip_schema
    }

    /// Validate filter configuration consistency.
    ///
    /// Returns an error message if filter_property is set but filter_value is not,
    /// or vice versa.
    pub fn validate_filter(&self) -> Result<(), String> {
        match (&self.filter_property, &self.filter_value) {
            (Some(_), None) => Err(format!(
                "field `{}`: filter_property requires filter_value",
                self.name()
            )),
            (None, Some(_)) => Err(format!(
                "field `{}`: filter_value requires filter_property",
                self.name()
            )),
            _ => Ok(()),
        }
    }
}

impl RlmTypeAttrs {
    /// Validate all fields and container-level attributes.
    ///
    /// Returns a list of validation errors, or empty vec if valid.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Validate iter_field exists if specified
        if let Some(ref iter_name) = self.iter_field {
            if self.iter_field_attrs().is_none() {
                errors.push(format!(
                    "iter field `{}` not found in struct `{}`",
                    iter_name,
                    self.ident
                ));
            }
        }

        // Validate index_field exists if specified
        if let Some(ref index_name) = self.index_field {
            if self.index_field_attrs().is_none() {
                errors.push(format!(
                    "index field `{}` not found in struct `{}`",
                    index_name,
                    self.ident
                ));
            }
        }

        // Validate each field
        for field in self.fields() {
            if let Err(e) = field.validate_filter() {
                errors.push(e);
            }
        }

        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use darling::FromDeriveInput;
    use syn::parse_quote;

    #[test]
    fn test_basic_parsing() {
        let input: syn::DeriveInput = parse_quote! {
            #[rlm(repr = "Trajectory({session_id})", iter = "steps")]
            pub struct Trajectory {
                #[rlm(desc = "Unique session identifier")]
                pub session_id: String,
                #[rlm(desc = "List of steps in the trajectory")]
                pub steps: Vec<Step>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        assert_eq!(attrs.ident.to_string(), "Trajectory");
        assert_eq!(attrs.repr, Some("Trajectory({session_id})".to_string()));
        assert_eq!(attrs.iter_field, Some("steps".to_string()));
        assert_eq!(attrs.python_class_name(), "Trajectory");
    }

    #[test]
    fn test_pyclass_name_override() {
        let input: syn::DeriveInput = parse_quote! {
            #[rlm(pyclass_name = "PyTrajectory")]
            pub struct Trajectory {
                pub id: String,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        assert_eq!(attrs.python_class_name(), "PyTrajectory");
    }

    #[test]
    fn test_field_attrs() {
        let input: syn::DeriveInput = parse_quote! {
            pub struct Session {
                #[rlm(desc = "Session ID", skip_python)]
                pub id: String,
                #[rlm(filter_property = "tool_calls", filter_value = "ToolCall")]
                pub steps: Vec<Step>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        let fields: Vec<_> = attrs.fields().collect();

        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].desc, Some("Session ID".to_string()));
        assert!(fields[0].skip_python);
        assert!(!fields[0].should_generate_getter());

        assert!(fields[1].has_filter());
        assert_eq!(fields[1].filter_property, Some("tool_calls".to_string()));
        assert_eq!(fields[1].filter_value, Some("ToolCall".to_string()));
    }

    #[test]
    fn test_computed_properties() {
        let input: syn::DeriveInput = parse_quote! {
            #[rlm(property(name = "all_steps", desc = "All steps flattened"))]
            #[rlm(property(name = "total_cost", desc = "Sum of all step costs"))]
            pub struct Session {
                pub trajectories: Vec<Trajectory>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        assert_eq!(attrs.property.len(), 2);
        assert_eq!(attrs.property[0].name, "all_steps");
        assert_eq!(
            attrs.property[0].desc,
            Some("All steps flattened".to_string())
        );
        assert_eq!(attrs.property[1].name, "total_cost");
    }

    #[test]
    fn test_validation_iter_field_exists() {
        let input: syn::DeriveInput = parse_quote! {
            #[rlm(iter = "steps")]
            pub struct Trajectory {
                pub steps: Vec<Step>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        let errors = attrs.validate();
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_validation_iter_field_missing() {
        let input: syn::DeriveInput = parse_quote! {
            #[rlm(iter = "nonexistent")]
            pub struct Trajectory {
                pub steps: Vec<Step>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        let errors = attrs.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("iter field `nonexistent` not found"));
    }

    #[test]
    fn test_validation_filter_incomplete() {
        let input: syn::DeriveInput = parse_quote! {
            pub struct Session {
                #[rlm(filter_property = "tool_calls")]
                pub steps: Vec<Step>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        let errors = attrs.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("filter_property requires filter_value"));
    }

    #[test]
    fn test_validation_all_valid() {
        let input: syn::DeriveInput = parse_quote! {
            #[rlm(iter = "steps", index = "steps")]
            pub struct Trajectory {
                #[rlm(filter_property = "tool_calls", filter_value = "ToolCall")]
                pub steps: Vec<Step>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).unwrap();
        let errors = attrs.validate();
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }
}
