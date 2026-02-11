extern crate self as dspy_rs;

pub mod adapter;
pub mod augmentation;
pub mod core;
pub mod data;
pub mod evaluate;
pub mod modules;
pub mod optimizer;
pub mod predictors;
pub mod trace;
pub mod utils;

pub use adapter::chat::*;
pub use augmentation::*;
pub use core::*;
pub use data::*;
pub use evaluate::*;
pub use modules::*;
pub use optimizer::*;
pub use predictors::*;
pub use utils::*;

pub use bamltype::BamlConvertError;
pub use bamltype::BamlType; // attribute macro
pub use bamltype::Shape;
pub use bamltype::baml_types::{
    BamlValue, Constraint, ConstraintLevel, ResponseCheck, StreamingMode, TypeIR,
};
pub use bamltype::internal_baml_jinja::types::{OutputFormatContent, RenderOptions};
pub use bamltype::jsonish::deserializer::deserialize_flags::Flag;
pub use dsrs_macros::*;
pub use facet::Facet;

/// Pre-built signature for use in doc examples. Not part of the public API.
#[doc(hidden)]
pub mod doctest {
    #[derive(crate::Signature, Clone, Debug)]
    /// Answer questions accurately and concisely.
    pub struct QA {
        #[input]
        pub question: String,
        #[output]
        pub answer: String,
    }
}

#[doc(hidden)]
pub mod __macro_support {
    pub use anyhow;
    pub use bamltype;
    pub use indexmap;
    pub use schemars;
    pub use serde;
    pub use serde_json;
}

#[macro_export]
macro_rules! field {
    // Example Usage: field! {
    //   input["Description"] => question: String
    // }
    //
    // Example Output:
    //
    // {
    //   "question": {
    //     "type": "String",
    //     "desc": "Description",
    //     "schema": ""
    //   },
    //   ...
    // }

    // Pattern for field definitions with descriptions
    { $($field_type:ident[$desc:literal] => $field_name:ident : $field_ty:ty),* $(,)? } => {{
        use $crate::__macro_support::serde_json::json;

        let mut result = $crate::__macro_support::serde_json::Map::new();

        $(
            let type_str = stringify!($field_ty);
            let schema = {
                let schema = $crate::__macro_support::schemars::schema_for!($field_ty);
                let schema_json = $crate::__macro_support::serde_json::to_value(schema).unwrap();
                // Extract just the properties if it's an object schema
                if let Some(obj) = schema_json.as_object() {
                    if obj.contains_key("properties") {
                        schema_json["properties"].clone()
                    } else {
                        "".to_string().into()
                    }
                } else {
                    "".to_string().into()
                }
            };
            result.insert(
                stringify!($field_name).to_string(),
                json!({
                    "type": type_str,
                    "desc": $desc,
                    "schema": schema,
                    "__dsrs_field_type": stringify!($field_type)
                })
            );
        )*

        $crate::__macro_support::serde_json::Value::Object(result)
    }};

    // Pattern for field definitions without descriptions
    { $($field_type:ident => $field_name:ident : $field_ty:ty),* $(,)? } => {{
        use $crate::__macro_support::serde_json::json;

        let mut result = $crate::__macro_support::serde_json::Map::new();

        $(
            let type_str = stringify!($field_ty);
            let schema = {
                let schema = $crate::__macro_support::schemars::schema_for!($field_ty);
                let schema_json = $crate::__macro_support::serde_json::to_value(schema).unwrap();
                // Extract just the properties if it's an object schema
                if let Some(obj) = schema_json.as_object() {
                    if obj.contains_key("properties") {
                        schema_json["properties"].clone()
                    } else {
                        "".to_string().into()
                    }
                } else {
                    "".to_string().into()
                }
            };
            result.insert(
                stringify!($field_name).to_string(),
                json!({
                    "type": type_str,
                    "desc": "",
                    "schema": schema,
                    "__dsrs_field_type": stringify!($field_type)
                })
            );
        )*

        $crate::__macro_support::serde_json::Value::Object(result)
    }};
}

#[macro_export]
macro_rules! sign {
    // Example Usage: signature! {
    //     question: String, random: bool -> answer: String
    // }
    //
    // Example Output:
    //
    // #[derive(Signature, Clone)]
    // struct InlineSignature {
    //     #[input]
    //     question: String,
    //     #[input]
    //     random: bool,
    //     #[output]
    //     answer: String,
    // }
    //
    // Predict::<InlineSignature>::new()

    // Pattern: input fields -> output fields
    { ($($input_name:ident : $input_type:ty),* $(,)?) -> $($output_name:ident : $output_type:ty),* $(,)? } => {{
        #[derive($crate::Signature, Clone)]
        struct __InlineSignature {
            $(
                #[input]
                $input_name: $input_type,
            )*
            $(
                #[output]
                $output_name: $output_type,
            )*
        }

        $crate::Predict::<__InlineSignature>::new()
    }};
}

/// Source: https://github.com/wholesome-ghoul/hashmap_macro/blob/master/src/lib.rs
/// Author: https://github.com/wholesome-ghoul
/// License: MIT
/// Description: This macro creates a HashMap from a list of key-value pairs.
/// Reason for Reuse: Want to avoid adding a dependency for a simple macro.
#[macro_export]
macro_rules! hashmap {
    () => {
        ::std::collections::HashMap::new()
    };

    ($($key:expr => $value:expr),+ $(,)?) => {
        ::std::collections::HashMap::from([ $(($key, $value)),* ])
    };
}
