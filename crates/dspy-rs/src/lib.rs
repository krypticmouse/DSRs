pub mod adapter;
pub mod core;
pub mod data;
pub mod evaluate;
pub mod predictors;
pub mod utils;

pub use core::*;
pub use data::*;
pub use evaluate::*;
pub use predictors::*;
pub use utils::*;

pub use dsrs_macros::*;

#[macro_export]
macro_rules! sign {
    // Example Usage: signature! {
    //     question: String, random: bool -> answer: String
    // }
    //
    // Example Output:
    //
    // #[derive(Signature)]
    // struct InlineSignature {
    //     question: In<String>,
    //     random: In<bool>,
    //     answer: Out<String>,
    // }
    //
    // InlineSignature::new()

    // Pattern: input fields -> output fields
    { ($($input_name:ident : $input_type:ty),* $(,)?) -> $($output_name:ident : $output_type:ty),* $(,)? } => {{
        use dspy_rs::core::MetaSignature;
        use serde_json::Value;
        
        let mut input_fields = serde_json::Map::new();
        let mut output_fields = serde_json::Map::new();

        #[Signature]
        struct InlineSignature {
            $(
                #[input]
                $input_name: $input_type,
            )*
            $(
                #[output]
                $output_name: $output_type,
            )*
        }
        
        InlineSignature::new()
    }};
}

#[macro_export]
macro_rules! example {
    // Pattern: { "key": "value", ... }
    { $($key:literal : $value:expr),* $(,)? } => {{
        use std::collections::HashMap;
        use dspy_rs::data::example::Example;

        let mut fields = HashMap::new();
        $(
            fields.insert($key.to_string(), $value.to_string());
        )*

        Example::new(
            fields,
            vec![],
            vec![],
        )
    }};
}
