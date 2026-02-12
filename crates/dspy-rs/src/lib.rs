//! Typed prompt engineering and LM program optimization.
//!
//! DSRs is a Rust port of [DSPy](https://github.com/stanfordnlp/dspy): you declare what
//! you want the LM to produce (a [`Signature`]), pick a prompting strategy (a [`Module`]
//! like [`Predict`] or [`ChainOfThought`]), and let an [`Optimizer`] tune the program's
//! instructions and demos on your training data. The type system enforces correctness
//! at every layer — field types, strategy swaps, and augmentation composition are all
//! compile-time checked.
//!
//! # The mental model
//!
//! Three concepts, three layers:
//!
//! | Layer | Concept | Key types | Who |
//! |-------|---------|-----------|-----|
//! | **Signatures** | "Given these inputs, produce these outputs" | [`Signature`], `#[derive(Signature)]` | Everyone |
//! | **Modules** | Prompting strategies that implement a signature | [`Module`], [`Predict`], [`ChainOfThought`] | Everyone |
//! | **Optimization** | Auto-tuning instructions and demos | [`Optimizer`], [`COPRO`], [`GEPA`], [`MIPROv2`] | When you need better results |
//!
//! A [`Predict`] is the leaf — the only thing that actually calls the LM. Every other
//! module ([`ChainOfThought`], custom pipelines) delegates to one or more `Predict` leaves.
//! Optimizers discover these leaves automatically via Facet reflection and mutate their
//! instructions and few-shot demos.
//!
//! # Quick start
//!
//! ```no_run
//! use dspy_rs::*;
//!
//! #[derive(Signature, Clone, Debug)]
//! /// Answer questions accurately and concisely.
//! struct QA {
//!     #[input] question: String,
//!     #[output] answer: String,
//! }
//!
//! # async fn example() -> Result<(), PredictError> {
//! // 1. Configure the LM
//! let lm = LM::builder()
//!     .model("openai:gpt-4o-mini".to_string())
//!     .build()
//!     .await
//!     .unwrap();
//! dspy_rs::configure(lm, ChatAdapter);
//!
//! // 2. Pick a strategy
//! let cot = ChainOfThought::<QA>::new();
//!
//! // 3. Call it
//! let result = cot.call(QAInput { question: "What is 2+2?".into() }).await?;
//! println!("{}", result.reasoning);  // chain-of-thought text
//! println!("{}", result.answer);     // the actual answer, via Deref
//! # Ok(())
//! # }
//! ```
//!
//! `ChainOfThought<QA>` returns [`Predicted<WithReasoning<QAOutput>>`](Predicted), not
//! `Predicted<QAOutput>`. You access `.reasoning` directly and `.answer` through auto-deref
//! ([`WithReasoning<O>`] derefs to `O`). This pattern holds for all augmentations — the
//! compiler tells you what changed when you swap strategies.
//!
//! # What doesn't work (yet)
//!
//! - **No dynamic graph / structural optimization.** The type-erased `ProgramGraph`,
//!   `DynModule`, `StrategyFactory` layer was prototyped and intentionally removed.
//!   Everything here is statically typed, which is both the strength and the constraint.
//! - **MIPRO is instruction-only.** It should also mutate demos per-predictor based on
//!   trace data — Python DSPy does this — but it doesn't yet (`TODO(trace-demos)`).
//! - **No `ReAct`, `BestOfN`, `Refine`, or other advanced modules** beyond `ChainOfThought`.
//!   The module trait and augmentation system are designed for them, but nobody's built
//!   them yet.
//! - **`CallMetadata` is not extensible.** Modules can't attach custom metadata (e.g.
//!   "which attempt won in BestOfN"). This should probably be a trait with associated
//!   types, but it isn't.
//! - **Container traversal is partial.** The optimizer walker handles `Option`, `Vec`,
//!   `HashMap<String, _>`, and `Box`. `Rc`/`Arc` containing `Predict` leaves return
//!   explicit container errors (not silent skips), and `Predict` discovery requires
//!   a valid shape-local accessor payload (`TODO(dsrs-shared-ptr-policy)`).
//!
//! # Crate organization
//!
//! - [`adapter`] — Prompt formatting and LM response parsing ([`ChatAdapter`])
//! - [`core`] — [`Module`] trait, [`Signature`] trait, [`SignatureSchema`], error types,
//!   LM client, [`Predicted`] and [`CallMetadata`]
//! - [`predictors`] — [`Predict`] (the leaf module) and typed [`Example`]
//! - [`modules`] — [`ChainOfThought`] and augmentation types
//! - [`evaluate`] — [`TypedMetric`] trait, [`evaluate_trainset`], scoring utilities
//! - [`optimizer`] — [`Optimizer`] trait, [`COPRO`], [`GEPA`], [`MIPROv2`]
//! - [`data`] — [`DataLoader`] for JSON/CSV/Parquet/HuggingFace datasets
//! - [`trace`] — Execution graph recording for debugging
//! - [`utils`] — Response caching

// TODO(dsrs-facet-lint-scope): remove this crate-level allow once Facet's generated
// extension-attr dispatch no longer triggers rust-lang/rust#52234 on in-crate usage.
#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]

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
pub use data::dataloader::*;
pub(crate) use data::example::Example as RawExample;
pub use data::prediction::*;
pub use data::serialize::*;
pub use data::utils::*;
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

/// Source: <https://github.com/wholesome-ghoul/hashmap_macro/blob/master/src/lib.rs>
/// Author: <https://github.com/wholesome-ghoul>
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
