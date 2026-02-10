### Summary
Slice 6 delivers the full V6 dynamic graph path on top of Slice 5: [NEW] `DynModule` + [NEW] strategy registry/factories, [NEW] `ProgramGraph` mutation/validation/execution, and typed-module projection with the locked snapshot-then-fit-back contract: [NEW] immutable `from_module(&module)` built on [NEW] `named_parameters_ref`, followed by [NEW] `graph.fit(&mut module)` for mutable write-back. This explicitly resolves the current API tension between existing mutable discovery (`/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:72`) and design-time immutable projection, while keeping C8 locked to annotation-first edge derivation (no trace-inferred wiring in this slice). The implementation path stays shortest-correct: reuse the existing accessor bridge where possible, and record all spec-divergent shortcuts as migration debt.

### Implementation Steps
1. Add immutable predictor discovery to support snapshot projection without mutably borrowing typed modules.
   - Files to modify:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs`
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs`
   - Existing signatures (copied):
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:33`
       ```rust
       pub struct PredictAccessorFns {
           pub accessor: fn(*mut ()) -> *mut dyn DynPredictor,
       }
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:48`
       ```rust
       pub fn register_predict_accessor(
           shape: &'static Shape,
           accessor: fn(*mut ()) -> *mut dyn DynPredictor,
       )
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:72`
       ```rust
       pub fn named_parameters<M>(
           module: &mut M,
       ) -> std::result::Result<Vec<(String, &mut dyn DynPredictor)>, NamedParametersError>
       where
           M: for<'a> Facet<'a>,
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs:34`
       ```rust
       fn predict_dyn_accessor<S>(value: *mut ()) -> *mut dyn DynPredictor
       where
           S: Signature,
       ```
   - New signatures:
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs`
       ```rust
       pub struct PredictAccessorFns {
           pub accessor_mut: fn(*mut ()) -> *mut dyn DynPredictor,
           pub accessor_ref: fn(*const ()) -> *const dyn DynPredictor,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs`
       ```rust
       pub fn register_predict_accessor(
           shape: &'static Shape,
           accessor_mut: fn(*mut ()) -> *mut dyn DynPredictor,
           accessor_ref: fn(*const ()) -> *const dyn DynPredictor,
       )
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs`
       ```rust
       pub fn named_parameters_ref<M>(
           module: &M,
       ) -> std::result::Result<Vec<(String, &dyn DynPredictor)>, NamedParametersError>
       where
           M: for<'a> Facet<'a>,
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs`
       ```rust
       fn predict_dyn_accessor_ref<S>(value: *const ()) -> *const dyn DynPredictor
       where
           S: Signature,
       ```
   - Imports needed:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs`: add `use bamltype::facet_reflect::{Peek, Poke};`
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs`: no new crate imports; update `register_predict_accessor(...)` call sites to pass mutable and immutable accessors.
   - Existing code that must change:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs:58` (`pub fn new() -> Self`) and `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs:288` (`pub fn build(self) -> Predict<S>`) must register both accessor function pointers.
   - Migration debt to record (explicit):
     - [NEW] Slice 5 currently resolves predictor accessor functions via a global `ShapeId -> PredictAccessorFns` registry, while S2's preferred end-state is shape-local Facet attr payload decoding (`attr.get_as::<PredictAccessorFns>()`).
   - Arbitration resolution:
     - Keep the global accessor registry bridge in V6 for shortest-correct delivery; do not migrate to shape-local attr payload decoding in this slice.
     - Keep this as explicit migration debt for the post-implementation cleanup pass.

2. Add schema cloning and field lookup APIs required by strategy factories and graph validation.
   - Files to modify:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs`
   - Existing signatures (copied):
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs:74`
       ```rust
       pub struct SignatureSchema {
           instruction: &'static str,
           input_fields: Box<[FieldSchema]>,
           output_fields: Box<[FieldSchema]>,
           output_format: Arc<OutputFormatContent>,
       }
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs:139`
       ```rust
       pub fn input_fields(&self) -> &[FieldSchema]
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs:143`
       ```rust
       pub fn output_fields(&self) -> &[FieldSchema]
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs:166`
       ```rust
       pub fn field_by_rust<'a>(&'a self, rust_name: &str) -> Option<&'a FieldSchema>
       ```
   - New signatures:
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs`
       ```rust
       pub(crate) fn from_parts(
           instruction: &'static str,
           input_fields: Vec<FieldSchema>,
           output_fields: Vec<FieldSchema>,
           output_format: Arc<OutputFormatContent>,
       ) -> Self
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs`
       ```rust
       pub fn input_field_by_rust<'a>(&'a self, rust_name: &str) -> Option<&'a FieldSchema>
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs`
       ```rust
       pub fn output_field_by_rust<'a>(&'a self, rust_name: &str) -> Option<&'a FieldSchema>
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs`
       ```rust
       pub fn with_fields(
           &self,
           input_fields: Vec<FieldSchema>,
           output_fields: Vec<FieldSchema>,
       ) -> Self
       ```
   - Imports needed:
     - Existing `use std::sync::{Arc, Mutex, OnceLock};` at `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs:3` remains sufficient.
   - Existing code that must change:
     - Add `Clone` to `SignatureSchema` derive so factories can snapshot and transform schemas without mutating the global cache entry returned by `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs:82` (`pub fn of<S: Signature>() -> &'static Self`).
     - Keep `from_parts` crate-private to avoid reintroducing manual public schema construction across P1/P2 boundaries (R3/R9).

3. Add the dynamic strategy layer (`DynModule`, `StrategyFactory`, registry APIs) with inventory auto-registration.
   - Files to create/modify:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs`
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml`
   - Existing signatures (copied):
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs:2`
       ```rust
       pub mod dyn_predictor;
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs:13`
       ```rust
       pub use dyn_predictor::*;
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml:16`
       ```toml
       [dependencies]
       ```
   - New signatures:
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs`
       ```rust
       pub enum StrategyError {
           UnknownStrategy { name: String },
           InvalidConfig { strategy: &'static str, reason: String },
           BuildFailed { strategy: &'static str, reason: String },
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs`
       ```rust
       pub type StrategyConfig = serde_json::Value;
       pub type StrategyConfigSchema = serde_json::Value;
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs`
       ```rust
       #[async_trait::async_trait]
       pub trait DynModule: Send + Sync {
           fn schema(&self) -> &SignatureSchema;
           fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)>;
           fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)>;
           async fn forward(
               &self,
               input: BamlValue,
           ) -> std::result::Result<Predicted<BamlValue>, PredictError>;
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs`
       ```rust
       pub trait StrategyFactory: Send + Sync {
           fn name(&self) -> &'static str;
           fn config_schema(&self) -> StrategyConfigSchema;
           fn create(
               &self,
               base_schema: &SignatureSchema,
               config: StrategyConfig,
           ) -> std::result::Result<Box<dyn DynModule>, StrategyError>;
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs`
       ```rust
       pub mod registry {
           pub fn get(name: &str) -> std::result::Result<&'static dyn StrategyFactory, StrategyError>;
           pub fn create(
               name: &str,
               schema: &SignatureSchema,
               config: StrategyConfig,
           ) -> std::result::Result<Box<dyn DynModule>, StrategyError>;
           pub fn list() -> Vec<&'static str>;
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs`
       ```rust
       pub struct StrategyFactoryRegistration {
           pub factory: &'static dyn StrategyFactory,
       }
       inventory::collect!(StrategyFactoryRegistration);
       ```
   - Imports needed:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_module.rs`: `use crate::{BamlValue, PredictError, Predicted, SignatureSchema}; use crate::core::DynPredictor;`
     - Add `inventory = "0.3"` under `/Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml` `[dependencies]` block.
   - Existing code that must change:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs` must add [NEW] `pub mod dyn_module;` and [NEW] `pub use dyn_module::*;`.

4. Implement concrete schema-driven dynamic strategy modules and factories (`predict`, `chain_of_thought`, `react`) and register them.
   - Files to create/modify:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs`
   - Execution order note:
     - Implement Step 5 before this step. `SchemaPredictor` and dynamic factory modules depend on new untyped adapter helpers for prompt/parse parity.
   - Existing signatures (copied):
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:11`
       ```rust
       pub trait DynPredictor: Send + Sync {
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/modules/chain_of_thought.rs:20`
       ```rust
       pub struct ChainOfThought<S: Signature> {
           predictor: Predict<Augmented<S, Reasoning>>,
       }
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/modules/react.rs:51`
       ```rust
       pub struct ReAct<S>
       where
           S: Signature,
           S::Input: BamlType + Clone,
           S::Output: BamlType,
       {
           action: Predict<ReActActionStep>,
           extract: Predict<ReActExtractStep<S::Output>>,
           #[facet(skip, opaque)]
           tools: Vec<Arc<dyn ToolDyn>>,
           #[facet(skip)]
           max_steps: usize,
       }
       ```
   - New signatures:
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs`
       ```rust
       pub struct SchemaPredictor {
           schema: SignatureSchema,
           demos: Vec<Example>,
           instruction_override: Option<String>,
           tools: Vec<std::sync::Arc<dyn rig::tool::ToolDyn>>,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs`
       ```rust
       #[async_trait::async_trait]
       impl DynPredictor for SchemaPredictor { /* full DynPredictor surface */ }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs`
       ```rust
       pub struct PredictDynModule {
           schema: SignatureSchema,
           predictor: SchemaPredictor,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs`
       ```rust
       pub struct ChainOfThoughtDynModule {
           schema: SignatureSchema,
           predictor: SchemaPredictor,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs`
       ```rust
       pub struct ReActDynModule {
           schema: SignatureSchema,
           action: SchemaPredictor,
           extract: SchemaPredictor,
           max_steps: usize,
           tools: Vec<std::sync::Arc<dyn rig::tool::ToolDyn>>,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs`
       ```rust
       pub struct PredictFactory;
       pub struct ChainOfThoughtFactory;
       pub struct ReActFactory;
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_factories.rs`
       ```rust
       inventory::submit! { StrategyFactoryRegistration { factory: &PredictFactory } }
       inventory::submit! { StrategyFactoryRegistration { factory: &ChainOfThoughtFactory } }
       inventory::submit! { StrategyFactoryRegistration { factory: &ReActFactory } }
       ```
   - Imports needed:
     - `use crate::core::{DynModule, DynPredictor, PredictState, StrategyConfig, StrategyConfigSchema, StrategyFactory, StrategyFactoryRegistration};`
     - `use crate::{BamlValue, Chat, ChatAdapter, Example, PredictError, Predicted, SignatureSchema, GLOBAL_SETTINGS};`
   - Existing code that must change:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs` must add [NEW] `pub mod dyn_factories;` and [NEW] `pub use dyn_factories::*;`.
   - Migration debt to record (explicit):
     - [NEW] `ReActFactory` config parsing is JSON-first (`StrategyConfig = serde_json::Value`) and does not yet provide typed tool deserialization; tools remain runtime-provided.

5. Add untyped adapter helpers so dynamic modules execute through the same prompt/parse path as typed modules.
   - Files to modify:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs`
   - Existing signatures (copied):
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs:463`
       ```rust
       pub fn build_system(
           &self,
           schema: &crate::SignatureSchema,
           instruction_override: Option<&str>,
       ) -> Result<String>
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs:563`
       ```rust
       pub fn format_input<I>(
           &self,
           schema: &crate::SignatureSchema,
           input: &I,
       ) -> String
       where
           I: BamlType + for<'a> facet::Facet<'a>,
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs:654`
       ```rust
       pub fn parse_output_with_meta<O>(
           &self,
           schema: &crate::SignatureSchema,
           response: &Message,
       ) -> std::result::Result<(O, IndexMap<String, FieldMeta>), ParseError>
       where
           O: BamlType + for<'a> facet::Facet<'a>,
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs:968`
       ```rust
       fn value_for_path_relaxed<'a>(
           value: &'a BamlValue,
           path: &crate::FieldPath,
       ) -> Option<&'a BamlValue>
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs:998`
       ```rust
       fn insert_baml_at_path(
           root: &mut bamltype::baml_types::BamlMap<String, BamlValue>,
           path: &crate::FieldPath,
           value: BamlValue,
       )
       ```
   - New signatures:
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs`
       ```rust
       pub fn format_input_baml(
           &self,
           schema: &crate::SignatureSchema,
           input: &BamlValue,
       ) -> String
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs`
       ```rust
       pub fn format_output_baml(
           &self,
           schema: &crate::SignatureSchema,
           output: &BamlValue,
       ) -> String
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs`
       ```rust
       pub fn parse_output_baml_with_meta(
           &self,
           schema: &crate::SignatureSchema,
           response: &Message,
       ) -> std::result::Result<(BamlValue, IndexMap<String, FieldMeta>), ParseError>
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs`
       ```rust
       pub fn parse_output_baml(
           &self,
           schema: &crate::SignatureSchema,
           response: &Message,
       ) -> std::result::Result<BamlValue, ParseError>
       ```
   - Imports needed:
     - Existing imports in `chat.rs` already include `BamlValue`, `Message`, `IndexMap`, `FieldMeta`; no new external crate dependency needed.
   - Existing code that must change:
     - `value_for_path_relaxed` and `insert_baml_at_path` become `pub(crate)` helpers or stay private but are called by the new public BAML APIs.

6. Implement `ProgramGraph` mutation/validation/execution and lock projection to immutable snapshot + mutable fit-back.
   - Files to create/modify:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs`
   - Existing signatures (copied):
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/trace/dag.rs:62`
       ```rust
       pub fn new() -> Self
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/trace/dag.rs:66`
       ```rust
       pub fn add_node(
           &mut self,
           node_type: NodeType,
           inputs: Vec<usize>,
           input_data: Option<Example>,
       ) -> usize
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:72`
       ```rust
       pub fn named_parameters<M>(
           module: &mut M,
       ) -> std::result::Result<Vec<(String, &mut dyn DynPredictor)>, NamedParametersError>
       where
           M: for<'a> Facet<'a>,
       ```
   - New signatures:
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       pub struct ProgramGraph {
           nodes: indexmap::IndexMap<String, Node>,
           edges: Vec<Edge>,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       pub struct Node {
           pub schema: SignatureSchema,
           pub module: Box<dyn DynModule>,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       pub struct Edge {
           pub from_node: String,
           pub from_field: String,
           pub to_node: String,
           pub to_field: String,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       pub enum GraphError { /* duplicate node, missing node, missing field, type mismatch, cycle, ambiguous sink, projection mismatch, execution */ }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       impl ProgramGraph {
           pub fn new() -> Self;
           pub fn add_node(&mut self, name: impl Into<String>, node: Node) -> Result<(), GraphError>;
           pub fn remove_node(&mut self, name: &str) -> Result<Node, GraphError>;
           pub fn connect(
               &mut self,
               from: &str,
               from_field: &str,
               to: &str,
               to_field: &str,
           ) -> Result<(), GraphError>;
           pub fn replace_node(&mut self, name: &str, node: Node) -> Result<(), GraphError>;
           pub fn insert_between(
               &mut self,
               from: &str,
               to: &str,
               inserted_name: impl Into<String>,
               inserted_node: Node,
               from_field: &str,
               to_field: &str,
           ) -> Result<(), GraphError>;
           pub async fn execute(&self, input: BamlValue) -> Result<BamlValue, GraphError>;
           pub fn from_module<M>(module: &M) -> Result<Self, GraphError>
           where
               M: for<'a> Facet<'a>;
           pub fn fit<M>(&self, module: &mut M) -> Result<(), GraphError>
           where
               M: for<'a> Facet<'a>;
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       pub struct GraphEdgeAnnotation {
           pub from_node: &'static str,
           pub from_field: &'static str,
           pub to_node: &'static str,
           pub to_field: &'static str,
       }
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       pub fn register_graph_edge_annotations(
           shape: &'static facet::Shape,
           annotations: &'static [GraphEdgeAnnotation],
       )
       ```
     - [NEW] `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/program_graph.rs`
       ```rust
       pub trait TypeIrAssignabilityExt {
           fn is_assignable_to(&self, to: &TypeIR) -> bool;
       }
       ```
   - Imports needed:
     - `use crate::core::{named_parameters, named_parameters_ref, DynModule, DynPredictor, PredictState};`
     - `use crate::{BamlValue, SignatureSchema, TypeIR};`
     - `use indexmap::IndexMap;`
     - `use std::collections::{HashMap, VecDeque};`
   - Existing code that must change:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs` must add [NEW] `pub mod program_graph;` and [NEW] `pub use program_graph::*;`.
     - Concrete lock for this slice:
       - `from_module(&module)` only snapshots predictor state via [NEW] `named_parameters_ref`.
       - `fit(&mut module)` is the only mutable write-back path and applies node predictor state back to typed leaves by path.
       - Edge derivation in `from_module` consumes only [NEW] registered annotations; no trace inference is implemented in V6.
     - Arbitration resolution:
       - Use a global edge-annotation registration table keyed by shape ID in V6 as the single annotation source.
       - Do not mix sources (no concurrent shape-local attr decoding in this slice).
   - Migration debt to record (explicit):
     - [NEW] `TypeIrAssignabilityExt::is_assignable_to` starts conservative (exact match + optional-nullable widening + identical unions). Broader subtyping stays deferred debt until a native method exists on `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs:29` (`TypeIR`).
     - [NEW] Edge annotations are runtime-registered in V6; migrating to shape-local Facet attr storage is deferred to cleanup.

7. Wire crate exports and keep API discoverable from the current top-level re-export path.
   - Files to modify:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs`
   - Existing signatures (copied):
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/mod.rs:12`
       ```rust
       pub use errors::{ConversionError, ErrorClass, JsonishError, LmError, ParseError, PredictError};
       ```
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs:16`
       ```rust
       pub use core::*;
       ```
   - New signatures:
     - [NEW] `pub mod dyn_module;`
     - [NEW] `pub mod dyn_factories;`
     - [NEW] `pub mod program_graph;`
     - [NEW] `pub use dyn_module::*;`
     - [NEW] `pub use dyn_factories::*;`
     - [NEW] `pub use program_graph::*;`
   - Imports needed:
     - None (module wiring only).
   - Existing code that must change:
     - No `lib.rs` edits required because `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs:16` already re-exports all of `core::*`.

8. Add regression and acceptance tests for registry, graph mutation/validation, execution, and snapshot-fit projection.
   - Files to create:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_named_parameters_ref.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_registry_dynamic_modules.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_mutation.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_execution.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_projection_fit.rs` [NEW]
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_annotations.rs` [NEW]
   - Existing signatures (copied) used by tests:
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:72` (`named_parameters`)
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs:58` (`pub fn new() -> Self`)
     - `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/settings.rs:27` (`pub fn configure(lm: LM, adapter: impl Adapter + 'static)`).
   - New signatures:
     - [NEW] test fns listed in Test Plan below.
   - Imports needed:
     - Test files use existing test LM scaffolding types from `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_dyn_predictor_forward_untyped.rs:3-43` (`LM`, `LMClient`, `TestCompletionModel`, `ChatAdapter`, `configure`).
   - Existing code that must change:
     - None outside newly added tests.

### Test Plan
1. [NEW] `named_parameters_ref_discovers_same_paths_as_named_parameters`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_named_parameters_ref.rs`
   - Asserts:
     - [NEW] `named_parameters_ref(&module)` returns the same ordered path list as existing `named_parameters(&mut module)` from `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/dyn_predictor.rs:72`.
     - [NEW] Immutable handles expose `instruction()` and `demos_as_examples()` but cannot mutate.
   - Setup/fixtures:
     - Reuse existing typed fixture pattern from `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_named_parameters.rs:7-37`.
   - Expected behavior:
     - Deterministic path parity and no mutable borrow requirement for projection-time discovery.

2. [NEW] `registry_list_contains_predict_chain_of_thought_react`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_registry_dynamic_modules.rs`
   - Asserts:
     - [NEW] `registry::list()` includes `"predict"`, `"chain_of_thought"`, and `"react"`.
     - [NEW] `registry::create(name, schema, config)` returns `Box<dyn DynModule>` for each built-in strategy.
   - Setup/fixtures:
     - Use existing schema source `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/schema.rs:82` (`SignatureSchema::of::<S>()`) via a local test signature.
   - Expected behavior:
     - Auto-registration works at link time and factories are instantiable by string name.

3. [NEW] `program_graph_connect_rejects_type_mismatch`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_mutation.rs`
   - Asserts:
     - [NEW] `ProgramGraph::connect(...)` returns [NEW] `GraphError::TypeMismatch` when source `TypeIR` is not assignable to target `TypeIR`.
   - Setup/fixtures:
     - Build two [NEW] `Node` values from two local test signatures with incompatible fields via `SignatureSchema::of::<S>()`.
   - Expected behavior:
     - Invalid edges are rejected at insertion time; graph state remains unchanged.

4. [NEW] `program_graph_replace_node_revalidates_incident_edges`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_mutation.rs`
   - Asserts:
     - [NEW] `replace_node` fails when existing incoming/outgoing edges become incompatible.
     - [NEW] On failure, original node and edges remain intact.
   - Setup/fixtures:
     - Start from a valid 2-node graph, then replace one node with incompatible schema.
   - Expected behavior:
     - Revalidation runs on all incident edges before commit.

5. [NEW] `program_graph_insert_between_rewires_edge_and_preserves_validity`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_mutation.rs`
   - Asserts:
     - [NEW] `insert_between(...)` removes the direct `from -> to` edge and inserts two validated edges through the inserted node.
     - [NEW] On validation failure, graph topology remains unchanged.
   - Setup/fixtures:
     - Start with a valid single edge graph, then insert a compatible node and an incompatible node.
   - Expected behavior:
     - F10 mutation affordance `insert_between` behaves atomically.

6. [NEW] `program_graph_execute_routes_fields_topologically`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_execution.rs`
   - Asserts:
     - [NEW] Execution computes topological order and routes edge fields into downstream inputs.
     - [NEW] Final returned `BamlValue` equals designated sink node output.
   - Setup/fixtures:
     - Use deterministic test LM fixture pattern from `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_dyn_predictor_forward_untyped.rs:27-43`.
     - Build 3-node graph (`predict -> chain_of_thought -> predict`) with explicit edges.
   - Expected behavior:
     - Stable order, correct piping, no reliance on insertion order.

7. [NEW] `program_graph_execute_cycle_errors`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_execution.rs`
   - Asserts:
     - [NEW] A cycle in edges returns [NEW] `GraphError::Cycle` before any node forward call.
   - Setup/fixtures:
     - Create 2 nodes and connect both directions.
   - Expected behavior:
     - Deterministic cycle rejection from topological-sort stage.

8. [NEW] `from_module_snapshot_then_fit_roundtrip`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_projection_fit.rs`
   - Asserts:
     - [NEW] `ProgramGraph::from_module(&module)` succeeds without mutable borrow.
     - [NEW] Mutating projected node predictor state does not mutate typed module immediately.
     - [NEW] `graph.fit(&mut module)` applies updated predictor state back to the typed module.
   - Setup/fixtures:
     - Use a typed module fixture like `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/modules/chain_of_thought.rs:20` (`ChainOfThought<S>`).
   - Expected behavior:
     - Lock is enforced: immutable projection + explicit mutable write-back.

9. [NEW] `from_module_uses_annotation_edges_only`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_annotations.rs`
   - Asserts:
     - [NEW] With registered [NEW] `GraphEdgeAnnotation`s, `from_module` creates the exact annotated edges.
     - [NEW] Without registered annotations, `from_module` creates nodes but no inferred edges.
   - Setup/fixtures:
     - Register annotations with [NEW] `register_graph_edge_annotations(...)` for a test module shape.
   - Expected behavior:
     - C8 lock holds: annotation-first only, trace inference deferred.

10. [NEW] `typed_dynamic_prompt_parity_for_predict_and_chain_of_thought`
   - File path: `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_program_graph_execution.rs`
   - Asserts:
     - [NEW] Dynamic `PredictDynModule` system/user prompt text matches typed path output from existing `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs:463` (`build_system`) and `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs:563` (`format_input`) for equivalent schema/input.
     - [NEW] Dynamic `ChainOfThoughtDynModule` prompt text matches typed `ChainOfThought<S>` prompt text for equivalent base signature/input.
   - Setup/fixtures:
     - One signature fixture + canonical `BamlValue` input; construct both predict and chain-of-thought strategy nodes from the same base schema.
   - Expected behavior:
     - R8 parity is preserved for both identity strategy (`predict`) and transformed schema strategy (`chain_of_thought`).
