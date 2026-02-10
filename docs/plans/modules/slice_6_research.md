### Spec Requirements
- U38: Implement `registry::create(name, &schema, config)` to return `Box<dyn DynModule>`.
- U39: Implement `registry::list()` to return registered strategy names.
- U40: Implement `DynModule::predictors()` and `DynModule::predictors_mut()` to expose internal `DynPredictor` handles.
- U41: Implement `ProgramGraph::new()` with empty node/edge stores.
- U42: Implement `ProgramGraph::add_node(name, node) -> Result`.
- U43: Implement `ProgramGraph::connect(from, from_field, to, to_field) -> Result` with edge validation.
- U44: Implement `ProgramGraph::replace_node(name, node) -> Result` with re-validation of affected edges.
- U45: Implement `ProgramGraph::execute(input).await -> Result<BamlValue>`.
- U46: Implement `ProgramGraph::from_module(&module) -> ProgramGraph` reusing the F6 walker.
- N17: Strategy factories must transform `SignatureSchema` (reasoning prepend, action/extract schema shaping, etc.).
- N24: Edge insertion/replacement must validate field type compatibility via `TypeIR::is_assignable_to(&to_type)` semantics.
- N25: Graph execution must compute topological order from `nodes` + `edges`.
- N26: Graph execution must pipe `BamlValue` output fields into downstream input fields by edges.
- N27: Strategy factories must auto-register at link time (inventory-style distributed registration).
- F9: Provide `DynModule` and `StrategyFactory` as the dynamic strategy layer.
- F10: Provide `ProgramGraph`, `Node`, and `Edge` with graph mutation APIs (`add_node`, `remove_node`, `replace_node`, `connect`, `insert_between`) and execution.
- R7: Dynamic graph must be constructable, mutable, validated, and executable.
- R8: Typed modules and dynamic graph nodes must produce identical prompts for the same logical signature.
- R14: Dynamic nodes must be instantiable from a name+schema+config strategy registry.
- Design §10: Registry must expose `get`, `create`, and `list`; factories define `name`, `config_schema`, and `create`.
- Design §11: Program graph nodes hold `(schema, module)`, edges are typed field routes, and execution delegates node internals to `DynModule::forward`.

### Existing Code Inventory
- [Module] `pub mod dyn_predictor;` — `crates/dspy-rs/src/core/mod.rs:2`
- [Module] `pub mod module;` — `crates/dspy-rs/src/core/mod.rs:5`
- [Module] `mod schema;` — `crates/dspy-rs/src/core/mod.rs:7`
- [Module] `pub mod signature;` — `crates/dspy-rs/src/core/mod.rs:9`
- [Module] `pub mod chain_of_thought;` — `crates/dspy-rs/src/modules/mod.rs:1`
- [Module] `pub mod react;` — `crates/dspy-rs/src/modules/mod.rs:2`
- [Module] `pub mod dag;` — `crates/dspy-rs/src/trace/mod.rs:2`
- [Module] `pub mod executor;` — `crates/dspy-rs/src/trace/mod.rs:3`

- [Trait] `crates/dspy-rs/src/core/module.rs:9`
```rust
pub trait Module: Send + Sync {
    type Input: BamlType + for<'a> Facet<'a> + Send + Sync;
    type Output: BamlType + for<'a> Facet<'a> + Send + Sync;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError>;

    async fn call(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        self.forward(input).await
    }
}
```

- [Trait] `crates/dspy-rs/src/core/module.rs:82`
```rust
pub trait Optimizable {
    fn get_signature(&self) -> &dyn MetaSignature {
        todo!()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable>;

    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        todo!()
    }
}
```

- [Trait] `crates/dspy-rs/src/core/dyn_predictor.rs:11`
```rust
pub trait DynPredictor: Send + Sync {
    fn schema(&self) -> &SignatureSchema;
    fn instruction(&self) -> String;
    fn set_instruction(&mut self, instruction: String);
    fn demos_as_examples(&self) -> Vec<Example>;
    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()>;
    fn dump_state(&self) -> PredictState;
    fn load_state(&mut self, state: PredictState) -> Result<()>;
    async fn forward_untyped(
        &self,
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError>;
}
```

- [Type] `crates/dspy-rs/src/core/dyn_predictor.rs:26`
```rust
pub struct PredictState {
    pub demos: Vec<Example>,
    pub instruction_override: Option<String>,
}
```

- [Type] `crates/dspy-rs/src/core/dyn_predictor.rs:33`
```rust
pub struct PredictAccessorFns {
    pub accessor: fn(*mut ()) -> *mut dyn DynPredictor,
}
```

- [Function] `crates/dspy-rs/src/core/dyn_predictor.rs:48`
```rust
pub fn register_predict_accessor(
    shape: &'static Shape,
    accessor: fn(*mut ()) -> *mut dyn DynPredictor,
)
```

- [Type] `crates/dspy-rs/src/core/dyn_predictor.rs:60`
```rust
pub enum NamedParametersError {
    Container { path: String, ty: &'static str },
    MissingAttr { path: String },
}
```

- [Function] `crates/dspy-rs/src/core/dyn_predictor.rs:72`
```rust
pub fn named_parameters<M>(
    module: &mut M,
) -> std::result::Result<Vec<(String, &mut dyn DynPredictor)>, NamedParametersError>
where
    M: for<'a> Facet<'a>,
```

- [Trait] `crates/dspy-rs/src/core/signature.rs:34`
```rust
pub trait Signature: Send + Sync + 'static {
    type Input: BamlType + for<'a> Facet<'a> + Send + Sync;
    type Output: BamlType + for<'a> Facet<'a> + Send + Sync;

    fn instruction() -> &'static str;

    fn schema() -> &'static SignatureSchema
    where
        Self: Sized,
    {
        SignatureSchema::of::<Self>()
    }

    fn input_shape() -> &'static Shape;
    fn output_shape() -> &'static Shape;

    fn input_field_metadata() -> &'static [FieldMetadataSpec];
    fn output_field_metadata() -> &'static [FieldMetadataSpec];

    fn output_format_content() -> &'static OutputFormatContent
    where
        Self: Sized,
    {
        Self::schema().output_format()
    }
}
```

- [Type] `crates/dspy-rs/src/core/schema.rs:52`
```rust
pub struct FieldSchema {
    pub lm_name: &'static str,
    pub rust_name: String,
    pub docs: String,
    pub type_ir: TypeIR,
    pub shape: &'static Shape,
    pub path: FieldPath,
    pub constraints: &'static [ConstraintSpec],
    pub format: Option<&'static str>,
}
```

- [Type] `crates/dspy-rs/src/core/schema.rs:74`
```rust
pub struct SignatureSchema {
    instruction: &'static str,
    input_fields: Box<[FieldSchema]>,
    output_fields: Box<[FieldSchema]>,
    output_format: Arc<OutputFormatContent>,
}
```

- [Function] `crates/dspy-rs/src/core/schema.rs:82`
```rust
pub fn of<S: Signature>() -> &'static Self
```

- [Function] `crates/dspy-rs/src/core/schema.rs:139`
```rust
pub fn input_fields(&self) -> &[FieldSchema]
```

- [Function] `crates/dspy-rs/src/core/schema.rs:143`
```rust
pub fn output_fields(&self) -> &[FieldSchema]
```

- [Function] `crates/dspy-rs/src/core/schema.rs:151`
```rust
pub fn navigate_field<'a>(
    &self,
    path: &FieldPath,
    root: &'a BamlValue,
) -> Option<&'a BamlValue>
```

- [Function] `crates/dspy-rs/src/core/schema.rs:166`
```rust
pub fn field_by_rust<'a>(&'a self, rust_name: &str) -> Option<&'a FieldSchema>
```

- [Type] `crates/dspy-rs/src/predictors/predict.rs:47`
```rust
pub struct Predict<S: Signature> {
    #[facet(skip, opaque)]
    tools: Vec<Arc<dyn ToolDyn>>,
    #[facet(skip, opaque)]
    demos: Vec<Demo<S>>,
    instruction_override: Option<String>,
    #[facet(skip, opaque)]
    _marker: PhantomData<S>,
}
```

- [Function] `crates/dspy-rs/src/predictors/predict.rs:34`
```rust
fn predict_dyn_accessor<S>(value: *mut ()) -> *mut dyn DynPredictor
where
    S: Signature,
```

- [Function] `crates/dspy-rs/src/predictors/predict.rs:58`
```rust
pub fn new() -> Self
```

- [Function] `crates/dspy-rs/src/predictors/predict.rs:472`
```rust
pub async fn forward_untyped(
    &self,
    input: BamlValue,
) -> Result<Predicted<BamlValue>, PredictError>
```

- [Impl] `crates/dspy-rs/src/predictors/predict.rs:494`
```rust
impl<S> DynPredictor for Predict<S>
where
    S: Signature,
    S::Input: BamlType,
    S::Output: BamlType,
```

- [Type] `crates/dspy-rs/src/modules/chain_of_thought.rs:20`
```rust
pub struct ChainOfThought<S: Signature> {
    predictor: Predict<Augmented<S, Reasoning>>,
}
```

- [Function] `crates/dspy-rs/src/modules/chain_of_thought.rs:25`
```rust
pub fn new() -> Self
```

- [Impl] `crates/dspy-rs/src/modules/chain_of_thought.rs:62`
```rust
impl<S> Module for ChainOfThought<S>
where
    S: Signature + Clone,
    S::Input: BamlType,
    S::Output: BamlType,
```

- [Type] `crates/dspy-rs/src/modules/react.rs:51`
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

- [Function] `crates/dspy-rs/src/modules/react.rs:71`
```rust
pub fn new() -> Self
```

- [Impl] `crates/dspy-rs/src/modules/react.rs:243`
```rust
impl<S> Module for ReAct<S>
where
    S: Signature,
    S::Input: BamlType + Clone,
    S::Output: BamlType,
```

- [Function] `crates/dspy-rs/src/modules/react.rs:309`
```rust
pub fn tool<F, Fut>(
    mut self,
    name: impl Into<String>,
    description: impl Into<String>,
    tool_fn: F,
) -> Self
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = String> + Send + 'static,
```

- [Type] `crates/dspy-rs/src/adapter/chat.rs:25`
```rust
pub struct ChatAdapter;
```

- [Function] `crates/dspy-rs/src/adapter/chat.rs:463`
```rust
pub fn build_system(
    &self,
    schema: &crate::SignatureSchema,
    instruction_override: Option<&str>,
) -> Result<String>
```

- [Function] `crates/dspy-rs/src/adapter/chat.rs:563`
```rust
pub fn format_input<I>(
    &self,
    schema: &crate::SignatureSchema,
    input: &I,
) -> String
where
    I: BamlType + for<'a> facet::Facet<'a>,
```

- [Function] `crates/dspy-rs/src/adapter/chat.rs:654`
```rust
pub fn parse_output_with_meta<O>(
    &self,
    schema: &crate::SignatureSchema,
    response: &Message,
) -> std::result::Result<(O, IndexMap<String, FieldMeta>), ParseError>
where
    O: BamlType + for<'a> facet::Facet<'a>,
```

- [Function] `crates/dspy-rs/src/adapter/chat.rs:824`
```rust
pub fn parse_output<O>(
    &self,
    schema: &crate::SignatureSchema,
    response: &Message,
) -> std::result::Result<O, ParseError>
where
    O: BamlType + for<'a> facet::Facet<'a>,
```

- [Function] `crates/dspy-rs/src/adapter/chat.rs:836`
```rust
pub fn parse_sections(content: &str) -> IndexMap<String, String>
```

- [Function] `crates/dspy-rs/src/adapter/chat.rs:968`
```rust
fn value_for_path_relaxed<'a>(
    value: &'a BamlValue,
    path: &crate::FieldPath,
) -> Option<&'a BamlValue>
```

- [Function] `crates/dspy-rs/src/adapter/chat.rs:998`
```rust
fn insert_baml_at_path(
    root: &mut bamltype::baml_types::BamlMap<String, BamlValue>,
    path: &crate::FieldPath,
    value: BamlValue,
)
```

- [Type] `crates/dspy-rs/src/trace/dag.rs:5`
```rust
pub enum NodeType {
    Root,
    Predict { signature_name: String },
    Operator { name: String },
    Map { mapping: Vec<(String, (usize, String))> },
}
```

- [Type] `crates/dspy-rs/src/trace/dag.rs:36`
```rust
pub struct Node {
    pub id: usize,
    pub node_type: NodeType,
    pub inputs: Vec<usize>,
    pub output: Option<Prediction>,
    pub input_data: Option<Example>,
}
```

- [Type] `crates/dspy-rs/src/trace/dag.rs:57`
```rust
pub struct Graph {
    pub nodes: Vec<Node>,
}
```

- [Function] `crates/dspy-rs/src/trace/dag.rs:62`
```rust
pub fn new() -> Self
```

- [Function] `crates/dspy-rs/src/trace/dag.rs:66`
```rust
pub fn add_node(
    &mut self,
    node_type: NodeType,
    inputs: Vec<usize>,
    input_data: Option<Example>,
) -> usize
```

- [Type] `crates/dspy-rs/src/trace/executor.rs:6`
```rust
pub struct Executor {
    pub graph: Graph,
}
```

- [Function] `crates/dspy-rs/src/trace/executor.rs:15`
```rust
pub async fn execute(&self, root_input: Example) -> Result<Vec<Prediction>>
```

- [Type] `crates/bamltype/src/facet_ext.rs:21`
```rust
pub struct WithAdapterFns {
    pub type_ir: fn() -> TypeIR,
    pub register: AdapterRegisterFn,
    pub apply: AdapterApplyFn,
}
```

- [Function] `crates/bamltype/src/facet_ext.rs:41`
```rust
pub fn with_adapter_fns(attrs: &'static [facet::Attr]) -> Option<&'static WithAdapterFns>
```

- [Type Alias] `vendor/baml/crates/baml-types/src/ir_type/mod.rs:127`
```rust
pub type TypeIR = TypeGeneric<type_meta::IR>;
```

- [Function] `vendor/baml/crates/baml-types/src/ir_type/mod.rs:136`
```rust
pub fn diagnostic_repr(&self) -> TypeIRDiagnosticRepr<'_>
```

- [Function] `crates/dspy-rs/src/core/settings.rs:20`
```rust
pub static GLOBAL_SETTINGS: LazyLock<RwLock<Option<Settings>>> =
    LazyLock::new(|| RwLock::new(None));
```

### Gap Analysis
- U38 `registry::create(name, &schema, config)`
  - [EXISTS] `crates/dspy-rs/src/core/schema.rs:74` — `SignatureSchema` exists and is used across typed path.
  - [NEW] — Add `DynModule`, `StrategyFactory`, `StrategyConfig`, `StrategyConfigSchema`, and `registry::create` surface.
- U39 `registry::list()`
  - [NEW] — Add global strategy registry store and list API.
- U40 `dyn_module.predictors()/predictors_mut()`
  - [EXISTS] `crates/dspy-rs/src/core/dyn_predictor.rs:11` — `DynPredictor` trait exists.
  - [NEW] — Add `DynModule` trait exposing predictor handles.
- U41 `ProgramGraph::new()`
  - [EXISTS] `crates/dspy-rs/src/trace/dag.rs:62` — existing graph constructor pattern.
  - [NEW] — Add dedicated `ProgramGraph` type for dynamic modules.
- U42 `graph.add_node(name, node)`
  - [EXISTS] `crates/dspy-rs/src/trace/dag.rs:66` — add-node pattern exists on trace graph.
  - [NEW] — Add named-node insertion for `ProgramGraph` with schema/module node payload.
- U43 `graph.connect(from, from_field, to, to_field)`
  - [NEW] — Add edge model and connect API.
  - [NEW] — Add type-compatibility check routine for field-to-field wiring.
- U44 `graph.replace_node(name, node)`
  - [NEW] — Add node replacement semantics and incident-edge revalidation.
- U45 `graph.execute(input).await -> Result<BamlValue>`
  - [EXISTS] `crates/dspy-rs/src/trace/executor.rs:15` — async graph executor scaffold exists (for trace replay, not typed dynamic modules).
  - [NEW] — Add real dynamic graph execution (node invocation + BamlValue routing + error handling).
- U46 `ProgramGraph::from_module(&module)`
  - [EXISTS] `crates/dspy-rs/src/core/dyn_predictor.rs:72` — walker already yields `(path, &mut dyn DynPredictor)`.
  - [MODIFY] `crates/dspy-rs/src/core/dyn_predictor.rs:72` — current walker requires `&mut`; design example uses `&module` and dynamic projection likely needs non-mutating discovery path or explicit mutable API decision.
  - [NEW] — Add predictor-to-node adapter (`DynPredictor` wrapper implementing `DynModule`) and projection builder.
- N17 schema transformation in factories
  - [EXISTS] `crates/dspy-rs/src/modules/chain_of_thought.rs:21` and `crates/dspy-rs/src/modules/react.rs:57` — typed strategies already encode transformed signatures internally.
  - [MODIFY] `crates/dspy-rs/src/core/schema.rs:74` — add schema transformation helpers (copy/update output/input fields) suitable for factory-time mutation.
- N24 `TypeIR::is_assignable_to(&to_type)` validation
  - [EXISTS] `vendor/baml/crates/baml-types/src/ir_type/mod.rs:127` — `TypeIR` type exists.
  - [NEW] — Add assignability function (method or graph-local helper); no such method exists today.
- N25 topological sort
  - [EXISTS] `crates/dspy-rs/src/trace/executor.rs:16` — current executor assumes topological order but does not compute it.
  - [NEW] — Implement deterministic topological sort + cycle detection for `ProgramGraph`.
- N26 BamlValue piping
  - [EXISTS] `crates/dspy-rs/src/adapter/chat.rs:968` and `crates/dspy-rs/src/adapter/chat.rs:998` — path-based BamlValue read/write utilities already exist (currently private to adapter).
  - [MODIFY] `crates/dspy-rs/src/adapter/chat.rs:968` — extract/share path read/write utility or duplicate in graph module for edge piping.
  - [NEW] — Graph execution-time input assembly by incoming edges.
- N27 inventory auto-registration
  - [NEW] — Add `inventory`-based registration type and collection for `StrategyFactory`.
- R7 dynamic graph construct/validate/mutate/execute
  - [EXISTS] `crates/dspy-rs/src/trace/dag.rs:57` and `crates/dspy-rs/src/trace/executor.rs:15` — graph-shaped scaffolding exists.
  - [NEW] — Implement dynamic module graph domain types and behaviors from F10.
- R8 typed/dynamic prompt parity
  - [EXISTS] `crates/dspy-rs/src/adapter/chat.rs:463` and `crates/dspy-rs/src/adapter/chat.rs:563` and `crates/dspy-rs/src/adapter/chat.rs:654` — schema-based prompt/parse building blocks exist.
  - [NEW] — Ensure all `DynModule` implementations route through these same `ChatAdapter` schema APIs.
- R14 registry instantiation by name + schema + config
  - [NEW] — Registry/factory/config contract is not implemented yet.
- F9 (`DynModule` + `StrategyFactory` + registry)
  - [EXISTS] `crates/dspy-rs/src/core/dyn_predictor.rs:11` — lower-level predictor abstraction is in place.
  - [NEW] — Add full dynamic-module/factory/registry layer.
- F10 (`ProgramGraph` + Node/Edge + mutation + execution)
  - [EXISTS] `crates/dspy-rs/src/trace/dag.rs:57` — graph data structure precedent exists.
  - [NEW] — Add dedicated `ProgramGraph` types/APIs (`remove_node`, `insert_between`, `connect`, `replace_node`, `execute`, projection).

### Patterns & Conventions
- Trait-first architecture with typed core + erased boundary:
  - `crates/dspy-rs/src/core/module.rs:9` (`Module`), `crates/dspy-rs/src/core/dyn_predictor.rs:11` (`DynPredictor`), `crates/dspy-rs/src/predictors/predict.rs:494` (typed `Predict<S>` implements erased trait).
- Global singleton state uses lock + lazy init:
  - `crates/dspy-rs/src/core/schema.rs:83` (`OnceLock<Mutex<HashMap<TypeId, ...>>>` cache), `crates/dspy-rs/src/core/settings.rs:20` (`LazyLock<RwLock<Option<Settings>>>`).
- Deterministic traversal and ordering are tested and expected:
  - `crates/dspy-rs/src/core/dyn_predictor.rs:111` (struct field order walk), `crates/dspy-rs/tests/test_named_parameters.rs:112` (deterministic order test).
- Explicitly unsupported paths are surfaced as typed errors (not silently skipped):
  - `crates/dspy-rs/src/core/dyn_predictor.rs:60` (`NamedParametersError`), `crates/dspy-rs/tests/test_named_parameters_containers.rs:27` (container error contract).
- Prompt/parsing pipeline is centralized in schema-driven adapter building blocks:
  - `crates/dspy-rs/src/adapter/chat.rs:463` (`build_system`), `crates/dspy-rs/src/adapter/chat.rs:563` (`format_input`), `crates/dspy-rs/src/adapter/chat.rs:654` (`parse_output_with_meta`).
- Function-pointer payload pattern via Facet attrs already exists and is runtime-decoded:
  - `crates/bamltype/src/facet_ext.rs:21` (`WithAdapterFns`), `crates/bamltype/src/facet_ext.rs:41` (`with_adapter_fns`).
- Builder APIs are the established module-construction style:
  - `crates/dspy-rs/src/predictors/predict.rs:246` (`PredictBuilder<S>`), `crates/dspy-rs/src/modules/react.rs:257` (`ReActBuilder<S>`).
- Tracing instrumentation (`#[tracing::instrument]`) is consistently used on core execution boundaries:
  - `crates/dspy-rs/src/core/dyn_predictor.rs:67`, `crates/dspy-rs/src/adapter/chat.rs:447`, `crates/dspy-rs/src/predictors/predict.rs:466`.

### Spec Ambiguities
- ~~`ProgramGraph::from_module(&module)` vs current walker mutability (`named_parameters(&mut module)`).~~
  - **Resolved:** snapshot-then-fit-back. `from_module(&module)` uses an immutable walker variant (`named_parameters_ref`) to read predictor schemas and state, creates independent owned `DynModule` graph nodes. Graph mutates freely during optimization. `graph.fit(&mut module)` writes optimized state back via mutable walker + `load_state`. Structural divergences surfaced explicitly. See tracker decision entry.
- F10 includes `remove_node` and `insert_between`, but breadboard U41-U46 does not enumerate them.
  - Proposed resolution: treat `remove_node` and `insert_between` as in-scope for slice completeness (F10 source-of-truth), but phase delivery after `new/add/connect/replace/execute` if time-boxing is needed.
- Edge derivation in `from_module` is underspecified (“trace or explicit annotation”).
  - Proposed resolution: follow tracker C8 lock: annotation-first deterministic edges in V6; trace inference explicitly deferred.
- `TypeIR::is_assignable_to` is named in spec but absent in codebase.
  - Proposed resolution: define a graph-local compatibility function first (exact/optional/union-safe rules), then optionally upstream to `TypeIR` extension trait.
- Strategy config model is unspecified (`StrategyConfig`, `StrategyConfigSchema` structure not defined).
  - Proposed resolution: use `serde_json::Value` plus JSON-schema metadata for v1; keep factory-specific typed decoding behind each factory.
- Graph execution output contract is underspecified for multi-sink graphs (`Result<BamlValue>` singular output).
  - Proposed resolution: require one designated terminal node in v1 (explicit graph output node), error on ambiguous sinks.
- Demo uses `connect("input", ...)` without defining an input node lifecycle.
  - Proposed resolution: model input as explicit virtual root node created by `execute` (not user-added), reserved name `__input` to avoid user collisions.

### Recommended Approach
1. Add F9 core surfaces first: `DynModule`, `StrategyFactory`, config/schema types, registry API (`get/create/list`), and error types.
2. Implement first two concrete factories (`chain_of_thought`, `react`) that wrap existing typed modules and expose predictor handles.
3. Build F10 graph data types (`ProgramGraph`, `Node`, `Edge`) and mutation APIs (`new/add/connect/replace/remove/insert_between`) with deterministic validation errors.
4. Implement N24 type compatibility helper and wire it into `connect` and `replace_node` edge checks.
5. Implement N25 topological sort + cycle errors; then N26 BamlValue edge routing and node input assembly.
6. Implement `execute` by invoking `DynModule::forward` in topo order using shared adapter formatting/parsing paths to preserve R8.
7. Implement `from_module` projection using F6 walker + adapter wrapper around discovered predictors; apply annotation-first edge derivation policy.
8. Add registration plumbing (`inventory` submit/collect), then test matrix: registry operations, graph validation/mutation, cycle handling, parity golden tests (typed vs dynamic prompts), and end-to-end V6 smoke.
