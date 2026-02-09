# Slice 3 Plan: Module authoring

## Summary
1. Replace the legacy `Module` surface with the V3/F4/F12 shape (`CallOutcome` + strongly typed `Input/Output`).
2. Finish the generic/flatten signature derive path so `SignatureSchema::of::<S>` can be built from Facet metadata and expose the helper parsing/formatting primitives that adapter-heavy modules re-use.
3. Ship the schema-aware adapter building blocks and end-to-end tests so future F5/F6 modules can compose adapters without reimplementing low-level prompt mechanics.
4. Validate that the P1→P2 ramp (e.g., `Map<M, F>` / `.and_then()`) stays Facet-transparent for the optimizer walker (breadboard `N18`, `U51`) and that each `Predict` leaf still publishes the `dsrs::parameter` accessor payload required by the optimizer bridge (design reference F5/F6/S2).

## Constraints
- Keep the work strictly within V3 (module authoring) targeted at the breadboard V3 story; do not add new augmentation, optimizer, or non-V3 features.
- Honor tracker decisions: the typed `forward` return must stay `CallOutcome<Self::Output>` and metadata must carry raw response and parsing context in the same call. Any new APIs must not regress `CallOutcome` ergonomics (e.g., `Try`/`into_result`).
- The new schema path must settle on the Facet-driven `SignatureSchema` builder; legacy `FieldMetadataSpec` helpers are to be retired from the public surface.
- Explicitly mention the `Facet::reflect` based metadata generation so there is no lingering parallel schema surface versus the spec-level requirement that all metadata derive from Facet (shapes F1/F2/F12).

## Key type signatures & imports
1. **Module trait (`crates/dspy-rs/src/core/module.rs`)**
   ```rust
   use async_trait::async_trait;
   use crate::core::call_outcome::CallOutcome;
   use facet::{BamlType, Facet};

   #[async_trait]
   pub trait Module: Send + Sync + 'static {
       type Input: BamlType + Facet + Send + Sync + 'static;
       type Output: BamlType + Facet + Send + Sync + 'static;

       async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output>;
   }
   ```
2. **Signature derive helpers (`crates/dsrs-macros/src/lib.rs`)**
   - Imports: `proc_macro::{TokenStream, TokenTree}`, `syn::{DeriveInput, Data}`, `quote::quote`, `facet::Facet`.
   - Function signature:
     ```rust
     pub fn derive_signature(input: TokenStream) -> TokenStream;
     ```
   - Must emit `struct FooInput<T>`/`FooOutput<T>` where `T` carries the same bounds as `Foo<T>`, and the generated `Facet` impl reflects `#[flatten]` fields with their flattened `FieldPath` (LM-visible names preserved).
3. **Signature schema builder (`crates/dspy-rs/src/core/schema.rs`)**
   ```rust
   pub fn schema_from_signature<S: Signature>() -> SignatureSchema;
   impl SignatureSchema {
       pub fn field_paths(&self) -> &[FieldPath];
       pub fn build_system_message(&self, override: Option<&str>) -> ChatMessage;
       pub fn format_input<S: Signature>(&self, input: &S::Input) -> Vec<ChatMessage>;
       pub fn format_output<S: Signature>(&self, output: &S::Output) -> Vec<ChatMessage>;
   }
   ```
   - Imports: `crate::core::signature::Signature`, `facet::{Facet, BamlValue, FieldPath}`, `adapter::chat::ChatMessage`.
4. **Adapter helpers exposed (`crates/dspy-rs/src/adapter/chat.rs`)**
   ```rust
   pub fn build_system(schema: &SignatureSchema, instruction_override: Option<&str>) -> ChatMessage;
   pub fn format_input<S: Signature>(schema: &SignatureSchema, input: &S::Input) -> Vec<ChatMessage>;
   pub fn format_output<O: Facet>(schema: &SignatureSchema, output: &O) -> Vec<ChatMessage>;
   pub fn parse_sections(content: &str) -> Vec<ChatSection>;
   pub fn parse_output<O: Facet>(schema: &SignatureSchema, response: &str) -> Result<O, ParseError>;
   ```
   - Imports: `crate::core::schema::SignatureSchema`, `crate::core::signature::Signature`, `facet::{Facet, BamlValue}`, `crate::adapter::chat::ChatMessage` etc.
5. **SignatureSchema cache** (`crates/dspy-rs/src/core/schema.rs`)
   ```rust
   pub fn schema_cache<S: Signature>() -> &'static SignatureSchema;
   ```
   - Guards are initialized via once-cell/`lazy_static` from Facet metadata.

## Ordered edit steps
1. **Macro & schema metadata plumbing**
   - Update `crates/dsrs-macros/src/lib.rs` to thread generics/nested bounds through generated `FooInput<FooBounds>` and `FooOutput<FooBounds>` helper types, emitting `#[facet(flatten(name = "foo.bar"))]` metadata for flattened fields.
   - Replace any use of `FieldMetadataSpec` arrays with Facet introspection – the derive must now call into helper functions such as `facet::reflect::<T>()` to build `FieldPath` lists. Add tests in the macro crate verifying `#[flatten]` yields multiple `FieldPath`s with unique LM-visible names.
   - Add a regression test proving wrappers like `Map<M, F>` / `.and_then()` expose their inner `Module` fields via Facet so the optimizer walker described in `N18` and `U51` still discovers Predict leaves. Walk the derived schema from a wrapped module and assert the flattened `FieldPath` to the inner predictor exists.
 2. **SignatureSchema builder rewrite**
    - Modify `crates/dspy-rs/src/core/schema.rs` so `SignatureSchema::build<S: Signature>()` walks the Facet tree (`facet::Facet::schema()`) and records each leaf's `FieldPath` (flattened names, type info). The builder should drop references to `FieldMetadataSpec` entirely.
    - Ensure `SignatureSchema::of::<S>()` caches the schema via `once_cell::sync::Lazy` and exposes helper methods referenced in later steps.
    - Move the `TypeId`-keyed cache initialization ahead of step 3 so `schema::of::<S>()` is statically memoized before any module trait changes rely on it; include an idiomatic helper that guarantees each monomorphized signature has its own entry (S1 failure mode alert).
3. **Module trait migration**
    - Replace `crate::core::module.rs` contents with the V3 trait above, keeping `batch`/helper functions as free functions (e.g., `pub async fn batch_forward<M: Module>(_...)`). Update imports to include `CallOutcome`, `facet` traits, and `async_trait`.
    - Update any file using `Module` (predictor modules, aggregator modules) to adopt the new type signature; plan point includes list of affected files such as `crates/dspy-rs/src/module/predict.rs`, `chains/chain_of_thought.rs`, etc., with exact type replacements.
    - Confirm `Predict` still carries its `dsrs::parameter` Facet attribute and accessor payload so the optimizer walker (F6) can rehydrate a `DynPredictor`. Document this in the plan to remind implementers not to drop the attribute when refactoring `Predict` in this slice (design reference section 8, spike S2).
4. **Expose adapter building blocks**
   - In `crates/dspy-rs/src/adapter/chat.rs` declare the public helper functions listed above; each should delegate to the existing typed helpers but accept a `SignatureSchema` rather than `MetaSignature`.
   - Update the `Adapter` impl to route through these new helpers so backward compatibility is preserved while enabling new modules to call them directly. Mention to keep existing helper names for now but mark them `pub(crate)`.
5. **Module implementations and composer migration**
   - Touch each `Module` implementor (`predict.rs`, `re_act.rs`, any aggregator) to accept typed inputs/outputs and return `CallOutcome`. Provide typically `type Input = PredictInput; type Output = PredictOutput;` etc. Document new `impl Module for Foo` snippet with associated type names.
   - Confirm modules still call `ChatAdapter` helpers to format/parse via the new schema functions (they now call `build_system(schema, ...)` etc.).
6. **Testing & documentation**
   - Add unit tests in `crates/dspy-rs/tests/module_authoring.rs` covering:
     * Generic/flatten `Signature` derive round-trip: instantiate a test signature with `#[flatten]`, assert `SignatureSchema::of::<TestSignature>().field_paths()` contains expected `FieldPath`s, and assert `build_system` includes flattened names.
     * Module chain: construct stub modules implementing the new trait, wire them through `CallOutcome`, invoke `ForwardPipeline::call` (or similar), and assert `outcome.value.answer == "expected"` plus `outcome.metadata.raw_response.is_some()` and `outcome.metadata.field_paths.contains("flattened.field")`.
     * Adapter helpers: feed a `SignatureSchema` and deterministic `response` string into `parse_output::<TestSignature>` and assert parsed struct equals the expected facet-derived data with `assert_eq!(parsed.answer, "ok");
     * Schema cache uniqueness: add `#[test] fn schema_cache_is_per_monomorphization()` asserting pointers of `SignatureSchema::of::<FlatSignature<i32>>()` and `SignatureSchema::of::<FlatSignature<String>>()` differ and at least one field path contains "inner".


## Migration sequencing
1. Nucleus (macro + schema) – foundational: ensures `SignatureSchema` can represent flattened generics before Module trait changes go live.
2. Module trait – once schema is stable, switch the trait to typed `Input/Output` to avoid mixing old/ new signatures mid-migration.
3. Adapter helpers – expose the new schema-aware API only after the trait relies on it so modules can swap implementations without breaking compatibility.
4. Module implementations – update each consumer after the helper surface is stable so downstream code compiles immediately.
5. Tests/doc – finish with smoke tests documenting the new workflow and lock behavior via assertions listed above.

## Tests (assertions + coverage)
1. `crates/dspy-rs/tests/test_signature_schema.rs`
   - `#[tokio::test] fn signature_schema_reflects_flattened_paths()`
     * `let schema = SignatureSchema::of::<FlatSignature>();`
     * `assert_eq!(schema.field_paths().iter().map(|p| p.lm_name()).collect::<Vec<_>>(), vec!["question", "context", "context.detail"]);`
2. `crates/dspy-rs/tests/module_authoring.rs`
   - `#[tokio::test] async fn call_outcome_round_trips()`
     * Construct `SimpleModule` returning `CallOutcome::with_value(SimpleOutput { answer: "ok" }, metadata)`.
     * Compose stub adapter that parses known string.
     * `assert_eq!(outcome.value.answer, "ok");`
     * `assert!(outcome.metadata.raw_response.contains("ok"));`
     * `assert!(outcome.metadata.field_paths.iter().any(|path| path.lm_name() == "instructions"));`
3. `crates/dspy-rs/tests/chat_adapter_schema.rs`
   - `#[test] fn adapter_helpers_round_trip()`
     * Feed a deterministic chat response referencing flattened fields.
     * `let parsed = parse_output::<FlatSignature>(&schema, "answer: ok\ncontext.detail: meta").unwrap();`
     * `assert_eq!(parsed.answer, "ok");`
     * `assert_eq!(parsed.context_detail, "meta");`
4. `crates/dspy-rs/tests/signature_derive.rs`
   - `#[test] fn derive_thread_generics()`
     * Derive `Signature` for `struct Context<T: Serialize>`.
     * `assert!(SchemaCache::of::<Context<i32>>().is_unique());`
     * `assert!(schema.field_paths().iter().any(|path| path.lm_name().starts_with("context")));`

## Next steps / validation
- After implementation, run `cargo test -p dspy-rs --lib --tests` plus `cargo test -p dsrs-macros --lib` to ensure new derive macros and module surfaces compile.
- Document the new workflow under `docs/specs/modules/module_authoring.md` (if not already present) referencing the new helper functions and trait signatures.
