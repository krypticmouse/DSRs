# Slice 3 Research: Module authoring

## Required outcomes (V3 focus)
1. Enable P2 developers to write new modules by wiring generic signatures + adapter helpers instead of reimplementing prompt mechanics, mirroring the breadboard V3 story (`docs/specs/modules/breadboard.md:318-374`).
2. Ship F4 module trait semantics with `Module::forward(input)` returning `CallOutcome<Output>` while exposing typed Input/Output associated types so swapping strategies is a compile-time substitution (`docs/specs/modules/shapes.md:60-85`, `docs/specs/modules/design_reference.md:359-388`).
3. Support F12 generic `#[derive(Signature)]` + `#[flatten]` so modules that orchestrate other strategies can reuse reusable signatures, per the S1 spike resolution (Option C: full Facet-derived schema replacement, `docs/specs/modules/design_reference.md:167-240`, `docs/specs/modules/spikes/S1-generic-signature-derive.md`).
4. Provide F7 adapter building blocks (`build_system`, `format_input`, `format_output`, `parse_sections`, `parse_output`) on `SignatureSchema` so advanced modules (ReAct, BestOfN, custom paths) can compose prompts/parse results without reimplementing parsing (`docs/specs/modules/design_reference.md:576-672`).

## Current code baseline
### Module trait (outdated shape)
- `crates/dspy-rs/src/core/module.rs:1-108` defines `pub trait Module: Send + Sync { async fn forward(&self, inputs: Example) -> CallOutcome<Prediction>; ... }`. The trait operates on the legacy `Example/Prediction` pair and contains `forward_untyped` + `batch` helpers rather than tying input/output to the trait itself.
- `CallOutcome` already exists (`crates/dspy-rs/src/core/call_outcome.rs:1-210`) and encapsulates metadata, but `Module` never surfaces the typed `CallOutcome<O>` the spec expects.

### Signature infrastructure
- `crates/dspy-rs/src/core/signature.rs:1-90`: `Signature` trait still exposes `input_shape`, `output_shape`, `input_field_metadata`, and static `FieldSpec` arrays. `MetaSignature` is used by legacy adapters, and the trait is heavy on manual metadata rather than Facet reflection.
- `crates/dspy-rs/src/core/schema.rs:1-199` already has `SignatureSchema` + `FieldPath`, but the builder still depends on the old metadata (collect_fields pulls from `S::input_field_metadata()`/`FieldMetadataSpec`). The schema cache is once-and-for-all, but `FieldPath` usage is limited to a handful of typed helpers.
- `crates/dsrs-macros/src/lib.rs` (per `docs/specs/modules/spikes/S1...`) does not forward generics or `#[flatten]` to the generated helper types, so the derive cannot produce the `BamlType`/`Facet` metadata the spec needs.

### ChatAdapter / adapter helpers
- `crates/dspy-rs/src/adapter/chat.rs:400-930` provides typed helpers such as `format_system_message_typed`, `format_user_message_typed`, `format_assistant_message_typed`, and `parse_response_typed`. They already use `SignatureSchema::of::<S>()` and `FieldPath` traversal, but the helper boundary still targets `Signature` (via `S::schema()`) rather than the canonical builder functions described for F7. `parse_sections` and `insert_baml_at_path` exist but are private utilities.
- The `Adapter` trait implementation in the same file still depends on `MetaSignature`/legacy `format`/`parse` APIs, so module authors wanting fine control cannot call the lower-level pieces directly.

## Required types/signatures (per spec)
- **Module trait**: `#[async_trait] pub trait Module { type Input: BamlType + Facet + Send + Sync; type Output: BamlType + Facet + Send + Sync; async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output>; }` with `CallOutcome` carrying metadata (`docs/specs/modules/design_reference.md:359-388`). This makes any `Module` composable in typed pipelines.
- **Signature / SignatureSchema**: The derive should only require `Signature::Input`, `::Output`, and `fn instructions()`. `SignatureSchema::of::<S>()` must be entirely Facet-derived and track `FieldPath`s so adapter builders can format/parse nested/flattened fields (`docs/specs/modules/design_reference.md:167-240`, `docs/specs/modules/design_reference.md:576-672`). `SignatureSchema` already exists (the builder is in `crates/dspy-rs/src/core/schema.rs:1-199`), but the foundational metadata needs to drop `FieldMetadataSpec` reliance and instead reflect on `Facet` shapes.
- **Adapter building blocks**: Public API around `ChatAdapter` should include `build_system(schema, override)`, `format_input(schema, &input)`, `format_output(schema, &output)`, `parse_sections(content)`, and `parse_output::<O>(schema, &response)` so V3 authors can assemble prompts/outputs without reusing `MetaSignature` (`docs/specs/modules/design_reference.md:576-672`). Helper internals already exist (`format_user_message_typed`, `parse_response_typed`, `parse_sections`, `insert_baml_at_path`), but they must be re-exposed in the new surface.
- **Generic Signature derive / flatten support**: Spread generics across generated `Input`/`Output` helper types and carry `#[facet(flatten)]` metadata through to runtime so flattened field paths are available (`docs/specs/modules/shapes.md:60-90`, `docs/specs/modules/spikes/S1-generic-signature-derive.md`).
- **CallOutcome**: Already defined in `crates/dspy-rs/src/core/call_outcome.rs:1-210`, it can be reused directly for `Module::forward` once the trait shifts to typed inputs/outputs.

## Gaps between spec and repo
1. **Module trait mismatch**: Spec wants typed Input/Output associated types and `CallOutcome<Output>`, but current trait works with `Example`/`Prediction` and exposes `forward_untyped`/`batch`. No typed composition surface exists (`crates/dspy-rs/src/core/module.rs:1-108`).
2. **Signature derive limitations**: The derive macros and runtime metadata still emit static `FieldSpec`/`FieldMetadataSpec`, not Facet-driven `SignatureSchema`. The macros do not thread generics or recognize `#[flatten]`, so generic signature authors cannot build modules yet (`docs/specs/modules/spikes/S1-generic-signature-derive.md`).
3. **Adapter building block access**: The typed `ChatAdapter` helpers live behind `format_*_typed` + `parse_response_typed`, which are high-level and tied to `Signature`. There is no public `build_system(schema, ...)` / `format_input(schema, &input)` surface for module authors to reuse, nor is `parse_output` exported as a schema-aware function (`crates/dspy-rs/src/adapter/chat.rs:400-930`).
4. **MetaSignature/legacy path still present**: The `Adapter` trait implementation uses `MetaSignature` (legacy) and retains `format_system_message`, `parse_response_strict`, etc., so migrating module authoring to the typed path requires carefully removing or aliasing the legacy surface.
5. **Flatten-aware runtime metadata**: While `SignatureSchema` stores `FieldPath`, the builder still depends on manual metadata arrays rather than computing them from `Facet`, so flattened signatures cannot be derived without manual `FieldMetadataSpec` hacks (`crates/dspy-rs/src/core/schema.rs:1-199` and `crates/dspy-rs/src/core/signature.rs:1-90`).

## Practical implementation approach for Slice 3
1. **Finalize Signature derive + schema plumbing (S1 Option C).**
   - Thread generics and bounds through the macro so `Foo<T>` produces `FooInput<T>`/`FooOutput<T>` with the same constraints and the derive emits `#[facet(flatten)]` or equivalent for `#[flatten]` fields. Update `SignatureSchema::build` to reflect on `Facet` shapes (`crates/dspy-rs/src/core/schema.rs`) so adapters/metadata use the new shape-based field lists instead of `FieldMetadataSpec`. Document this as the key enabler for module authoring; when macros can handle generics + flatten, modules can chain arbitrary signatures.
2. **Rewrite `Module` trait to the spec shape.**
   - Replace the legacy `Module` trait in `crates/dspy-rs/src/core/module.rs` with the async trait that binds `type Input`/`type Output` and returns `CallOutcome<Self::Output>`. Keep or move `batch` helpers elsewhere (e.g., free function `dsrs::forward_all`) so the trait stays minimal. Ensure existing modules such as `Predict`, `ChainOfThought`, `ReAct`, and future composites implement the new trait.
3. **Surface adapter building blocks**.
   - Refactor `crates/dspy-rs/src/adapter/chat.rs` to expose the schema-aware helpers the spec calls out: `build_system(schema, instruction_override)`, `format_input(schema, &input)`, `format_output(schema, &output)`, `parse_sections(content)`, and `parse_output::<O>(schema, &response)`. Internally reuse the existing implementations of these behaviors but decouple them from `Signature`. This surfacing lets module authors call the same primitives used by `Predict` (F7). Keep the legacy `Adapter` trait implementation intact during migration but route its implementations through the new helpers.
4. **Lock the typed path to `CallOutcome`.**
   - Update `Predict`, `ChainOfThought`, and other modules to use the new module trait return value and to forward metadata via `CallOutcome`. Ensure the typed path still populates `CallMetadata` (raw response, field meta, tool usage) so module authors / optimizers can inspect it.
5. **Test the authoring path end-to-end.**
   - Add a smoke module (e.g., `SimpleRAG`) in tests that composes two modules with generic signatures, uses adapter builder functions directly, and asserts the `CallOutcome` metadata flows correctly. Include new tests for generic + flattened signature derive behavior (per S1) plus `ChatAdapter` parse/format coverage for flattened fields.

### Next steps
1. Coordinate Slice 3 with Slice 2 deliverables already in flight (augmentations + ChainOfThought) so the module authoring surface can reuse those components once the trait and signature derive stabilize.
2. Once Slice 3 implements the spec surface, remove `MetaSignature`/legacy adapters via the plan laid out in Slice 1/2 documents to avoid dual metadata systems.
3. Document the new module authoring workflow (module trait + adapter helpers) in the `docs/` tree so future contributors know how to compose modules without rewriting macros.
