# S1 Spike: Generic `#[derive(Signature)]` with `#[flatten]`

## Context

S1 is explicitly called out as a high-priority spike in shaping/design docs: generic `Signature` derive with `#[flatten]` is required for F12 and module authoring (`docs/specs/modules/shapes.md:140`, `docs/specs/modules/design_reference.md:117`, `docs/specs/modules/design_reference.md:1006`).

Current implementation spans macro codegen in `crates/dsrs-macros/src/lib.rs` and typed runtime consumption in `crates/dspy-rs/src/core/signature.rs` + `crates/dspy-rs/src/adapter/chat.rs`.

## Goal

Determine whether the current macro/runtime architecture can support generic signatures with flattened fields, and identify a concrete implementation path that satisfies S1 without introducing inconsistent metadata behavior.

## Questions

| ID | Question |
|---|---|
| **S1-Q1** | Does the current `Signature` derive thread generics and where-bounds into generated helper types + impls? |
| **S1-Q2** | Can the current derive accept and represent `#[flatten]` fields? |
| **S1-Q3** | Do runtime signature metadata and adapter parsing/formatting support flattened field paths? |
| **S1-Q4** | What does Facet already provide for flatten + shape metadata (NIA-backed evidence)? |
| **S1-Q5** | What implementation path best satisfies S1 while staying aligned with F2 direction? |

## Findings

1. **S1-Q1: current derive does not forward generics into generated types/impls.**
   - Evidence: helper identifiers are monomorphic (`FooInput`, `__FooOutput`, `__FooAll`) and emitted as non-generic structs (`crates/dsrs-macros/src/lib.rs:397`, `crates/dsrs-macros/src/lib.rs:398`, `crates/dsrs-macros/src/lib.rs:399`, `crates/dsrs-macros/src/lib.rs:408`, `crates/dsrs-macros/src/lib.rs:413`, `crates/dsrs-macros/src/lib.rs:418`).
   - Evidence: `impl BamlType for #name` and `impl Signature for #name` are emitted without `impl_generics/ty_generics/where_clause` forwarding (`crates/dsrs-macros/src/lib.rs:594`, `crates/dsrs-macros/src/lib.rs:652`).
   - Evidence: there is no generics plumbing (`split_for_impl`, `impl_generics`, `ty_generics`, `where_clause`) in this file.
   - Conclusion: generic signatures cannot be derived correctly with the current codegen shape.

2. **S1-Q2: `#[flatten]` is not accepted or represented by `#[derive(Signature)]`.**
   - Evidence: derive helper attribute whitelist excludes `flatten` (`crates/dsrs-macros/src/lib.rs:22`).
   - Evidence: field parsing only branches on `input`, `output`, `alias`, `format`, `check`, `assert` (`crates/dsrs-macros/src/lib.rs:198`, `crates/dsrs-macros/src/lib.rs:199`, `crates/dsrs-macros/src/lib.rs:204`, `crates/dsrs-macros/src/lib.rs:209`, `crates/dsrs-macros/src/lib.rs:211`, `crates/dsrs-macros/src/lib.rs:219`, `crates/dsrs-macros/src/lib.rs:221`).
   - Evidence: field codegen emits docs + `pub field: Ty` only; no flatten marker reaches generated helper structs (`crates/dsrs-macros/src/lib.rs:424`, `crates/dsrs-macros/src/lib.rs:439`, `crates/dsrs-macros/src/lib.rs:440`).
   - Conclusion: flattened signatures are blocked at macro boundary today.

3. **S1-Q3: typed runtime metadata/parsing is top-level-key based, not path-based.**
   - Evidence: `FieldSpec` contains `name/rust_name/description/type_ir/constraints/format` and no path field (`crates/dspy-rs/src/core/signature.rs:6`, `crates/dspy-rs/src/core/signature.rs:8`, `crates/dspy-rs/src/core/signature.rs:10`, `crates/dspy-rs/src/core/signature.rs:12`).
   - Evidence: typed formatting does `fields.get(field_spec.rust_name)` for input and output (`crates/dspy-rs/src/adapter/chat.rs:530`, `crates/dspy-rs/src/adapter/chat.rs:531`, `crates/dspy-rs/src/adapter/chat.rs:555`, `crates/dspy-rs/src/adapter/chat.rs:556`).
   - Evidence: `baml_value_fields` exposes only top-level `Class/Map` maps (`crates/dspy-rs/src/adapter/chat.rs:861`, `crates/dspy-rs/src/adapter/chat.rs:865`, `crates/dspy-rs/src/adapter/chat.rs:866`).
   - Evidence: parsing writes a flat `output_map` keyed by `rust_name` and then constructs `S::Output` from that map (`crates/dspy-rs/src/adapter/chat.rs:601`, `crates/dspy-rs/src/adapter/chat.rs:607`, `crates/dspy-rs/src/adapter/chat.rs:720`, `crates/dspy-rs/src/adapter/chat.rs:739`).
   - Conclusion: adding flatten only in macro codegen will still fail typed adapter roundtrip behavior.

4. **S1-Q4: local runtime already uses Facet-driven primitives compatible with flatten-aware direction.**
   - Evidence: `baml_type_ir::<T>()` delegates to shape-based type derivation (`crates/bamltype/src/runtime.rs:96`, `crates/bamltype/src/runtime.rs:97`).
   - Evidence: schema/type building entrypoints are shape-based (`crates/bamltype/src/schema_builder.rs:17`, `crates/bamltype/src/schema_builder.rs:30`).
   - Evidence: runtime serialization iterates reflect traversal (`fields_for_serialize`) (`crates/bamltype/src/convert.rs:611`).
   - Evidence: `serde(flatten)` is explicitly rejected in derive compatibility layer, enforcing Facet-native flatten instead (`crates/bamltype-derive/src/lib.rs:970`, `crates/bamltype-derive/src/lib.rs:973`, `crates/bamltype-derive/src/lib.rs:1003`, `crates/bamltype-derive/src/lib.rs:1006`).
   - NIA evidence: external Facet docs/runtime citations for `#[facet(flatten)]` and flatten-aware reflect traversal are already captured in sibling spikes (`docs/specs/modules/spikes/S3-augmentation-deref-composition.md:47`, `docs/specs/modules/spikes/S3-augmentation-deref-composition.md:50`, `docs/specs/modules/spikes/S3-augmentation-deref-composition.md:51`).

5. **S1-Q5: current gap is dual-surface (macro + typed runtime), not macro-only.**
   - Evidence: design decision D2 already calls static `FieldSpec` arrays the legacy model for flatten/generic scenarios (`docs/specs/modules/design_reference.md:991`).
   - Evidence: S1 remains marked High-priority and blocking F12/module authoring (`docs/specs/modules/design_reference.md:1006`).
   - Conclusion: a viable S1 plan must include generic codegen plus typed path metadata/access changes in the same delivery track.

6. **Current tests do not exercise S1 behavior end-to-end.**
   - Evidence: positive derive tests cover non-generic, non-flatten signatures only (`crates/dsrs-macros/tests/signature_derive.rs:4`, `crates/dsrs-macros/tests/signature_derive.rs:16`).
   - Evidence: trybuild UI harness covers validation failures only (`crates/dsrs-macros/tests/ui.rs:3`, `crates/dsrs-macros/tests/ui`).
   - Validation run (2026-02-09): `cargo test -p dsrs_macros --tests` passes, confirming no existing regression signal for S1.

## Options And Tradeoffs

### Option A: Macro-only generic threading
- **What**: forward generics in generated helper structs and impl blocks only.
- **Pros**: smallest code change.
- **Cons**: does not satisfy flatten path behavior in typed adapter; fails S1-Q3.

### Option B: Generic threading + path-aware runtime metadata (incremental)
- **What**: ship generic forwarding and flatten parsing/codegen, plus typed-path shape/path metadata and adapter path traversal. Keep legacy `FieldSpec`/`MetaSignature` surface for compatibility while typed path migrates.
- **Pros**: satisfies S1 with bounded blast radius; aligns with F2 direction without forcing full-system cutover.
- **Cons**: temporary dual metadata paths increase short-term maintenance complexity.

### Option C: Full F2 move now (Facet-derived `SignatureSchema`)
- **What**: complete replacement of static field arrays with full Facet-derived `SignatureSchema` across typed and dynamic paths.
- **Pros**: cleanest end-state; removes bridge complexity.
- **Cons**: largest immediate surface area and integration risk for a single spike exit.

## Decision

**Option C: full F2 replacement.** Build `SignatureSchema` from Facet, replace `FieldSpec` everywhere, delete the old system. No incremental migration, no compatibility shims.

Reasoning: Option A cannot satisfy S1-Q3. Option B was the spike's original recommendation (smallest correct path), but it contradicts the design principle "no parallel schema systems" and creates temporary dual metadata surfaces. Option C matches the architecture: Facet Shapes are the single source of truth (D2). S6 (incremental migration) is subsumed — there is no migration, just replacement.

## Concrete Implementation Steps

1. **Macro: thread generics end-to-end (`crates/dsrs-macros/src/lib.rs`)**
   - Capture `input.generics` in `generate_signature_code`.
   - Pass generics into `generate_helper_structs`, `generate_baml_delegation`, and `generate_signature_impl`.
   - Use `split_for_impl()` and emit:
     - `struct FooInput<T...> ...`
     - `impl<...> BamlType for Foo<...> where ...`
     - `impl<...> Signature for Foo<...> where ...`
   - Exit condition: a generic non-flatten signature compiles and existing non-generic tests still pass.

2. **Macro: add flatten parsing/validation/codegen (`crates/dsrs-macros/src/lib.rs`)**
   - Add `flatten` to derive helper attr whitelist.
   - Extend `ParsedField` with `is_flatten`.
   - Parse `#[flatten]` in `parse_single_field`.
   - Add explicit validation rules (at minimum): duplicate `flatten` rejected; `flatten` cannot be combined with leaf-only metadata (`alias`, `check`, `assert`, `format`).
   - Emit `#[facet(flatten)]` on generated helper struct field when `is_flatten` is true.
   - Exit condition: flatten marker survives macro expansion into helper types.

3. **Typed runtime: add path-aware metadata bridge (`crates/dspy-rs/src/core/signature.rs`)**
   - Introduce path-aware field metadata type (e.g. `FieldPathSpec`) carrying leaf name + logical path + `TypeIR`.
   - Add typed-schema derivation from `S::Input` / `S::Output` shapes for typed adapter use (do not rely on static top-level `FieldSpec` arrays for flatten cases).
   - Keep current `FieldSpec` APIs available for compatibility during migration.
   - Exit condition: flattened leaf fields are enumerable with deterministic paths.

4. **Typed adapter: path-based format + parse (`crates/dspy-rs/src/adapter/chat.rs`)**
   - Formatting path: replace direct `fields.get(rust_name)` lookups with path navigation.
   - Parsing path: replace flat `output_map.insert(rust_name, value)` with path insertion into nested `BamlValue` prior to `try_from_baml_value`.
   - Preserve current behavior for non-flatten signatures.
   - Exit condition: flattened output roundtrips with no dropped/misplaced fields.

5. **Tests: add explicit S1 coverage**
   - `crates/dsrs-macros/tests/signature_derive.rs`: add a generic+flatten positive case (including `where` bounds propagation assertions).
   - `crates/dsrs-macros/tests/ui/*.rs`: add compile-fail cases for invalid flatten placements.
   - `crates/dspy-rs` typed adapter tests: add flatten path format/parse roundtrip for generic signatures.
   - Verification command: `cargo test -p dsrs_macros --tests` and targeted `dspy-rs` typed adapter tests.

## Open Risks

- Name-collision semantics for flattened leaves are not yet specified (e.g. sibling `answer` + flattened `inner.answer`).
- Constraint/alias behavior on flattened leaves needs a single rule path (inherit-from-inner vs explicit-on-wrapper).
- Temporary dual metadata surfaces (`FieldSpec` + path-aware typed schema) must be tightly scoped to avoid drift before S6 cutover.

## Acceptance

S1 is complete when:

- A generic signature with flattened field(s) compiles and derives `Signature` successfully.
- Generated `Input`/`Output` helper types preserve generic parameters and bounds.
- Typed adapter formatting and parsing behave correctly for flattened fields (no field loss/mismatch).
- Existing non-generic signature behavior remains unchanged.
- Tests explicitly cover generic+flatten signatures in macro and runtime paths.
