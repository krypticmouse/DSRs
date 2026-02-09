# Slice 3 Adversarial Review (Module Authoring)

Date: 2026-02-09
Scope: Slice 3 only (`F4`, `F12`, `U21–U27`, `N15`) with emphasis on:
- Module trait migration
- Generic/flatten signature derive behavior
- Adapter schema helper exposure
- Example authoring syntax

Authority docs used:
- `docs/specs/modules/breadboard.md`
- `docs/specs/modules/shapes.md`
- `docs/specs/modules/design_reference.md`
- Spikes: `S1`, `S2`, `S3`, `S6`, `S7`

## Findings

### High

1. **Option-C full replacement is not complete; legacy `MetaSignature`/`LegacyPredict` path remains active.**
- Spec expectation:
  - `docs/specs/modules/design_reference.md:32` says no parallel schema systems.
  - `docs/specs/modules/design_reference.md:997` and `docs/specs/modules/design_reference.md:1002` (plus `docs/specs/modules/shapes.md:139`, `docs/specs/modules/shapes.md:144`) resolve S1/S6 to full replacement (no migration bridge).
- Current code:
  - `crates/dspy-rs/src/core/signature.rs:23` keeps `MetaSignature` as a primary trait.
  - `crates/dspy-rs/src/adapter/mod.rs:15` keeps adapter contract centered on `&dyn MetaSignature`.
  - `crates/dspy-rs/src/predictors/predict.rs:538` keeps `LegacyPredict` as an active type.
  - `crates/dspy-rs/src/predictors/predict.rs:472` keeps `Predict<S>: MetaSignature` bridge.
- Impact:
  - Slice 3 migration ships dual runtime surfaces (typed schema path + legacy meta-signature path), conflicting with the selected “full replacement” architecture.
- Remediation:
  - If Option C is authoritative: remove `LegacyPredict`/`MetaSignature` adapter path and move remaining consumers to schema-first APIs.
  - If compatibility is intentionally retained: update specs to explicitly permit a bounded compatibility bridge and define removal gates.

2. **Generic bounds are stripped from generated helper types, contradicting F12’s “thread bounds through generated types.”**
- Spec expectation:
  - `docs/specs/modules/design_reference.md:122` requires threading generic parameters **and bounds** through generated types/impls.
- Current code:
  - `crates/dsrs-macros/src/lib.rs:522` (`unconstrained_generics`) clears type-param bounds and removes the where clause.
  - `crates/dsrs-macros/src/lib.rs:443` uses unconstrained generics for generated input/output/all helper structs.
- Impact:
  - Generated helper type declarations can diverge from the user’s declared generic contract and are no longer a faithful projection of the source signature.
- Remediation:
  - Preserve original generic bounds/where clauses on generated helper structs.
  - Keep separate handling only for truly unused params (phantom marker), without dropping declared constraints globally.

### Medium

3. **Generated `__phantom` field leaks into public helper API and is user-visible in generic signatures.**
- Spec expectation:
  - `docs/specs/modules/design_reference.md:985` (D7) emphasizes “zero framework tax” module authoring ergonomics.
- Current code:
  - `crates/dsrs-macros/src/lib.rs:545` adds a public `__phantom` field for unused generics.
  - `crates/dsrs-macros/tests/signature_derive.rs:84` shows callers must manually initialize `__phantom` for `__GenericFlattenSigOutput<_>`.
- Impact:
  - Internal macro machinery leaks into author-facing types and complicates demo/output construction.
- Remediation:
  - Make marker fields private and auto-initialized in generated constructors/conversions.
  - Avoid requiring struct-literal initialization of generated output internals.

4. **`Module` trait allows untyped `Example`/`Prediction` modules, diverging from the design-reference typed F4 contract.**
- Spec expectation:
  - `docs/specs/modules/design_reference.md:364` and `docs/specs/modules/design_reference.md:365` constrain `Module::Input/Output` to typed `BamlType + Facet`.
- Current code:
  - `crates/dspy-rs/src/core/module.rs:10` and `crates/dspy-rs/src/core/module.rs:11` only require `Send + Sync + 'static`.
  - `crates/dspy-rs/examples/01-simple.rs:61` and `crates/dspy-rs/examples/01-simple.rs:62` continue `Module<Input=Example, Output=Prediction>` authoring.
- Impact:
  - Weakens compile-time composition guarantees and blurs Slice 3’s typed module-authoring boundary.
- Remediation:
  - Either tighten `Module` bounds to the typed contract, or explicitly codify a separate legacy/untyped module surface in the specs.

### Low

5. **Adapter building-block API shape drifts from spec on `build_system` return type.**
- Spec expectation:
  - `docs/specs/modules/design_reference.md:583`–`docs/specs/modules/design_reference.md:586` specifies `build_system(...) -> String`.
- Current code:
  - `crates/dspy-rs/src/adapter/chat.rs:463`–`crates/dspy-rs/src/adapter/chat.rs:467` exposes `build_system(...) -> Result<String>`.
- Impact:
  - Spec/API mismatch for P2 affordance `U23`; authors must handle a failure mode not described in Slice 3 docs.
- Remediation:
  - Align implementation to `String` or update specs and slice examples to document fallible behavior.

## Validation notes

Commands run:
- `cargo check -p dspy-rs --examples`
- `cargo test -p dsrs_macros --tests`
- `cargo test -p dspy-rs --lib`
- `cargo test -p dspy-rs --test test_signature_macro --test test_signature_schema --test test_chat_adapter_schema`
- `cargo test -p dspy-rs --test test_flatten_roundtrip --test test_typed_prompt_format --test test_with_reasoning_deref`

Result: all commands passed in current workspace state.
