# Slices 1–3 Closure Audit

Date: 2026-02-09
Scope: Breadboard vertical slices `V1`, `V2`, `V3` from `docs/specs/modules/breadboard.md`.

## Audit Method
- Re-read `docs/specs/modules/breadboard.md` slice details and `docs/specs/modules/shapes.md` / `docs/specs/modules/design_reference.md` constraints for `F1–F4`, `F5`, `F7`, `F11(CoT)`, and `F12`.
- Verify implementation in repo code and tests.
- Classify each slice affordance group as `Implemented`, `Partially Implemented`, or `Deferred`.
- For every non-implemented item, assign an explicit target follow-up phase.

## New Process Phase
- Added post-commit phase: **Closure Audit**.
- Purpose: explicit bookkeeping pass that confirms each in-scope slice requirement is either implemented or deferred to a named follow-up phase with an owner and exit criteria.

## Planned Phase 4.5

- **Phase 4.5: Cleanup / API Surface Pass** is explicitly scheduled.
- Scope:
  - remove or quarantine legacy compatibility surfaces that are no longer needed,
  - normalize public API to the intended typed-first authoring model,
  - reconcile module/adapter/signature surfaces with spec language,
  - reduce transitional glue and tighten invariants before further feature expansion.

## Long-Term Architecture Position (Recorded)

Rollout assessment: current slices 1–3 shape is acceptable for incremental delivery.

End-state assessment: current slices 1–3 shape is not the intended final architecture.

Current compatibility-heavy surfaces (cross-referenced):
- Legacy schema/predict path still active:
  - `crates/dspy-rs/src/core/signature.rs` (`MetaSignature`)
  - `crates/dspy-rs/src/predictors/predict.rs` (`LegacyPredict`)
  - `crates/dspy-rs/src/adapter/mod.rs` and `crates/dspy-rs/src/adapter/chat.rs` (`&dyn MetaSignature` call/format flow)
- Module trait still permits untyped compatibility shapes:
  - `crates/dspy-rs/src/core/module.rs` (`Module::Input/Output` are `Send + Sync + 'static` only)
  - Examples still demonstrate legacy-untyped composition:
    - `crates/dspy-rs/examples/01-simple.rs`
- Wrapper discoverability for future F6 walker is incomplete:
  - `crates/dspy-rs/src/modules/chain_of_thought.rs` (no module-level `Facet` derive yet)
- Generic helper authoring ergonomics remain transitional:
  - `crates/dsrs-macros/src/lib.rs` (`unconstrained_generics` helper strategy and generated marker field handling)
  - `crates/dsrs-macros/tests/signature_derive.rs` (`__phantom` initialization still visible in same-module test construction)
- Adapter building blocks are available but include a fallible `build_system` surface:
  - `crates/dspy-rs/src/adapter/chat.rs`

Target end-state direction:
1. Remove `MetaSignature`/`LegacyPredict` after schema-first consumer parity and migration gates.
2. Tighten `Module` to typed bounds (`BamlType + Facet`) for library-facing authoring.
3. Make wrapper module discoverability (`ChainOfThought` and combinators) Facet-walker ready.
4. Complete macro helper contract hardening (generic bounds + marker ergonomics).
5. Keep adapter helper fallibility explicit and spec-aligned (implementation/spec convergence).

## Slice 1 (V1 Typed Call) Accounting

| Affordance(s) | Status | Evidence |
|---|---|---|
| `U1,U2,U3` Signature derive + markers + doc extraction | Implemented | `crates/dsrs-macros/src/lib.rs`, `crates/dsrs-macros/tests/signature_derive.rs` |
| `U4,U5` generated typed input/output helper types | Implemented | `crates/dsrs-macros/src/lib.rs`, `crates/dsrs-macros/tests/signature_derive.rs` |
| `U6,U7,U8` `Predict` construction/builder/demo | Implemented | `crates/dspy-rs/src/predictors/predict.rs`, `crates/dspy-rs/examples/01-simple.rs` |
| `U9,U10,U11` typed call path (`forward`/`call` + `CallOutcome` + field access) | Implemented | `crates/dspy-rs/src/core/module.rs`, `crates/dspy-rs/src/predictors/predict.rs`, `crates/dspy-rs/tests/test_call_outcome.rs` |
| `U49` parse/error visibility on call path | Implemented | `crates/dspy-rs/src/predictors/predict.rs`, `crates/dspy-rs/src/core/call_outcome.rs` |
| `N1,N2` compile-time macro expansion and instruction extraction | Implemented | `crates/dsrs-macros/src/lib.rs` |
| `N3` `SignatureSchema` derivation/cache | Implemented | `crates/dspy-rs/src/core/schema.rs`, `crates/dspy-rs/tests/test_signature_schema.rs` |
| `N8` schema-driven adapter pipeline | Implemented | `crates/dspy-rs/src/adapter/chat.rs`, `crates/dspy-rs/tests/test_chat_adapter_schema.rs` |
| `N13` typed conversion boundary (`try_from_baml_value`) | Implemented | `crates/dspy-rs/src/adapter/chat.rs`, `crates/dspy-rs/src/predictors/predict.rs` |
| `S1,S2,S3` schema cache + demos + instruction state | Implemented | `crates/dspy-rs/src/core/schema.rs`, `crates/dspy-rs/src/predictors/predict.rs` |

Slice 1 verdict: **Implemented**.

## Slice 2 (V2 Augmentation + ChainOfThought) Accounting

| Affordance(s) | Status | Evidence |
|---|---|---|
| `U12` Deref access to augmented output fields | Implemented | `crates/dspy-rs/src/augmentation.rs`, `crates/dspy-rs/tests/test_with_reasoning_deref.rs` |
| `U13` `ChainOfThought::new()` and builder | Implemented | `crates/dspy-rs/src/modules/chain_of_thought.rs`, `crates/dspy-rs/tests/test_chain_of_thought_swap.rs` |
| `U16` strategy swap ergonomics (`Predict` -> `ChainOfThought`) | Implemented | `crates/dspy-rs/tests/test_chain_of_thought_swap.rs` |
| `U17,U18,U19,U20` augmentation derive and wrapper model | Implemented | `crates/dsrs-macros/src/lib.rs`, `crates/dspy-rs/src/augmentation.rs` |
| `U28` internal `Predict<Augmented<...>>` module composition | Implemented | `crates/dspy-rs/src/modules/chain_of_thought.rs` |
| `U29` module-level Facet discoverability (`#[derive(Facet)]` on module struct) | Deferred | `crates/dspy-rs/src/modules/chain_of_thought.rs` (currently no `Facet` derive) |
| `N14` augmentation macro mechanics | Implemented | `crates/dsrs-macros/src/lib.rs`, `crates/dspy-rs/tests/test_flatten_roundtrip.rs` |

Slice 2 verdict: **Partially Implemented** (`U29` deferred).

## Slice 3 (V3 Module Authoring) Accounting

| Affordance(s) | Status | Evidence |
|---|---|---|
| `U21,U22` generic signature derive + flatten behavior | Partially Implemented | `crates/dsrs-macros/src/lib.rs`, `crates/dsrs-macros/tests/signature_derive.rs` (functional flatten/generics pass; helper-generic bound threading mismatch deferred) |
| `U23,U24,U25,U26` adapter building blocks | Partially Implemented | `crates/dspy-rs/src/adapter/chat.rs` (`build_system`, `format_input`, `parse_sections`, `parse_output` are exposed; `build_system` return type differs from design-reference sketch) |
| `U27` custom `impl Module` authoring | Partially Implemented | `crates/dspy-rs/src/core/module.rs`, `crates/dspy-rs/examples/92-smoke-slice3-module-authoring.rs` (typed surface works; strict `BamlType + Facet` trait bounds deferred) |
| `N15` generic signature macro support | Partially Implemented | `crates/dsrs-macros/src/lib.rs`, `crates/dsrs-macros/tests/signature_derive.rs` |

Slice 3 verdict: **Partially Implemented** (three explicit hardening items deferred).

## Named/Labeled Smoke Artifacts

- Kept as stable, labeled examples:
  - `crates/dspy-rs/examples/90-smoke-slice1-typed-predict.rs`
  - `crates/dspy-rs/examples/91-smoke-slice2-chain-of-thought.rs`
  - `crates/dspy-rs/examples/92-smoke-slice3-module-authoring.rs`

## Explicit Deferral Ledger

| Deferred item | Why deferred now | Target phase | Exit criteria |
|---|---|---|---|
| `U29` module-level Facet discoverability for `ChainOfThought` | No active F6 walker consumption in slices 1–3; implementing in isolation risks churn before V5 optimizer boundary lands | **V5 Implement (Optimizer Interface)** | `ChainOfThought` and wrapper modules expose Facet shapes that are consumed by `named_parameters`/walker tests |
| Strict typed `Module` bounds (`Input/Output: BamlType + Facet`) | Compatibility layer still supports legacy `Example/Prediction` modules and examples | **Phase 4.5 Cleanup / API Surface Pass** | `Module` trait bounds tightened; examples/tests updated to typed module inputs/outputs only |
| F12 helper generic bounds threading in generated helper structs | Direct change caused macro trait-resolution breakage; requires deliberate macro redesign, not a one-line patch | **Phase 4.5 Cleanup / API Surface Pass** | Helper type declarations preserve source generic contract while `dsrs-macros` tests remain green |
| `__phantom` helper-field authoring ergonomics | Field is private now, but same-module struct literal ergonomics can still surface it | **Phase 4.5 Cleanup / API Surface Pass** | No user-visible phantom initialization burden in signature macro tests/examples |
| `build_system` return shape mismatch vs design sketch (`Result<String>` vs `String`) | Existing implementation legitimately propagates schema render failures; changing now would hide errors | **Phase 4.5 Cleanup / API Surface Pass** | Either spec updated to fallible API or implementation changed with explicit error-handling policy |
| Option-C full legacy cutover (`MetaSignature`/`LegacyPredict` still active) | Optimizers and adapter compatibility still consume legacy path; removing immediately would break active flows | **Phase 4.5 Cleanup / API Surface Pass** | All consumers migrated to schema-first typed surfaces; legacy path removed behind clear migration note |

## Validation Run During Closure Audit

- `cargo test -p dsrs_macros --tests`
- `cargo test -p dspy-rs --test test_call_outcome --test test_signature_schema --test test_chat_adapter_schema --test test_flatten_roundtrip --test test_chain_of_thought_swap --test test_with_reasoning_deref`

Both command groups passed in current workspace state.
