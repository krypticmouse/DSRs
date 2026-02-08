# S2 Spike: DynPredictor Handle Discovery

## Context

S2 asks for the concrete mechanism that lets a Facet-based walker discover predictor leaves and return usable optimizer handles (`&dyn DynPredictor` / `&mut dyn DynPredictor`) without manual traversal boilerplate. It is explicitly marked high-priority and blocks R4 in shaping/design (`docs/specs/modules/shapes.md:241`, `docs/specs/modules/design_reference.md:1007`).

The current runtime still uses `Optimizable::parameters()` with manual `#[derive(Optimizable)]` + `#[parameter]`, so S2 must bridge from that model to automatic Facet discovery.

## Goal

Identify the most practical first implementation for S2 that:
- discovers `Predict` leaves automatically from module structure,
- yields stable dotted paths + mutable optimizer handles,
- is compatible with current optimizer call patterns,
- and is backed by real Facet traversal/attribute capabilities.

## Questions

| ID | Question |
|---|---|
| **S2-Q1** | What exact handle contract do current optimizers require from discovered predictors? |
| **S2-Q2** | How does current discovery work, and where does it fail relative to S2/R4? |
| **S2-Q3** | What Facet walker/traversal primitives already exist for structs/lists/maps/options/cycles? |
| **S2-Q4** | Can Facet attributes carry typed runtime payloads sufficient for handle extraction (including function-pointer payloads)? |
| **S2-Q5** | Which mechanism is best for obtaining `DynPredictor` handles: shape attribute payload, global registry, or another pattern? |
| **S2-Q6** | What is the concrete migration path from `Optimizable::parameters()` to Facet walker discovery without breaking optimizers? |

## Findings

1. Current optimizers require mutable, path-addressable handles, not just discovery metadata.
   - `Optimizable` requires `parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable>` (`crates/dspy-rs/src/core/module.rs:89`).
   - Optimizers repeatedly look up dotted names and mutate predictors (`crates/dspy-rs/src/optimizer/copro.rs:221`, `crates/dspy-rs/src/optimizer/mipro.rs:419`, `crates/dspy-rs/src/optimizer/gepa.rs:442`).

2. Current discovery is manual and annotation-driven.
   - `Optimizable` derive is keyed on `#[parameter]` (`crates/dsrs-macros/src/lib.rs:17`).
   - Macro extraction only includes fields with that annotation (`crates/dsrs-macros/src/optim.rs:72`, `crates/dsrs-macros/src/optim.rs:78`).
   - Flattening uses unsafe casts to create leaf handles (`crates/dsrs-macros/src/optim.rs:49`, `crates/dsrs-macros/src/optim.rs:54`).

3. Predictor leaves are currently exposed only through `Optimizable` leaf behavior.
   - `Predict<S>` and `LegacyPredict` both return empty child maps (`crates/dspy-rs/src/predictors/predict.rs:503`, `crates/dspy-rs/src/predictors/predict.rs:667`).
   - There is no concrete `DynPredictor` trait in current runtime code.

4. Test coverage validates nested struct flattening, but not container traversal (`Option`/`Vec`/`Map`) or Facet auto-discovery.
   - Existing tests exercise nested named-field flattening (`crates/dspy-rs/tests/test_optimizable.rs:39`, `crates/dspy-rs/tests/test_optimizable.rs:64`).
   - `cargo test -p dspy-rs --test test_optimizable` passes (3/3) as of February 9, 2026.

5. Local code proves Facet attrs can carry typed function-pointer payloads and execute them at runtime.
   - `WithAdapterFns` stores function pointers and is Facet-derived (`crates/bamltype/src/facet_ext.rs:19`, `crates/bamltype/src/facet_ext.rs:23`).
   - Attr extraction uses typed decoding via `attr.get_as` (`crates/bamltype/src/facet_ext.rs:41`, `crates/bamltype/src/facet_ext.rs:47`).
   - Runtime paths invoke those payloads for conversion/schema work (`crates/bamltype/src/convert.rs:300`, `crates/bamltype/src/schema_builder.rs:390`).

6. Local conversion code demonstrates the exact reflection primitives needed for S2/S5 container traversal.
   - `convert.rs` imports Facet `Def` plus reflection `Partial`/`Peek` (`crates/bamltype/src/convert.rs:8`, `crates/bamltype/src/convert.rs:9`).
   - Deserialization handles pointer/option/list/map recursively (`crates/bamltype/src/convert.rs:125`, `crates/bamltype/src/convert.rs:133`, `crates/bamltype/src/convert.rs:191`, `crates/bamltype/src/convert.rs:216`).
   - Serialization handles option/list/map through recursive `Peek` traversal (`crates/bamltype/src/convert.rs:539`, `crates/bamltype/src/convert.rs:593`, `crates/bamltype/src/convert.rs:602`).

7. Typed attr-payload decoding is already a normal pattern in schema paths.
   - `schema_builder` resolves typed extension attrs with `attr.get_as::<facet_ext::Attr>()` (`crates/bamltype/src/schema_builder.rs:629`, `crates/bamltype/src/schema_builder.rs:636`).
   - This supports reusing the same pattern for DynPredictor accessor payloads instead of introducing a separate global registry first.

## Candidate Mechanisms + Tradeoffs

Decision criteria: satisfy Q1 handle contract, preserve S5 container recursion, keep migration risk low, and keep unsafe surface auditable.

| Mechanism | Q1: mutable handle contract | Q3/Q4/S5: Facet traversal + typed payload fit | Migration risk | Verdict |
|---|---|---|---|---|
| **A. Shape-local accessor payload (`dsrs::parameter` + fn ptr payload)** | **Strong**: direct cast to `&mut dyn DynPredictor` at leaf | **Strong**: matches existing typed attr payload pattern and recursive reflection model | **Medium**: requires one audited unsafe boundary | **Best first implementation** |
| **B. Global registry (shape/type id → accessor)** | **Strong**: can return mutable handles | **Medium**: traversal still works, but handle resolution depends on external registration | **High**: init-order, registration drift, harder debugging | **Fallback only** |
| **C. Store dyn handle inside `Predict` state** | **Medium**: contract works but via extra indirection | **Weak**: bypasses Facet metadata path and adds ownership complexity | **High**: invasive runtime state changes | **Reject for V1** |

## Recommended Approach

**Decision:** implement **Mechanism A** for S2 V1.

**Scope for this spike outcome:**
- **In:** shape-local accessor payload on `Predict<S>`, Facet walker discovery, compatibility shim for current optimizers.
- **Deferred:** registry-based indirection (Mechanism B) unless later required by cross-crate runtime loading.
- **Out:** interior dyn-handle state in `Predict` (Mechanism C).

Why this path is crisp:
- It reuses a proven local pattern (`WithAdapterFns` + typed attr payload extraction).
- It keeps discovery + extraction colocated with the type shape (fewer moving parts).
- It preserves deterministic traversal semantics needed for optimizer naming.

## Concrete Implementation Steps

| # | Implementation step | Testable exit criterion |
|---|---|---|
| 1 | Introduce `DynPredictor` trait and `PredictAccessorFns` payload type (opaque, fn-pointer based) | Compile-time check that `Predict<S>: DynPredictor`; payload type is `'static + Copy` and can be stored in Facet attr grammar |
| 2 | Add `dsrs` attr grammar entries for predictor marker + accessor payload | Unit test can read `Predict::<TestSig>::SHAPE` attrs and decode payload via typed `get_as` |
| 3 | Implement `DynPredictor` for `Predict<S>` and attach payload on `Predict` shape | Unit test obtains payload from shape and successfully reads/updates predictor instruction through returned dyn handle |
| 4 | Implement `named_predictors_mut` walker over Facet-reflect values (struct/list/map/option/pointer; stop descent at predictor leaves) | Snapshot test returns expected dotted paths for nested fixture module (e.g. `retrieve`, `answer.predict`) |
| 5 | Define deterministic path encoding (`field`, `[idx]`, `['key']`) + cycle guard behavior | Repeated runs (e.g. 100 iterations) return identical order/paths for the same module instance |
| 6 | Add compatibility shim from new discovery output to current optimizer mutation flow | Existing optimizer tests/smokes still mutate instructions by dotted name without changing optimizer call sites |
| 7 | Add container and failure-path tests | Tests cover `Option<Predict<_>>`, `Vec<Predict<_>>`, `Map<String, Predict<_>>`, and missing/invalid payload decode errors |
| 8 | Migrate module examples from derive-driven parameter discovery to walker discovery | Example modules no longer require `#[derive(Optimizable)]` + `#[parameter]` for predictor discovery |

## Acceptance

S2 is complete when:

- A Facet-based walker can discover all predictor leaves in nested modules without `#[parameter]` annotations.
- Discovery returns stable dotted paths and mutable handles that optimizers can use to read/update predictor state.
- `Option`/`Vec`/`Map` containment of predictors is covered by automated tests.
- Container recursion is cycle-safe and deterministic.
- Existing optimization flows remain functional through the compatibility shim.
- The mechanism is documented with clear unsafe boundaries and invariants.
- Baseline compatibility remains green (`cargo test -p dspy-rs --test test_optimizable`).

## Open Risks

- Unsafe cast boundary for payload-based handle extraction must be tightly documented and audited.
- Map-key ordering policy for dotted paths must be explicit to avoid optimizer cache churn across runs.
- If structural optimization later requires loading strategies from crates not linked at compile time, Mechanism B (registry fallback) may still be needed.
