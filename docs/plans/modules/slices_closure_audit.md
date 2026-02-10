# Slices 1-6 Closure Audit

Date: 2026-02-10  
Scope: Breadboard vertical slices `V1`, `V2`, `V3`, `V4`, `V5`, `V6` from `docs/specs/modules/breadboard.md`.

## Audit Method
- Re-checked `docs/specs/modules/breadboard.md` slice definitions and `docs/specs/modules/shapes.md` / `docs/specs/modules/design_reference.md` constraints.
- Verified implementation in the live codebase (not docs-only) with file-level evidence.
- Classified each slice requirement as `Implemented` or `Deferred` with explicit follow-up mapping.

## Slices 1-3 Baseline
- Baseline accounting for `V1`-`V3` remains in `docs/plans/modules/slices_1_3_closure_audit.md`.
- This document extends that ledger through `V6` and updates deferred-item routing with current post-V6 status.

## Slice 4 (V4 ReAct + Operational) Accounting

| Affordance(s) | Status | Evidence |
|---|---|---|
| `U14` ReAct builder with tools (`.tool("name", "desc", fn)`) | Implemented | `crates/dspy-rs/src/modules/react.rs:53`, `crates/dspy-rs/src/modules/react.rs:273`, `crates/dspy-rs/src/modules/react.rs:325`, `crates/dspy-rs/tests/test_react_builder.rs:93` |
| `U14` ReAct action/extract composition (two `Predict` leaves + loop in `forward`) | Implemented | `crates/dspy-rs/src/modules/react.rs:61`, `crates/dspy-rs/src/modules/react.rs:64`, `crates/dspy-rs/src/modules/react.rs:148`, `crates/dspy-rs/tests/test_react_builder.rs:66` |
| `U14` ReAct trajectory parity without extra API (`CallOutcome` metadata carries trace, no `call_with_trajectory`) | Implemented | `crates/dspy-rs/src/modules/react.rs:85`, `crates/dspy-rs/src/modules/react.rs:135`, `crates/dspy-rs/src/modules/react.rs:236`, `crates/dspy-rs/examples/93-smoke-slice4-react-operational.rs:89` |
| `U48` standalone `forward_all(&module, inputs, concurrency)` | Implemented | `crates/dspy-rs/src/core/module.rs:22`, `crates/dspy-rs/src/evaluate/evaluator.rs:24`, `crates/dspy-rs/tests/test_module_forward_all.rs:21` |
| `U51` module combinators (`.map()`, `.and_then()`) | Implemented | `crates/dspy-rs/src/core/module_ext.rs:5`, `crates/dspy-rs/src/core/module_ext.rs:28`, `crates/dspy-rs/src/core/module_ext.rs:55`, `crates/dspy-rs/tests/test_module_ext.rs:37` |
| `S4` tool storage for operational modules | Implemented | `crates/dspy-rs/src/modules/react.rs:66`, `crates/dspy-rs/src/modules/react.rs:152`, `crates/dspy-rs/src/modules/react.rs:335` |

Slice 4 verdict: **Implemented**.

## Slice 5 (V5 Optimizer Interface) Accounting

| Affordance(s) | Status | Evidence |
|---|---|---|
| `U30`, `U31` Facet-powered discovery entry + handle vector (`named_parameters(&mut module) -> Vec<(String, &mut dyn DynPredictor)>`) | Implemented | `crates/dspy-rs/src/core/dyn_predictor.rs:72`, `crates/dspy-rs/src/core/dyn_predictor.rs:91`, `crates/dspy-rs/tests/test_named_parameters.rs:50`, `crates/dspy-rs/tests/test_named_parameters.rs:112` |
| `N18` recursive struct walker with explicit container errors (`Vec`, `Option`, `HashMap`, pointer/Box-like) | Implemented | `crates/dspy-rs/src/core/dyn_predictor.rs:110`, `crates/dspy-rs/src/core/dyn_predictor.rs:128`, `crates/dspy-rs/src/core/dyn_predictor.rs:172`, `crates/dspy-rs/tests/test_named_parameters_containers.rs:27`, `crates/dspy-rs/tests/test_named_parameters_containers.rs:46` |
| `U32` schema access via dyn handle (`predictor.schema()`) | Implemented | `crates/dspy-rs/src/core/dyn_predictor.rs:12`, `crates/dspy-rs/src/predictors/predict.rs:500`, `crates/dspy-rs/src/optimizer/mipro.rs:594`, `crates/dspy-rs/src/optimizer/copro.rs:73` |
| `U33`, `U34`, `N21`, `N22` demos as `Example` + typed roundtrip mutation | Implemented | `crates/dspy-rs/src/core/dyn_predictor.rs:15`, `crates/dspy-rs/src/predictors/predict.rs:514`, `crates/dspy-rs/src/predictors/predict.rs:521`, `crates/dspy-rs/src/predictors/predict.rs:378`, `crates/dspy-rs/src/predictors/predict.rs:388`, `crates/dspy-rs/tests/test_named_parameters.rs:73` |
| `U35` instruction get/set through dyn handle | Implemented | `crates/dspy-rs/src/core/dyn_predictor.rs:13`, `crates/dspy-rs/src/predictors/predict.rs:504`, `crates/dspy-rs/src/predictors/predict.rs:510`, `crates/dspy-rs/tests/test_named_parameters.rs:57`, `crates/dspy-rs/examples/94-smoke-slice5-optimizer-interface.rs:37` |
| `U36` predictor state persistence (`dump_state` / `load_state`) | Implemented | `crates/dspy-rs/src/core/dyn_predictor.rs:17`, `crates/dspy-rs/src/predictors/predict.rs:529`, `crates/dspy-rs/src/predictors/predict.rs:536`, `crates/dspy-rs/tests/test_named_parameters.rs:73` |
| `U37`, `N23` untyped forward bridge (`forward_untyped(BamlValue)`) | Implemented | `crates/dspy-rs/src/core/dyn_predictor.rs:19`, `crates/dspy-rs/src/predictors/predict.rs:472`, `crates/dspy-rs/src/predictors/predict.rs:542`, `crates/dspy-rs/tests/test_dyn_predictor_forward_untyped.rs:63` |
| Optimizer internals rewired to new surface (`named_parameters` + dyn handle mutation) | Implemented | `crates/dspy-rs/src/optimizer/copro.rs:98`, `crates/dspy-rs/src/optimizer/mipro.rs:569`, `crates/dspy-rs/src/optimizer/gepa.rs:452` |
| `U50` compile entrypoint fidelity (`optimizer.compile(&mut module, trainset, metric)`) | Deferred | Current Rust surface remains `compile(&mut module, trainset)` bound to legacy `Evaluator` (`crates/dspy-rs/src/optimizer/mod.rs:22`, `crates/dspy-rs/src/evaluate/evaluator.rs:7`). Explicit metric arg / typed evaluator migration deferred to cleanup (C4 debt). |
| `S2` Mechanism A strict fidelity (shape-local `dsrs::parameter` payload extraction) | Deferred | Current discovery uses `shape.type_identifier == \"Predict\"` + accessor registry (`crates/dspy-rs/src/core/dyn_predictor.rs:188`, `crates/dspy-rs/src/core/dyn_predictor.rs:45`). Direct generic payload attachment hit compile constraints in current derive expansion; tracked as migration debt. |

Slice 5 verdict: **Partially Implemented** (core F6/F8 behavior shipped; U50/C4 and strict S2 mechanism deferred with explicit cleanup targets).

## Slice 6 (V6 Dynamic Graph) Accounting

| Affordance(s) | Status | Evidence |
|---|---|---|
| `U38`, `U39` strategy registry (`registry::create`, `registry::list`) | Implemented | `crates/dspy-rs/src/core/dyn_module.rs:53`, `crates/dspy-rs/src/core/dyn_module.rs:79`, `crates/dspy-rs/src/core/dyn_module.rs:88`, `crates/dspy-rs/tests/test_registry_dynamic_modules.rs:45` |
| `U40` dynamic predictor exposure (`predictors`, `predictors_mut`) | Implemented | `crates/dspy-rs/src/core/dyn_module.rs:29`, `crates/dspy-rs/src/core/dyn_factories.rs:329`, `crates/dspy-rs/src/core/dyn_factories.rs:333`, `crates/dspy-rs/tests/test_registry_dynamic_modules.rs:63` |
| `U41`, `U42` graph construction (`new`, `add_node`) including direct registry node insertion | Implemented | `crates/dspy-rs/src/core/program_graph.rs:162`, `crates/dspy-rs/src/core/program_graph.rs:181`, `crates/dspy-rs/tests/test_registry_dynamic_modules.rs:68` |
| `U43`, `N24` edge insertion with validation, including breadboard input pseudo-node wiring | Implemented | `crates/dspy-rs/src/core/program_graph.rs:209`, `crates/dspy-rs/src/core/program_graph.rs:601`, `crates/dspy-rs/tests/test_program_graph_mutation.rs:86`, `crates/dspy-rs/tests/test_program_graph_execution.rs:231` |
| `U44` node replacement + incident-edge revalidation | Implemented | `crates/dspy-rs/src/core/program_graph.rs:226`, `crates/dspy-rs/tests/test_program_graph_mutation.rs:100` |
| `U45`, `N25`, `N26` topological execution and BamlValue piping | Implemented | `crates/dspy-rs/src/core/program_graph.rs:349`, `crates/dspy-rs/src/core/program_graph.rs:657`, `crates/dspy-rs/tests/test_program_graph_execution.rs:143`, `crates/dspy-rs/tests/test_program_graph_execution.rs:198` |
| `U46` typed→graph projection + fit-back | Implemented | `crates/dspy-rs/src/core/program_graph.rs:453`, `crates/dspy-rs/src/core/program_graph.rs:512`, `crates/dspy-rs/tests/test_program_graph_projection_fit.rs:33` |
| `N17` schema-transforming factories (`chain_of_thought` reasoning prepend, `react` action/extract schemas) | Implemented | `crates/dspy-rs/src/core/dyn_factories.rs:449`, `crates/dspy-rs/src/core/dyn_factories.rs:552`, `crates/dspy-rs/src/core/dyn_factories.rs:617`, `crates/dspy-rs/tests/test_registry_dynamic_modules.rs:95` |
| `N27` distributed factory auto-registration (`inventory::submit!`) | Implemented | `crates/dspy-rs/src/core/dyn_factories.rs:540`, `crates/dspy-rs/src/core/dyn_factories.rs:544`, `crates/dspy-rs/src/core/dyn_factories.rs:548` |
| `R8` typed/dynamic prompt parity and dynamic graph real-model smoke | Implemented | `crates/dspy-rs/tests/test_program_graph_execution.rs:269`, `crates/dspy-rs/examples/95-smoke-slice6-dynamic-graph.rs:18`, `crates/dspy-rs/examples/95-smoke-slice6-dynamic-graph.rs:33` |

Slice 6 verdict: **Implemented** (with explicit post-implementation debt retained for strict S2 attr payload, edge-annotation storage mechanism, and broader TypeIR assignability semantics).

## Consolidated Deferred Ledger (Post-Implementation Cleanup)

| Deferred item | Why deferred | Target phase | Exit criteria |
|---|---|---|---|
| Strict typed `Module` bounds (`Input/Output: BamlType + Facet`) | Compatibility with legacy/untyped module surfaces still present | **Post-Implementation Cleanup** | `Module` bounds tightened and impacted examples/tests migrated |
| F12 helper generic bounds threading in generated helper structs | Macro helper constraints still use transitional strategy | **Post-Implementation Cleanup** | Generic helper declarations preserve source generic contract with `dsrs-macros` tests green |
| `__phantom` helper-field authoring ergonomics | Generic helper phantom initialization still leaks into same-module literals | **Post-Implementation Cleanup** | No user-facing phantom initialization burden in macro tests/examples |
| Option-C full legacy cutover (`MetaSignature`/`LegacyPredict`) | Legacy compatibility surfaces still active for older flows | **Post-Implementation Cleanup** | Schema-first typed path is sole default path and legacy surfaces are removed/quarantined with migration notes |
| `V5` walker discoverability for additional wrappers/combinators | Deferred by earlier closure audits; only Slice 4 ReAct discoverability addressed now | **Post-Implementation Cleanup** (prep) + **V5 Implement** (completion) | Walker traverses wrapper module trees end-to-end with tests for nested combinator/module stacks |
| `V5` strict S2 mechanism (`dsrs::parameter` payload extraction) | Current generic payload attachment path is blocked in current derive expansion; registry fallback was used to keep V5 green | **Post-Implementation Cleanup** | Replace registry/type-name discovery with shape-local typed attr payload extraction or finalize audited equivalent and update spec debt note |
| `V5` U50 typed metric surface (`compile(..., metric)`) | Optimizer compile remains coupled to legacy `Evaluator` / `Example`→`Prediction` IO boundary | **Post-Implementation Cleanup** | Optimizer compile path accepts typed metric/evaluator surface and no longer requires legacy compile bounds |
| GEPA uniform compile entrypoint | `GEPA::compile` intentionally bails and redirects to `compile_with_feedback`; inconsistent with uniform U50 contract | **Post-Implementation Cleanup** | GEPA exposes a functional uniform compile surface (or officially documented trait split) without runtime bailout |
| `V6` edge annotation storage mechanism | V6 uses global shape-id keyed registration for annotations; shape-local Facet attr storage remains deferred | **Post-Implementation Cleanup** | Move edge annotations to shape-local Facet attrs (or ratify global registration path in spec) and remove dual-path ambiguity |
| `V6` TypeIR assignability breadth | Current `is_assignable_to` is conservative (exact, nullable widening, simple unions) | **Post-Implementation Cleanup** | Replace with native/complete TypeIR subtyping semantics that cover richer unions/classes/aliases |
| Typed example loading (Shape A) | Training data remains untyped `Vec<Example>` — typed loading (`Vec<S>` where `S: Signature`) requires coercing DataLoader, macro-generated `.input()` extractor, and field mapping. Shares boundary with U50 typed metric surface and legacy `Evaluator`/`Example`→`Prediction` coupling. | **Post-Implementation Cleanup** (with U50) | Training data is `Vec<S>` where `S: Signature`; DataLoader produces typed examples with coercion (R11) and graceful error handling (R12); Signature macro generates `.input() -> S::Input` extractor; entire legacy `Example`/`Prediction` optimizer boundary replaced in single pass. Shaping doc: conversation record (2026-02-09). |

## Cleanup Kickoff Reference

Phase 4.5 execution planning and decision arbitration checkpoints are now tracked in:

- `docs/plans/modules/phase_4_5_cleanup_kickoff.md`

Use that doc as the active decision matrix for:
- strict typed-bound migration strategy,
- legacy-surface cutover gates,
- optimizer/evaluator contract migration boundaries,
- wrapper/combinator walker completion scope.

## Post-Implementation Cleanup Resolved Items
- `U29` (`ChainOfThought` Facet discoverability) resolved in code: `crates/dspy-rs/src/modules/chain_of_thought.rs:16`.
- `build_system` API/spec mismatch resolved by spec alignment to fallible return (`Result<String>`): `docs/specs/modules/breadboard.md:101`, `docs/specs/modules/design_reference.md:583`.

## Validation During Slice 5-6 Closure Audit
- `cargo check -p dspy-rs`
- `cargo check -p dspy-rs --examples`
- `cargo test -p dspy-rs --lib --tests`
- `cargo test -p dspy-rs --test test_named_parameters --test test_named_parameters_containers --test test_dyn_predictor_forward_untyped`
- `cargo test -p dspy-rs --test test_registry_dynamic_modules --test test_program_graph_execution --test test_program_graph_mutation --test test_program_graph_annotations --test test_program_graph_projection_fit --test test_named_parameters_ref`
- `set -a && source .env && set +a && cargo run -p dspy-rs --example 93-smoke-slice4-react-operational`
- `set -a && source .env && set +a && cargo run -p dspy-rs --example 94-smoke-slice5-optimizer-interface`
- `set -a && source .env && set +a && cargo run -p dspy-rs --example 95-smoke-slice6-dynamic-graph`

Observed smoke outputs:
- Slice 4 calculator trajectory parity pass: `tool_calls: 3`, `tool_executions: 5`, trajectory printed with `Step 1..4`, `answer: 70`.
- Slice 5 optimizer-interface pass: `named_parameters: ["predictor"]`, instruction mutation applied, `answer: smoke-ok`.
- Slice 6 dynamic-graph pass: registry-created node + input pseudo-edge execution returned `answer: smoke-ok`.
