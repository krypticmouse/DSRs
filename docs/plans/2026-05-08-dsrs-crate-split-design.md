# DSRs · Crate Split Design

**Date:** 2026-05-08
**Status:** Approved (design phase)
**Successor of:** `docs/specs/modules/design_reference.md`, `docs/specs/modules/breadboard.md`

> Companion artifact: [`2026-05-08-dsrs-crate-split-topology.html`](2026-05-08-dsrs-crate-split-topology.html) (interactive React view of the same plan).

---

## 1. Motivation

The current `crates/dspy-rs` is a monolith. Splitting it pays off on four axes simultaneously, all of which were chosen as motivators:

1. **Layer enforcement.** The breadboard's L0 / L1 / L2 + Place P1 / P2 / P3 topology is currently social, not mechanical. A user file can `use dspy_rs::optimizer::*` from a P1 codebase. Crate boundaries make the topology load-bearing.
2. **Optional features.** Today everyone pays for `parquet`, `arrow`, `hf-hub`, `foyer`, `minijinja`, `rig-core`, `csv` whether they call them or not. Splitting lets light users skip what they don't use.
3. **Compile times.** The monolith pulls every heavy dep into every rebuild. Per-crate cargo caching + parallel codegen across small crates is a strict win.
4. **Public API hygiene.** Users get narrow, named imports per area instead of one giant `dspy_rs::*`.

A fifth motivator emerged during design: **leaven readiness**. The user has [`leaven`](../../../leaven) — a separate Rust library for optimizing arbitrary artifacts — with `leaven-core` (cold algebra), `leaven-engine`, and concrete optimizers (`leaven-gepa`). The split prepares DSRs to be *a thing leaven optimizes* rather than its own optimizer host.

---

## 2. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **No facade crate.** The current `dspy-rs` aggregator is dissolved; users depend on the leaf crates explicitly. | Cleanest API hygiene. The cost is a one-time migration of import paths. |
| D2 | **12 crates total** in the workspace. | Layer-aligned (L0, L1, L2, adjacent, integration) plus the three existing support crates. Coarser splits don't enforce L1/L2; finer splits add ceremony without payoff. |
| D3 | **`DynPredictor`, `TraceSink`, `CacheBackend`, `LmClient` traits live in `dsrs-core`.** | Slight deviation from the breadboard's "L2 defines the interface" framing. With Cargo crates, putting abstract bridge traits in core gives a clean DAG with no dependency inversion gymnastics. |
| D4 | **Trace and cache are their own crates** (`dsrs-trace`, `dsrs-cache`). | Maximum modularity — each can be swapped or disabled. Both implement abstract traits in `dsrs-core`. |
| D5 | **`bamltype` + `bamltype-derive` stay as-is.** | Already external-shaped; the names signal BAML lineage; renaming buys nothing. |
| D6 | **GEPA-only optimization.** COPRO and MIPROv2 source files are deleted. | Per current direction, those optimizers are outdated. `dsrs-optimize` is renamed `dsrs-gepa` for clarity. |
| D7 | **`dsrs-evaluate` is permanent and separate.** | It's the metric surface `leaven-dsrs` adapts. Even after `dsrs-gepa` sunsets, evaluate stays as the canonical typed-metric API. |
| D8 | **`dsrs-gepa` is a sunset candidate.** | Survives until the leaven path (next decision) is real. |
| D9 | **DSRs implements leaven's capability traits directly** in a `dsrs-leaven` crate inside the DSRs workspace. The skeleton `leaven-dsrs` crate in leaven's workspace is dropped or repointed. | Aligns with leaven's own routing rule ("backend crates depend on the capability crate, not on internals"). DSRs *is* a leaven-compatible target, not a third-party project that needs a bridge owned by leaven. |
| D10 | **Zero compatibility shims.** Hard cutover for the import-path change. | Per project standard: no parallel old/new paths, no `pub use` redirects, no deprecated wrappers. |

---

## 3. Crate inventory

12 crates. Three existing, eight new (extracted from `dspy-rs`), one new integration crate.

### L0 · Foundation (existing, untouched)

| Crate | Role |
|-------|------|
| `bamltype` | Typed value system, jsonish coercion, BAML schema rendering. |
| `bamltype-derive` | `#[derive(BamlType)]` proc-macro. |
| `dsrs-macros` | `#[derive(Signature)]`, `#[derive(Augmentation)]`, `#[derive(Module)]` proc-macros. Emitted paths get rewritten to reference the new crate names. |

### L1 · Typed substrate (new, extracted from `dspy-rs`)

| Crate | Public surface | Depends on |
|-------|---------------|------------|
| `dsrs-core` | `Signature`, `Module`, `SignatureSchema`, `Augmentation`, `Predicted<O>`, `CallMetadata`, `Demo<S>`, `Example`, `Prediction`, `PredictError` / `ParseError` / `ConversionError` / `LmError`. Abstract bridge traits: `DynPredictor`, `TraceSink`, `CacheBackend`, `LmClient`. The Facet walker (`visit_named_predictors_mut`). | `bamltype` |
| `dsrs-lm` | Concrete `LM` (rig-core wrapper), `ChatAdapter`, `GLOBAL_SETTINGS`, `configure`, `with_lm`. Implements `dsrs-core::LmClient`. | `dsrs-core` |
| `dsrs-trace` | `ExecutionGraph`, `TraceContext`, span/event types. Implements `dsrs-core::TraceSink`. | `dsrs-core` |
| `dsrs-cache` | Foyer-backed LM response cache. Implements `dsrs-core::CacheBackend`. | `dsrs-core` |
| `dsrs-predict` | `Predict<S>` (impls `DynPredictor`), `ChainOfThought<S>`, `ReAct<S>`, `forward_all`, `Map` / `AndThen` combinators, library modules. | `dsrs-core`, `dsrs-lm` |

### L2 · Evaluation & optimization

| Crate | Public surface | Depends on | Status |
|-------|---------------|------------|--------|
| `dsrs-evaluate` | `TypedMetric<S, M>`, `MetricOutcome`, `FeedbackMetric`, `ExecutionTrace`, `evaluate_trainset`, feedback helpers (`retrieval_feedback`, `code_pipeline_feedback`, `multi_objective_feedback`, `string_similarity_feedback`, `classification_feedback`). | `dsrs-core` | Permanent. |
| `dsrs-gepa` | `Optimizer` trait, `GEPA`, `GEPACandidate`, `GEPAResult`, `ParetoFrontier`. | `dsrs-core`, `dsrs-predict`, `dsrs-evaluate` | **Sunset candidate.** |

COPRO and MIPROv2 source files (`optimizer/copro.rs`, `optimizer/mipro.rs`) are deleted as part of the split.

### Adjacent

| Crate | Public surface | Depends on |
|-------|---------------|------------|
| `dsrs-data` | `DataLoader`. Format readers (csv / json / parquet / hf-hub) behind feature flags so light users don't pull arrow/parquet/hf-hub. | `dsrs-core` |

### Integration · the future

| Crate | Public surface | Depends on |
|-------|---------------|------------|
| `dsrs-leaven` | `DsrsProgramArtifact` (impl `leaven_core::Artifact`), `DsrsProgramChange`, `DsrsProgramSurface` (impl `leaven_surface::EditSurface`), `DsrsEvaluator` (impl `leaven_engine::Evaluator<P>`), `DsrsEvidence` (impl `leaven_core::Evidence` + capability traits for `Casewise` and `Attributable`). | `dsrs-core`, `dsrs-evaluate`, `dsrs-predict`, `leaven-core`, `leaven-surface`, `leaven-engine`, `leaven-evidence` |

---

## 4. Dependency DAG

```
                      bamltype-derive
                              ▼
                          bamltype  ◄──── dsrs-macros
                              ▲ ▲
                              │ │
                              ▼ │
                          dsrs-core ◄── dsrs-trace
                            ▲ ▲ ▲    ◄── dsrs-cache
                            │ │ │
                            │ │ └── dsrs-evaluate ──┐
                            │ │                      │
                            │ └── dsrs-lm            │
                            │       ▲                │
                            │       │                │
                            └── dsrs-predict ────────┤
                                  ▲                  │
                                  │                  │
                            dsrs-gepa  ◄─────────────┘   (sunset)

                            dsrs-data ──► dsrs-core

                            dsrs-leaven ──► dsrs-core
                                          ► dsrs-evaluate
                                          ► dsrs-predict
                                          ► leaven-{core, surface, engine, evidence}
```

Cargo-enforced invariants:

- **`dsrs-core` is foundational and small.** No LM, no rig-core, no foyer, no parquet, no minijinja. Pure types + traits + Facet walker + abstract bridges.
- **`dsrs-trace` and `dsrs-cache` only depend on `dsrs-core`.** They're swap-points; concrete impls don't infect anything else.
- **`dsrs-predict` depends on `dsrs-core` + `dsrs-lm`.** You cannot construct a Predict without an LM client. This is right.
- **`dsrs-evaluate` depends only on `dsrs-core`.** Metrics are pure over typed I/O — no Predict, no LM, no optimizer.
- **`dsrs-gepa` is a leaf consumer.** Nothing depends on it. Deletion is safe.
- **`dsrs-leaven` is the only crate that pulls leaven types into the DSRs workspace.** Users who don't use leaven don't pay.

---

## 5. The leaven integration story

DSRs becomes a leaven-compatible optimization target. The user retains a typed `M: Module<...>` instance and hands it to leaven. Leaven owns the optimization run loop; DSRs owns module evaluation and the prompt format.

Concrete shape of `dsrs-leaven`:

```rust
// Wraps a typed module + signature; identity is content-hash of (instructions, demos)
// across all Predict leaves.
pub struct DsrsProgramArtifact<S: Signature, M: Module<...>> { ... }
impl<S, M> leaven_core::Artifact for DsrsProgramArtifact<S, M> { type Change = DsrsProgramChange; ... }

// Structured edit: (predict_path, op) where op is set-instruction or set-demos.
pub struct DsrsProgramChange { edits: Vec<(PredictPath, Edit)> }

// EditSurface — lets leaven proposers select Predict leaves by address and
// render them either inline (for one-shot LM proposers) or as a workspace
// directory (for agentic proposers that want selective read access).
pub struct DsrsProgramSurface;
impl leaven_surface::EditSurface for DsrsProgramSurface { ... }

// Evaluator — runs the user's typed metric against a batch of examples through
// a (snapshot of the) module and produces leaven Assessments.
pub struct DsrsEvaluator<S, M, MT: TypedMetric<S, M>> { ... }
impl<P> leaven_engine::Evaluator<P> for DsrsEvaluator<...> { ... }

// Evidence — wraps DSRs's MetricOutcome (scalar score + optional textual
// FeedbackMetric + metadata). Implements Casewise (per-example feedback for
// Pareto) and Attributable (which Predict caused which signal — for credit
// assignment).
pub enum DsrsEvidence { ... }
impl leaven_core::Evidence for DsrsEvidence {}
impl leaven_evidence::CasewiseEvidence for DsrsEvidence { ... }
impl leaven_evidence::AttributableEvidence<PredictPath> for DsrsEvidence { ... }
```

**Render/materialize separation** (per leaven principle 3.2): `ChatAdapter` stays internal to DSRs. Leaven only sees "call this module with this input → get a `Predicted` and its metadata." Prompt rendering happens inside `dsrs-predict` when the module executes; leaven never observes the prompt format.

**Where state lives:** Hybrid. The user's original `M: Module` is mutable through `DynPredictor`. Leaven proposes changes that produce *new* `DsrsProgramArtifact` snapshots via `apply_change`, and the run graph carries those snapshots for lineage and caching.

**What leaven currently lacks** (research from sub-agent investigation, 2026-05-08):
- `leaven-dsrs` crate (in leaven workspace) — empty stubs at v0.0.0
- `leaven-mipro` — skeleton (not needed for our GEPA-only path)
- `leaven-textgrad` — skeleton (feedback aggregation needed by GEPA)
- `leaven-gepa` — partial: strategy composition layer (CandidateSelector / PartSelector / Gate slots) but no runnable optimizer, no reflection-based mutation wired
- No ergonomic `optimize(artifact, proposer, evaluator, population) -> ...` entry point

These are what need to land in leaven before `dsrs-gepa` can be deleted.

---

## 6. Sunset trigger for `dsrs-gepa`

`dsrs-gepa` is deleted from the workspace when **both** are true:

1. **`leaven-gepa` is a runnable optimizer.** Strategy slots are filled, reflection-based mutation works, candidate selection / gate / parts-picker are wired. Not a strategy-composition skeleton.
2. **`dsrs-leaven` ships real implementations** of `DsrsProgramArtifact` / `DsrsProgramSurface` / `DsrsEvaluator` / `DsrsEvidence`, and a parity test confirms equal-or-better optimization results vs `dsrs-gepa` on a sample DSRs program (e.g. one of the `examples/` programs).

Originally six conditions covering MIPRO, COPRO, textgrad, etc — collapsed to two because GEPA is the only optimizer we keep.

---

## 7. Migration sequence

The split is a hard cutover. Drafted as one PR per crate extraction, in dependency order so each step compiles and tests pass.

1. **Create `dsrs-core`.** Move `core/`, `augmentation.rs`, the legacy `Example` / `Prediction` types from `data/`, the bridge trait stubs, and the Facet walker. Update `dsrs-macros` emitted paths. Verify `cargo test -p dsrs-core` and downstream.
2. **Create `dsrs-trace`.** Move `trace/`. Verify it depends only on `dsrs-core`.
3. **Create `dsrs-cache`.** Move `utils/cache.rs` (and `telemetry.rs` if it's only used here). Verify dep on `dsrs-core` only.
4. **Create `dsrs-lm`.** Move `core/lm/`, `adapter/`, `core/settings.rs`. Implements `dsrs-core::LmClient`. Wire `dsrs-trace` and `dsrs-cache` via core's traits, not concrete deps.
5. **Create `dsrs-evaluate`.** Move `evaluate/`. Verify dep on `dsrs-core` only.
6. **Create `dsrs-predict`.** Move `predictors/`, `modules/`. Update `Predict<S>`'s `impl DynPredictor` to use the trait from `dsrs-core`.
7. **Create `dsrs-gepa`.** Move `optimizer/gepa.rs` and `optimizer/pareto.rs`. **Delete `optimizer/copro.rs` and `optimizer/mipro.rs`** along with any tests that target them.
8. **Create `dsrs-data`.** Move `data/dataloader.rs`, `data/serialize.rs`, `data/utils.rs`. Add format feature flags (`csv`, `parquet`, `hf-hub`).
9. **Create `dsrs-leaven`.** Initial skeleton — type signatures only, `unimplemented!()` bodies. Real implementations land in a follow-up plan once the first leaven-side piece is ready.
10. **Delete `crates/dspy-rs`.** Remove from workspace `Cargo.toml`. Update `README.md`, `CURRENT_PLAN.md`, `CURRENT_SPEC.md`, doc references.
11. **Update consumers.** `examples/`, `tests/` outside crates, vendor dirs, anything that does `use dspy_rs::*`.
12. **In leaven workspace** (separate PR there): delete `crates/leaven-dsrs/` or repoint as a thin re-export pointer to DSRs's `dsrs-leaven`.

Tests pass after each step. No step leaves the workspace in a non-compiling state.

---

## 8. Open questions deferred to implementation plan

- **Feature flag granularity in `dsrs-data`.** Default features = none vs default = `csv`+`json`? Probably default to none and document the four feature combos.
- **`dsrs-leaven` initial scope.** The first cut is type skeletons; what's the first end-to-end smoke test? Probably "GEPA-equivalent run on the QA example using leaven-gepa once it's a real optimizer."
- **MSRV alignment with leaven** (`rust-version = 1.85` in leaven's workspace). Make `dsrs-leaven` match.
- **Does `dsrs-macros` need feature flags** to emit different paths depending on whether you're targeting the new crate layout? Likely no — hard cutover, paths are unconditional.

---

## 9. References

- `docs/specs/modules/breadboard.md` — original L0 / L1 / L2 + P1 / P2 / P3 topology
- `docs/specs/modules/design_reference.md` — design principles (Facet shapes, parse-don't-validate, structure-IS-declaration, modules-as-strategies, typed-path-primary, one-adapter)
- `CURRENT_SPEC.md` — superseded baseline (Phase 2 typed-native runtime)
- [`leaven/AGENTS.md`](../../../leaven/AGENTS.md) — routing rules and the "backends depend on capability crates" principle
- [`leaven/docs/specs/guiding_principles.md`](../../../leaven/docs/specs/guiding_principles.md) — artifact-shape neutrality, render-materialize separation, evidence-shape neutrality
- [`2026-05-08-dsrs-crate-split-topology.html`](2026-05-08-dsrs-crate-split-topology.html) — interactive React view of this design
