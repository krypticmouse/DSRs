# Slice 6 Plan Refinery (Ground-Truth Check)

Verified against:
- `/Users/darin/src/personal/DSRs/docs/specs/modules/breadboard.md` (including V6)
- `/Users/darin/src/personal/DSRs/docs/specs/modules/shapes.md`
- `/Users/darin/src/personal/DSRs/docs/specs/modules/dspy_module_system_reference/`
- `/Users/darin/src/personal/DSRs/docs/specs/modules/design_reference.md`
- `/Users/darin/src/personal/DSRs/docs/specs/modules/calling_convention_revision.md`
- `/Users/darin/src/personal/DSRs/docs/specs/modules/spikes/S2-dynpredictor-handle-discovery.md`
- `/Users/darin/src/personal/DSRs/docs/specs/modules/spikes/S5-facet-walker-containers.md`
- `/Users/darin/src/personal/DSRs/docs/specs/modules/spikes/S8-facet-flatten-metadata.md`

## Per-Criterion Findings

### 1. Spec fidelity: **FAIL**
- The plan now covers core V6 obligations (F9/F10): registry/factories, graph mutation/validation/execution, typed projection, and explicit snapshot-then-fit-back (`from_module` + `fit`).
- Calling-convention alignment is preserved (`Result<Predicted<...>, PredictError>` for dynamic forward surfaces).
- Remaining mismatch vs ground truth: S2's preferred mechanism is shape-local Facet attr payload decoding for accessor handles; current plan intentionally keeps the global accessor registry bridge and records it as migration debt.
- Remaining mismatch vs C8 implementation detail: edge-annotation storage mechanism (shape-local attrs vs global registry) is not resolved; the plan now marks this explicitly for arbitration.

### 2. Shape compliance: **FAIL**
- Good: `DynModule`, `StrategyFactory`, `ProgramGraph`, and edge-validation surfaces match Shape F9/F10 intent.
- Good: `SignatureSchema::from_parts` was tightened to `pub(crate)` to avoid reopening public manual schema authoring, which helps R3/R9 boundaries.
- Blocking uncertainty remains on two shape-level mechanisms:
  - Predictor accessor extraction path (S2 mechanism A vs registry bridge).
  - Edge-annotation storage location for C8.

### 3. Breadboard consistency: **PASS**
- The plan now explicitly respects owner-resolved lock semantics:
  - Immutable projection: `ProgramGraph::from_module(&module)`.
  - Explicit mutable application: `graph.fit(&mut module)`.
- C8 lock is explicit: annotation-first edge derivation, no trace inference in V6.
- Added `insert_between` coverage so F10 mutation affordances are not under-specified.

### 4. Sequencing: **PASS**
- Original hidden dependency (dynamic factories needing untyped adapter helpers) is now addressed with explicit execution ordering: implement adapter helper step before factory implementations.
- Remaining step order is coherent: discovery/schema → dynamic trait layer → adapter untyped helpers → factory modules → graph → exports → tests.

### 5. API design: **PASS**
- Public dynamic registry/factory APIs now use typed errors (`StrategyError`) instead of unscoped `anyhow::Result`.
- Added missing `format_output_baml(...)` helper for dynamic demo formatting parity, which completes the adapter-building-block surface for untyped execution.
- Parity tests now include both identity strategy (`predict`) and transformed strategy (`chain_of_thought`).

### 6. Over-engineering: **PASS**
- Scope stays focused on V6: only `predict`, `chain_of_thought`, and `react` factories are planned.
- Public migration scaffolding was reduced (`from_parts` no longer public).
- Shortcuts that intentionally diverge from end-state spec are called out as migration debt instead of hidden complexity.

## Arbitration Required Before Coding
- Resolved: keep the global accessor registry bridge for V6; defer shape-local Facet attr payload decoding migration to post-implementation cleanup debt.
- Resolved: use the global edge-annotation registry keyed by shape ID in V6 as the single annotation source; defer shape-local annotation storage migration to cleanup debt.
