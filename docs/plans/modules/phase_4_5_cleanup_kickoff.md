# Phase 4.5-lite: Prerequisite Cleanup

Date: 2026-02-09
Status: Completed (executed 2026-02-10)
Revised: 2026-02-09 (descoped from full 4.5 to prerequisites-only)

## Revision History

Original Phase 4.5 planned a full cleanup cycle with legacy quarantine, compatibility
wrappers, staged deletion gates, and optimizer ABI prep work. After reviewing the
actual code against the breadboard/design reference, the conclusion was:

**Most of Phase 4.5's planned work is either V5 in disguise (C3 optimizer ABI, C4
evaluator adapter) or unnecessary intermediate scaffolding (C2 quarantine) that V5
replaces.** Building compatibility wrappers for a system you're about to replace is
waste.

Revised plan: do only the hard prerequisites for V5, build V5 and V6, then do one
kill pass to delete all legacy surfaces in a single sweep.

## Revised Execution Roadmap

| Phase | Scope | What Dies |
|---|---|---|
| **4.5-lite** (this doc) | V5 prerequisites only: bounds, annotations, macro naming | Nothing deleted yet |
| **V5** (Slice 5) | F6 walker + F8 DynPredictor + typed optimizer/evaluator | Legacy system becomes fully replaceable |
| **V6** (Slice 6) | F9 DynModule + F10 ProgramGraph + registry | Dynamic graph realized |
| **Kill Pass** (post-V6) | Delete all legacy surfaces, rewrite examples | `MetaSignature`, `LegacyPredict`, `Optimizable`, `#[LegacySignature]`, `#[parameter]`, `Example`/`Prediction` coupling, examples 01-10 |

Rationale: the legacy system is ugly but stable. It compiles, passes tests, and
doesn't block V5/V6 development. Every intermediate cleanup pass (quarantine,
compatibility wrappers, staged deletion) is throwaway work if V5 provides the actual
replacement. Build the replacement first, then delete the old thing in one sweep.

## Ground Truth

Primary references:
- `docs/plans/modules/tracker.md`
- `docs/plans/modules/slices_closure_audit.md`
- `docs/specs/modules/breadboard.md`
- `docs/specs/modules/shapes.md`
- `docs/specs/modules/design_reference.md`

## Locked Decisions (Do Not Re-open)

1. **S1/S6 direction is Option C full replacement**: Facet-native `SignatureSchema` is the target; legacy `FieldSpec`/`MetaSignature` are transitional only.
2. **Single call surface**: `Module::call` returns `Result<Predicted<O>, PredictError>`. `Predicted<O>` carries output + metadata with `Deref<Target = O>`. `forward` remains as a compatibility hook/alias for implementers. Revision brief: `docs/specs/modules/calling_convention_revision.md`.
3. **Typed path is primary; dynamic is escape hatch**: user-facing APIs should optimize for typed modules first.
4. **`build_system` fallibility is intentional**: spec/docs were aligned to `Result<String>`.
5. **Post-slice reconciliations already completed**: `ChainOfThought` Facet discoverability and ReAct parity fixes are done.

## C1-C8 Arbitration Outcomes

All eight decision checkpoints from the original kickoff are resolved below.
Three are in scope for 4.5-lite. Five are rerouted.

| ID | Decision | Resolution | Phase |
|---|---|---|---|
| **C1** | `Module` trait bounds | **Accept A (hard tighten now).** No compatibility wrappers — the legacy `Module<Input=Example, Output=Prediction>` impls stay on the old types until the kill pass deletes them. New code uses tight bounds. Wrappers were only needed if we were doing staged migration; we're not. | **4.5-lite** |
| **C2** | Legacy surface cutover | **Skip quarantine. Defer to kill pass.** Legacy surfaces (`MetaSignature`, `LegacyPredict`, `Optimizable`) remain untouched until V5 provides complete replacement. No intermediate quarantine module, no feature gates. Straight deletion after V5+V6. | **Kill Pass** |
| **C3** | Optimizer core contract | **This IS V5.** Migrating optimizer ABI from `Example/Prediction` to `DynPredictor`/`SignatureSchema` is the V5 slice definition. Not prep work. | **V5** |
| **C4** | Evaluator/feedback substrate | **This IS V5.** Typed evaluator surface replaces `Evaluator: Module<Input=Example, Output=Prediction>`. | **V5** |
| **C5** | F12 helper generics + `__phantom` | **Accept B (redesign).** Fix macro to generate `QAOutput` not `__QAOutput`. Hide phantom from struct literal construction. | **4.5-lite** |
| **C6** | Wrapper/combinator discoverability | **Accept B (full matrix).** Fix `#[facet(opaque)]` to transparent on predictor fields. Add walker traversal tests for shipped combinators. This is a hard V5 prerequisite. | **4.5-lite** |
| **C7** | Container traversal | **Accept A (defer).** Add error-path contract tests during V5 when the walker exists to test against. No point writing walker tests before the walker. | **V5** |
| **C8** | Typed->graph edge derivation | **Accept B (lock strategy).** Annotation-first with optional trace inference later. Record as design note in V6 planning. | **V6 planning** |

## Phase 4.5-lite Scope (Three Items)

### Item 1: Module Trait Bounds (C1)

Tighten `Module` associated type bounds from:
```rust
pub trait Module: Send + Sync {
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;
```
To:
```rust
pub trait Module: Send + Sync {
    type Input: BamlType + for<'a> Facet<'a> + Send + Sync;
    type Output: BamlType + for<'a> Facet<'a> + Send + Sync;
```

Impact:
- Existing typed modules (`Predict<S>`, `ChainOfThought<S>`, `ReAct<S>`, `Map`, `AndThen`) already satisfy these bounds. No changes needed.
- Legacy impls that use `Module<Input=Example, Output=Prediction>` will break IF `Example`/`Prediction` don't implement `BamlType + Facet`. Two options:
  - (a) Add `BamlType + Facet` derives to `Example`/`Prediction` temporarily so legacy compiles. Kill pass removes them.
  - (b) Remove `Module` impl from legacy types now; legacy flows use `Predictor` trait (which they already do).
- Decision on (a) vs (b) during implementation based on blast radius.

Files to touch:
- `crates/dspy-rs/src/core/module.rs` (trait definition)
- `crates/dspy-rs/src/core/module_ext.rs` (Map/AndThen bounds propagation)
- Any examples that `impl Module` with untyped I/O

### Item 2: Macro Naming and Phantom Cleanup (C5)

Current problems:
- Generated output type is `__QAOutput` (double-underscore prefix leaks into user code)
- Generic signatures require `__phantom: std::marker::PhantomData` in struct literals
- Type alias workarounds like `type ReActActionStepOutput = __ReActActionStepOutput;`

Target:
- Generated types use clean names: `QAOutput`, `QAInput`
- Phantom field is either hidden from construction (builder/`Default`) or eliminated
- No double-underscore types in public API or example code

Files to touch:
- `crates/dsrs-macros/src/lib.rs` (macro code generation)
- `crates/dsrs-macros/tests/signature_derive.rs` (macro tests)
- `crates/dspy-rs/src/modules/react.rs` (type aliases, phantom construction)
- `crates/dspy-rs/examples/01-simple.rs` (`__QAOutput` references)
- Any other files referencing `__`-prefixed generated types

### Item 3: Facet Annotation Fixes for Walker Transparency (C6)

Current problems:
- `ChainOfThought.predictor` is `#[facet(opaque)]` — walker can't see the Predict inside
- `ReAct.action` and `ReAct.extract` are `#[facet(opaque)]` — same problem
- No tests verify walker traversal through wrappers

Target:
- Predictor fields on library modules use the correct Facet annotation for walker transparency
- Walker traversal tests cover: `ChainOfThought<S>`, `ReAct<S>`, `Map<M, F>`, `AndThen<M, F>`
- Tests verify correct dotted-path output (e.g. `predictor` for CoT, `action` + `extract` for ReAct, `inner.predictor` for `Map<ChainOfThought<S>>`)

Note: the full F6 walker runtime ships in V5. These tests may use a minimal test walker or verify Facet shape metadata directly, depending on what infrastructure exists. The point is that the annotations are correct so V5 doesn't immediately break.

Files to touch:
- `crates/dspy-rs/src/modules/chain_of_thought.rs` (annotation fix)
- `crates/dspy-rs/src/modules/react.rs` (annotation fix)
- `crates/dspy-rs/tests/` (new walker/shape traversal tests)

## Exit Gates (Phase 4.5-lite)

1. **Bounds gate**: `Module` trait requires `BamlType + Facet` on associated types.
2. **Naming gate**: No `__`-prefixed types in public API, examples, or tests. No user-facing `PhantomData` initialization.
3. **Annotation gate**: All predictor fields on shipped library modules use walker-transparent Facet annotations. Shape traversal tests pass.
4. **Regression gate**: `cargo check -p dspy-rs && cargo check -p dspy-rs --examples && cargo test`
5. **Smoke gate**: Examples 90-93 still pass.
6. **Legacy untouched gate**: `MetaSignature`, `LegacyPredict`, `Optimizable`, evaluator traits, and legacy examples are not modified (they'll be dealt with in the kill pass).

## What Happens to Legacy During V5/V6

The legacy system stays in the codebase untouched. It compiles, it passes its tests, it works for the optimizer examples. It is not canonical — the 90-93 smoke examples are the reference for how the API should look.

During V5/V6 development:
- Do NOT use examples 01-10 as reference. Use 90-93.
- Do NOT add new `MetaSignature`/`Optimizable` impls on new code.
- Do NOT extend the legacy evaluator trait surface.
- The legacy system is frozen. No new features, no fixes, no attention.

## Kill Pass Checklist (Post-V6)

Kept here for future reference. Execute after V5 + V6 are complete.

- [ ] `MetaSignature` trait: zero-reference check, delete
- [ ] `LegacyPredict` struct: zero-reference check, delete
- [ ] `Optimizable` trait + `#[derive(Optimizable)]`: zero-reference check, delete
- [ ] `#[LegacySignature]` proc macro: zero-reference check, delete
- [ ] `#[parameter]` attribute: zero-reference check, delete
- [ ] `Predictor` trait (legacy forward path): zero-reference check, delete
- [ ] `Evaluator: Module<Input=Example, Output=Prediction>` constraint: replaced by V5 typed surface
- [ ] `FeedbackEvaluator` / `ExecutionTrace` Example/Prediction coupling: replaced by V5
- [ ] Triple-impl blocks on `Predict` (`Module` + `MetaSignature` + `Optimizable`): reduce to `Module` only
- [ ] Triple-impl blocks on `ChainOfThought`: same
- [ ] Examples 01-10: rewrite against typed path or delete
- [ ] `Example` / `Prediction` types: evaluate if anything remains; delete or move behind legacy feature
- [ ] Container traversal error-path contract tests (C7)
- [ ] Graph edge derivation strategy doc (C8)
- [ ] Final: `cargo check -p dspy-rs && cargo check -p dspy-rs --examples && cargo test`
- [ ] Final: all 90-93+ smoke examples pass
- [ ] Update `tracker.md` and `slices_closure_audit.md`

## Superseded Sections (Historical Reference)

The following content from the original Phase 4.5 kickoff is preserved for context
but is no longer the active plan.

<details>
<summary>Original Phase 4.5 scope (superseded)</summary>

### Original Scope Guardrails

Phase 4.5 was originally scoped as a full API cleanup and contract-hardening phase
including legacy quarantine, compatibility wrappers, staged deletion gates, and
optimizer ABI prep work. This was descoped because:

1. C3 (optimizer ABI migration) and C4 (evaluator adapter) are V5 feature work, not cleanup.
2. C2 (legacy quarantine) creates intermediate scaffolding that V5 replaces entirely.
3. The compatibility wrappers needed for staged C1 migration are unnecessary if legacy
   impls are left untouched until the kill pass.

### Original Execution Order (Superseded)

- Stage A: Contract Freeze (resolve C1-C8)
- Stage B: API Surface Cleanup (bounds + macro)
- Stage C: Legacy Quarantine and Cutover (quarantine + optimizer migration)
- Stage D: Walker and Discoverability Hardening (wrapper coverage)

Replaced by: 4.5-lite (prerequisites) -> V5 -> V6 -> Kill Pass.

### Original Confusion Points (Resolved)

1. "Where optimizer migration stops in Phase 4.5" — Answer: it doesn't start. It's V5.
2. "Evaluator/feedback ownership" — Answer: V5 replaces the evaluator contract.
3. "`Module` bound tightening blast radius" — Answer: hard tighten; legacy impls stay on old types until kill pass.
4. "Legacy deletion gate" — Answer: deletion happens in one sweep after V5+V6, not staged.
5. "Combinator walker guarantees" — Answer: annotation fixes in 4.5-lite; walker runtime in V5.

</details>
