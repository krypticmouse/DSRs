### Spec fidelity — Pass
- The plan hits the breadboard must-haves (U14 ReAct builder + tools, U48 `forward_all` as a standalone utility, U51 `.map`/`.and_then` output combinators) and records the surface-level behaviors called out in `/Users/darin/src/personal/DSRs/docs/specs/modules/breadboard.md:40-110` and the V4 table at `:410-430`.
- Remaining ambiguity: the spec doesn’t say whether `.and_then` should re-run the tool loop or simply replay the returned `CallOutcome`, so I flagged that assumption for arbitration before code lands.

### Shape compliance — Pass
- ModuleExt wrappers still implement `Facet` so the walker can reach `inner` predictors, and the new `ReAct` module derives `Facet` while attaching `ActionStep<S>`/`ExtractStep<S>` signatures built via the generic/flatten-friendly derive path (F12, S1, S8 from `/Users/darin/src/personal/DSRs/docs/specs/modules/shapes.md:60-90`).
- Using `Signature`-derived structs for those steps keeps the type boundaries aligned with the design reference’s typed modules (F4/F11) and ensures `CallOutcome` metadata stays the single return surface (see `/Users/darin/src/personal/DSRs/docs/specs/modules/design_reference.md:360-460`).

### Breadboard consistency — Pass
- ReAct stays in P1/Layer 1, `forward_all` remains a free utility rather than a new Module method, and ModuleExt provides the P1 ramp without hiding inner modules behind trait objects, exactly as the affordance notes demand (`docs/specs/modules/breadboard.md:40-80`, `:90-110`).
- The added `#[facet(skip)]` placeholder plus manual `Facet` impls keep the walker’s path namespace stable, satisfying the requirement that combinators expose their `inner` fields (U51 + N18 in the same document).

### Sequencing — Pass
- The steps follow a sensible order: tighten `forward_all`, add the combination helpers, then build the ReAct module that relies on those foundations, and finally add regression tests. Nothing in the plan introduces hidden dependencies needing reordering.

### API design — Pass
- Every new API matches existing patterns: `forward_all(module, inputs, concurrency)` mirrors the spec’s call, `ModuleExt::map/and_then` return wrapper modules that preserve `CallOutcome` metadata, and the ReAct builder exposes `.tool(name, desc, async_fn)` plus typed action/extract stages derived from `Signature` (see `/Users/darin/src/personal/DSRs/docs/specs/modules/design_reference.md:919-940`). The only open question is how the `.and_then` metadata should merge inside the ReAct loop, which is already marked for arbitration.

### Over-engineering — Pass
- Nothing extra is being built beyond the spec’s deliverables; the tests explicitly exercise the three new surfaces so we can fix runtime behavior ahead of coding. No additional layers or extraneous APIs are introduced.
