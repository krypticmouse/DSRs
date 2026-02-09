# Slice 1 Refinery Critique

## Spec Fidelity
- The current plan now tracks the primary requirements (R0–R16) from `docs/specs/modules/shapes.md` via the schema/call-outcome rewrites, but it still leaves two speculative gaps: alias/constraint behavior for flattened fields and the `Try` ergonomics around `CallOutcome`. I flagged both with exact arbitration comments because the spike docs (`S1-...`, `S8-...`, `splice S2 etc.`) advocate for concrete rules before locking any builder logic or trait changes.
- The plan otherwise aligns with `docs/specs/modules/breadboard.md` and `docs/specs/modules/design_reference.md` on the default return surface being `CallOutcome<O>`, the single surface for `Module::forward`, and the caching/instruction invariants (U9–U10, N8, F1–F7). No additional specs were violated.

## Shape Compliance
- Shape F2 demands a Facet-derived `SignatureSchema` with flattened paths and cache sharing; the plan now explicitly calls out a TypeId-keyed cache (with a per-`S` `OnceLock`), matching the breadboard's S1 invariant that schema state is immutable and accessible from all places.
- Generic signature derive (F12) and augmentation semantics (F3) are respected by storing demos as `Vec<Demo<S>>` and by insisting the macro emit `input_shape`/`output_shape`. However, the plan still needs to confirm whether flattened aliases keep parent path prefixes; I captured that as an arbitration marker so engineering doesn't proceed with an assumption that might break `FieldPath` invariants from `S8`.

## Breadboard Consistency (Places/Affordances/Boundaries)
- P1 affordances (`U1`–`U10`) remain intact because the plan retains the `Predict` builder/adapter path while clearly differentiating typed vs. legacy consumers through the note on re-exported legacy modules. The plan makes no modifications to P2–P4 flows yet, so the Place boundaries still match the breadboard descriptions.
- The `CallOutcome` single calling convention is reinforced per the locked decision. No new affordances (e.g., extra `forward_result` helpers) were introduced, keeping the plan within the breadboard's cognitive boundary for P1.

## Sequencing & Hidden Dependencies
- The execution order already codifies the correct sequence (schema/outcome → macro → adapter → Predict/tests). I added notes explaining the TypeId cache requirement and the demo reshaping, making those dependencies explicit.
- Hidden dependency: the `Try` implementation depends on the stable toolchain supporting `try_trait_v2`. Instead of assuming it works, the plan now explicitly requests arbitration before finalizing the `Try` integration, ensuring sequencing won't fail mid-implementation.

## API Design Consistency
- Aligning `Module::forward` and `DynPredictor::forward_untyped` on `CallOutcome` keeps the API surface uniform. Refs to `SignatureSchema::of::<S>()` and `CallOutcome` in the adapter section replay the design reference narrative and keep the API consistent with the typed path.
- The plan also makes explicit that demos will live as typed `Demo<S>` pairs, which both respects the new augmentation strategy and avoids future API mutations (no more `S::from_parts`).

## Over-engineering
- No new over-engineered abstractions were introduced. The cache change is a simple map-per-type to avoid a known bug (S1). The plan deliberately defers optimizer/augmentation rewrites (Slice 2+ work), so Slice 1 remains lean and focused on the typed path.
- I kept the legacy `CallResult`/`MetaSignature` shims in the plan but clearly marked them as deprecated, so their inclusion is a compatibility layer rather than over-engineering.
