# Slice 2 Plan Refinery

## Spec fidelity (V2 only)
- The shaping docs charge Slice 2 with delivering F3 (augmentation combinators) and F11 (ChainOfThought) on top of the existing Slice 1 surface without re-opening earlier work. The current plan partly addresses this, but the augmentation section still treats `Augmented` as an `Augmentation` implementation target rather than the Signature-level combinator described in `docs/specs/modules/design_reference.md` §4 and `docs/specs/modules/shapes.md` (F3). In practice that means the plan risks hardwiring augmentation metadata in the wrong module and leaving the signature combinator under-specified. The critique below updates the plan accordingly.
- The ChainOfThought module is described at the right level (predictor wrapper + builder + re-export) but needs to explicitly reuse `PredictBuilder::<Augmented<_, Reasoning>>` so that the API matches the breadboard affordances (slice narrative at `docs/specs/modules/breadboard.md`, U13/U16). Otherwise type-switching becomes more invasive than intended.

## Shape compliance (F3 + F11)
- F3 requires an augmentation derive that emits a wrapper type with `#[flatten]` and `Deref`, generating FieldPaths before the wrapped output when `prepend` is used. The plan already describes building `WithReasoning` and flattening metadata, but the Augmentation trait description must drop the `DerefMut` requirement and clarify that `Augmented<S, A>` implements `Signature` (not `Augmentation`) so the combinator stays at the type level. That matches the Shape doc's resolution of S3 and S7.
- F11's ChainOfThought needs to be discoverable as a Module, keep the same `CallOutcome` metadata, and fit into the `Strategy swap` affordance (U16) by letting the user swap `Predict<QA>` for `ChainOfThought<QA>` with minimal plumbing. The plan already mirrors `Predict` in structure, but call/delegate logic should be spelled out to avoid divergence.

## Breadboard consistency (U12/U13/U16/U17-20/U28/U29/N14)
- U12/U17–20 insist that the augmented reasoning field behaves like any other output field via Deref after `#[derive(Augmentation)]`. Removing `DerefMut` and ensuring the macro emits the wrapper and `Augmentation` implementation keeps the user-facing behavior aligned with the breadboard narrative.
- U13/U16 highlight the `ChainOfThought` builder and strategy swap. The plan must keep the `ChainOfThought` constructor shape matching `PredictBuilder` (so demos/instructions/tools still look familiar) and call `WithReasoning` outputs through the same `CallOutcome` plumbing (N14 + U28/U29). The plan touches these points but should explicitly link them in the implementation steps.

## Sequencing and dependencies
- Step order is reasonable: signature cleanup → augmentation/macro → predictor/adapters → ChainOfThought → tests → docs. It respects the dependency chain, but the plan should remind readers that removing `Signature::from_parts/into_parts` must happen before the permutation of `Demo` helpers is deleted (`Step 1` already covers this). Adding a short cross-reference will keep readers from accidentally reintroducing the old helpers later.

## API design consistency with current code
- The plan commits to keeping `CallOutcome`, metadata, and `ChatAdapter` helper semantics unchanged, which matches the requirement from the breadboard (U9–U10) and the design reference (same adapter/prompt format). The new `ChainOfThought` module should remain a simple wrapper so existing call sites only need to change a type annotation and the `call`/`forward` pattern stays consistent.
- On augmentation, the new module must expose `Augmentation`, `Augmented`, and the concrete `WithReasoning` wrapper in a public API (per design reference §4). The plan already re-exports these from the top-level crate, so remaining work is to ensure the combinator shape is documented and `WithReasoning<O>` exposes helper accessors for the reasoning fields (i.e., provide `fn reasoning(&self)` even before relying on Deref).

## Over-engineering
- The plan currently adds a dedicated test for `ChainOfThought` swapping, flatten round-trips, and Deref ergonomics. Those are high-signal verification points for Slice 2, so they are appropriate; the risk of over-engineering would arise if we added unrelated tooling or extra layering beyond the spec, which the plan avoids.
- The only quibble is the `augmentation.rs` snippet, which tries to make `Augmented` also implement `Augmentation`. That duplication increases cognitive load for reviewers without changing behavior. Reframing `Augmented` as the signature combinator eliminates the extra indirection.

## Arbitration outcomes
- Resolved in arbitration (2026-02-09): `ChainOfThoughtBuilder` exposes the delegated `PredictBuilder` DSL (`demo`, `with_demos`, `instruction`, `add_tool`, `with_tools`) in addition to `ChainOfThought::new()` so both ergonomic entry points are supported without API bifurcation.
