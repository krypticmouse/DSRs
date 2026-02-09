# Slice 3 Refinery: Module authoring drilling

## Sources
- `docs/specs/modules/breadboard.md`
- `docs/specs/modules/shapes.md`
- `docs/specs/modules/design_reference.md`

## 1. Spec fidelity
- The shaping & design specs (F1/F2/F3/F4/F12) require that signatures, schemas, and modules share a single Facet-driven metadata stack rather than the legacy `FieldMetadataSpec` arrays. The current plan explicitly rewrites the macro and schema builder to walk `Facet::schema()` data (see `Ordered edit steps` 1‑2) but the text never states that the old helpers must be deleted; callouts to `Facet::reflect::<T>()` should highlight that `FieldMetadataSpec` can be retired in this slice, otherwise there is risk of lingering dual schemas.
- Breadboard P1/P2/U51 expectations specify that module combinators exist as a P1 ramp without leaking `impl Module` complexity. The plan touches the new trait and tests modules, but it does not mention re-validating `Map<M, F>`/`.and_then()` combinators or the Facet transparency they require (see `Breadboard` discussion around `N18` errors). Add an explicit step or test so the P1→P2 ramp stays wired.
- The design reference (F5/F6/F8) insists that any `Predict` leaf registers `dsrs::parameter` metadata so the optimizer walker can find it. Step 3 only mentions keeping helper `batch` functions in the module crate; please state that `Predict` continues carrying the attribute (and that the discovery payload is unchanged) so front-line schema tests confirm optimizer bridges stay intact.

## 2. Shape compliance (F4/F12)
- `Module` now matches the spec signature: `async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output>` with `Input/Output: BamlType + Facet + Send + Sync + 'static`. The plan uses the same trait signature in section 1, so compliance is satisfied as long as `CallOutcome` remains the sole return surface (no parallel `Result`, no `forward_result`).
- The `Signature` derive rewrite fights the F12 requirement for generic/flattened outputs. The plan already calls for testing the generated `FieldPath`s and LM names, but make sure the new derive also publishes `Facet` metadata that includes the `FieldPath` primer (see `design_reference`, F2) for downstream adapter helpers.

## 3. Breadboard consistency (places/affordances/boundaries)
- Place P1 (module consumers) and P2 (module authors) appear throughout the plan: summary points 1‑3 focus on typed calling, schema helpers, and adapter building blocks, which align with U1‑U9/U48/U51. The plan currently lacks any mention of P3/P4 affordances (optimizer or graph), which is fine for this slice, but capture in the refinement notes that the optimizer bridge (`N18`/`S2`) and program graph (`F9`/`F10`) need zero regression as downstream steps. This helps keep the breadboard boundary map alive.

## 4. Sequencing sanity & hidden dependencies
- The ordered steps go macro→schema→trait→adapter→implementations→tests, which respects the dependency graph (schema must exist before the trait, the trait must exist before consumers, tests last). Ensure that step 2 explicitly establishes the `SignatureSchema` cache (`TypeId` → `'static`) before step 3 runs so `schema::of::<S>()` can be a static faster path that all modules call. Likewise, adapter helper exposure must wait until the schema surface is stabilized to avoid interim `MetaSignature` references.

## 5. API design consistency with repo patterns
- The plan keeps `async_trait` and the existing `CallOutcome` ergonomics and even preserves `forward_all` as a free function — this matches the async/utility style of `crates/dspy-rs`. The adapter helpers are rewritten to accept `SignatureSchema` instead of `MetaSignature`, mirroring the design document's `SignatureSchema` builder helpers. Just call out that new helper names (e.g., `format_input(schema, input)` vs `format_input_typed`) should follow the existing naming convention (snake_case, descriptive) and stay in `adapter::chat` so the rest of the crate sees them the same way.

## 6. Over-engineering check
- The plan sticks to a single slice of work (module authoring) without introducing extra features (no new optimizers or graph mechanics). The testing matrix is comprehensive but proportionate to the new surface: 4 tests cover schema reflection, `CallOutcome`, adapter helpers, and derive generic bounds. No additional scaffolding is proposed, so over-engineering does not appear to be a risk here.

## 7. Test comprehensiveness
- The authored tests hit the right guardrails: flatten path coverage, `CallOutcome` metadata, adapter parse round-trips, and generic signature caching. One missing assertion is the `SignatureSchema` cache key: the shaping doc warns about a `OnceLock` per monomorphized signature (S1). Add an explicit test that `schema_cache::<Foo<i32>>()` and `schema_cache::<Foo<String>>()` return distinct addresses to prevent the old bug where a generic `OnceLock` is shared across all monomorphizations.
