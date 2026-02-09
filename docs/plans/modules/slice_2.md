# Slice 2 Implementation Plan: Augmentation + ChainOfThought (V2)

## Scope & Locked Decisions
- **Slice 2 scope only**: implement F3 (augmentation combinator/macro) and F11 (ChainOfThought module) on top of the Slice 1 surface. Do not reopen Slice 1 deliverables; keep the `CallOutcome<O>` default return, the metadata-rich error plumbing, and the typed `ChatAdapter`/`Predict` contract exactly as shipped after Slice 1.
- **Architectural constraints**: the new code must build on the current `Signature`/`FieldPath` metadata, use the same `CallOutcome`/`CallMetadata` semantics, not introduce new public variants or `Module::forward` signatures, and keep the breadth of the `FieldSchema` walker (no reversion to legacy `FieldSpec`).

## Ordered implementation sequence (minimize compile breakage)
1. **Signature cleanup prep (keep compile passing while refactoring)**
   - Update `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs` and `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs` to stop relying on `Signature::into_parts`/`from_parts` by operating on `Demo<S>`'s stored `input`/`output` fields directly.
   - Keep `Signature` trait methods temporarily, but mark `demo_signature`, `with_demo_signatures`, and `demo_from_signature` as wrappers that call the new helpers.
   - After all call sites use `Demo` fields instead of `into_parts`, remove `Signature::into_parts` & `from_parts` from `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/signature.rs` and the macro generation in `/Users/darin/src/personal/DSRs/crates/dsrs-macros/src/lib.rs` in the same patch to maintain compile stability.
   - Run `cargo check -p dspy-rs -p dsrs_macros` once the trait removal is staged to ensure macro changes and callers stay in sync.
2. **Augmentation trait + macro metadata**
   - Create `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/augmentation.rs` that defines the augmentation contract (`Augmentation`, `Augmented`) and exports `WithReasoning` helpers for ChainOfThought. The contract should leave augmentation-specific wrappers as strictly typed `BamlType + Facet + Deref` adapters (no `DerefMut`) and expose `Augmented<S, A>` as the `Signature` combinator that reuses `S::Input` while wrapping `S::Output` with `A::Wrap`.
  - Extend `/Users/darin/src/personal/DSRs/crates/dsrs-macros/src/lib.rs` so `#[derive(Augmentation)]` emits `WithReasoning<O>` wrappers, the `Augmentation` impl, `Deref` helpers, and a `#[flatten]`d `output` field whose `FieldPath` metadata puts augmentation-specific fields before the wrapped output when `#[augment(output, prepend)]` is used.
   - Ensure the macro also implements `Facet`/`BamlType` for `WithReasoning<O>` and emits `FieldSchema` metadata consistent with the flattened layout; test by running `cargo check -p dsrs_macros` before adding consumers.
3. **Predict/ChatAdapter adjustments for augmentation**
   - Update `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs` to let `Predict<Augmented<S, Reasoning>>` carry `CallOutcome<WithReasoning<S::Output>>`, to convert between `Example` and `Demo` without `into_parts`, and to keep metadata as-is.
   - Update `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs` so `format_demo_typed` accepts `&Demo<S>` (no `Signature::into_parts`), and verify `parse_response_typed` correctly traverses `FieldPath`s for `WithReasoning` fields generated via augmentation.
   - After these adjustments compile, re-run `cargo check -p dspy-rs` to ensure typed parsing still succeeds.
4. **ChainOfThought module implementation**
   - Implement `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/modules/chain_of_thought.rs` containing `ChainOfThought<S>`, the `Reasoning` facet, and a builder that wraps `Predict<Augmented<S, Reasoning>>`.
   - Wire `ChainOfThought` into `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs` (e.g., add `pub mod modules;` and `pub use modules::chain_of_thought::{ChainOfThought, Reasoning};`).
   - Implement `ChainOfThought<S>::call` mirroring `Predict::call` but returning `CallOutcome<WithReasoning<S::Output>>` and call metadata, and implement `Module`/`MetaSignature`/`Optimizable` in exactly the same shape as `Predict`, so `ChainOfThought` can replace `Predict` in graphs without reworking `CallOutcome` behaviors.
5. **Validation tests for deref, flatten roundtrip, and CoT swap**
   - Create dedicated tests (see section below) and run `cargo test -p dspy-rs --tests` to confirm deref ergonomics, schema round-trips, and `ChainOfThought` swap.
6. **Re-export + documentation updates (post-implementation)**
   - Add doc comments summarizing the augmentation workflow and update `docs/plans/modules/slice_2_research.md` or other relevant docs to mention the `BamlValue` reconstruction choice and `Deref` guidance (already started in the research doc). No new API surface beyond the planned module.

## File-specific tasks and exact signatures/macro outputs

### `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/core/signature.rs`
- Remove `fn from_parts(input: Self::Input, output: Self::Output) -> Self` and `fn into_parts(self) -> (Self::Input, Self::Output)` from `trait Signature`.
- Keep the rest of the trait intact so `SignatureSchema` still consumes `instruction()`, `input_shape`, `output_shape`, `input/output_field_metadata`, and `output_format_content`.
- After trait removal, ensure no other file calls the deleted methods by updating `PredictBuilder::demo_signature`, `with_demo_signatures`, and any adapters; once the callers are rewritten, delete those helper methods entirely.

### `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/predictors/predict.rs`
- Replace `demo.into_parts()` usage in `ChatAdapter` calls with direct access to `demo.input`/`demo.output`.
- Delete `demo_from_signature` and the deprecated builder helpers once trait cleanup is finished.
- Ensure `Predict::call` is generic over `S::Input: BamlType` and `S::Output: BamlType` as today, but add a helper `fn wrap_with_reasoning<A: Augmentation>(output: S::Output) -> A::Wrap<S::Output>` when building `Augmented`-based predictors.
- When `ChainOfThought` plugs in, call `self.predictor.call(input)` (where `predictor` is `Predict<Augmented<S, Reasoning>>`) and return the resulting `CallOutcome<WithReasoning<S::Output>>`.

### `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/adapter/chat.rs`
- Change `format_demo_typed` signature to `pub fn format_demo_typed<S: Signature>(&self, demo: &Demo<S>) -> (String, String)` and call `format_user_message_typed(&demo.input)`/`format_assistant_message_typed(&demo.output)`.
- Verify the `FieldPath` navigation helpers (`value_for_path`, `insert_baml_at_path`) continue to work for `WithReasoning` by ensuring the generated schema still emits the reasoning field’s path (`reasoning`) and the flattened output fields (e.g., `answer`/`confidence`).
- No changes to the parsing logic are necessary beyond confirming config because `SignatureSchema::output_fields()` derives from the new `WithReasoning` shape created by the augmentation macro.

### `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/augmentation.rs` (new file)
Define the augmentation primitives with these exact signatures:
```rust
use std::marker::PhantomData;
use std::ops::Deref;
use crate::{BamlType, Facet, Signature};

pub trait Augmentation: Send + Sync + 'static {
    type Wrap<T: BamlType + Facet>: BamlType + Facet + Deref<Target = T>;
}

pub struct Augmented<S: Signature, A: Augmentation> {
    _marker: PhantomData<(S, A)>,
}

impl<S: Signature, A: Augmentation> Signature for Augmented<S, A> {
    type Input = S::Input;
    type Output = A::Wrap<S::Output>;

    fn instructions() -> &'static str {
        S::instructions()
    }
}
```
- Also expose helper traits to fetch the wrapped output (e.g., `pub type AugmentedOutput<S, A> = <A::Wrap<S::Output> as Deref>::Target;`).
- Re-export this module from `crates/dspy-rs/src/lib.rs` so `ChainOfThought` and other consumers can `use dspy_rs::augmentation::{Augmentation, Augmented};`.

### `/Users/darin/src/personal/DSRs/crates/dsrs-macros/src/lib.rs`
- Add support for `#[derive(Augmentation)]` with an optional `#[augment(output, prepend)]` attribute. The macro should:
  1. Declare `pub struct With<ReasoningName><O: BamlType + Facet> { pub reasoning: <ReasoningFields>, #[flatten] pub output: O }` with `reasoning` fields derived directly from the annotated struct.
  2. Implement `Deref` so `With...` transparently forwards to the wrapped output and a `From<A::Wrap<O>> for With...` if needed.
  3. Implement `Augmentation` for the original struct: `impl Augmentation for Reasoning { type Wrap<T> = WithReasoning<T>; }`.
  4. Apply `#[derive(BamlType, Facet)]` to `WithReasoning<O>` so the generated `FieldSchema`s carry flattened metadata; the macro should insert the `output` field with `#[flatten]` and ensure the augmentation fields (reasoning) appear ahead of the flattened `output` when `prepend` is specified.
  5. Propagate the doc comments (`collect_doc_comment`) from the annotated struct to the generated `reasoning` fields so LM instructions stay consistent.
- The macro should also emit an inherent `impl` for `WithReasoning<O>` that exposes `pub fn reasoning(&self) -> &Reasoning` so ergonomic access works even before `Deref` (helpful for pipe/resizable code).

### `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/modules/chain_of_thought.rs` (new file)
Define:
```rust
use crate::{CallMetadata, CallOutcome, CallOutcomeErrorKind, ChatAdapter, Example, FieldSchema, Module, Optimizable, Predict, Prediction, Signature};
use crate::augmentation::{Augmentation, Augmented};

pub struct Reasoning {
    #[output]
    pub reasoning: String,
}

#[derive(Default)]
pub struct ChainOfThought<S: Signature> {
    predictor: Predict<Augmented<S, Reasoning>>,
}
```
- Provide `impl<S: Signature> ChainOfThought<S> { pub fn new() -> Self; pub fn with_predict(predictor: Predict<Augmented<S, Reasoning>>) -> Self; pub fn builder() -> ChainOfThoughtBuilder<S>; pub async fn call(&self, input: S::Input) -> CallOutcome<WithReasoning<S::Output>> { self.predictor.call(input).await } }`. `new()` must construct with `Predict::<Augmented<S, Reasoning>>::new()` to match U13 (`ChainOfThought::<S>::new()`), and `ChainOfThoughtBuilder` must expose the full delegated `PredictBuilder<Augmented<S, Reasoning>>` DSL (demos, instruction overrides, tools) for swap ergonomics.
- Implement `Module`/`MetaSignature`/`Optimizable` the same way `Predict` currently does, forwarding `forward`, `forward_untyped`, and the metadata methods to the internal `Predict<Augmented<S, Reasoning>>` so the optimizer/walker sees the same `CallOutcome` flow and field schema as `Predict`.
- Re-export `ChainOfThought` and `Reasoning` through `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs` so modules can `use dspy_rs::modules::chain_of_thought::ChainOfThought;` and swap it in place of `Predict`.

## Macro expansion example for `#[derive(Augmentation)]` + `#[augment(output, prepend)]`
Given
```rust
#[derive(Augmentation, Facet, BamlType)]
#[augment(output, prepend)]
pub struct ReasoningFacet {
    #[output]
    pub reasoning: String,
}
```
the macro should expand to roughly:
```rust
pub struct WithReasoningFacet<O: BamlType + Facet> {
    pub reasoning: String,
    #[flatten]
    pub output: O,
}

impl<O: BamlType + Facet> Deref for WithReasoningFacet<O> {
    type Target = O;
    fn deref(&self) -> &Self::Target { &self.output }
}

impl Augmentation for ReasoningFacet {
    type Wrap<T: BamlType + Facet> = WithReasoningFacet<T>;
}
```
Because the `output` field is annotated with `#[flatten]`, `SignatureSchema::output_fields()` emits two `FieldSchema`s: one for `reasoning` with `FieldPath::new(["reasoning"])` and one for every field inside `O` with `FieldPath`s such as `["answer"]`. The `prepend` flag instructs the macro to insert the reasoning `FieldSchema` before the flattened `output` fields so collection order matches the spec.

## ChainOfThought wiring details
- `ChainOfThought<S>` stores a `Predict<Augmented<S, Reasoning>>` and exposes `call(input: S::Input)` returning `CallOutcome<WithReasoning<S::Output>>` by delegating to the inner predictor. It reuses the same `CallOutcome`/`CallMetadata` as `Predict` so metadata stays intact (raw text, token counts, field checks).
- `Module::forward`/`forward_untyped` for `ChainOfThought` mirror the implementations in `Predict`, converting `Example`/`BamlValue` to typed inputs, calling `Predict::call`, and transforming the `WithReasoning` output into a `Prediction` while preserving metadata.
- `MetaSignature` implementation forwards to the inner `Predict` so the optimizer/walker sees the same schema (including the augmented `reasoning` field) and the same `CallOutcome` machinery.
- Expose a builder (`ChainOfThoughtBuilder<S>`) that wraps `PredictBuilder<Augmented<S, Reasoning>>` so demos/instruction overrides continue to work.

## Migration steps (Signature cleanup + Demo/adapter integration)
1. **Demo helpers**: remove `demo_signature`, `with_demo_signatures`, and `demo_from_signature`; update `PredictBuilder` to accept only `Demo<S>` values.
2. **Example conversion**: `example_from_demo` no longer relies on `Signature::into_parts`; keep using `demo.input`/`demo.output` for serialization/deserialization.
3. **ChatAdapter**: update `format_demo_typed` to accept `&Demo<S>` and call `format_user_message_typed(&demo.input)`/`format_assistant_message_typed(&demo.output)` so `Recursive FieldPath` logic continues to work after trait cleanup.
4. **Signature trait**: remove `from_parts/into_parts` and ensure `dsrs-macros` no longer tries to synthesize them; add release notes if ratio instructs.
5. **Field schema verification**: ensure `SignatureSchema` still builds `FieldPath`s for new `WithReasoning` output by running `cargo test -p dspy-rs tests::test_chat_adapter_schema` (existing test) once new augmentation derived fields exist.

## Explicit test matrix
| File | Focus | Key assertions |
|------|-------|----------------|
| `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_with_reasoning_deref.rs` | S3 deref ergonomics | instantiate `Reasoning`/`WithReasoning<QAOutput>` via the macro, assert `result.reasoning` is reachable, `Deref` lets you call methods on the inner `QAOutput`, and pattern matching without `.reasoning` requires destructuring (the test can assert that `let WithReasoning { reasoning, output: _ } = result;` compiles).|
| `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_flatten_roundtrip.rs` | Flatten metadata + ChatAdapter roundtrip | Build a `Demo<Augmented<QA, Reasoning>>`, call `ChatAdapter::format_demo_typed`, feed the formatted strings back into `ChatAdapter::parse_response_typed::<Augmented<QA, Reasoning>>()`, and assert the returned `WithReasoning<QAOutput>` has both `answer`/`confidence` (flattened) and `reasoning` fields populated. Ensure `FieldPath`s from `schema.output_fields()` include `reasoning` before the QA output paths.|
| `/Users/darin/src/personal/DSRs/crates/dspy-rs/tests/test_chain_of_thought_swap.rs` | ChainOfThought swap type check | Define a helper that accepts any `impl Module` and show that `ChainOfThought::<QA>` builds via its builder and satisfies the `Module` bound. Additionally construct a `Predict<Augmented<QA, Reasoning>>` call and assert `CallOutcome<WithReasoning<QAOutput>>` returns the reasoning string and can be `.into_result().unwrap()`. This test verifies that swapping `Predict<QA>` for `ChainOfThought<QA>` compiles and exposes the augmented output.|

## Recommended verification cadence
- After signature cleanup and demo rewrites: `cargo check -p dspy-rs`.
- After augmentation macro changes: `cargo check -p dsrs_macros`.
- After ChainOfThought implementation: `cargo test -p dspy-rs --tests test_chain_of_thought_swap test_flatten_roundtrip test_with_reasoning_deref`.
- Before merging: full `cargo test -p dspy-rs --tests`.

## Next steps
1. Implement the trait/macro changes in the order outlined above.
2. Add the ChainOfThought module and tests once the augmentation macro is complete.
3. Run the verification commands and capture any new diagnostics in `/Users/darin/src/personal/DSRs/docs/plans/modules/tracker.md` if they affect later slices.
