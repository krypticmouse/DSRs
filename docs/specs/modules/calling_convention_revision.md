# Calling Convention Revision: `CallOutcome<O>` -> `Result<Predicted<O>, PredictError>`

Date: 2026-02-09
Status: Approved and integrated (spec updates applied 2026-02-10)
Scope: Spec-only changes across `breadboard.md`, `design_reference.md`, `shapes.md`

---

## Context: How DSPy (Python) Works

DSPy is the reference implementation we're porting to Rust. In DSPy, every module
call returns a `Prediction` object. This is the single, universal return type.

### DSPy's `Prediction`

`Prediction` inherits from `Example` (a dict-like container). It carries:
- **Output fields** via attribute access: `result.answer`, `result.reasoning`
- **Metadata** as methods/properties: `result.get_lm_usage()`, `result.completions`
- **Extra module-specific fields**: `result.trajectory` (for ReAct)

There is no `Result` wrapper. Errors are Python exceptions.

### DSPy user experience

```python
# P1: Simple call
result = predict(question="What is 2+2?")
print(result.answer)                  # direct field access
print(result.get_lm_usage())          # metadata on same object

# P1: Chain of thought
result = cot(question="What is 2+2?")
print(result.reasoning)               # augmented field
print(result.answer)                  # original field (via dict)

# P1: ReAct
result = react(question="Who won the 2024 election?")
print(result.answer)                  # output field
print(result.trajectory)              # trajectory metadata (dict of steps)

# P2: Module authoring
class HopModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.Predict("question -> query")
        self.predict2 = dspy.Predict("query -> answer")

    def forward(self, question):
        query = self.predict1(question=question).query
        return self.predict2(query=query)
```

Key observations:
1. Output and metadata travel together on one object.
2. Field access is direct — no unwrapping, no `.into_result()`.
3. `__call__` wraps `forward` and adds token tracking. No return type difference.
4. Module composition chains `.call()` invocations. The return value from one
   module feeds naturally into the next.

---

## Our Current Design (What We Have)

### `CallOutcome<O>`

Defined in `crates/dspy-rs/src/core/call_outcome.rs`:

```rust
pub struct CallOutcome<O> {
    metadata: CallMetadata,
    result: Result<O, CallOutcomeErrorKind>,
}
```

`CallOutcome` wraps BOTH the success/failure result AND metadata in one struct.
The Module trait returns it directly:

```rust
pub trait Module: Send + Sync {
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;
    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output>;
}
```

### The ergonomics problem

To access the output, users must unwrap the Result inside CallOutcome:

```rust
// Current P1 code — ugly
let output = predict.call(input).await.into_result()?;
println!("{}", output.answer);

// Or with explicit parts destructuring
let (result, metadata) = outcome.into_parts();
let output = result.map_err(|e| /* ... */)?;
```

The `?` operator does not work directly on `CallOutcome` because it's not a `Result`.
There's a nightly `Try` trait impl behind `#[cfg(feature = "nightly-try")]`, but
`try_trait_v2` has been unstable since 2021 with no stabilization timeline.

### How this violates Place separation

The breadboard defines four Places (P1-P4) with strict dependency direction.
P1 (User Code) should never need to understand metadata, adapter internals, or
optimizer concerns.

But `CallOutcome` forces every P1 user to interact with a metadata-carrying wrapper
type just to get their output. The `.into_result()?` ceremony exists because the
return type was designed for P2/P3's metadata needs, not P1's "call and get result"
needs.

In DSPy, metadata is available on the Prediction but never gets in the way — you
access `result.answer` directly without unwrapping anything. The metadata is there
if you want it, invisible if you don't.

---

## The New Design (What To Change To)

### `Predicted<O>` — the success type

```rust
/// The successful result of a module call.
/// Carries the typed output alongside call metadata.
/// Deref to O for direct field access — like DSPy's Prediction.
pub struct Predicted<O> {
    output: O,
    metadata: CallMetadata,
}

impl<O> Deref for Predicted<O> {
    type Target = O;
    fn deref(&self) -> &O { &self.output }
}

impl<O> Predicted<O> {
    pub fn new(output: O, metadata: CallMetadata) -> Self {
        Self { output, metadata }
    }

    pub fn metadata(&self) -> &CallMetadata { &self.metadata }

    pub fn into_inner(self) -> O { self.output }

    pub fn into_parts(self) -> (O, CallMetadata) {
        (self.output, self.metadata)
    }
}
```

### The Module trait

```rust
pub trait Module: Send + Sync {
    type Input: BamlType + for<'a> Facet<'a> + Send + Sync;
    type Output: BamlType + for<'a> Facet<'a> + Send + Sync;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError>;
}
```

### `PredictError` — the error type

`PredictError` already exists and already carries error-path metadata (raw_response,
lm_usage on parse failures). No changes needed to the error type.

### Why this is better

| Concern | `CallOutcome<O>` (old) | `Result<Predicted<O>, PredictError>` (new) |
|---|---|---|
| P1 field access | `outcome.into_result()?.answer` | `result?.answer` (via Deref) |
| `?` on stable Rust | Doesn't work | Works (it's a `Result`) |
| Metadata on success | `outcome.metadata()` before unwrap | `result.metadata()` after `?`-less bind |
| Metadata on error | `outcome.into_parts()` then match | In `PredictError` variants |
| DSPy parity | No equivalent | `Predicted<O>` ≈ `Prediction` |
| Nightly dependency | Needs `try_trait_v2` for ergonomics | None |

### User experience after the change

```rust
// P1: Simple call — ? just works
let result = predict.call(input).await?;
println!("{}", result.answer);              // Deref to QAOutput
println!("{:?}", result.metadata().lm_usage); // metadata if you want it

// P1: Chain of thought
let result = cot.call(input).await?;
println!("{}", result.reasoning);           // Deref to WithReasoning<QAOutput>
println!("{}", result.answer);              // Deref chain through WithReasoning -> QAOutput

// P1: Batching
let results = forward_all(&module, inputs, 5).await;
for result in results {
    match result {
        Ok(output) => println!("{}", output.answer),
        Err(err) => eprintln!("failed: {err}"),
    }
}
```

```rust
// P2: Module authoring — ChainOfThought (simple delegation)
impl<S: Signature> Module for ChainOfThought<S> {
    type Input = S::Input;
    type Output = WithReasoning<S::Output>;

    async fn forward(&self, input: S::Input) -> Result<Predicted<Self::Output>, PredictError> {
        self.predictor.call(input).await
    }
}

// P2: Module authoring — ReAct (needs sub-call metadata)
impl<S: Signature> Module for ReAct<S> {
    type Input = S::Input;
    type Output = S::Output;

    async fn forward(&self, input: S::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let mut merged_metadata = CallMetadata::default();

        for step in 0..self.max_steps {
            let action = self.action.call(action_input).await?;
            // action is Predicted<ActionStepOutput>
            // action.thought via Deref — direct field access
            // action.metadata() for token tracking
            merged_metadata.merge(action.metadata());

            if is_terminal(&action.action) { break; }
            let observation = self.execute_tool(&action.action, &action.action_input).await;
            trajectory.push_str(&format_step(step, &action, &observation));
        }

        let extract = self.extract.call(extract_input).await?;
        merged_metadata.merge(extract.metadata());

        Ok(Predicted::new(extract.into_inner().output, merged_metadata))
    }
}

// P2: Module authoring — BestOfN (wraps any Module)
impl<M: Module> Module for BestOfN<M> where M::Input: Clone {
    type Input = M::Input;
    type Output = M::Output;

    async fn forward(&self, input: M::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let mut best: Option<Predicted<M::Output>> = None;
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..self.n {
            let result = self.module.call(input.clone()).await?;
            let score = (self.reward_fn)(&input, &result);  // Deref to M::Output
            if score >= self.threshold {
                return Ok(result);
            }
            if score > best_score {
                best_score = score;
                best = Some(result);
            }
        }

        Err(PredictError::AllAttemptsFailed)
    }
}
```

```rust
// P2: Module combinators
impl<M, F, T> Module for Map<M, F> where M: Module, F: Fn(M::Output) -> T {
    type Input = M::Input;
    type Output = T;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let result = self.inner.call(input).await?;
        let (output, metadata) = result.into_parts();
        Ok(Predicted::new((self.map)(output), metadata))
    }
}
```

```rust
// P3: Optimizer interface (V5 — DynPredictor)
pub trait DynPredictor: Send + Sync {
    fn schema(&self) -> &SignatureSchema;
    fn instruction(&self) -> String;
    fn set_instruction(&mut self, instruction: String);
    fn demos_as_examples(&self) -> Vec<Example>;
    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()>;
    fn dump_state(&self) -> PredictState;
    fn load_state(&mut self, state: PredictState) -> Result<()>;
    async fn forward_untyped(&self, input: BamlValue) -> Result<Predicted<BamlValue>, PredictError>;
}
```

### What `call` vs `forward` means after this change

`call` is the canonical user-facing entry point. It returns
`Result<Predicted<O>, PredictError>`.

`forward` remains the implementation hook for module authors. The default `call`
method delegates to `forward`, mirroring DSPy's model where callers invoke the
module while implementers define forward logic.

The locked decision "call_with_meta is folded into call" is still superseded:
there is no `call_with_meta` split because metadata always travels with the output
inside `Predicted<O>`.

### What gets deleted

- `CallOutcome<O>` struct
- `CallOutcomeError` struct
- `CallOutcomeErrorKind` enum (may be partially absorbed into `PredictError`)
- `into_result()`, `into_parts()`, `try_into_result()` methods on CallOutcome
- The nightly `Try` / `FromResidual` impls
- `Deref<Target = Result<O, CallOutcomeErrorKind>>` impl on CallOutcome
- All references to `CallOutcome` in specs, plans, and tracker

---

## Spec Files to Update

### File 1: `docs/specs/modules/breadboard.md`

**Location: Line 51** — Batching resolved gap text.
References `Vec<CallOutcome<Output>>` in the `forward_all` description.
Change to `Vec<Result<Predicted<Output>, PredictError>>`.

**Location: Line 58** — "CallOutcome undecided" resolved gap.
Full rewrite. Currently reads:
> N8 returns a metadata-first wrapper by default and treats `forward` as the
> canonical invocation path.

Replace with:
> N8 returns `Result<Predicted<O>, PredictError>`. `Predicted<O>` carries output +
> call metadata (like DSPy's `Prediction`), with `Deref<Target = O>` for direct field
> access and `.metadata()` for call metadata. `?` works on stable Rust — no nightly
> `Try` trait needed. `Module::call` is the canonical user-facing entrypoint, and
> `Module::forward` remains the implementation hook.

**Location: Line 84** — U10 affordance row.
Change `CallOutcome<S::Output>` to `Predicted<S::Output>` and update the description
text from "single return surface; carries Result + metadata" to "output + metadata
wrapper; Deref to Output for field access".

**Location: Line 90** — U48 affordance row.
Change `Vec<CallOutcome<Output>>` to `Vec<Result<Predicted<Output>, PredictError>>`.
Change `→ Vec\<CallOutcome\>` in the Returns To column.

**Location: Line 92** — U51 affordance row.
If it references `CallOutcome`, update. Verify the combinator description doesn't
assume `CallOutcome` return semantics.

**Location: Line 137** — N8 code affordance row.
Change "Returns `CallOutcome<O>`" to "Returns `Result<Predicted<O>, PredictError>`"
in the affordance description.

**Location: Line 191** — P1 wiring narrative.
Change `→ U10 (CallOutcome<Output>)` to `→ U10 (Result<Predicted<Output>, PredictError>)`.

**Location: Line 192** — P1 wiring narrative, error line.
Change `→ on error: U49 (PredictError with raw response + stage)` — this stays mostly
the same, but verify the wiring makes sense with `Result`'s `Err` path.

**Location: Line 197** — Batching wiring narrative.
Change `→ Vec<CallOutcome<Output>>` to `→ Vec<Result<Predicted<Output>, PredictError>>`.

**Location: Line 342** — V1 slice detail table.
Change `forward(), CallOutcome, field access` to `forward(), Predicted<O>, field access`.

**Location: ~Line 360** — V1 demo program code block.
Currently uses `?` which is correct. Verify it reads naturally:
```rust
let result = predict.call(QAInput { question: "What is 2+2?".into() }).await?;
println!("{}", result.answer);  // typed field access via Deref
```

**Location: Line 403** — V3 demo program Module impl.
Already returns `Result<Self::Output, PredictError>`. Update to
`Result<Predicted<Self::Output>, PredictError>`.

### File 2: `docs/specs/modules/design_reference.md`

**Location: Section 5 (line ~362-398)** — Module trait definition + explanation.

Replace the trait definition:
```rust
// Old
async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output>;

// New
async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError>;
```

Replace the `CallOutcome<O>` explanation paragraph (line 371) entirely. This currently
reads:
> `CallOutcome<O>` is the default return surface for N8. It carries both outcome
> (`Result<O, PredictError>`) and call metadata (raw response, usage, tool calls,
> field parse metadata). There is no separate convenience API (for example
> `forward_result()`); ergonomics come from trait impls on `CallOutcome` itself
> (`Try` when available on toolchain, otherwise at least
> `Deref<Target = Result<...>>` + `into_result()`).

Replace with an explanation of `Predicted<O>`:
> `Module::forward` returns `Result<Predicted<O>, PredictError>`. `Predicted<O>`
> carries the typed output alongside call metadata (raw response, usage, tool calls,
> field parse metadata). It implements `Deref<Target = O>` so output fields are
> accessible directly: `result.answer`, `result.reasoning`. Metadata is available
> via `result.metadata()`. This mirrors DSPy's `Prediction` object where output
> fields and metadata coexist on the same value. `?` works on stable Rust because
> the outer type is `Result`.

Add the `Predicted<O>` struct definition, Deref impl, and key methods as a new code
block in this section (see "The New Design" section above for the definition).

**Location: Section 6 (~lines 440-480)** — Predict::call pipeline code sketch.

Update the code sketch. Key changes:
- Method signature: `pub async fn call(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError>`
  (Note: in the new design, `call` is just an alias or doesn't exist separately —
  Predict implements Module::forward. The code sketch should show `forward` or note
  that `call` delegates to the same logic.)
- Error returns: change `CallOutcome::from_error(PredictError::Lm { ... })` to
  `return Err(PredictError::Lm { ... })`
- Success return: change `CallOutcome::from_parts(output, ...)` to
  `Ok(Predicted::new(typed_output, CallMetadata::new(...)))`

**Location: Section 9 (~line 699)** — DynPredictor trait definition.

Change:
```rust
async fn forward_untyped(&self, input: BamlValue) -> CallOutcome<BamlValue>;
```
To:
```rust
async fn forward_untyped(&self, input: BamlValue) -> Result<Predicted<BamlValue>, PredictError>;
```

**Location: Section 9 (~lines 728-735)** — DynPredictor impl code sketch.

Update the `forward_untyped` implementation:
- Error: `return Err(PredictError::Conversion { ... })` instead of
  `CallOutcome::from_error(...)`
- Success: `Ok(Predicted::new(output.to_baml_value(), metadata))` instead of
  the `CallOutcome` map/into_result chain

**Location: Section 12 (~line 881)** — ChainOfThought forward signature.

Change:
```rust
async fn forward(&self, input: S::Input) -> CallOutcome<WithReasoning<S::Output>> {
    self.predict.call(input).await
}
```
To:
```rust
async fn forward(&self, input: S::Input) -> Result<Predicted<WithReasoning<S::Output>>, PredictError> {
    self.predict.call(input).await
}
```

**Location: Section 12 (~lines 905-914)** — BestOfN forward signature and body.

Change:
```rust
async fn forward(&self, input: M::Input) -> CallOutcome<M::Output> {
    // ...
    if score >= self.threshold { return CallOutcome::ok(output); }
    // ...
    CallOutcome::from_error(PredictError::AllAttemptsFailed)
}
```
To:
```rust
async fn forward(&self, input: M::Input) -> Result<Predicted<M::Output>, PredictError> {
    // ...
    if score >= self.threshold { return Ok(result); }
    // ...
    Err(PredictError::AllAttemptsFailed)
}
```

**Location: Section 10 (~line 761)** — DynModule::forward.
Already returns `Result<BamlValue>`. Update to `Result<Predicted<BamlValue>, PredictError>`
for consistency, or leave as-is if the dynamic path intentionally strips metadata.
Decision: update for consistency.

### File 3: `docs/specs/modules/shapes.md`

**Location: Line 60** — F4 Module trait part description.

Currently reads:
> `trait Module { type Input; type Output; async fn forward(&self, input) -> CallOutcome<Output> }`.
> `CallOutcome` is the single return surface (result + metadata), with trait-based
> ergonomics for `?`-style consumption so there is no parallel convenience API.

Replace with:
> `trait Module { type Input; type Output; async fn forward(&self, input) -> Result<Predicted<Output>, PredictError> }`.
> `Predicted<O>` carries output + metadata with `Deref<Target = O>` for direct field
> access. `?` works on stable Rust. Mirrors DSPy's `Prediction` return convention.

---

## Plan Files to Update

### `docs/plans/modules/phase_4_5_cleanup_kickoff.md`

**Location: Locked Decisions section, item 2.**
Currently reads:
> **Single call surface**: `CallOutcome<O>` is the default call contract; no parallel
> convenience call path.

Replace with:
> **Single call surface**: `Module::call` returns `Result<Predicted<O>, PredictError>`.
> `Predicted<O>` carries output + metadata. `forward` remains the implementation hook.

### `docs/plans/modules/tracker.md`

Add a decision entry in the Decisions & Architectural Notes section:
> **Calling convention revision (2026-02-09):** Replaced `CallOutcome<O>` with
> `Result<Predicted<O>, PredictError>` as the canonical `Module::call` return type
> (delegating to `forward`).
> `Predicted<O>` implements `Deref<Target = O>` for direct field access and carries
> `CallMetadata` (like DSPy's `Prediction`). Rationale: `CallOutcome` required
> `.into_result()?` on stable Rust, violating P1 ergonomics goals. The nightly `Try`
> trait (`try_trait_v2`) has no stabilization timeline. `Predicted<O>` + `Result`
> gives DSPy-parity ergonomics on stable: `module.call(input).await?.answer`.
> `call` is canonical for users; `forward` is the implementation hook. Former locked
> decision "call_with_meta folded into call" is superseded.

---

## Files NOT to Change

- **Spike docs** (`spikes/S1-S8`): Historical findings. Do not retroactively edit.
- **DSPy module system reference** (`dspy_module_system_reference/`): Reference docs
  about the Python DSPy system. Not our design specs.
- **Plan docs** other than kickoff and tracker: Historical records of slice execution.
- **Code files**: This revision is spec-only. Code changes happen during implementation.

---

## Validation After Spec Updates

After all spec changes are made, verify:

1. **No orphan `CallOutcome` references** in breadboard.md, design_reference.md, or
   shapes.md. Grep for `CallOutcome` — should return zero hits in these three files.
2. **`Predicted<O>` is defined** in design_reference.md Section 5 with struct
   definition, Deref impl, and key methods.
3. **All code sketches compile conceptually** — return types match, error handling
   uses `?` and `Err(...)`, success uses `Ok(Predicted::new(...))`.
4. **Demo programs use `?`** — V1-V6 demo code blocks show the clean P1 experience.
5. **No legacy split-call or `into_result` references** remain in the spec files.
6. **F4 description** in shapes.md matches the trait in design_reference.md.
