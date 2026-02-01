# AGENTS.md - dspy-rs

## Boundary

Main DSPy-RS library: typed LLM signatures, predictors, optimizers, and tracing.

**Depends on:** `dsrs_macros` (derive macros), `baml-bridge` (type system/rendering), `rlm-*` (Python interop, feature-gated)
**Depended on by:** User applications (leaf library crate)
**NEVER:** Add provider-specific logic here; break `Signature` trait contract; use pyo3 outside `rlm` feature

---

## How to work here

### Core Abstractions

1. **`Signature` trait** (`core/signature.rs`): Compile-time LLM I/O contract via `#[derive(Signature)]`
2. **`Predict<S>`** (`predictors/predict.rs`): Main predictor. `call()` for typed, `forward()` for Module compat
3. **`Module` trait** (`core/module.rs`): Async `forward(Example) -> Prediction` for composition/optimizers
4. **`Adapter` trait** (`adapter/mod.rs`): Formats signatures to prompts. `ChatAdapter` is default
5. **`LM`** (`core/lm/mod.rs`): LLM client. Builder pattern, async: `LM::builder().model("...").build().await`

### Patterns to copy

```rust
#[derive(dspy_rs::Signature, Clone)]  // Generates QAInput, QAOutput
pub struct QA {
    #[input] pub question: String,
    /// Field description as doc comment
    #[output] pub answer: String,
}

configure(LM::builder().model("openai:gpt-4o-mini").build().await?, ChatAdapter);
let predict = Predict::<QA>::builder().instruction("...").build();
let result = predict.call(QAInput { question: "..." }).await?;

// Tracing (task-local scope)
let (result, graph) = trace(|| async { predict.call(input).await }).await;
```

### Golden files
- `examples/01-simple.rs` - typed API, demos, module composition
- `tests/test_compiled_signature.rs` - modern `#[derive(Signature)]` patterns
- `examples/rlm_trajectory.rs` - RLM with `#[rlm_type]` inputs

---

## Verification

```bash
cargo test -p dspy-rs --lib              # Unit tests (no API keys)
cargo test -p dspy-rs test_signature     # Signature macro tests
cargo test -p dspy-rs --features rlm     # RLM + Python interop
cargo build --examples -p dspy-rs        # All examples compile
```

---

## Don't do this

### Deprecated (do NOT copy)
- `example!` / `prediction!` macros - Use typed structs: `QAInput { ... }`
- `LegacyPredict` / `LegacySignature` - Use `Predict<S>` with `#[derive(Signature)]`
- `TypedRlm<S>` - Use `Rlm<S>` directly

### Forbidden
- `unwrap()` in library code (except tests)
- Direct `GLOBAL_SETTINGS` access - use `configure()` or `.with_lm()`
- `async-trait` on new traits - use `#[allow(async_fn_in_trait)]` pattern

---

## Gotchas

1. **`LM::builder().build()` is async** - Must `.await`. Common mistake.
2. **`configure()` must run first** - Without it, `Predict::call()` panics on missing global LM.
3. **Derive generates types** - `#[derive(Signature)]` on `Foo` creates `FooInput`, `FooOutput`.
4. **`call()` vs `forward()`** - `call()` = typed `CallResult<S>`, `forward()` = untyped `Prediction`.
5. **Tracing is task-local** - `trace()` only captures calls in that async task scope.
6. **Trace executor limitation** - `Executor` can't re-execute `Predict` nodes; only `Map` nodes work.
7. **Optimizers use LegacyPredict internally** - COPRO/MIPROv2/GEPA use legacy APIs; this is expected.
8. **GEPA requires FeedbackEvaluator** - Not just `Evaluator`; use `compile_with_feedback()`.
9. **RLM needs `#[rlm_type]`** - Input types need `RlmInputFields` trait via `#[rlm_type]` derive.
10. **Reasoning models** - Set `temperature: None` for o1/o3 (they reject temperature param).

---

## References

- `baml-bridge` crate - Type system and serialization internals
- `dsrs_macros` crate - Derive macro implementations (`Signature`, `LegacySignature`)
- `rlm-derive` crate - `#[rlm_type]` macro for RLM Python interop
