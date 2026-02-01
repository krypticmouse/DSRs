# AGENTS.md - predictors/

## Boundary

This directory: Typed prediction over LLM signatures. Single-step inference only.
Depends on: `core::Signature`, `adapter::ChatAdapter`, `baml_bridge` types
Depended on by: User code, optimizers, modules that compose predictors
NEVER: Implement multi-step pipelines here (use `modules/`); add provider logic; handle module composition

---

## How to work here

### Core Types

- **`Predictor` trait** (`mod.rs`): Async `forward(Example) -> Prediction` + batch methods
- **`Predict<S>`** (`predict.rs`): Main typed predictor parameterized by `Signature`

### `Predict<S>` Usage

```rust
// Builder pattern (preferred)
let predict = Predict::<MySignature>::builder()
    .instruction("custom instruction")
    .demo(example_signature)
    .with_lm(custom_lm)
    .build();

// Direct construction
let predict = Predict::<MySignature>::new().with_lm(lm);
```

### `call()` vs `forward()`

| Method | Input | Output | Use when |
|--------|-------|--------|----------|
| `call()` | `S::Input` | `CallResult<S>` | Typed API, normal usage |
| `call_with_example()` | `Example` | `CallResult<S>` | Untyped input, typed output |
| `forward()` | `Example` | `Prediction` | Module trait compat, optimizers |

### Golden file

- `predict.rs` lines 48-129 - `call()` implementation shows full flow

---

## Verification

```bash
cargo test -p dspy-rs test_predictor     # Basic predictor test
cargo test -p dspy-rs --lib predictors   # All predictor tests
```

---

## Don't do this

- **`LegacyPredict`** - Exists for optimizer internals only. New code uses `Predict<S>`.
- **Direct `GLOBAL_SETTINGS` access** - Use `configure()` or `.with_lm()` method.
- **`DummyPredict` in production** - Testing stub only; always echoes input as output.

---

## Gotchas

1. **Generic bounds are strict** - `S: Signature + Clone`, `S::Input: ToBamlValue`, etc. Missing bounds cause cryptic errors.
2. **Demos are typed** - `PredictBuilder::demo()` takes `S` (full signature), not `Example`. Convert with helper functions if needed.
3. **`call()` requires `configure()`** - Without global LM setup, panics. Use `.with_lm()` to bypass.
4. **`forward()` loses type info** - Returns untyped `Prediction`; use `call()` when you need `S::Output`.
5. **Batch methods preserve order** - `batch()` runs concurrent but returns results in input order.
