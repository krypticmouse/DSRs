# AGENTS.md - evaluate

## Boundary

This module provides evaluation infrastructure: metrics, feedback, and batch evaluation for DSPy modules.

**Depends on:** `crate::core` (Module trait), `crate::data` (Example, Prediction), `futures` (concurrency)
**Depended on by:** optimizers (COPRO, MIPROv2, GEPA), user evaluation scripts
**NEVER:** Put optimizer logic here; implement prediction logic; leak optimizer-specific types

---

## How to work here

### Core traits

1. **`Evaluator`** - Extend `Module` with `metric()` method and `evaluate()` batch runner
2. **`FeedbackEvaluator`** - Rich feedback with `FeedbackMetric` (score + explanation + metadata)

### Pattern: Implement Evaluator

```rust
impl Evaluator for MyModule {
    const MAX_CONCURRENCY: usize = 16;  // Controls parallel evaluation
    const DISPLAY_PROGRESS: bool = true;

    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        // Compare prediction to example's expected output
        // Return 0.0-1.0 score
    }
}
```

### Golden file
- `examples/03-evaluate-hotpotqa.rs` - Complete Evaluator impl with HotpotQA dataset

### When to use which
- **Simple metrics** - Implement `Evaluator::metric()`, returns `f32`
- **GEPA optimizer** - Implement `FeedbackEvaluator::feedback_metric()`, returns `FeedbackMetric`
- **Pre-built helpers** - Use `feedback_helpers.rs` functions (retrieval, classification, similarity)

---

## Verification

```bash
cargo test -p dspy-rs evaluate      # Module unit tests
cargo build --example 03-evaluate-hotpotqa -p dspy-rs --features dataloaders
```

---

## Don't do this

- **Don't call `batch()` directly** - `evaluate()` handles batching with proper concurrency
- **Don't ignore MAX_CONCURRENCY** - API rate limits exist; default 32 may be too high

---

## Gotchas

1. **`evaluate()` clones examples** - Required for concurrent iteration; large datasets = memory
2. **Metric evaluation is also concurrent** - `metric()` runs in `buffer_unordered(MAX_CONCURRENCY)`
3. **FeedbackMetric for GEPA** - GEPA's `compile_with_feedback()` requires `FeedbackEvaluator`, not plain `Evaluator`
4. **Empty metrics.rs** - Placeholder file; use `feedback_helpers.rs` for built-in metric functions
