# Trace Module

## Boundary

This directory: Execution DAG capture and replay for LLM pipelines.

- **context.rs** - `task_local!` storage, `trace()` wrapper, `record_node()`, `record_output()`
- **dag.rs** - `Graph`, `Node`, `NodeType` (Root, Predict, Operator, Map)
- **executor.rs** - Replays a captured `Graph` with new inputs
- **value.rs** - `TrackedValue`, `IntoTracked` for data lineage

Depends on: `crate::Example`, `crate::Prediction`
Depended on by: `predictors/`, macros (`example!`)
NEVER: Store graph state outside `task_local!`; break node ID referential integrity

## How to work here

**Golden example:** `examples/12-tracing.rs`
- Shows `trace::trace(|| async { ... })` wrapper pattern
- Demonstrates `get_tracked()` for lineage preservation
- Expected graph structure: Root -> Predict -> Map -> Predict

**Pattern: Wrapping execution for capture**
```rust
let (result, graph) = trace::trace(|| async {
    module.forward(example).await
}).await;
```

**Pattern: Preserving data lineage**
```rust
let answer = prediction.get_tracked("answer");  // TrackedValue with source
let inputs = example! {
    "question": "input" => question.clone(),
    "answer": "input" => answer.clone()  // Lineage flows through
};
```

**Adding a new NodeType:**
1. Add variant to `NodeType` in `dag.rs`
2. Update `record_node()` callsites (usually in predictors)
3. Handle in `Executor::execute()` if replayable
4. Update example expected graph comments

## Verification

```bash
cargo run --example 12-tracing  # Should show Root->Predict->Map->Predict structure
cargo test -p dspy-rs           # General test suite
```

No dedicated unit tests for trace module yet; the example serves as integration test.

## Gotchas

- **task_local storage**: `CURRENT_TRACE` only exists inside `trace::trace()` scope. Calling `record_node()` outside returns `None` silently.
- **Arc unwrap fallback**: If graph is still shared when `trace()` ends (orphaned tasks), it clones instead of unwrapping. This shouldn't happen in normal usage.
- **TrackedValue.source is skipped in serde**: Lineage info doesn't serialize. It's runtime-only for building the DAG.
- **Executor can't replay Predict nodes**: `Executor::execute()` errors on Predict nodes because signature data isn't stored. Only Map/Root replay works currently.
- **node_id on Prediction**: Predictions carry their originating node ID. Use `prediction.get_tracked()` to extract values with lineage, not raw `prediction.data.get()`.

## References

- `examples/12-tracing.rs` - canonical usage
- `predictors/predict.rs` - where tracing hooks into forward passes (see `is_tracing()` checks)
- `data/prediction.rs` - `get_tracked()` implementation
