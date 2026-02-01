# rlm module

## Boundary

Python/REPL-based language model interaction via iterative code execution. **REQUIRES `rlm` feature flag.**

**Depends on:** `rlm-core` (type introspection), `baml-bridge` (parsing), `pyo3` (Python FFI), `core/` (LM, Predict)
**Depended on by:** User code via `Rlm::<S>::call()`
**NEVER:** pyo3 code without `#![cfg(feature = "rlm")]`; blocking Python calls on async runtime without Handle

## How to work here

### Key Types

- **`Rlm<S>`** (`rlm.rs`): Main entry. Generic over `S: Signature`. Builder pattern: `Rlm::builder().max_iterations(10).build()`
- **`RlmConfig`** (`config.rs`): Iteration limits, LLM call limits, fallback behavior
- **`RlmResult<S>`** (`config.rs`): Typed result with trajectory, field metadata, constraint summary
- **`RlmAdapter`** (`adapter.rs`): Builds prompts from signature, formats variable previews
- **`REPLHistory`** (`history.rs`): Immutable trajectory container with `#[render]` template
- **`LlmTools`** (`tools.rs`): `#[pyclass]` exposing `llm_query()` and `llm_query_batched()` to Python
- **`StorableRlmResult`** (`storage.rs`): Serializable snapshot for persistence/analysis

### Execution Flow

1. `Rlm::call()` injects input variables into Python globals via `RlmInputFields::inject_into_python()`
2. LLM generates Python code via `Predict<RlmActionSig>`
3. `execute_repl_code()` runs code in embedded Python, captures output
4. Loop until `SUBMIT()` called or max iterations
5. On success: convert `BamlValue` to typed `S::Output`
6. On timeout: optionally run extraction fallback via `Predict<RlmExtractSig<S>>`

### Patterns to copy

```rust
// Basic usage
let rlm = Rlm::<MySignature>::new();
let result = rlm.call(input).await?;

// With configuration
let rlm = Rlm::<MySignature>::builder()
    .max_iterations(15)
    .max_llm_calls(30)
    .strict_assertions(false)
    .build();

// Access trajectory
for entry in result.trajectory.entries() {
    println!("Code: {}\nOutput: {}", entry.code, entry.output);
}

// Serialize for storage
let storable = result.to_storable()?;
let json = storable.to_json_pretty()?;
```

## Golden examples

- `examples/rlm_trajectory.rs` - Full RLM workflow with `#[rlm_type]` inputs, trajectory inspection
- `examples/rlm_storage.rs` - Serialization patterns for RLM results

## Verification

```bash
cargo test -p dspy-rs --features rlm          # All RLM tests
cargo test -p dspy-rs --features rlm rlm_     # Pattern-match RLM tests
cargo run --example rlm_trajectory --features rlm  # Requires OPENAI_API_KEY
```

## Don't do this

- **pyo3 without feature gate** - Every file MUST have `#![cfg(feature = "rlm")]` at top
- **`TypedRlm<S>`** - Deprecated alias; use `Rlm<S>` directly
- **Blocking async in Python callbacks** - Use `Handle::block_on()` pattern (see `tools.rs:61-66`)
- **`unwrap()` in library code** - Propagate errors via `RlmError`

## Gotchas

1. **Python environment required** - Tests need working Python with `pyo3` bindings
2. **GIL handling** - Use `Python::attach()` for short operations; never hold GIL across await
3. **Tokio runtime** - `LlmTools` requires a `Handle` to block on async LM calls from sync Python
4. **History is immutable** - `REPLHistory::append()` returns new instance, preserves original
5. **Feature compilation** - Without `--features rlm`, all files in this dir are skipped entirely
6. **`SUBMIT()` validation** - Python callable validates against signature schema; errors shown to LLM for retry
7. **Output truncation** - `max_output_chars` and `max_history_output_chars` prevent context overflow
8. **Extraction fallback** - If enabled, max iterations triggers `RlmExtractSig` instead of error

## References

- `rlm-core` crate - `RlmDescribe`, `RlmInputFields` traits
- `rlm-derive` crate - `#[rlm_type]` macro for input types
- `submit.rs` - `SubmitHandler` pyclass (not public API)
- `signatures.rs` - Internal `RlmActionSig`, `RlmExtractSig` definitions
