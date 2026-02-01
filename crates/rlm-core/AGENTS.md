# rlm-core

## Boundary

Core traits for describing Rust types to LLMs. Pure type introspection - no runtime LLM calls.

**Depends on:** `std` only. Optional: `pyo3`, `baml-types`
**Depended on by:** `rlm-derive` (codegen), `dspy-rs` (prompt generation)
**NEVER:** PyO3 types outside `#[cfg(feature = "pyo3")]`; derive logic here (use `rlm-derive`); LLM calls

## How to work here

**Traits** (`describe.rs`):
- `RlmTypeInfo` - compile-time: `TYPE_NAME`, `IS_OPTIONAL`, `IS_COLLECTION`, `IS_DESCRIBABLE`
- `RlmDescribe` - runtime: `type_name()`, `fields()`, `properties()`, `describe_value()`, `describe_type()`

**Structs:**
- `RlmFieldDesc` - field metadata (name, type, description, flags)
- `RlmPropertyDesc` - computed property metadata
- `RlmVariable` (`variable.rs`) - wraps values for prompt injection

**PyO3** (`input.rs`, feature-gated):
- `RlmInputFields`: `rlm_py_fields()`, `rlm_variables()`

**Golden example:** `describe.rs:404-420` (String impl) - canonical type_name/describe_value/describe_type pattern.

**Adding stdlib impls:**
1. Add `RlmTypeInfo` with const `TYPE_NAME`
2. Add `RlmDescribe` with required methods
3. Add tests in `mod tests` at bottom of `describe.rs`

## Verification

```bash
cargo test -p rlm-core
```

Skip `--features pyo3` unless touching `input.rs` (requires Python env).

## Don't do this

- `RlmDescribe` without `RlmTypeInfo` for containers
- Python code outside `#[cfg(feature = "pyo3")]`
- `unwrap()`/`expect()` in library code
- Change `describe_value()` format without checking `dspy-rs/src/rlm/prompt.rs`

## Gotchas

**Failing test:** `test_vec_describe_empty` expects `"Vec<String> (empty)"` but gets `"list[String] (empty)"`. Vec uses `"list"` for Python interop; test written for Rust naming.

**Truncation:**
- `String::describe_value()`: 100 chars then `"...{N} chars"`
- `RlmVariable` preview: 500 chars
- `Vec`/`HashMap`: first 3 items, then `"... and {N} more"`

**WIP:** `Vec`/`HashMap` impls have TODO at lines 324, 489 - output format not finalized.

**Edition 2024** - use latest Rust syntax.

## See also

- `../rlm-derive/AGENTS.md` - proc-macro that generates `RlmDescribe` impls
- `../dspy-rs/AGENTS.md` - consumes these traits for prompt generation
