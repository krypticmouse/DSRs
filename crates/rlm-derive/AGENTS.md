# rlm-derive

## Boundary

Proc-macro crate for RLM-compatible PyO3 types. Generates Python bindings + `RlmDescribe` impls.

**Depends on:** `syn`, `quote`, `darling` (compile-time); generated code uses `::dspy_rs::{pyo3, baml_bridge, rlm_core}`

**Depended on by:** Any crate with `#[rlm_type]` structs (e.g., `dspy-rs` examples/tests)

**NEVER:** Generate code without `#[pyclass]`; emit runtime panics in generated code

## How to work here

**Golden examples:**
- Attribute parsing: `src/attrs.rs` (unit tests at lines 268-408)
- Getter generation: `src/generators/pyclass.rs::getter_strategy()` (line 118)
- End-to-end: `crates/dspy-rs/examples/rlm_trajectory.rs`, `crates/dspy-rs/tests/rlm_test.rs`

**Module structure:**
- `lib.rs` - entry points for `#[rlm_type]` and `#[derive(RlmType)]`
- `attrs.rs` - darling-based attribute parsing (`RlmTypeAttrs`, `RlmFieldAttrs`)
- `rlm_attr.rs` - `#[rlm_type]` expansion (adds `#[pyclass]` + injects derives)
- `rlm_type.rs` - `#[derive(RlmType)]` orchestration
- `generators/` - code gen (pyclass, repr, iter, properties, schema, describe)

**Adding a new attribute:**
1. Add field to `RlmTypeAttrs` or `RlmFieldAttrs` in `attrs.rs`
2. Add validation in `RlmTypeAttrs::validate()` if needed
3. Add code generation in `generators/` submodule
4. Wire into `rlm_type.rs::derive()`
5. Add unit test in relevant module
6. Add UI test if compile-fail matters (`tests/ui/`)

**Container attrs** (`#[rlm(...)]` on struct): `repr`, `iter`, `index`, `pyclass_name`, `property(name, desc)`

**Field attrs** (`#[rlm(...)]` on field): `desc`, `skip_python`, `filter_property`+`filter_value`, `flatten_property`

## Verification

```bash
cargo test -p rlm-derive                           # Unit tests
cargo test -p rlm-derive --test ui                 # UI tests (compile-fail)
cargo test -p dspy-rs --features rlm --test rlm_test  # Integration
```

Run UI tests when changing error messages. The `.stderr` files must match exactly.

## Don't do this

- **Don't use `#[derive(RlmType)]` directly** - use `#[rlm_type]` which adds `#[pyclass]`
- **Don't assume Clone** - `getter_strategy()` handles Copy/String/Clone separately
- **Don't `unwrap()` in generated code** - emit compile errors via `syn::Error`
- **Don't hardcode paths** - use `::dspy_rs::` prefix for all re-exports

## Gotchas

- **PyO3 re-export path:** Generated code uses `::dspy_rs::pyo3::*`. If dspy-rs re-export changes, all generated code breaks.
- **Derive order:** `#[rlm_type]` merges into existing `#[derive(...)]`, so RlmType sees other derives.
- **UI test fragility:** `trybuild` compares exact stderr. Rust version changes can break tests.
- **filter_property requires Vec<T>** - non-Vec fields error in `generators/describe.rs`.
- **Edition 2024:** Cargo.toml uses `edition = "2024"` - ensure toolchain supports it.

## Generated traits

`RlmType` generates:
- `#[pymethods]` block: getters, `__repr__`, `__baml__()`, `__rlm_schema__()`
- Optional: `__len__`, `__iter__`, `__getitem__` (via `iter`/`index` attrs)
- Optional: filter/flatten property methods

`RlmDescribe` (from `rlm-core`): `type_name()`, `fields()`, `properties()`, `is_iterable()`, `is_indexable()`, `describe_value()`
