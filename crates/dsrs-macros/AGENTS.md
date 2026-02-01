# dsrs-macros

## Boundary

Compile-time procedural macros for DSPy-rs signatures. **No runtime code**; output is generated Rust consumed by `dspy-rs`.

**Exports:**
- `#[derive(Signature)]` - generates `Signature` impl + helper structs (`{Name}Input`, `__{Name}Output`, `__{Name}All`)
- `#[derive(Optimizable)]` - parameter traversal for optimizers (`optim.rs`)
- `#[LegacySignature]` - **deprecated**, do not extend

**Depends on:** `syn`, `quote`, `proc-macro2`, `serde_json` (LegacySignature only)

**Depended on by:** `dspy-rs` (re-exports these macros)

**NEVER:**
- Import from `dspy-rs` (circular); use `::dspy_rs::` paths in generated code
- Emit `unwrap`/`expect` in generated code

---

## How to work here

### Golden examples

- **Signature derive:** `tests/signature_derive.rs` - alias, check, render attrs
- **Real-world usage:** `crates/dspy-rs/examples/01-simple.rs` - `QA` and `Rate` structs
- **UI error tests:** `tests/ui/check_missing_label.rs`, `render_conflict.rs`

### Field parsing flow (`src/lib.rs`)

```
expand_signature() -> parse_signature_fields() -> parse_single_field() -> parse_field_render_attr()
                   -> generate_signature_code() -> generate_helper_structs()
                                                -> generate_field_specs()
                                                -> generate_baml_delegation()
                                                -> generate_signature_impl()
```

### Adding a field attribute

1. Add to `FieldRenderAttr` struct (~line 71)
2. Parse in `parse_render_attr()` with duplicate checking (~line 434)
3. Generate tokens in `generate_field_specs()` (~line 721)
4. Add UI test in `tests/ui/` (trybuild auto-generates `.stderr`)

### Patterns

- Errors: `syn::Error::new_spanned(node, "message")`
- Duplicates: track with `Option<T>`, error on `Some`
- Generated paths: always `::dspy_rs::*` (absolute)
- Feature gates: `#[cfg(feature = "rlm")]` + stub for `#[cfg(not(...))]`

---

## Verification

```bash
cargo test -p dsrs_macros              # Core tests
cargo test -p dsrs_macros --test ui    # Compile-fail error matching
cargo test -p dsrs_macros --features rlm
cargo expand -p dspy-rs --example 01-simple  # Debug expansion (needs cargo-expand)
```

**Regenerate `.stderr`:** `TRYBUILD=overwrite cargo test -p dsrs_macros --test ui`

---

## Don't do this

- **`#[format]`** - removed; use `#[render(style = "...")]` (see `tests/ui/format_removed.rs`)
- **`#[LegacySignature]`** - deprecated
- **`anyhow` in macro code** - use `syn::Result<T>` + `to_compile_error()`
- **New helper struct fields** without updating `field_tokens()`, `generate_baml_delegation()`, `generate_signature_impl()`

---

## Gotchas

- **`r#fn` syntax:** `#[render(r#fn = path)]` because `fn` is keyword (line 34 in `tests/signature_derive.rs`)
- **trybuild:** Never edit `.stderr` manually; use `TRYBUILD=overwrite`
- **RLM feature:** `generate_rlm_input_impl()` has two cfg versions; both must compile
- **Static `type_ir`:** Uses `fn() -> TypeIR` for static array lifetimes
- **`#[check]` requires label, `#[assert]` does not** - enforced in `parse_constraint_attr()`
- **Visibility:** `{Name}Input` inherits parent; `__{Name}Output`/`__{Name}All` always `pub`

## References

- `crates/dspy-rs/src/core/signature.rs` - Signature trait
- `crates/baml-bridge/` - BAML traits (generated code uses `::dspy_rs::baml_bridge::*`)
