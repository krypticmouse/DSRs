# baml-bridge

A small set of crates that vendors the minimal BAML components needed for:
- schema rendering via `OutputFormatContent::render()`
- fuzzy LLM output parsing via `jsonish::from_str()`
- a thin Rust integration layer (`BamlType`, render/parse helpers)

## Layout

- `crates/baml-bridge` - public API surface (BamlType trait + helpers)
- `crates/baml-bridge-derive` - proc-macro for `#[derive(BamlType)]`
- `crates/baml-types` - TypeIR, constraints, BamlValue (yoinked)
- `crates/internal-baml-jinja` - output_format renderer only (trimmed)
- `crates/jsonish` - jsonish parser and coercer (trimmed)
- `crates/internal-baml-diagnostics` - dependency for `baml-types`
- `crates/baml-ids` - dependency for `baml-types`
- `crates/bstd` - dependency for `jsonish`

## Quick start

```
cargo check -p baml-bridge
cargo run -p baml-bridge --example manual
```

## Example

```rust
use baml_bridge::{parse_llm_output, render_schema, BamlType, HoistClasses, RenderOptions};

/// A user record returned by the model.
#[derive(Debug, Clone, PartialEq, BamlType)]
struct User {
    /// Full name for display.
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
}

fn main() {
    let schema = render_schema::<User>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();
    println!("{schema}");

    let raw = r#"{ "fullName": "Ada Lovelace", "age": 36 }"#;
    let parsed = parse_llm_output::<User>(raw, true).expect("parse failed");
    println!("{:?}", parsed.value);
}
```

## Attribute quick notes

- `#[baml(map_key_repr = "string")]` parses non-String map keys from JSON objects.
- `#[baml(map_key_repr = "pairs")]` represents maps as `[{ key: ..., value: ... }]`.
- `RenderOptions::hoist_classes(HoistClasses::All)` hoists all classes for compact unions.

## Yoink sources

These crates are copied from the BAML engine repo under:
- `engine/baml-lib/baml-types`
- `engine/baml-lib/jinja-runtime`
- `engine/baml-lib/jsonish`
- `engine/baml-lib/diagnostics`
- `engine/baml-ids`
- `engine/bstd`

We only trimmed modules that depend on the full compiler/runtime stack. The core
rendering and parsing logic is kept intact.

## Notes

- `internal-baml-jinja` is trimmed to `output_format::types` only.
- `jsonish` excludes streaming helpers and internal tests that depend on
  `internal-baml-core`.

## TODOs

See `TODO.md`.
