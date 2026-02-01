# baml-bridge

## Boundary

**Purpose**: Bridge Rust types to BAML for LLM output parsing.

**Dependencies**: `baml-types` (TypeIR), `internal-baml-jinja` (schema rendering), `jsonish` (fuzzy parsing)

**Exports**: `BamlType`, `BamlTypeInternal`, `BamlValueConvert`, `ToBamlValue`, `BamlAdapter`, `Registry`, `Parsed<T>`, `parse_llm_output()`, `render_schema()`

**NEVER**: Use `TypeIR::Top` as output type (panics) | Bypass `Registry` for registration

---

## How to work here

### Core Flow
1. `#[derive(BamlType)]` generates `BamlTypeInternal` + `BamlValueConvert` + `ToBamlValue` impls
2. `render_schema::<T>()` -> TypeIR -> OutputFormatContent -> Jinja
3. `parse_llm_output::<T>(raw, is_done)` -> jsonish -> BamlValueWithFlags -> BamlValue -> T

### Key Traits
- **`BamlTypeInternal`**: `baml_internal_name()`, `baml_type_ir()`, `register(reg)`
- **`BamlValueConvert`**: `try_from_baml_value(value, path)` -> Result<Self, BamlConvertError>
- **`BamlType`**: Combines both + caches `OutputFormatContent`

### Golden Examples
- **Feature showcase**: `tests/integration.rs` - DocUser (alias), Shape (tagged enum), CheckedValue (constraints), BigIntString (int_repr), MapKeyPairs (map_key_repr)
- **Property roundtrip**: `tests/property.rs::RoundTripUser` - fuzz testing pattern
- **Schema snapshots**: `tests/golden.rs` - expect_test assertions
- **Compile-fail patterns**: `tests/ui/*.rs` + `.stderr`

---

## Verification

```bash
cargo test -p baml-bridge                    # Full suite
cargo test -p baml-bridge --test golden      # Schema snapshots
cargo test -p baml-bridge --test property    # Roundtrip fuzzing
cargo test -p baml-bridge --test ui          # Compile-time rejections
```

Update snapshots: `UPDATE_EXPECT=1 cargo test -p baml-bridge --test golden`

---

## Don't do this

| Pattern | Use Instead |
|---------|-------------|
| Tuple structs | Named-field struct |
| `#[serde(untagged)]` | `#[baml(tag = "...")]` |
| `#[serde(flatten)]` | Explicit fields |
| `serde_json::Value` | Concrete type or `#[baml(with = "...")]` |
| `u64`/`i128` without repr | `#[baml(int_repr = "string"\|"i64")]` |
| Non-String map keys | `#[baml(map_key_repr = "string"\|"pairs")]` |
| Unit structs / tuple variants | Struct with fields / struct variant |
| `#[serde(default = "path")]` | `#[serde(default)]` or `#[baml(default)]` |
| `#[serde(skip)]` on variants | Remove variant |

---

## Gotchas

**Constraints**: `@check` populates `Parsed::checks` (non-fatal) | `@assert` causes `BamlParseError::ConstraintAssertsFailed`

**Recursion**: Detected via `Registry::compute_recursive_classes()` (Tarjan's SCC)

**Name collisions**: Types get unique internal names via module path; `rendered_name()` is user-facing, `real_name()` is Rust field name

**Parsed<T>**: Contains `value`, `baml_value`, `flags` (coercions), `checks` (constraint results), `explanations` (parse trace)

---

## References

- `README.md` - Quick start and example usage
- `../baml-bridge-derive/AGENTS.md` - Derive macro internals and attribute parsing
- `tests/ui/*.stderr` - Exact error messages for forbidden patterns
