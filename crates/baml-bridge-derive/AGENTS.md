# baml-bridge-derive

## Boundary

Proc macro crate that generates `#[derive(BamlType)]`. Produces four trait impls:
- `BamlTypeInternal` - TypeIR for schema generation
- `BamlValueConvert` - BamlValue -> Rust type
- `ToBamlValue` - Rust type -> BamlValue
- `BamlType` - builds OutputFormatContent (top-level)

**Depends on:** `syn`, `quote`, `proc_macro2`, `minijinja`, `convert_case`
**Used by:** Any crate deriving `BamlType`
**NEVER:** Generate code with `unwrap`/`expect` in conversion paths

---

## How to work here

**Golden patterns:**
- Struct derivation: `src/lib.rs::derive_struct` (line ~156)
- Enum derivation: `src/lib.rs::derive_enum` (line ~360)
- Integration tests: `../baml-bridge/tests/integration.rs` - shows all attribute combinations
- Golden snapshots: `../baml-bridge/tests/golden.rs` - schema output examples

**Attribute namespaces:** `#[baml(...)]`, `#[serde(...)]`, `#[render(...)]`

**Derivation flow (struct):**
1. Parse `ContainerAttrs` -> reject unit/tuple structs
2. Per field: parse `FieldAttrs` -> `type_ir_for_field` -> conversion tokens
3. Emit four trait impls

**Derivation flow (enum):**
- Unit variants only -> BAML enum OR `as_union` -> literal union
- Named-field variants -> tagged union via `#[baml(tag = "...")]`

**Supported types:**
- Primitives: `String`, `bool`, `f32/f64`, `i8..i64`, `isize`, `u8..u32`
- Large ints (`u64`, `usize`, `i128`, `u128`): require `#[baml(int_repr = "string"|"i64")]`
- Containers: `Option<T>`, `Vec<T>`, `Box<T>`, `Arc<T>`, `Rc<T>`, `HashMap<K, V>`, `BTreeMap<K, V>`
- Non-string map keys: require `#[baml(map_key_repr = "string"|"pairs")]`

**Adding attributes:**
1. Add to `ContainerAttrs`/`FieldAttrs`/`VariantAttrs` struct
2. Parse in `parse_*_attrs` function
3. Use in `derive_struct`/`derive_enum`
4. Add UI test in `tests/ui/` (this crate) or `../baml-bridge/tests/ui/` (type-level)

---

## Verification

```bash
cargo test -p baml-bridge-derive              # full suite
cargo test -p baml-bridge-derive --test ui    # UI tests only
cargo test -p baml-bridge                     # runtime + integration tests
```

Run after: attribute parsing, type matching, code generation changes.

---

## Don't do this

**Compile-time rejections:**
- `union` types, unit structs, tuple structs, tuple enum variants
- `serde(untagged)` -> use `#[baml(tag)]`
- `serde(flatten)` -> model explicitly
- `serde(skip)` on variants -> remove variant
- `serde(default = "path")` -> use `#[baml(default)]`
- Bare `serde_json::Value` -> use `#[baml(with)]` adapter
- Function types, trait objects

**Render restrictions:**
- `#[render(fn = ...)]` forbidden on fields/variants (type-level only)
- `#[render(...)]` on variants not supported for `as_union`/data enums

---

## Gotchas

- **Recursive types:** require `Box<Self>` or `Arc<Self>`
- **Template validation:** happens at macro time; invalid Jinja = compile error
- **Field refs in templates:** `{{ value.missing }}` fails unless `allow_dynamic = true`
- **`rename_all` precedence:** `#[baml(rename_all)]` beats `#[serde(rename_all)]`
- **Internal names:** default is `module_path!() + "::" + type_name`
- **Constraints:** require both `label` and `expr`: `#[baml(check(label = "x", expr = "..."))]`

---

## References

- `../baml-bridge/src/lib.rs` - runtime traits (`BamlType`, `BamlValueConvert`)
- `../baml-bridge/src/registry.rs` - `Registry` that collects generated defs
- `tests/ui/*.rs` - render attribute compile-fail tests
- `../baml-bridge/tests/ui/*.rs` - type-level compile-fail tests (serde_json::Value, large ints, etc.)
