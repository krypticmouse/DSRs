# internal-baml-jinja - Output Schema Renderer

## Boundary

**Purpose:** Vendored BAML crate for rendering output schemas to LLM prompts. Converts type definitions into human-readable schema instructions that guide LLM structured output.

**This is vendored code.** Modifications should be extremely rare and only for critical fixes.

**Key Types:**
- `OutputFormatContent` - Container for enums, classes, and target type to render
- `RenderOptions` - Controls schema formatting (hoisting, prefixes, map style)
- `Builder` - Constructs `OutputFormatContent` from type metadata
- `Class`, `Enum`, `Name` - Schema representations with alias support

**Dependencies:** `baml-types` (TypeIR, constraints), `minijinja` (template engine), `indexmap`.

**Depended on by:** `baml-bridge` (schema rendering), `jsonish` (type coercion).

## How to Work Here

**You probably shouldn't.** This is vendored upstream BAML code.

**Key Files:**
- `output_format/types.rs` - Core rendering logic, hoisting, prefix generation
- `format.rs` - `format_baml_value()` for JSON/YAML/TOON output

**Rendering Flow:**
1. `OutputFormatContent::target(TypeIR)` creates a `Builder`
2. Builder accumulates enums/classes, then `.build()`
3. `OutputFormatContent::render(RenderOptions)` produces the schema string

**Hoisting:** Recursive classes and large enums are "hoisted" - defined at the top of the schema and referenced by name to avoid infinite recursion and token waste.

## Verification

```bash
cargo test -p internal-baml-jinja
cargo check -p baml-bridge  # Ensure downstream still compiles
```

## Don't Do This

- **Don't add new features.** This renders existing BAML types, not new ones.
- **Don't refactor the rendering logic.** It's intricate and handles edge cases.
- **Don't modify hoisting heuristics** without understanding recursive class handling.
- **Don't change `RenderOptions` defaults** - downstream depends on them.

## Gotchas

- `OutputFormatContent::mk_fake()` exists for expression functions that need a placeholder
- Enum hoisting triggers when >6 variants or any variant has a description
- `MapStyle::TypeParameters` renders `map<K, V>`, `ObjectLiteral` renders `{K: V}`
- `inner_type_render` is entry point; `render_possibly_hoisted_type` handles nested recursive types
- The `or_splitter` default is `" or "` (used for union type rendering)

## References

- Upstream usage in DSRs: `crates/baml-bridge/src/lib.rs` - `render_schema<T: BamlType>()`
- Registry building: `crates/baml-bridge/src/registry.rs` - `RegistryBuilder::build(target)`
