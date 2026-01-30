# Nested SigTypeMeta Implementation Spec

**Status: PARTIAL - Infrastructure ready, nested #[render] not yet implemented**

## Summary

Added `SigTypeMeta` for structured type traversal while preserving `type_name`/`schema` for backward-compatible templates.

## Current State

`SigFieldMeta` has **both** legacy fields AND new structured type:

```rust
pub struct SigFieldMeta {
    pub llm_name: &'static str,
    pub rust_name: &'static str,
    pub description: Option<&'static str>,
    pub type_name: String,           // KEPT: for simple template access
    pub schema: Option<String>,      // KEPT: BAML-rendered schema
    pub r#type: SigTypeMeta,         // NEW: for nested #[render] introspection
}
```

## Implemented Types (`signature.rs`)

```rust
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SigTypeMeta {
    Any,
    Primitive { name: String },
    Literal { value: String },
    List { item: Box<SigTypeMeta> },
    Map { key: Box<SigTypeMeta>, value: Box<SigTypeMeta> },
    Tuple { items: Vec<SigTypeMeta> },
    Union { options: Vec<SigTypeMeta>, nullable: bool },
    Enum { name: String, dynamic: bool, values: Vec<SigEnumValue> },
    Class { name: String, dynamic: bool, recursive: bool, fields: Vec<SigClassField> },
    Ref { name: String },  // cycle breaker
    Other { description: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct SigClassField {
    pub name: String,
    pub description: Option<String>,
    pub r#type: SigTypeMeta,
    // FUTURE: pub render: Option<FieldRenderSpec>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SigEnumValue {
    pub name: String,
    pub alias: Option<String>,
    pub description: Option<String>,
}
```

## Key Implementation Details

### Type Traversal (`build_type_meta`)
- Recursively traverses `TypeIR` to build `SigTypeMeta`
- Uses `OutputFormatContent` to look up class/enum definitions
- **Cycle detection** via `VisitedTypes` struct:
  - `classes: HashSet<String>` - prevents class recursion
  - `aliases: HashSet<String>` - prevents recursive type alias loops

### SigMeta Construction
- `SigMeta::from_format<S: Signature>(format: &OutputFormatContent)`
- Populates BOTH legacy fields (`type_name`, `schema`) AND new `r#type`

### Default Template (unchanged)
Uses simple field access - no Jinja macros:
```jinja
{% for f in sig.outputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% if f.schema %}
{{ f.schema }}
{% endif %}
{% endfor %}
```

## What's NOT Implemented Yet

1. **Nested `#[render]` parsing** - `#[render]` on BamlType fields
2. **Render spec in SigClassField** - `pub render: Option<FieldRenderSpec>`
3. **PromptValue integration** - respecting nested specs during traversal
4. **`__type__` injection** - exposing SigTypeMeta to templates

See `handoffs/nested-render-handoff.md` for the full design.

## Files
- `crates/dspy-rs/src/core/signature.rs` - SigTypeMeta + legacy fields
- `crates/dspy-rs/src/core/signature/compiled.rs` - simple template (no macros)
