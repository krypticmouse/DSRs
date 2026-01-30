# Investigation: Nested SigMeta / CompiledSignature

## Summary

`SigMeta` and `CompiledSignature` are currently "flat" because they only store stringified type information (`type_name: String`, `schema: Option<String>`) rather than preserving the nested structure from `TypeIR`. Templates cannot traverse type hierarchies - they can only display pre-rendered strings.

## Symptoms

- `SigFieldMeta.type_name` is a simplified string like `"InsuranceClaim"` or `"list[Step]"`
- `SigFieldMeta.schema` is a pre-rendered text blob (for outputs only)
- Templates iterate flat lists: `{% for f in sig.inputs %}` with no access to nested field structure
- Input fields have no schema at all (`schema: None`)
- No way for templates to conditionally render based on type structure (class vs enum vs primitive)

## Investigation Log

### Phase 1 - Current Implementation Analysis

**Location:** `crates/dspy-rs/src/core/signature.rs:37-84`

**Findings:**

1. **`SigFieldMeta` is flat by design:**
```rust
pub struct SigFieldMeta {
    pub llm_name: &'static str,
    pub rust_name: &'static str,
    pub type_name: String,      // Just a string!
    pub schema: Option<String>, // Pre-rendered text, not structured
}
```

2. **`SigMeta::from_signature()` stringifies types immediately:**
```rust
// Lines 56-80
let ty = (field.type_ir)();
SigFieldMeta {
    type_name: simplify_type_name(&ty),  // Converts TypeIR → String
    schema: Some(build_field_schema(...)), // Renders to text
}
```

3. **`simplify_type_name()` deliberately loses structure:**
   - Strips module paths (`foo::bar::Baz` → `Baz`)
   - Removes `class `/`enum ` prefixes
   - Replaces ` | ` with ` or `
   - Result is human-readable but not machine-traversable

4. **`build_field_schema()` produces text, not data:**
   - Calls `OutputFormatContent::render()` → `Option<String>`
   - Good for display, unusable for conditional logic

### Phase 2 - Compilation Flow Analysis

**Location:** `crates/dspy-rs/src/core/signature/compiled.rs:215-240`

**Findings:**

1. **`CompileExt::compile()` builds a rich type registry but doesn't use it for SigMeta:**
```rust
let mut registry = Registry::new();
<Self::Input as BamlTypeInternal>::register(&mut registry);
<Self::Output as BamlTypeInternal>::register(&mut registry);

let (output_format, renderer_seed) = registry.build_with_renderers(TypeIR::string());
// output_format has full nested type info!

// But SigMeta ignores it:
sig_meta: SigMeta::from_signature::<Self>(), // Uses S::output_format_content() instead
```

2. **The registry's `output_format` contains both input AND output types**, but `SigMeta::from_signature()` calls `S::output_format_content()` which only includes output types.

3. **`PromptWorld` receives the rich type DB** but it's not exposed to templates via `sig`.

### Phase 3 - Template Limitations

**Location:** `crates/dspy-rs/src/core/signature/compiled.rs:22-42`

**Current templates:**
```jinja
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% endfor %}
```

**What templates CAN'T do today:**
- Check if a type is a class/enum/union/list
- Iterate over class fields
- Access enum variants
- Render differently for optional vs required
- Show nested type definitions inline

## Root Cause

**Architectural gap:** The `TypeIR` → `SigMeta` transformation discards structure too early.

```
TypeIR (rich, nested)
    ↓ simplify_type_name() / build_field_schema()
String (flat, display-only)
    ↓
SigFieldMeta (no structure)
    ↓
Jinja templates (can't traverse)
```

The information exists in `TypeIR` and `OutputFormatContent`, but it's converted to strings before reaching templates.

## Recommendations

### 1. Add `SigTypeMeta` - A Structured Type Representation

```rust
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SigTypeMeta {
    Primitive { display: String },
    List { item: Box<SigTypeMeta>, display: String },
    Map { key: Box<SigTypeMeta>, value: Box<SigTypeMeta>, display: String },
    Union { options: Vec<SigTypeMeta>, nullable: bool, display: String },
    Enum { name: String, values: Vec<SigEnumValueMeta>, display: String },
    Class { name: String, fields: Vec<SigClassFieldMeta>, recursive: bool, display: String },
    Ref { name: String, display: String }, // For cycles/depth limits
}

#[derive(Debug, Clone, Serialize)]
pub struct SigClassFieldMeta {
    pub name: String,
    pub description: Option<String>,
    pub r#type: SigTypeMeta,
}
```

### 2. Add `type_meta` to `SigFieldMeta`

```rust
pub struct SigFieldMeta {
    pub llm_name: &'static str,
    pub rust_name: &'static str,
    pub type_name: String,        // Keep for backwards compat
    pub schema: Option<String>,   // Keep for backwards compat
    pub type_meta: SigTypeMeta,   // NEW: structured type info
}
```

### 3. Build `SigMeta` from Compiled Registry

In `CompileExt::compile()`:
```rust
let (output_format, renderer_seed) = registry.build_with_renderers(TypeIR::string());

// NEW: Build sig_meta from the registry's output_format (has both input+output types)
let sig_meta = SigMeta::from_signature_with_format::<Self>(&output_format, opts);
```

### 4. Implement `build_type_meta()` Traversal

Similar to existing `format_baml_shape()` in `rlm/prompt.rs`, but returns structured data instead of strings:

```rust
fn build_type_meta(
    format: &OutputFormatContent,
    ty: &TypeIR,
    opts: SigMetaBuildOptions,
    depth: usize,
    visited: &mut HashSet<String>,
) -> SigTypeMeta
```

### 5. Enable Rich Templates

```jinja
{% macro render_type(t) %}
  {% if t.kind == "class" %}
{{ t.name }} {
    {% for f in t.fields %}
  {{ f.name }}: {{ render_type(f.type) }}
    {% endfor %}
}
  {% elif t.kind == "enum" %}
{{ t.name }} (one of: {% for v in t.values %}{{ v.name }}{% if not loop.last %}, {% endif %}{% endfor %})
  {% else %}
{{ t.display }}
  {% endif %}
{% endmacro %}
```

## Preventive Measures

1. **Keep structure as long as possible** - Only stringify at the final render step
2. **Provide both** - `display` for simple cases, structured data for advanced templates
3. **Share traversal logic** - Factor out the `TypeIR` → nested metadata conversion to avoid duplicating `format_baml_shape` logic

## Implementation Priority

1. `SigTypeMeta` enum + `SigClassFieldMeta`/`SigEnumValueMeta` structs
2. `build_type_meta()` traversal function
3. `SigMeta::from_signature_with_format()` 
4. Update `CompileExt::compile()` to use registry's output_format
5. (Optional) Update default templates to demonstrate capability
