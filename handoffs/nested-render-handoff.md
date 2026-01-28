# Handoff: Nested `#[render]` for Input Fields

**Date:** 2026-01-30  
**Status:** Partially implemented, needs revert + redesign

---

## What We Were Trying To Do

**Goal:** Enable `#[render]` attributes on nested struct fields, not just top-level signature inputs.

```rust
#[derive(BamlType)]
struct Customer {
    name: String,
    #[render(style = "json")]  // <-- THIS should work
    metadata: HashMap<String, Value>,
}

#[derive(Signature)]
struct MySignature {
    #[input]
    customer: Customer,  // metadata should render as JSON when customer is rendered
    #[output]
    result: String,
}
```

**Current state:** `#[render]` only works on top-level `#[input]` fields in signatures.

---

## What We Built (and What's Wrong)

### What Was Built

1. **`SigTypeMeta` enum** - Structured type metadata with nested field info:
   ```rust
   pub enum SigTypeMeta {
       Primitive { name: String },
       List { item: Box<SigTypeMeta> },
       Class { name: String, fields: Vec<SigClassField>, ... },
       // ... other variants
   }
   ```

2. **`build_type_meta()` traversal** - Recursively builds `SigTypeMeta` from `TypeIR` + `OutputFormatContent`

3. **Cycle detection** - `VisitedTypes` struct prevents stack overflow on recursive types

4. **`SigMeta::from_format()`** - Builds metadata from registry's `OutputFormatContent` (has both input + output types)

### What's Wrong

**We broke the default template.** 

The original template used:
- `f.type_name` - simple string like `"Citation[]"`
- `f.schema` - BAML-rendered schema block from `OutputFormatContent::render()`

We replaced it with janky Jinja macros that re-implement BAML's type rendering badly:
```jinja
{%- macro render_type(t) -%}
{%- if t.kind == "primitive" %}{{ t.name }}
...
```

**This was wrong because:**
1. `#[render]` is for INPUT VALUE rendering, not OUTPUT SCHEMA display
2. BAML's `OutputFormatContent::render()` already handles output schemas well
3. We don't need `SigTypeMeta` for output schema rendering at all
4. The Jinja macros produce different output format (`list[T]` vs `T[]`)

---

## What To Delete/Revert

### Files to Modify

#### `crates/dspy-rs/src/core/signature.rs`

**DELETE these changes to `SigFieldMeta`:**
```rust
// WRONG - we removed type_name and schema
pub struct SigFieldMeta {
    pub llm_name: &'static str,
    pub rust_name: &'static str,
    pub description: Option<&'static str>,
    pub r#type: SigTypeMeta,  // ONLY this
}
```

**RESTORE to:**
```rust
pub struct SigFieldMeta {
    pub llm_name: &'static str,
    pub rust_name: &'static str,
    pub description: Option<&'static str>,
    pub type_name: String,        // RESTORE
    pub schema: Option<String>,   // RESTORE
    pub r#type: SigTypeMeta,      // KEEP
}
```

**RESTORE these deleted functions:**
- `simplify_type_name()` - produces `type_name` string
- `build_field_schema()` - produces `schema` string via BAML's render

**KEEP these new additions:**
- `SigTypeMeta` enum (all variants)
- `SigClassField` struct
- `SigEnumValue` struct
- `VisitedTypes` struct
- `build_type_meta()` function
- `lookup_class()` function
- `build_enum_values()` function
- `literal_to_string()` function (with serde_json fix)
- `simplify_name()` function
- `SigMeta::from_format()` method
- RecursiveTypeAlias cycle detection

**ADD:** `SigTypeMeta::display()` method to derive type name string:
```rust
impl SigTypeMeta {
    pub fn display(&self) -> String {
        match self {
            SigTypeMeta::Any => "any".to_string(),
            SigTypeMeta::Primitive { name } => name.clone(),
            SigTypeMeta::Literal { value } => value.clone(),
            SigTypeMeta::List { item } => format!("{}[]", item.display()),
            SigTypeMeta::Map { key, value } => format!("map[{}, {}]", key.display(), value.display()),
            SigTypeMeta::Tuple { items } => {
                let inner = items.iter().map(|i| i.display()).collect::<Vec<_>>().join(", ");
                format!("({})", inner)
            }
            SigTypeMeta::Union { options, nullable } => {
                let base = options.iter().map(|o| o.display()).collect::<Vec<_>>().join(" | ");
                if *nullable { format!("{} | null", base) } else { base }
            }
            SigTypeMeta::Enum { name, .. } => name.clone(),
            SigTypeMeta::Class { name, .. } => name.clone(),
            SigTypeMeta::Ref { name } => name.clone(),
            SigTypeMeta::Other { description } => description.clone(),
        }
    }
}
```

#### `crates/dspy-rs/src/core/signature/compiled.rs`

**DELETE the Jinja macros in `DEFAULT_SYSTEM_TEMPLATE`:**
```jinja
{%- macro render_type(t) -%}
...
{%- endmacro -%}

{%- macro render_class_fields(cls) -%}
...
{%- endmacro -%}

{%- macro render_enum_values(enm) -%}
...
{%- endmacro -%}

{%- macro render_type_details(t) -%}
...
{%- endmacro -%}
```

**RESTORE `DEFAULT_SYSTEM_TEMPLATE` to original:**
```rust
pub const DEFAULT_SYSTEM_TEMPLATE: &str = r#"
Your input fields are:
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% endfor %}

Your output fields are:
{% for f in sig.outputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% if f.schema %}
{{ f.schema }}
{% endif %}
{% endfor %}
"#;
```

**UPDATE `compile_signature_inner()`** to build both old fields AND new `r#type`.

#### `crates/dspy-rs/tests/test_typed_prompt_format.rs`

**RESTORE original test expectations:**
- `"- keywords: string[]"` not `"- keywords: list[string]"`
- Schema blocks with `doc_id`, `quote` fields
- Original assertions for `test_array_renders_with_brackets`, `test_schema_is_separate_from_type_line`, `test_nested_struct_with_comments`

---

## Design Decisions (FINAL)

### 1. `#[render]` is for INPUT VALUE rendering ONLY

- **NEVER** affects output schema display in system messages
- Output schemas use BAML's `OutputFormatContent::render()` unchanged
- Default template uses `f.type_name` and `f.schema` (BAML-rendered)

### 2. Innermost render spec wins

```rust
struct Inner {
    #[render(style = "compact")]
    value: String,
}

struct Outer {
    #[render(style = "json")]
    data: Inner,
}
```

When rendering `data.value`, **"compact" wins** (innermost).

### 3. `#[render]` should work on all types

Not just structs - strings, enums, primitives, collections. Examples:
- `#[render(max_string_chars = 100)]` on a String field
- `#[render(max_list_items = 5)]` on a Vec field

### 4. `SigTypeMeta` is for carrying nested render specs

NOT for rendering output schemas. It will eventually include:
```rust
pub struct SigClassField {
    pub name: String,
    pub description: Option<String>,
    pub r#type: SigTypeMeta,
    pub render: Option<FieldRenderSpec>,  // FUTURE: nested render override
}
```

### 5. Derive `type_name` from `SigTypeMeta::display()`

Don't store redundant `type_name` string. Use `SigTypeMeta::display()` method.

Actually, wait - we still need `type_name` for the template since Jinja can't call Rust methods. Options:
- A) Keep `type_name` field (redundant but simple)
- B) Serialize `display` as a field during serde
- C) Add a Jinja filter

**Decision needed** - but for now, keep `type_name` field for simplicity.

---

## Next Steps (P0 items to figure out in plan)

### 1. Where does `#[render]` on nested types get parsed?

Options:
- In `baml-bridge-derive` during `#[derive(BamlType)]`?
- A new attribute macro?
- Somewhere else?

Need to figure out:
- How render specs get attached to type metadata
- How they flow into `SigTypeMeta.fields[].render`

### 2. How does PromptValue find nested render specs at render time?

Current PromptValue rendering traverses BamlValue tree. To respect nested `#[render]`:
- Look up from global registry by type name + field path?
- Carry specs through SigTypeMeta passed at render call?
- Encode specs in BamlTypeInternal registration?

### 3. Integration with existing render pipeline

Current render chain: Field Override → Type Renderer → Builtin Style → Structural

Where do nested render specs fit? Probably:
- When traversing into a class field, check if that field has a render spec
- If yes, apply it before continuing the chain

---

## Files Changed (current state to revert from)

```
crates/dspy-rs/src/core/signature.rs             - PARTIALLY REVERT
crates/dspy-rs/src/core/signature/compiled.rs    - MOSTLY REVERT  
crates/dspy-rs/tests/test_typed_prompt_format.rs - FULLY REVERT
docs/docs/specs/nested-sigmeta-impl.md           - UPDATE
docs/docs/investigations/nested-sigmeta.md       - KEEP (historical)
```

---

## End State After Revert

1. **Default template works exactly like before** - `f.type_name`, `f.schema`
2. **`SigTypeMeta` structure exists** - ready for nested `#[render]` feature
3. **All original tests pass** - same output format as before
4. **`SigTypeMeta::display()` method available** - for future use
5. **No functional changes to user-facing behavior** - purely internal prep work

---

## UPDATED VISION (2026-01-30 late)

### The Real Goal: RLM-Style Programmatic Rendering

Look at how RLM renders `REPLHistory`:

```rust
#[render(default = r#"
{%- for entry in value.entries -%}
=== Step {{ loop.index }} ===
{{ entry.reasoning }}
```python
{{ entry.code }}
```
{% if entry.output.raw | length > ctx.max_output_chars %}
{{ entry.output.raw | slice_chars(ctx.max_output_chars) }}
... (truncated)
{% else %}
{{ entry.output }}
{% endif %}
{% endfor -%}
"#)]
pub struct REPLHistory { ... }
```

**Key features:**
- Templates traverse structured data: `value.entries`, `entry.output.raw`
- Runtime context: `ctx.max_output_chars`
- Custom filters: `slice_chars()`, `format_count()`
- Conditional rendering based on value inspection

### What SigTypeMeta Is Actually For

**NOT for:**
- `type_name` display strings (don't need it)
- Output schema rendering (BAML handles this)
- Default template simplification

**YES for:**
- Letting templates introspect type structure at render time
- Carrying `#[render]` specs on nested fields
- Enabling custom renderers that traverse the type tree

### Target UX

Users write:
```rust
#[derive(BamlType)]
struct Customer {
    name: String,
    #[render(max_string_chars = 50)]
    bio: String,
    #[render(style = "json")]
    metadata: HashMap<String, Value>,
}
```

Templates can do:
```jinja
{% for field in value.__type__.fields %}
  {% if field.render.max_string_chars %}
    {{ value[field.name] | truncate(field.render.max_string_chars) }}
  {% else %}
    {{ value[field.name] }}
  {% endif %}
{% endfor %}
```

Or the default renderer automatically respects nested `#[render]` specs when traversing.

### Revised Architecture

1. **`#[render]` parsed on BamlType fields** (in `baml-bridge-derive`?)
2. **Render specs stored in type metadata** (via `BamlTypeInternal::register`?)
3. **`SigTypeMeta` carries specs** for template introspection
4. **PromptValue rendering respects nested specs** when traversing
5. **Innermost spec wins** when specs conflict

### What To Do Now

1. **REVERT template changes** - restore `f.type_name`, `f.schema` for default
2. **KEEP `SigTypeMeta` structure** - but don't expose to default template yet
3. **KEEP cycle detection fixes** - RecursiveTypeAlias, etc.
4. **PLAN the full feature:**
   - Where to parse `#[render]` on BamlType fields
   - How specs flow into PromptValue rendering
   - How to expose `__type__` to templates

### Questions Still Open (for planning phase)

1. Does `#[render]` on BamlType go in `baml-bridge-derive` or new macro?
2. How does PromptValue get nested render specs at render time?
3. Should `__type__` be auto-injected or opt-in?
4. How do custom filters get registered per-type?

### Files Summary

**Revert:**
- `compiled.rs` DEFAULT_SYSTEM_TEMPLATE → original
- `test_typed_prompt_format.rs` → original expectations

**Keep:**
- `SigTypeMeta`, `SigClassField`, `SigEnumValue`
- `build_type_meta()`, cycle detection
- `SigMeta::from_format()`

**Restore:**
- `simplify_type_name()` for `type_name` field
- `build_field_schema()` for `schema` field
- `SigFieldMeta` with all three: `type_name`, `schema`, `r#type`

**Future (not now):**
- `SigClassField.render: Option<FieldRenderSpec>`
- Parse `#[render]` on BamlType fields
- PromptValue integration
