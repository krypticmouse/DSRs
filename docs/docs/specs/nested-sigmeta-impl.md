# Nested SigTypeMeta Implementation Spec

## Goal

Replace flat `SigMeta` (stores `type_name: String`, `schema: Option<String>`) with fully nested `SigTypeMeta` that templates can traverse.

## New Types (`signature.rs`)

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
    Class { name: String, mode: String, dynamic: bool, recursive: bool, fields: Vec<SigClassField> },
    Ref { name: String },  // cycle breaker
    Other { description: String },  // Arrow, unknown variants
}

#[derive(Debug, Clone, Serialize)]
pub struct SigClassField {
    pub name: String,
    pub description: Option<String>,
    pub r#type: SigTypeMeta,
}

#[derive(Debug, Clone, Serialize)]
pub struct SigEnumValue {
    pub name: String,
    pub alias: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SigFieldMeta {
    pub llm_name: &'static str,
    pub rust_name: &'static str,
    pub description: Option<&'static str>,
    pub r#type: SigTypeMeta,
}

#[derive(Debug, Clone, Serialize)]
pub struct SigMeta {
    pub inputs: Vec<SigFieldMeta>,
    pub outputs: Vec<SigFieldMeta>,
}
```

## Traversal Function (`signature.rs`)

```rust
fn build_type_meta(
    format: &OutputFormatContent,
    ty: &TypeIR,
    visited: &mut HashSet<(String, StreamingMode)>,  // cycle detection
) -> SigTypeMeta
```

### TypeIR Mapping

| TypeIR variant | SigTypeMeta |
|----------------|-------------|
| `Top` | `Any` |
| `Primitive::String/Int/Float/Bool` | `Primitive { name }` |
| `Primitive::Media(_)` | `Primitive { name: "image" }` etc |
| `Literal { value }` | `Literal { value: value.to_string() }` |
| `List { inner }` | `List { item: build(inner) }` |
| `Map { key, value }` | `Map { key: build(key), value: build(value) }` |
| `Tuple { items }` | `Tuple { items: items.map(build) }` |
| `Union` | Use `union.view()` → `Union { options, nullable }` |
| `Enum { name, dynamic }` | Lookup `format.enums.get(name)` → `Enum { values }` |
| `Class { name, mode, dynamic }` | Lookup `format.classes.get((name, mode))` with fallback → `Class { fields }` |
| `RecursiveTypeAlias { name }` | Lookup `format.structural_recursive_aliases.get(name)` → recurse |
| `Arrow` | `Other { description }` |

### Class Lookup Fallback (match `format_baml_shape`)

1. Exact `(name, mode)`
2. `(name, NonStreaming)`
3. `(name, Streaming)`

### Cycle Detection

- Before expanding a class: check `visited.contains((name, mode))`
- If yes → return `Ref { name }`
- If no → insert, recurse, remove after

## SigMeta Builder (`signature.rs`)

```rust
impl SigMeta {
    pub fn from_format<S: Signature>(format: &OutputFormatContent) -> Self {
        let mut visited = HashSet::new();
        
        let inputs = S::input_fields().iter().map(|field| {
            let ty = (field.type_ir)();
            SigFieldMeta {
                llm_name: field.name,
                rust_name: field.rust_name,
                description: if field.description.is_empty() { None } else { Some(field.description) },
                r#type: build_type_meta(format, &ty, &mut visited),
            }
        }).collect();
        
        // Same for outputs
        
        Self { inputs, outputs }
    }
}
```

## Update `compile()` (`compiled.rs`)

```rust
impl<T: Signature> CompileExt for T {
    fn compile() -> CompiledSignature<Self> {
        let mut registry = Registry::new();
        <Self::Input as BamlTypeInternal>::register(&mut registry);
        <Self::Output as BamlTypeInternal>::register(&mut registry);

        let (output_format, renderer_seed) = registry.build_with_renderers(TypeIR::string());
        
        // NEW: Build SigMeta from registry's output_format (has both input + output types)
        let sig_meta = SigMeta::from_format::<Self>(&output_format);
        
        let mut world = PromptWorld::from_registry(/* ... */)?;
        // ...
        
        CompiledSignature { world, sig_meta, /* ... */ }
    }
}
```

## Update Default Templates (`compiled.rs`)

```jinja
{# System template #}
Your input fields are:
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {{ f.type | render_type }}
{% endfor %}

Your output fields are:
{% for f in sig.outputs -%}
- {{ f.llm_name }}: {{ f.type | render_type }}
{% endfor %}
```

Or inline type rendering:

```jinja
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {% if f.type.kind == "class" %}{{ f.type.name }}{% elif f.type.kind == "list" %}list[...]{% else %}{{ f.type.name }}{% endif %}
{% endfor %}
```

## Files to Modify

1. `crates/dspy-rs/src/core/signature.rs` - new types + `build_type_meta()` + `SigMeta::from_format()`
2. `crates/dspy-rs/src/core/signature/compiled.rs` - update `compile()`, update templates
3. `crates/dspy-rs/tests/test_compiled_signature.rs` - update test assertions

## Delete

- `simplify_type_name()` - no longer needed
- `build_field_schema()` - no longer needed
- `SigFieldMeta.type_name` / `SigFieldMeta.schema` - replaced by `type: SigTypeMeta`

## Reference

- `format_baml_shape` in `rlm/prompt.rs` - similar traversal pattern
- `OutputFormatContent` in `vendor/baml/.../output_format/types.rs` - type DB structure
- `TypeIR` in `vendor/baml/.../baml-types/src/ir_type/mod.rs` - all variants
