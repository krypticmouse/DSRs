# Investigation: Facet ↔ BAML Bridge Redundancy

## Summary

The codebase has **two independent paths** that produce BAML `TypeIR` from facet type metadata, causing divergence. BAML's native rendering is already used — the problem isn't that we're re-implementing rendering, it's that `SignatureSchema` builds its own TypeIR from raw `facet::Shape` while `bamltype`'s `SchemaBuilder` builds a richer, field-attr-aware TypeIR. These can disagree silently.

## The Two Paths (Root Problem)

### Path 1: `bamltype` SchemaBuilder (full-fidelity)
```
#[BamlType] struct → facet::Facet derive → facet attrs (bamltype::*)
    → SchemaBuilder::build_field_type_ir(field, owner, variant)
    → checks field attrs: with adapters, int_repr, map_key_repr
    → registers Classes/Enums into SchemaRegistry
    → builds OutputFormatContent (with recursive class detection)
    → cached in BamlSchema::baml_schema() as SchemaBundle
```
**Location:** `crates/bamltype/src/schema_builder.rs:453-490` (`build_field_type_ir`)

This path sees `#[baml(with="Codec")]`, `#[baml(int_repr="string")]`, `#[baml(map_key_repr="pairs")]` and transforms the TypeIR accordingly. An adapter can completely replace a field's type. A map can become a list of generated entry classes.

### Path 2: `SignatureSchema` (shape-only, loses field attrs)
```
#[derive(Signature)] struct → facet::Shape for Input/Output
    → collect_fields() iterates struct fields
    → emit_field() calls build_type_ir_from_shape(field.shape())
    → TypeIR built from shape alone, NO field attr awareness
    → stored in FieldSchema.type_ir
```
**Location:** `crates/dspy-rs/src/core/schema.rs:337` — the critical line:
```rust
let mut type_ir = build_type_ir_from_shape(field.shape());
```

This calls `schema_builder::build_type_ir_from_shape()` which creates a **fresh SchemaBuilder** and calls `build_type_ir(shape)` — NOT `build_field_type_ir(field, ...)`. It never sees field-level attributes.

### Where They're Used Together (Mismatch Surface)

| Consumer | Uses FieldSchema.type_ir (Path 2) | Uses OutputFormatContent (Path 1) |
|----------|----------------------------------|----------------------------------|
| `ChatAdapter::parse_structured_output_with_meta` | ✅ `jsonish::from_str(..., &field.type_ir, ...)` | ✅ `schema.output_format()` |
| RLM Output Contract (prompt.rs:99) | ✅ `field.type_ir.diagnostic_repr()` | ❌ |
| RLM py_bridge kwargs coercion | ✅ `field.type_ir` for dispatch | ✅ `output_format` for class/enum lookups |
| ChatAdapter field schema rendering | ❌ | ✅ `OutputFormatContent::render()` |

When Path 1 and Path 2 disagree (e.g., a field has `int_repr="string"` or `with="Codec"`), `jsonish` gets a TypeIR that says "int" while OutputFormatContent says "string" (or a completely different adapter type). This is a silent correctness bug.

## What BAML "Native" Actually Means Here

BAML's native rendering (`internal-baml-jinja`) is **already used**:
- `OutputFormatContent::render(options)` — schema prompt text ✅
- `jsonish::from_str(...)` — LLM output parsing ✅
- `format_baml_value(...)` — value formatting ✅

The custom bridge (`crates/bamltype`) builds the **inputs** to native rendering:
- `facet::Shape` → `TypeIR` (the type graph)
- `facet::Shape` → `OutputFormatContent` (class/enum registry)

You can't remove this bridge without a replacement source of truth (e.g., a BAML compiler, or user-authored BAML schemas).

## RlmType: Not a Schema Divergence

`#[rlm_type]` is a composition macro, not a competing schema system:
```rust
// rlm_attr.rs:43-45 — it literally just adds these:
input.attrs.push(syn::parse_quote!(#[pyclass(...)]));
input.attrs.push(syn::parse_quote!(#[BamlType]));
merge_derive(&mut input.attrs, &[syn::parse_quote!(RlmType)]);
```

`RlmType` derive adds Python interop methods (`__baml__`, `__repr__`, `__iter__`, etc.) that delegate to `BamlType` for conversion. There's no schema divergence here — it's a pure consumer of `bamltype`.

## Internal Name Drift

There's a subtle naming inconsistency between two functions that compute BAML internal names:

**`schema_builder::internal_name_for_shape(shape)`** (schema_builder.rs:44-55):
```rust
// Uses module_path::type_identifier
format!("{module}::{}", shape.type_identifier)
```

**`runtime::baml_internal_name::<T>()`** (runtime.rs:80-94):
```rust
// Falls back to std::any::type_name::<T>()
std::any::type_name::<T>()
```

`std::any::type_name` returns e.g. `my_crate::my_module::MyType` while `internal_name_for_shape` returns `my_module::MyType`. These could drift in edge cases, causing class lookup failures in value conversion or formatting.

## Complexity Hotspots

### 1. Adapter Function Pointers in Facet Attrs
`bamltype-derive` encodes function pointers (`WithAdapterFns`) into facet attribute metadata. These are `fn()` pointers stored as `&'static dyn Any` in compile-time reflection data. This works but is deeply non-obvious and makes the bridge hard to replace.

### 2. Map Key Repr "pairs" Generates Phantom Classes
`map_key_repr="pairs"` lowers `Map<K,V>` → `List<GeneratedMapEntry>` and registers a generated class. Any code that assumes maps stay maps will break.

### 3. Two Value Conversion Engines
- `bamltype/src/convert.rs`: Rust value ↔ BamlValue (facet Peek-based)
- `rlm/py_bridge.rs`: Python value → BamlValue (TypeIR + OutputFormatContent-aware)

Both walk value trees against schemas, both have relaxed parsing heuristics, both could diverge.

## Recommendations

### Fix 1: Make SignatureSchema source TypeIR from bamltype's SchemaBundle (HIGH PRIORITY)

Instead of:
```rust
let mut type_ir = build_type_ir_from_shape(field.shape());
```

Do one of:
- **Option A**: Look up the field's TypeIR from `<Output as BamlType>::baml_schema().output_format` class definitions
- **Option B**: Expose `SchemaBuilder::build_field_type_ir` as a public API that `SignatureSchema` can call

This eliminates the "two sources of truth" problem entirely.

### Fix 2: Unify internal name computation

Change `runtime::baml_internal_name::<T>()` fallback from `type_name::<T>()` to `internal_name_for_shape(T::SHAPE)`.

### Fix 3: Use OutputFormatContent::render for RLM Output Contract

Instead of `field.type_ir.diagnostic_repr()` (which uses the divergent Path 2 TypeIR), render the contract using the same native rendering used for structured output prompts.

### Fix 4 (Optional): Consolidate py_bridge coercion through jsonish

Normalize Python values to JSON, then use `jsonish::from_str(output_format, type_ir, ...)` instead of a parallel walker. Keeps one coercion engine.
