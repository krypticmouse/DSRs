# AGENTS.md - ir_type (VENDORED)

## Boundary

Core TypeIR implementation: the `TypeGeneric<T>` ADT representing all BAML types.

**Depends on**: Parent crate internals (`BamlMediaType`, `Constraint`)
**Depended on by**: Everything using TypeIR - this is the foundation

## How to work here

**This is the most sensitive part of baml-types. Modifications have wide impact.**

Key abstractions:
- `TypeGeneric<T>` - parameterized by metadata type, has 12 variants (Top, Primitive, Enum, Literal, Class, List, Map, RecursiveTypeAlias, Tuple, Arrow, Union)
- `UnionTypeGeneric<T>` - union container with `view()` returning `UnionTypeViewGeneric` (Null/Optional/OneOf/OneOfOptional)
- Three metadata modes in `type_meta.rs`: `IR` (full), `NonStreaming`, `Streaming`

File responsibilities:
- `mod.rs`: TypeGeneric enum, UnionTypeGeneric, type aliases (TypeIR/TypeNonStreaming/TypeStreaming)
- `type_meta.rs`: Three metadata types + `MayHaveMeta` trait
- `builder.rs`: Factory methods (`TypeIR::string()`, `TypeIR::class("Foo")`, etc.)
- `union_type.rs`: `UnionConstructor` trait, `optional()` helpers
- `converters/streaming.rs`: IR -> Streaming conversion (adds nullability)
- `converters/non_streaming.rs`: IR -> NonStreaming conversion
- `simplify/ir.rs` + `simplify/non_streaming.rs`: Union flattening/deduplication

Critical invariants:
1. Unions with `@check` constraints are NOT flattened - preserves constraint metadata
2. `UnionTypeGeneric::new_unsafe` panics if all types are null
3. `UnionConstructor::union` auto-simplifies; use `new_unsafe` to bypass
4. Streaming conversion wraps non-optional types in `T | null`

## Verification

```bash
cargo test -p baml-types ir_type
```

Tests in `mod.rs` cover: simplify, flatten, partialize (streaming conversion), constraint preservation.

## Don't do this

- Do NOT add TypeGeneric variants without updating ALL match arms across the crate
- Do NOT bypass `UnionConstructor::union` unless you understand simplification implications
- Do NOT modify streaming conversion without verifying constraint propagation tests pass

## Gotchas

1. `iter_skip_null()` vs `iter_include_null()` - choose carefully for union iteration
2. `StreamingMode` on Class/RecursiveTypeAlias determines codegen behavior, not just metadata
3. `map_meta` is recursive - transforms metadata throughout nested types
