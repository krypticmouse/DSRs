# AGENTS.md - baml-types (VENDORED)

## Boundary

**This is vendored upstream code from BoundaryML's BAML. Do not modify unless absolutely necessary.**

TypeIR (Type Intermediate Representation) - foundational type system for LLM output schemas.

**Purpose**: Defines `TypeGeneric<T>`, the core ADT representing all BAML types (primitives, classes, enums, unions, maps, lists, tuples, arrows).

**Depends on**: `baml-ids`, `internal-baml-diagnostics`, minijinja

**Depended on by**:
- `baml-bridge`: Uses `TypeIR` for schema generation, parsing, Rust type derivation
- `jsonish`: Uses `TypeIR` for coercing LLM output into typed values

## How to work here

**You should almost never modify this crate.**

Golden patterns:
- Type construction: `builder.rs` - factory methods like `TypeIR::string()`, `TypeIR::class("Foo")`
- Type metadata: `type_meta.rs` - three modes: `IR`, `NonStreaming`, `Streaming`
- Union handling: `mod.rs` defines `UnionTypeViewGeneric` to classify union variants

Key concepts:
- `TypeGeneric<T>` parameterized by metadata type (IR/NonStreaming/Streaming)
- `TypeIR` = `TypeGeneric<type_meta::IR>` - base representation with streaming behavior
- Streaming types add nullability for partial LLM responses
- Constraints (`@check`, `@assert`) stored in type metadata

Module layout:
- `ir_type/mod.rs`: Core `TypeGeneric` enum, `UnionTypeViewGeneric`
- `ir_type/type_meta.rs`: Metadata variants (IR, NonStreaming, Streaming)
- `ir_type/builder.rs`: Factory methods for type construction
- `ir_type/converters/`: IR -> Streaming/NonStreaming transforms
- `ir_type/simplify/`: Union flattening/deduplication
- `baml_value.rs`: Runtime values (distinct from TypeIR)

## Verification

```bash
cargo test -p baml-types
```

## Don't do this

- Do NOT add new public APIs without extreme justification
- Do NOT modify `TypeGeneric` enum variants
- Do NOT change serialization formats (breaks BAML ecosystem compatibility)
- Do NOT add dependencies (Cargo.toml requests minimal deps)

**Local patches**: If necessary, tag with `// DSRs-LOCAL:` and list below.

**Current local patches**: None

## Gotchas

1. **TypeIR vs BamlValue**: `TypeIR` = schema-time types; `BamlValue` = runtime values.

2. **Streaming nullability**: IR -> Streaming makes primitives `Optional<T>` for partial output.

3. **Union flattening**: Unions with `@check` are NOT flattened to preserve constraint metadata.

4. **MayHaveMeta trait**: Unifies operations across `TypeGeneric<T>` variants.

5. **Upstream**: From `github.com/BoundaryML/baml`. Check `TypeGeneric`/`type_meta` for breaking changes when syncing.
