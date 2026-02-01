# jsonish - BAML's LLM Output Parser

## Boundary

**Purpose:** Parse malformed/incomplete JSON from LLM outputs into typed BAML values.

**Depends on:** `baml-types`, `internal-baml-jinja`, `bstd`

**Depended on by:** BAML runtime (via `from_str()` in `lib.rs`)

**VENDORED CODE:** Minimize changes. Understand existing flow before modifying.

**NEVER:** Add parsing strategies without understanding cascade; change `Value` enum variants

---

## Two-Phase Architecture

1. **JSONish Parsing** (`src/jsonish/`) - Raw string to intermediate `Value`
   - Entry: `lib.rs` calls `jsonish::parse()` (in `jsonish/mod.rs` -> `parser/entry.rs`)
   - Strategy cascade: serde JSON -> markdown extraction -> multi-object finder -> fixing parser
   - Produces `Value::AnyOf` when multiple interpretations exist

2. **Type Coercion** (`src/deserializer/`) - `Value` to `BamlValueWithFlags`
   - Entry: `TypeCoercer::coerce()` impl on `TypeIR` in `coercer/field_type.rs`
   - Dispatches by target type: primitives, classes, enums, unions, maps
   - Uses scoring (`Flag` system) to pick best interpretation from `AnyOf`

### Key Files

- `src/lib.rs` - Public API: `from_str(of, target, raw_string, is_done)`
- `src/jsonish/value.rs` - `Value` enum (String, Number, Boolean, Null, Object, Array, AnyOf, FixedJson, Markdown)
- `src/jsonish/parser/entry.rs` - `parse_func()` with strategy cascade
- `src/deserializer/coercer/field_type.rs` - `impl TypeCoercer for TypeIR` (main dispatch)
- `src/deserializer/coercer/array_helper.rs` - `pick_best()` scoring logic for `AnyOf`
- `src/deserializer/score.rs` - `WithScore` trait, per-`Flag` penalty values
- `src/deserializer/deserialize_flags.rs` - `Flag` enum (all transformation markers)

---

## Verification

```bash
cargo check -p jsonish                    # Build check
cargo bench -p jsonish                    # Benchmarks (tests are disabled)
```

---

## Don't do this

- Don't add `Flag` variants without updating scoring in `score.rs`
- Don't simplify `AnyOf` handling - ambiguity exists for good reasons
- Don't add `unwrap()`/`expect()` in coercion paths (untrusted LLM input)
- Don't change strategy order in `entry.rs` (fallback cascade is load-bearing)

---

## Gotchas

- **Streaming:** `CompletionState` on `Value` tracks parse completion. Coercers propagate `Flag::Incomplete`.
- **Circular refs:** `ParsingContext::visit_class_value_pair()` tracks visited pairs to prevent infinite loops.
- **Scoring:** Lower score wins. `array_helper::pick_best()` does the actual selection, `score.rs` defines per-flag penalties.
- **String short-circuit:** `from_str()` returns raw string for `TypeIR::Primitive(String)` without parsing.
