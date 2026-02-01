# deserializer/ - Type Coercion Phase

## Boundary

**Purpose:** Convert parsed `jsonish::Value` into typed `BamlValueWithFlags` for a target `TypeIR`.

**Depends on:** `../jsonish/` (Value enum), `baml-types` (TypeIR, BamlValue)

**Depended on by:** `../lib.rs` calls `TypeIR::coerce()` after parsing

**NEVER:** Modify parsing logic here - parsing is in `../jsonish/`; this is coercion only

---

## How to work here

**Entry point:** `impl TypeCoercer for TypeIR` in `coercer/field_type.rs`
- `coerce()` - Full coercion with fallbacks and transformations
- `try_cast()` - Fast path for exact matches (returns `None` to fall through to `coerce`)

**Adding a new type coercer:**
1. Create `coerce_xxx.rs` in `coercer/`
2. Export `coerce_xxx()` and optionally `try_cast_xxx()`
3. Wire into match arms in `field_type.rs`

**Golden examples:**
- Simple type: `coerce_primitive.rs`
- Complex container: `coerce_array.rs`
- IR reference: `ir_ref/coerce_class.rs`

---

## Scoring system (critical for AnyOf)

**Lower score = better match.** See `score.rs` for per-`Flag` penalties.

- `array_helper::pick_best()` selects winner from multiple candidates
- Complex tiebreakers exist: prefer non-default values, non-coerced strings, lists without parse errors
- When target is `Union`, special logic prefers structured over coerced-string matches

---

## Don't do this

- Don't add `Flag` variants without updating `score.rs` penalties
- Don't use `unwrap()`/`expect()` - input comes from untrusted LLM output
- Don't bypass `ParsingContext::visit_class_value_pair()` - it prevents infinite loops
- Don't assume `try_cast` returning `None` is an error - it signals "try coerce instead"

---

## Gotchas

- **Map has double conditions:** `BamlValueWithFlags::Map` stores `(DeserializerConditions, BamlValueWithFlags)` per entry - outer conditions are for the key, inner for value
- **`DefaultValue` trait:** Returns `None` if the default would fail asserts, not just if no default exists
- **Streaming:** Check `value.completion_state()` and propagate `Flag::Incomplete` - don't fail incomplete values
- **Union hint:** `ParsingContext::union_variant_hint` optimizes arrays of unions by remembering which variant worked
