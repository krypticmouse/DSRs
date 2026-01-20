# ADR-001: BAML-Bridge Typed DSPy-RS Integration Decisions

Status: Accepted
Date: 2026-01-09

## Context
DSPy-RS is migrating from an untyped HashMap-based API to a typed Rust API backed by
BAML-Bridge. The integration required a set of architectural decisions to balance
macro ergonomics, runtime behavior, and backward compatibility with existing modules
and optimizers.

## Decisions

### 1) Three Generated Structs Pattern

Context: `derive(Signature)` cannot add derives to the original struct.

Decision: Generate three structs per signature:
- `QAInput` (public) with input fields only
- `__QAOutput` (hidden) with output fields only
- `__QAAll` (hidden) with all fields for BamlType delegation

Rationale:
- Procedural macros can add impls but not new derives on the original type.
- Shadow structs allow us to reuse `derive(BamlType)` without manual impls.

Alternatives considered:
- Manual `BamlType` impls: more macro complexity and boilerplate.
- Require users to derive `BamlType` manually: worse UX and higher burden.

### 2) ToBamlValue Instead of PromptParts

Context: Typed input rendering needed structured access to field values.

Decision: Use `ToBamlValue` for all prompt rendering instead of adding `PromptParts`.

Rationale:
- `ToBamlValue` is already required for demo serialization.
- `PromptParts` would duplicate field extraction logic.
- Fewer traits keeps the API simpler.

### 3) Function Pointers for TypeIR

Context: `TypeIR` is not const-constructible, but `FieldSpec` needs static lifetime.

Decision: Store `type_ir: fn() -> TypeIR` in `FieldSpec`.

Rationale:
- Deferred construction keeps the static field list viable.
- Avoids const-eval limitations for heap allocations.

Example:
```rust
fn __qa_confidence_type_ir() -> TypeIR {
    <f32 as BamlTypeInternal>::baml_type_ir()
}

static __QA_OUTPUT_FIELDS: &[FieldSpec] = &[
    FieldSpec { type_ir: __qa_confidence_type_ir, .. },
];
```

### 4) ParseError::Multiple with Partial Results

Context: LLM responses often contain multiple field errors.

Decision: Aggregate all field errors in `ParseError::Multiple` and return partial
results for successfully parsed fields.

Rationale:
- Improves debugging by surfacing all issues at once.
- Enables partial recovery when some fields parse successfully.

### 5) Keep Marker Protocol `[[ ## field ## ]]`

Context: Alternatives included JSON-only responses or other formatting strategies.

Decision: Preserve the existing marker-based protocol.

Rationale:
- Proven in current DSPy-RS prompts.
- Allows natural language plus structured fields in one response.
- Fewer escaping pitfalls than JSON-only responses.

### 6) BamlValue as Interchange Format

Context: Optimizers need to work generically across signatures.

Decision: Use `BamlValue` as the untyped interchange format.

Rationale:
- Single conversion layer (typed <-> BamlValue).
- Optimizers can inspect and modify field values uniformly.
- Avoids leaking `serde_json::Value` into user-facing APIs.

### 7) Check vs Assert Constraint Semantics

Context: Constraints can be informational or hard failures.

Decision:
- `#[check]`: soft constraint, recorded in metadata.
- `#[assert]`: hard constraint, fails parse.

Rationale:
- Aligns with BAML semantics.
- Lets users track violations without failing if desired.

### 8) Separate ToBamlValue Trait (Not on BamlType)

Context: Spec suggested adding `to_baml_value()` to `BamlType`.

Decision: Keep `ToBamlValue` as a standalone trait.

Rationale:
- Avoids modifying baml-bridge core traits.
- Maintains clear separation of "to" vs "from" conversions.

### 9) No PromptParts Trait

Context: Early plan included a PromptParts trait for formatting.

Decision: Removed PromptParts (bead dsrs-ohh.4.5); use `ToBamlValue` instead.

Rationale:
- Reduces conceptual overhead.
- Same capability via `BamlValue::Class` field inspection.

## Implementation Notes

- ChatAdapter no longer depends on macro completion; handwritten signatures are
  sufficient for early development.
- Tracing integration uses a lightweight Predict node that stores signature name only.
- JsonishError wrappers keep `anyhow::Error` out of the public API.
