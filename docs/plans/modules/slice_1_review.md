# Slice 1 adversarial review

## High
- **Legacy MetaSignature prompts now emit dotted field names while the parser only accepts `\w+`.** `schema_fields_to_value` builds the JSON maps shown to `MetaSignature` using `FieldSchema::rust_name`, which is the dotted `FieldPath` (e.g. `detail.note`). That means the legacy prompts now request `[[ ## detail.note ## ]]`, but `FIELD_HEADER_PATTERN` (and therefore `parse_sections`) only matches `[[:word:]]+`, so the parser never sees those headers and `parse_response_strict` immediately claims `MissingField`. Any signature that flattens inputs/outputs is now uncompilable through `LegacyPredict` / the adapter path, which regresses the S1/S8 flatten guarantee in `docs/specs/modules/shapes.md:137-146`. Please align the legacy representation with the LM-facing name (`FieldSchema::lm_name`) or broaden the header regex so that dotted names survive, otherwise GEPA/optimizer tooling that still uses the legacy path cannot handle flattened signatures at all.— `crates/dspy-rs/src/predictors/predict.rs:295-303`, `crates/dspy-rs/src/adapter/chat.rs:27-87`, `docs/specs/modules/shapes.md:137-146`

## Medium
- None.

## Low
- None.
