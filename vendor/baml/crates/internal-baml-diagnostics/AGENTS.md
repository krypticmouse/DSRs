# internal-baml-diagnostics

## Boundary

Diagnostic infrastructure for BAML parser/validator: source files, spans, errors/warnings, pretty-printing.

**VENDORED from BoundaryML. Do not modify unless upstreaming.**

- Depends on: `pest`, `colored`, `strsim`, `serde`
- Depended on by: Other `internal-baml-*` crates needing error reporting
- NEVER: Add DSRs-specific logic here

## Key Types

| Type | File | Purpose |
|------|------|---------|
| `SourceFile` | `source_file.rs` | BAML source file (path + contents) |
| `Span` | `span.rs` | Location in source (file + byte range), converts to line/col |
| `SerializedSpan` | `span.rs` | JSON-serializable span for JS playground |
| `DatamodelError` | `error.rs` | Parser/validation error with span |
| `DatamodelWarning` | `warning.rs` | Non-fatal warning with span |
| `Diagnostics` | `collection.rs` | Accumulates errors/warnings for batch reporting |

## Patterns

- Error constructors: `DatamodelError::new_*` family for specific error types
- Fuzzy suggestions: `sort_by_match()` uses OSA distance for "did you mean?" hints
- Multi-span errors: Some errors reference multiple locations (e.g., duplicates)

## If You Must Modify

1. Ensure changes are generic enough to upstream
2. Run: `cargo test -p internal-baml-diagnostics`
3. Visually verify error formatting - pretty-print output matters for UX
