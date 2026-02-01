# AGENTS.md - baml-ids

## Boundary

**Purpose:** Type-safe prefixed UUIDv7 identifiers for BAML.

**Vendored from BoundaryML's BAML project. Modifications should be minimal.**

Depends on: `type-safe-id`, `uuid`, `serde`, `time`, `anyhow`
Depended on by: Other BAML crates needing stable, prefixed identifiers
NEVER: Add new ID types without upstream consideration

## ID Types

All defined via `define_id!` macro in `src/lib.rs`:
- `FunctionCallId` (`bfcall`) - top-level function call IDs
- `FunctionEventId` (`bfevent`) - content span IDs
- `HttpRequestId` (`breq`) - internal HTTP request IDs
- `ProjectId` (`proj`) - project identifiers
- `TraceBatchId` (`tracebatch`) - trace batch identifiers

## Verification

```bash
cargo check -p baml-ids
cargo test -p baml-ids
```

## Don't Do This

- Don't modify vendored code unless necessary
- Don't add business logic; this is pure ID generation
- Don't change existing prefixes (breaks serialization)
