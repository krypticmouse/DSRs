# TODO

- [x] Add `baml-bridge-derive` proc-macro crate and wire `#[derive(BamlType)]`
- [x] Implement a registry (dependency collection + recursion detection + stable ordering)
- [x] Generate conversions from `BamlValue` with path-aware errors
- [x] Enforce representability invariants (ints, maps, tuples) in the derive macro
- [x] Doc comment extraction → BAML descriptions (type/field/variant)
- [x] Enum descriptions in output renderer
- [x] Add end-to-end integration tests for render/parse/convert flows
- [x] Add list-of-pairs map key representation

- [x] Add compile-fail UI tests for unsupported serde patterns and types
- [ ] Re-enable jsonish streaming helpers behind a feature flag
- [ ] Port or trim jsonish tests to run without full compiler stack
- [x] Add round-trip property tests and golden schema snapshots
