# Upstream reference

Source repo: /Users/darin/vendor/github.com/BoundaryML/baml
Commit: 51662aa4fa0620a7ecba1699f2cf1626deaaed95

Yoinked crates:
- engine/baml-lib/baml-types
- engine/baml-lib/jinja-runtime
- engine/baml-lib/jsonish
- engine/baml-lib/diagnostics
- engine/baml-ids
- engine/bstd

Local trims:
- internal-baml-jinja: keep only output_format renderer (lib.rs + output_format/mod.rs).
- jsonish: drop helpers/tests modules and streaming helpers from compilation; replace
  internal-baml-core hooks with a local jinja predicate evaluator.
