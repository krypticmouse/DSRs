# S3 Spike: Augmentation Deref Composition

## Context

S3 is the explicit open question for F3/R13: whether nested augmentation wrappers can expose fields ergonomically via deref chaining, and what fallback is needed if that breaks (`docs/specs/modules/shapes.md:142`, `docs/specs/modules/shapes.md:45`, `docs/specs/modules/shapes.md:60`).

The technical design already encodes tuple composition as nested wrappers and leaves multi-layer deref behavior as the unresolved spike (`docs/specs/modules/design_reference.md:357`, `docs/specs/modules/design_reference.md:362`, `docs/specs/modules/design_reference.md:370`).

## Goal

Pin down real Rust ergonomics for composed wrappers (read/method/mutation/pattern behavior) and identify the minimum architecture changes required to ship tuple augmentation safely.

## Questions

| ID | Question |
|---|---|
| **S3-Q1** | Do field reads and method calls auto-deref through multiple augmentation wrapper layers? |
| **S3-Q2** | What are the exact limits for mutation and pattern matching through wrapper layers? |
| **S3-Q3** | What in the current local macro + adapter pipeline blocks composed augmentation rollout? |
| **S3-Q4** | What does the target F2/F7 design require for path-aware flatten parsing? |

## Findings

1. F3 is intentionally built around nested wrappers + `Deref`; S3 is specifically about the remaining ergonomics risk.
   - Evidence: `Augmentation::Wrap` requires `Deref<Target = T>` (`docs/specs/modules/design_reference.md:259`).
   - Evidence: tuple composition is nested (`A::Wrap<B::Wrap<T>>`) (`docs/specs/modules/design_reference.md:361`, `docs/specs/modules/design_reference.md:362`).
   - Evidence: S3 is explicitly called out for multi-layer deref behavior (`docs/specs/modules/design_reference.md:370`).

2. Field reads and method calls through nested wrappers work with current Rust deref coercions.
   - Evidence: probe assertions for `result.reasoning`, `result.confidence`, `result.answer` pass (`/tmp/s3_deref_probe.rs:66`, `/tmp/s3_deref_probe.rs:67`, `/tmp/s3_deref_probe.rs:68`).
   - Evidence: method call through nested wrappers passes (`/tmp/s3_deref_probe.rs:71`).
   - Validation: `rustc /tmp/s3_deref_probe.rs -o /tmp/s3_deref_probe_check && /tmp/s3_deref_probe_check` succeeded.

3. Mutation is conditional: it works only when every wrapper layer implements `DerefMut`.
   - Evidence: mutation passes in probe where both wrappers implement `DerefMut` (`/tmp/s3_deref_probe.rs:48`, `/tmp/s3_deref_probe.rs:74`).
   - Evidence: without `DerefMut`, mutation fails with E0596 (`/tmp/s3_deref_no_mut.rs:46`).
   - Evidence: current design sketch only shows `Deref` impl on generated wrapper (`docs/specs/modules/design_reference.md:289`).

4. Pattern matching is not auto-deref ergonomic.
   - Evidence: direct match of inner wrapper from outer value fails with E0308 (`/tmp/s3_pattern_fail.rs:37`).
   - Evidence: explicit layer-by-layer destructuring works (`/tmp/s3_deref_probe.rs:78`, `/tmp/s3_deref_probe.rs:79`, `/tmp/s3_deref_probe.rs:80`).
   - Evidence: explicit deref in pattern contexts is limited for non-`Copy` data (E0507) (`/tmp/s3_pattern_double_deref.rs:21`).

5. The real rollout blocker is current top-level-only `FieldSpec` metadata/parsing, not deref read ergonomics.
   - Evidence: current derive attribute set excludes `flatten` (`crates/dsrs-macros/src/lib.rs:22`).
   - Evidence: generated metadata is `FieldSpec` with no field path (`crates/dsrs-macros/src/lib.rs:551`, `crates/dspy-rs/src/core/signature.rs:6`).
   - Evidence: typed formatting looks up only top-level `field_spec.rust_name` (`crates/dspy-rs/src/adapter/chat.rs:530`, `crates/dspy-rs/src/adapter/chat.rs:531`).
   - Evidence: typed parse inserts output values at top-level rust names (`crates/dspy-rs/src/adapter/chat.rs:607`, `crates/dspy-rs/src/adapter/chat.rs:720`, `crates/dspy-rs/src/adapter/chat.rs:739`).

6. The target design already defines the required path-aware model; this is what must land before tuple composition is considered done.
   - Evidence: `FieldSchema` includes `path` (`docs/specs/modules/design_reference.md:175`).
   - Evidence: flatten reconstruction requires `["inner", "answer"]`-style paths (`docs/specs/modules/design_reference.md:334`, `docs/specs/modules/design_reference.md:336`).
   - Evidence: F7 parse/format path navigation is the intended mechanism (`docs/specs/modules/design_reference.md:642`, `docs/specs/modules/design_reference.md:681`, `docs/specs/modules/design_reference.md:692`).
   - Evidence: design decision D2 explicitly rejects static `FieldSpec` arrays for this use case (`docs/specs/modules/design_reference.md:991`).
   - Evidence: local conversion already uses flatten-aware reflective serialization (`crates/bamltype/src/convert.rs:611`, `crates/bamltype/src/convert.rs:612`).

## Options + Risks

| Option | Approach | Risks |
|---|---|---|
| **A. Keep nested wrappers + ship path-aware schema/adapter** | Keep `A::Wrap<B::Wrap<T>>`; guarantee read/method ergonomics via `Deref`; migrate formatting/parsing to path-aware metadata (F2/F7). | Requires coordinated macro + signature + adapter migration; biggest integration surface but matches selected design. |
| **B. Generate a flat tuple wrapper** | Special-case `(A, B, ...)` into one merged wrapper instead of nesting. | More proc-macro complexity, weaker composability, drifts from F3 design and D1/D8 decisions. |
| **C. Keep nested wrappers + explicit accessor traits** | Preserve nesting, avoid deref field ergonomics by requiring trait methods for composed fields. | Regresses R2/R13 ergonomics and adds trait/coherence boilerplate. |

## Recommended Approach

Proceed with **Option A**.

Refined conclusions:
- Treat nested deref as sufficient for primary ergonomics: field reads and method calls.
- Treat pattern matching as intentionally explicit API surface (not auto-deref ergonomic).
- Decide mutability contract explicitly:
  - If outputs are read-only at call sites, keep `Deref` only.
  - If mutable field edits are part of supported ergonomics, add `DerefMut` generation and test it.
- Gate R13 completion on path-aware parse/format migration; do not ship tuple augmentation on top-level-only `FieldSpec`.

## Concrete Implementation Steps

1. Add compile-pass tests for nested wrapper read and method-call ergonomics (`result.reasoning`, `result.confidence`, `result.answer`, method dispatch).
2. Add compile-fail tests for unsupported ergonomics:
   - direct inner-wrapper pattern match from outer wrapper,
   - mutation through wrappers when `DerefMut` is absent.
3. Finalize augmentation mutability contract (`Deref`-only vs `Deref` + `DerefMut`) and codify it in derive output + docs.
4. Implement/land path-aware signature metadata (`FieldPath`) in the typed path.
5. Migrate typed adapter formatting/parsing to path navigation/insertion instead of top-level `rust_name` map access.
6. Add integration roundtrip tests for two-layer composed augmentations ensuring no dropped fields and deterministic field ordering.
7. Document pattern-matching limitations and explicit destructuring guidance for module authors.

## Open Risks

- Path-aware migration spans macro output, signature traits, and adapter internals; partial rollout risks mixed metadata models.
- S1 (generic `#[derive(Signature)]` + flatten) remains a dependency for some high-value module authoring flows (`docs/specs/modules/shapes.md:140`).
- If mutable composed outputs become part of the public ergonomic promise, derive codegen and trait bounds will expand.

## Acceptance

S3 is complete when:

- Tests prove nested wrapper field reads and method calls work for composed augmentations.
- Tests/docs explicitly capture pattern-matching limits and supported destructuring patterns.
- The mutability contract is explicit and test-backed (`Deref`-only or `Deref` + `DerefMut`).
- Typed prompt formatting/parsing roundtrips composed augmentation outputs using path-aware metadata.
- R13 is evaluated against the path-aware implementation, not against the legacy top-level `FieldSpec` path.
