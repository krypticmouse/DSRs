# S7 Spike: `#[derive(Augmentation)]` Feasibility + `Augmented<S, A>` Phantom Type

## Context

F3 specifies `#[derive(Augmentation)]` which generates a generic wrapper type from a non-generic input struct, and `Augmented<S, A>` as a phantom type-level combinator. Neither was verified against the existing derive infrastructure. The design reference has a literal `todo!()` in `Augmented::from_parts`.

## Goal

Determine if `#[derive(Augmentation)]` is feasible given existing Facet/BamlType derive capabilities, and resolve the `Augmented` phantom type design.

## Questions

| ID | Question |
|---|---|
| **S7-Q1** | Can a proc macro generate a generic struct (`WithReasoning<O>`) from a non-generic input (`Reasoning`)? |
| **S7-Q2** | Do Facet and BamlType derives handle generic types with trait bounds? |
| **S7-Q3** | Does anything in the call pipeline actually call `from_parts`/`into_parts` on a Signature? |
| **S7-Q4** | What is the correct design for `Augmented<S, A>` given the phantom type problem? |

## Findings

1. **Generating a generic struct from non-generic input is standard proc macro practice.** The macro receives a token stream and can emit any valid Rust. The `#[derive(Signature)]` macro already generates new structs (`{Name}Input`, `__{Name}Output`) from the input. Adding a generic parameter is a small incremental change.

2. **Facet derive handles generics.** Evidence from `facet-macros-impl/src/derive.rs:46-66` and `process_struct.rs:1459-1471`: it detects `has_type_or_const_generics`, uses `TypeOpsIndirect` for generic types, and generates proper `where` clauses via `build_where_clauses()` that automatically adds `T: Facet<'f>` bounds. It skips static SHAPE generation for generic types (can't monomorphize a static), but `<T as Facet>::SHAPE` works at call sites.

3. **BamlType derive handles generics.** `bamltype-derive/src/lib.rs:139` already calls `input.generics.split_for_impl()` and threads `impl_generics`, `ty_generics`, `where_clause` through generated code. `OnceLock` statics work per-monomorphization — each concrete `WithReasoning<QAOutput>` gets its own cache entry.

4. **`from_parts`/`into_parts` ARE called in 5 places:**

   | Location | Method | Purpose |
   |----------|--------|---------|
   | `predict.rs:179` | `S::from_parts(input, typed_output)` | Reassemble full signature struct from input + parsed output |
   | `predict.rs:332` | `S::from_parts(input, output)` | Construct typed signature from untyped Example |
   | `predict.rs:340` | `signature.into_parts()` | Convert typed signature to untyped Example |
   | `predict.rs:410` | `call_result.output.into_parts()` | Extract output for Module::forward |
   | `chat.rs:575` | `demo.into_parts()` | Split demo into input + output for formatting |

5. **All 5 call sites exist to support the pattern of combining input+output into one struct, then splitting back apart.** The design reference already calls for demos to be `Vec<Demo<S>>` (typed input + output pairs, not `Vec<S>`). If Predict stores pairs and returns `Output` directly, the round-trip is unnecessary.

## Decision

**Remove `from_parts`/`into_parts` from the `Signature` trait.**

The new trait:
```rust
pub trait Signature: Send + Sync + 'static {
    type Input: BamlType + Facet + Send + Sync;
    type Output: BamlType + Facet + Send + Sync;
    fn instructions() -> &'static str { "" }
}
```

`Augmented<S, A>` becomes a clean type-level combinator:
```rust
pub struct Augmented<S: Signature, A: Augmentation>(PhantomData<(S, A)>);

impl<S: Signature, A: Augmentation> Signature for Augmented<S, A> {
    type Input = S::Input;
    type Output = A::Wrap<S::Output>;
    fn instructions() -> &'static str { S::instructions() }
}
```

No `todo!()`, no `unreachable!()`.

The user's `#[derive(Signature)]` still generates the combined struct with all fields (for ergonomic `result.question` / `result.answer` access), but that's a convenience on the user's type, not a requirement of the trait.

Downstream changes:
- `Predict::call()` returns `S::Output` directly
- Demos stored as `Demo<S> { input: S::Input, output: S::Output }`
- `format_demo_typed` takes `(&S::Input, &S::Output)` instead of `&S`

## Acceptance

S7 is complete when:
- We can confirm all three derive macros (Facet, BamlType, Augmentation) support generating/handling generic types
- The `Augmented` phantom type works without `from_parts`/`into_parts`
- The `Signature` trait simplification is reflected in the design reference
