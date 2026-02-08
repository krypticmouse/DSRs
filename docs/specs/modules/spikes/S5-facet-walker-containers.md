# S5 Spike: Facet Walker Containers

## Context

S5 asks how Facet-based parameter discovery should traverse containers (`Option`, `Vec`, `HashMap`, `Box<dyn Module>`) while still yielding optimizer-usable dotted paths and leaf handles. It is called out as high-priority in shaping/design because it blocks reliable automatic predictor discovery for non-trivial module trees.

Relevant framing:
- `docs/specs/modules/shapes.md:63` defines F6 as recursive container traversal for predictor discovery.
- `docs/specs/modules/shapes.md:244` defines S5 scope explicitly (`Option`/`Vec`/`HashMap`/`Box<dyn Module>`).
- `docs/specs/modules/design_reference.md:530` sketches expected walker behavior by `shape.def`.

## Goal

Establish a concrete first-pass container traversal strategy for S5, grounded in:
- current repo behavior (what works today vs gaps),
- Facet primitives/capabilities (NIA evidence),
- and explicit limits that affect path determinism and trait-object handling.

## Questions

| ID | Question |
|---|---|
| **S5-Q1** | What container traversal behavior exists today in runtime/optimizer code? |
| **S5-Q2** | Which Facet primitives are available for container traversal at shape level and value level? |
| **S5-Q3** | How should `Option<Predict<_>>` be traversed and represented in dotted paths? |
| **S5-Q4** | How should `Vec<Predict<_>>` and `HashMap<K, Predict<_>>` traversal preserve deterministic, stable naming? |
| **S5-Q5** | What concrete limits apply to `Box<dyn Module>` in the current stack? |
| **S5-Q6** | Which implementation shape is best for a first pass (shape-only, value-only, or hybrid)? |

## Findings (with Evidence)

1. Current optimizer discovery is still manual `Optimizable` recursion, not Facet walker recursion.
   - `Optimizable` requires `parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable>`: `crates/dspy-rs/src/core/module.rs:84`.
   - `#[derive(Optimizable)]` only includes fields tagged `#[parameter]` and recursively flattens by calling child `parameters()`; no explicit container branching logic exists in the derive: `crates/dsrs-macros/src/optim.rs:41`, `crates/dsrs-macros/src/optim.rs:50`, `crates/dsrs-macros/src/optim.rs:72`, `crates/dsrs-macros/src/optim.rs:92`.
   - Existing tests cover nested struct flattening only (`a`, `b.predictor`, `p.b.predictor`), not `Option`/`Vec`/`HashMap`: `crates/dspy-rs/tests/test_optimizable.rs:39`, `crates/dspy-rs/tests/test_optimizable.rs:64`, `crates/dspy-rs/tests/test_optimizable.rs:103`.

2. Local runtime already uses Facet value traversal primitives for `Option`/list/map/pointer conversion.
   - Option/list/map read-paths are used in `to_baml_value`: `crates/bamltype/src/convert.rs:539`, `crates/bamltype/src/convert.rs:593`, `crates/bamltype/src/convert.rs:602`.
   - Pointer and option are handled in `from_baml_value` with explicit recurse/unwrap behavior: `crates/bamltype/src/convert.rs:125`, `crates/bamltype/src/convert.rs:133`, `crates/bamltype/src/convert.rs:137`.
   - Shape-level `Def` matching for `Option`/`List`/`Map`/`Pointer` is already used by schema building: `crates/bamltype/src/schema_builder.rs:121`, `crates/bamltype/src/schema_builder.rs:123`, `crates/bamltype/src/schema_builder.rs:127`, `crates/bamltype/src/schema_builder.rs:129`, `crates/bamltype/src/schema_builder.rs:135`.

3. NIA evidence confirms Facet has the needed container primitives and traversal controls.
   - `Facet` exposes static shape via `const SHAPE`: `facet-rs/facet/facet-core/src/lib.rs:39`, `facet-rs/facet/facet-core/src/lib.rs:46`.
   - `Def` includes container/pointer variants (`Map`, `List`, `Option`, `Pointer`, `DynamicValue`): `facet-rs/facet/facet-core/src/types/def/mod.rs:24`, `facet-rs/facet/facet-core/src/types/def/mod.rs:37`, `facet-rs/facet/facet-core/src/types/def/mod.rs:47`, `facet-rs/facet/facet-core/src/types/def/mod.rs:67`, `facet-rs/facet/facet-core/src/types/def/mod.rs:75`.
   - Shape walker supports deterministic DFS + skip/stop + cycle detection and explicit container steps: `facet-rs/facet/facet-path/src/walk.rs:25`, `facet-rs/facet/facet-path/src/walk.rs:49`, `facet-rs/facet/facet-path/src/walk.rs:55`, `facet-rs/facet/facet-path/src/walk.rs:63`, `facet-rs/facet/facet-path/src/walk.rs:93`, `facet-rs/facet/facet-path/src/walk.rs:105`.
   - Value walker primitives exist for list/map/option/pointer access (`Peek*`): `facet-rs/facet/facet-reflect/src/peek/value.rs:15`, `facet-rs/facet/facet-reflect/src/peek/value.rs:18`, `facet-rs/facet/facet-reflect/src/peek/value.rs:49`; `facet-rs/facet/facet-reflect/src/peek/list.rs:35`, `facet-rs/facet/facet-reflect/src/peek/list.rs:54`; `facet-rs/facet/facet-reflect/src/peek/map.rs:28`, `facet-rs/facet/facet-reflect/src/peek/map.rs:41`; `facet-rs/facet/facet-reflect/src/peek/option.rs:44`; `facet-rs/facet/facet-reflect/src/peek/pointer.rs:32`.

4. NIA evidence also shows a key limit: shape-only paths encode container structure, not runtime multiplicity.
   - Shape walk emits placeholder map path steps (`MapKey(0)`, `MapValue(0)`) and option/pointer semantic steps: `facet-rs/facet/facet-path/src/walk.rs:49`, `facet-rs/facet/facet-path/src/walk.rs:55`, `facet-rs/facet/facet-path/src/walk.rs:63`, `facet-rs/facet/facet-path/src/walk.rs:93`.
   - Formatted paths for maps/options use generic markers like `[key#i]`, `[value#i]`, `::Some`, not concrete runtime keys: `facet-rs/facet/facet-path/src/lib.rs:47`, `facet-rs/facet/facet-path/src/lib.rs:54`, `facet-rs/facet/facet-path/src/lib.rs:61`.

5. Trait-object limits are hard constraints in the typed path and must be treated as explicit out-of-scope for first-pass S5 support.
   - BAML derive explicitly rejects trait-object fields: `crates/bamltype-derive/src/lib.rs:418`, `crates/bamltype-derive/src/lib.rs:421`.
   - UI test locks in that behavior (`Box<dyn Debug>` compile-fails): `crates/bamltype/tests/ui/trait_object.rs:5`, `crates/bamltype/tests/ui/trait_object.stderr:1`.
   - Design guidance explicitly chooses associated-type modules over trait-object composition for typed workflows: `docs/specs/modules/design_reference.md:992`.
   - Consequence for S5: `Box<T>` with concrete `T` can be traversed through pointer semantics; `Box<dyn Module>` cannot be a typed walker leaf/container and should be modeled as Layer-3 dynamic graph nodes instead.

## Container Handling Matrix

| Container Case | First-pass Status | Traversal Rule | Canonical Path Form | Determinism Policy | Explicit Limits |
|---|---|---|---|---|---|
| `Option<Predict<S>>` | **Supported** | Recurse only when `Some(inner)`; emit no path for `None` | `retriever` (no `::Some`) | Stable for unchanged value/state; path appears/disappears only when option toggles `Some`/`None` | No phantom leaves for `None`; absence is expected, not error |
| `Vec<Predict<S>>` | **Supported** | Iterate index order `0..len-1` and recurse per element | `retrievers[0]`, `retrievers[1]` | Stable if vector order/content is unchanged | Index renumbering on insert/remove/reorder is accepted churn in first pass |
| `HashMap<String, Predict<S>>` | **Supported** | Iterate entries sorted by canonical key order, recurse on values | `weights['foo']` | Sort by raw string key bytes before emitting paths; do not depend on map iteration order | First pass is string-key only for canonical naming; non-string keys are out-of-scope |
| `Box<Predict<S>>` (or concrete `Box<T>` with traversable `T`) | **Supported** | Dereference pointer and continue traversal on concrete pointee shape | Same as inner path (for example `leaf`) | Stable under same pointee graph + cycle guard | Requires concrete type metadata (`Facet`/shape); cycle protection required |
| `Box<dyn Module>` | **Unsupported (typed walker)** | Do not descend; report unsupported at compile/design boundary | None | N/A | Rejected by BAML derive and by typed-path design; represent this case as dynamic graph nodes (`DynModule`) instead of typed container traversal |

## Options + Tradeoffs

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A. Shape-only walker (`walk_shape`)** | Use type-level traversal only, deriving leaf locations from `Shape` graph. | Deterministic, cycle-safe, simple to reason about at schema level. | Cannot enumerate runtime multiplicity (`Vec` length, actual map keys, `Option::None` presence). |
| **B. Value-only `Peek` recursion** | Traverse live values only (`into_option`, `into_list_like`, `into_map`, `into_pointer`). | Captures real runtime leaves and concrete dotted paths (`[i]`, `['key']`). | Needs explicit cycle guards, ordering policy, and shape-based semantics for stable naming. |
| **C. Hybrid (shape-guided + value-driven)** | Use `Shape/Def` to choose traversal protocol, then `Peek` to enumerate runtime children for multiplicity containers. | Best match for S5: deterministic protocol + real runtime coverage. | More moving parts; requires clear path canonicalization rules. |

## Decision

**Deferred.** Container traversal (`Option`/`Vec`/`HashMap`/`Box`) is not needed for V1 library modules — all use struct-field recursion only (ChainOfThought has `predict: Predict<...>`, ReAct has `action: Predict<...>`, BestOfN wraps `module: M`). Container traversal will be implemented when a concrete use case requires it. The spike findings and tradeoff analysis are preserved below for when that happens.

## Original Recommendation (not adopted)

The spike originally recommended Option C (hybrid walker):

Rationale:
- S5 requires container *runtime* handling, not just type graph coverage.
- Local code already demonstrates both sides (`Def` matching + `Peek` traversal).
- NIA evidence shows shape-only map/list steps are intentionally abstract (`key#i`/`value#i`), so runtime enumeration is required for optimizer-meaningful handles.

## Deterministic Path Policy (First Pass)

1. Canonical grammar:
   - `<path> := <field>(.<field> | [<index>] | ['<escaped_key>'])*`
   - `<field>` uses Rust field names as declared in the owning struct.
2. Container representation:
   - `Option<T>` is transparent for naming; emit `x`, never `x::Some`.
   - `Vec<T>` uses zero-based decimal indices (`items[0]`, `items[1]`), no leading zeros.
   - `HashMap<String, T>` uses single-quoted keys with escaping (`weights['foo']`).
3. Key escaping for map paths:
   - Escape `\` as `\\`.
   - Escape `'` as `\'`.
   - Escape control characters as `\u{HEX}`.
4. Ordering policy:
   - Struct fields: declaration order.
   - Lists: index order.
   - Maps: sort by raw UTF-8 bytes of the original key before formatting/escaping.
5. Disallowed forms in emitted paths:
   - No Facet placeholder tokens (`[key#i]`, `[value#i]`).
   - No semantic tags (`::Some`, pointer-only markers).

## Concrete Implementation Steps

1. Introduce `named_predictors` (or equivalent) walker API that accepts a reflectable root value and returns `(path, handle)` pairs.
2. Implement a shared path formatter that enforces the deterministic path policy above (grammar, escaping, ordering assumptions).
3. Resolve predictor leaf markers from shape metadata (S2 mechanism) before descending into children.
4. Implement container traversal arms:
   - `Option`: recurse only for `Some`; emit nothing for `None`.
   - `List/Vec`: recurse each index in `0..len-1`.
   - `Map<String, _>`: collect entries, sort keys by raw key bytes, recurse on values.
   - `Pointer/Box<T>`: recurse into concrete pointee when available.
5. Add explicit unsupported handling for trait-object pointers (`Box<dyn Module>`) with clear compile/design-time diagnostics and dynamic-graph fallback guidance.
6. Add cycle protection for pointer/self-referential graphs to avoid infinite recursion.
7. Add tests for each matrix row: positive cases (`Option`, `Vec`, `HashMap<String, _>`, `Box<T>`) and negative trait-object coverage.
8. Add compatibility shim from current `Optimizable::parameters()` callers to the new walker so optimizers can migrate incrementally.

## Acceptance

S5 is complete when:
- The new walker discovers predictor leaves under `Option`, `Vec`, and `HashMap` containers with deterministic path output.
- Emitted paths follow one canonical grammar (`a.b`, `items[3]`, `weights['foo']`) with required escaping.
- Paths are stable under repeated traversal of unchanged structures and never include placeholder markers (`key#i`, `value#i`) or `::Some`.
- `Option::None` is handled without panic and without emitting phantom leaves.
- Map traversal ordering is deterministic and documented (sorted by raw key bytes, not hash iteration order).
- `Box<dyn Module>` is explicitly unsupported in typed traversal with a documented dynamic-graph alternative.
- Tests cover positive and negative container cases and are integrated with existing optimizer discovery flows.

## Open Risks

- Non-string map keys remain out-of-scope for first pass; adding them later will require a canonical key-rendering spec that is collision-safe.
- Vector index paths are deterministic but not identity-stable under insert/reorder; optimizers that persist paths across structure edits will need remapping logic.
