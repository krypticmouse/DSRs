# DSRs Module System — Shaping Document

**Selected shape:** F (Facet-native typed modules with dynamic graph escape hatch)

---

## Frame

### Problem

- DSRs has typed signatures and a working Predict, but no module system — users cannot compose prompting strategies (ChainOfThought, ReAct, BestOfN) or write custom multi-step pipelines
- The current `LegacyPredict` / `MetaSignature` path is stringly typed and parallel to the typed path — two systems doing the same job, neither complete
- Macro-emitted static `FieldSpec` arrays duplicate what Facet already knows about types
- Optimizers require manual `Optimizable` trait impls with hand-written `named_parameters()` traversal
- No mechanism for signature augmentation — ChainOfThought cannot add `reasoning` to an arbitrary signature's output
- No path toward structural optimization (changing program topology, not just leaf parameters)

### Outcome

- Module authors can express all DSPy module patterns (signature extension, multi-signature orchestration, module wrapping, aggregation) as idiomatic typed Rust
- Users compose modules by swapping types, not rewriting code — `Predict<QA>` → `ChainOfThought<QA>` is a type change, everything else works
- Optimizers discover all tunable parameters automatically from program structure
- A dynamic graph layer exists for structural optimization and PyO3 interop, sharing the same adapter and prompt format as the typed path
- Facet is the single source of truth for type metadata — no parallel schema systems

---

## Requirements (R)

| ID | Requirement | Status |
|----|-------------|--------|
| **R0** | Users define signatures as typed Rust structs with `#[derive(Signature)]`, getting typed Input/Output types and full IDE support | Core goal |
| **R1** | Library-provided modules (ChainOfThought, ReAct, BestOfN, Refine, ProgramOfThought, MultiChainComparison) are generic over any compatible Signature | Core goal |
| **R2** | Module output includes augmented fields (e.g. `reasoning`) accessible via natural field access (`result.reasoning`, `result.answer`) without the user naming wrapper types | Must-have |
| **R3** | Module authoring (by library or advanced users) requires: define augmentation fields OR signature struct, implement forward logic — no traversal boilerplate, no manual schema construction | Must-have |
| **R4** | Optimizers discover all Predict leaves in an arbitrary module tree automatically — no `#[parameter]` annotations, no manual `named_parameters()` impl | Must-have |
| **R5** | All four DSPy augmentation patterns are expressible: signature extension, multi-signature orchestration, module wrapping, aggregation | Must-have |
| **R6** | Custom user types (enums, structs, nested) work in signatures via `#[BamlType]`, including inside augmented signatures | Must-have |
| **R7** | A dynamic program graph can be constructed, validated, mutated, and executed — nodes are module instances, edges are validated by field type compatibility | Must-have |
| **R8** | Typed modules and dynamic graph nodes produce identical prompts for the same logical signature — one adapter, one prompt format | Must-have |
| **R9** | Type metadata (field names, types, docs, constraints, ordering) is derived from Facet Shapes at runtime — not from macro-emitted static arrays | Must-have |
| **R10** | Parameter state (demos, instruction overrides) serializes to JSON and loads back into an existing typed module without serializing type structure | Must-have |
| **R11** | ReAct-style modules accept tools as plain Rust async functions with a builder API (`.tool("name", "desc", fn)`) | Must-have |
| **R12** | Module composition is type-safe: passing wrong input type to a module is a compile error | Must-have |
| **R13** | Augmentations compose: `(Reasoning, Confidence)` produces output with both `reasoning` and `confidence` fields, all accessible | Nice-to-have |
| **R14** | Dynamic graph nodes can be instantiated from a strategy registry by name + schema + config (e.g. `registry::create("react", schema, config)`) | Must-have |
| **R16** | The typed path is the default and primary experience — the dynamic/untyped path exists but is not required for normal use | Must-have |

---

## Shape F: Facet-native typed modules with dynamic graph escape hatch

### Parts

| Part | Mechanism | Flag |
|------|-----------|:----:|
| **F1** | **Signature trait + derive macro** — `#[derive(Signature)]` on a struct with `#[input]`/`#[output]` fields generates `Input`/`Output` helper types, implements `Signature` trait. Supports generic type parameters and `#[flatten]` for composition. Doc comments become LM instructions/descriptions. | |
| **F2** | **SignatureSchema (Facet-derived, cached)** — `SignatureSchema::of::<S>()` walks `S::Input` and `S::Output` Facet Shapes to produce an ordered flat field list with TypeIR, docs, constraints, and flatten paths. Cached in `OnceLock`. Used by adapter for prompt formatting/parsing AND by dynamic graph for edge validation. Replaces macro-emitted `FieldSpec` arrays. | |
| **F3** | **Augmentation derive + combinator** — `#[derive(Augmentation)]` on a small struct (e.g. `Reasoning { reasoning: String }`) generates: a wrapper type (`WithReasoning<O>`) with `#[flatten]` on inner + `Deref` to inner, and the `Augmentation` trait impl. `Augmented<S, A>` is a generic signature combinator (same input, wrapped output). Eliminates per-augmentation signature boilerplate. | |
| **F4** | **Module trait** — `trait Module { type Input; type Output; async fn forward(&self, input) -> Result<Output> }`. All prompting strategies implement this: `Predict<S>`, `ChainOfThought<S>`, `ReAct<S>`, `BestOfN<M>`, `Refine<M>`, user-defined modules. This is the swapping/composition interface. | |
| **F5** | **Predict as leaf parameter** — `Predict<S: Signature>` holds typed demos `Vec<Demo<S>>`, optional instruction override, tools. Only thing that calls the LM. Marked with Facet attribute `dsrs::parameter` for automatic discovery. Implements both `Module` and `DynPredictor` (type-erased optimizer interface). | |
| **F6** | **Facet-powered parameter discovery** — A walker reflects over any `Facet` value, recurses through struct fields, yields `(dotted_path, &dyn DynPredictor)` for every value whose Shape carries `dsrs::parameter`. No manual traversal code. Replaces `#[derive(Optimizable)]` + `#[parameter]`. Container traversal (`Option`/`Vec`/`HashMap`/`Box`) is deferred (S5) — struct-field recursion covers all V1 library modules. | |
| **F7** | **Adapter building blocks** — ChatAdapter exposes public composable functions: `build_system()`, `format_input()`, `parse_sections()`, `parse_output()`. Modules that need fine-grained control (ReAct action loop) call these directly. Standard modules go through the high-level `format_system_message_typed::<S>()` which calls building blocks internally. All operate on `SignatureSchema` (F2). | |
| **F8** | **DynPredictor vtable** — Type-erased interface for optimizer operations on a Predict leaf: get/set demos (as `Vec<Example>`), get/set instruction, get schema, `forward_untyped(BamlValue) -> BamlValue`. Obtained via shape-local accessor payload: `Predict<S>` carries `PredictAccessorFns` as a typed Facet attribute, extracted at discovery time by the walker. Bridges typed Predict to untyped optimizer. | |
| **F9** | **DynModule + StrategyFactory** — `DynModule` is the dynamic equivalent of `Module` (BamlValue in/out, exposes internal predictors). `StrategyFactory` creates a `DynModule` from a `SignatureSchema` + config. Each module type (ChainOfThought, ReAct, etc.) registers a factory. Factories perform schema transformations (prepend reasoning, build action schema from tools, etc.) on `SignatureSchema` directly. | |
| **F10** | **ProgramGraph** — Dynamic graph of `Node` (holds `DynModule` + `SignatureSchema`) and `Edge` (from_node.field → to_node.field). Edges validated by TypeIR compatibility at insertion time. Supports `add_node`, `remove_node`, `replace_node`, `connect`, `insert_between`. Execution follows topological order, piping `BamlValue` between nodes. Typed modules can be projected into a graph (via F6 walker) and graph nodes can wrap typed modules internally. | |
| **F11** | **Library modules** — Concrete implementations of DSPy's module zoo: `ChainOfThought<S>` (F3 augmentation + Predict), `ReAct<S>` (two Predicts + tool loop + builder API), `BestOfN<M>` (wraps any Module), `Refine<M>` (BestOfN + feedback, scoped context mechanism TBD), `ProgramOfThought<S>` (three ChainOfThought + code interpreter), `MultiChainComparison<S>` (M sources + comparison Predict). Each is generic over Signature, implements Module, and is discoverable via F6. | ⚠️ |
| **F12** | **Generic Signature derive** — `#[derive(Signature)]` works on structs with generic type parameters (e.g. `ActionStep<I: BamlType + Clone>`) and `#[flatten]` fields. The generated `Input`/`Output` types carry the generic parameters through. Required for module authors who define custom multi-field signatures. Implementation path: generic forwarding in macro + path-aware runtime metadata bridge + path-based adapter format/parse (see S1). | |

**Flag notes:**
- **F11 ⚠️**: ChainOfThought and BestOfN have concrete designs. Remaining unknowns: Refine's scoped context mechanism (S4 deferred — `tokio::task_local!` vs explicit parameter TBD when Refine is built), ReAct's tool builder API (concrete `ToolDyn` trait, action/extract loop wiring), and MultiChainComparison's attempt format (how M source outputs are aggregated into the comparison prompt). These require prototyping during implementation.

---

## Fit Check (R × F)

| Req | Requirement | Status | F |
|-----|-------------|--------|---|
| **R0** | Users define signatures as typed Rust structs with `#[derive(Signature)]`, getting typed Input/Output types and full IDE support | Core goal | ✅ |
| **R1** | Library-provided modules are generic over any compatible Signature | Core goal | ✅ |
| **R2** | Module output includes augmented fields accessible via natural field access without naming wrapper types | Must-have | ✅ |
| **R3** | Module authoring requires: define fields + implement forward — no traversal boilerplate, no manual schema construction | Must-have | ✅ |
| **R4** | Optimizers discover all Predict leaves automatically — no annotations, no manual traversal | Must-have | ✅ |
| **R5** | All four DSPy augmentation patterns are expressible | Must-have | ✅ |
| **R6** | Custom user types work in signatures, including inside augmented signatures | Must-have | ✅ |
| **R7** | Dynamic program graph can be constructed, validated, mutated, and executed | Must-have | ✅ |
| **R8** | Typed and dynamic paths produce identical prompts | Must-have | ✅ |
| **R9** | Type metadata derived from Facet Shapes, not macro-emitted static arrays | Must-have | ✅ |
| **R10** | Parameter state serializes/deserializes without serializing type structure | Must-have | ✅ |
| **R11** | ReAct accepts tools as plain async functions with builder API | Must-have | ✅ |
| **R12** | Module composition is type-safe | Must-have | ✅ |
| **R13** | Augmentations compose via tuples | Nice-to-have | ✅ |
| **R14** | Dynamic graph nodes instantiated from registry by name + schema + config | Must-have | ✅ |
| **R16** | Typed path is default, dynamic is opt-in | Must-have | ✅ |

**Notes:**
- R2 satisfied by `Deref` coercion on wrapper types — `result.reasoning` is a direct field, `result.answer` resolves via Deref to inner type. S3 confirmed: auto-deref works through multiple layers for field reads and method calls. Pattern matching requires explicit layer-by-layer destructuring (acceptable — documented limitation).
- R4 satisfied by Facet walker (F6) using shape-local accessor payloads (S2: Mechanism A). `#[derive(Facet)]` on the module struct is the only requirement. V1 walker recurses through struct fields only; container traversal deferred (S5).
- R8 satisfied by both paths using `SignatureSchema` (F2) → same adapter building blocks (F7) → same prompt format.

---

## Layers (how parts compose)

```
Layer 0: Types
  Rust types with Facet + BamlType derives
  Source of truth. Never serialized. Lives in the binary.
  Parts: (foundation for everything)

Layer 1: Typed Modules
  Signature trait (F1) + SignatureSchema (F2) + Augmentation (F3)
  + Module trait (F4) + Predict (F5) + Library modules (F11)
  Compile-time checked. IDE-supported. 90% of programs live here.
  Parts: F1, F2, F3, F4, F5, F11, F12

Layer 2: Optimization Bridge
  Facet parameter discovery (F6) + DynPredictor vtable (F8)
  + Adapter building blocks (F7)
  Minimal type erasure for optimizer access to typed leaves.
  Parts: F6, F7, F8

Layer 3: Dynamic Graph
  DynModule + StrategyFactory (F9) + ProgramGraph (F10)
  Opt-in for structural optimization and PyO3 interop.
  Parts: F9, F10
```

Each layer only exists if needed. A simple `Predict::<QA>::new().call(input)` touches Layers 0-1 and the adapter from Layer 2. The graph layer is never instantiated unless structural optimization or PyO3 is in play.

---

## Spikes (Resolved)

All spikes have been investigated and resolved. Full findings in `spikes/S{n}-*.md`.

| # | Question | Decision | Spike doc |
|---|----------|----------|-----------|
| **S1** | Can `#[derive(Signature)]` handle generic type parameters with `#[flatten]` fields? | **Option C: full replacement.** Build `SignatureSchema` from Facet, replace `FieldSpec` everywhere, delete the old system. No incremental migration. | `S1-generic-signature-derive.md` |
| **S2** | How does the Facet walker obtain a usable optimizer handle from a discovered Predict? | **Mechanism A**: shape-local accessor payload (`dsrs::parameter` + fn-pointer `PredictAccessorFns`). Reuses existing `WithAdapterFns` typed-attr pattern. | `S2-dynpredictor-handle-discovery.md` |
| **S3** | Does Rust auto-Deref chain resolve field access through nested augmentation wrappers? | **Yes for reads/methods**, no for pattern matching (don't care). `Deref`-only unless `DerefMut` is proven necessary. | `S3-augmentation-deref-composition.md` |
| **S4** | What scoped-context mechanism for Refine's hint injection? | **Deferred.** Mechanism chosen when Refine is built. Findings preserved in spike doc. | `S4-refine-scoped-context.md` |
| **S5** | How does the Facet walker handle Option/Vec/HashMap/Box containers? | **Deferred.** Struct-field recursion covers all V1 library modules. Container traversal when a concrete use case requires it. | `S5-facet-walker-containers.md` |
| **S6** | Migration path from FieldSpec/MetaSignature to Facet-derived SignatureSchema? | **Subsumed by S1 → Option C.** No migration — full replacement. | `S6-migration-fieldspec-to-signatureschema.md` |
| **S7** | Can `#[derive(Augmentation)]` generate a generic wrapper from a non-generic struct? What about the `Augmented` phantom type? | **Yes, feasible.** All three derives handle generics. `from_parts`/`into_parts` removed from `Signature` trait — `Augmented` becomes a clean type-level combinator. | `S7-augmentation-derive-feasibility.md` |
| **S8** | How does Facet flatten manifest in Shape metadata? | **`field.is_flattened()` flag check + `field.shape()` recurse.** Facet ships `fields_for_serialize()` as reference. Direct mapping to design pseudocode. | `S8-facet-flatten-metadata.md` |

---

## Fit Check — F × R (Parts as rows, Requirements as columns)

✅ = this part directly satisfies this requirement. Blank = does not.

| Part | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R16 |
|------|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| **F1** Signature derive | ✅ | | | | | | ✅ | | | | | | ✅ | | | ✅ |
| **F2** SignatureSchema | | | | ✅ | | | ✅ | | ✅ | ✅ | | | | | | |
| **F3** Augmentation | | ✅ | ✅ | ✅ | | ✅ | ✅ | | | | | | ✅ | ✅ | | |
| **F4** Module trait | | ✅ | | ✅ | | ✅ | | | | | | | ✅ | | | ✅ |
| **F5** Predict leaf | | | | | | | | | | | ✅ | | | | | ✅ |
| **F6** Facet discovery | | | | ✅ | ✅ | | | | | ✅ | | | | | | |
| **F7** Adapter blocks | | | | ✅ | | ✅ | | | ✅ | ✅ | | ✅ | | | | |
| **F8** DynPredictor | | | | | ✅ | | | ✅ | | | ✅ | | | | | |
| **F9** DynModule+Factory | | | | | | | | ✅ | ✅ | | | | | | ✅ | |
| **F10** ProgramGraph | | | | | | | | ✅ | ✅ | | | | | | ✅ | |
| **F11** Library modules | | ✅ | ✅ | | | ✅ | ✅ | | | | | ✅ | ✅ | | | ✅ |
| **F12** Generic Sig derive | ✅ | ✅ | | ✅ | | ✅ | ✅ | | | | | | ✅ | | | ✅ |

---

### Coverage analysis

**Every R is covered by at least one part.** Reading the columns:

| Req | Description | Satisfied by |
|-----|-------------|-------------|
| R0 | Typed signature derive | F1, F12 |
| R1 | Modules generic over Signature | F3, F4, F11, F12 |
| R2 | Augmented fields accessible naturally | F3, F11 |
| R3 | No traversal boilerplate, no manual schema | F2, F3, F4, F6, F7, F12 |
| R4 | Automatic optimizer discovery | F6, F8 |
| R5 | All four augmentation patterns | F3, F4, F5 (implicitly), F7, F11, F12 |
| R6 | Custom types in signatures | F1, F2, F3, F11, F12 |
| R7 | Dynamic program graph | F8, F9, F10 |
| R8 | Typed and dynamic produce identical prompts | F2, F7, F9, F10 |
| R9 | Metadata from Facet not static arrays | F2, F6, F7 |
| R10 | Parameter state serialization | F5, F8 |
| R11 | ReAct tool builder API | F7, F11 |
| R12 | Composition is type-safe | F1, F3, F4, F11, F12 |
| R13 | Augmentations compose via tuples | F3 |
| R14 | Dynamic nodes from registry | F9, F10 |
| R16 | Typed path is default | F1, F4, F5, F11, F12 |

**Observations:**

**R13 (augmentation composition) has the thinnest coverage** — only F3. S3 confirmed auto-deref works for reads/methods, so the risk is mitigated. Pattern matching through nested wrappers requires explicit destructuring — acceptable for a Nice-to-have.

**R4 (automatic discovery) depends on F6 + F8 together.** F6 finds the values, F8 makes them operable. S2 resolved the handle mechanism (shape-local accessor payload). Container traversal deferred (S5) — struct-field recursion is sufficient for V1.

**R7 (dynamic graph) is the heaviest requirement** — needs F8, F9, AND F10. All three are Layer 3. This is expected — it's the most complex capability.

**F3 (Augmentation) is the most load-bearing part** — covers 8 requirements. S3 confirmed the core deref ergonomics work. Flatten round-trips depend on `SignatureSchema` path-aware adapter (S1/C).

**F7 (Adapter building blocks) quietly covers 5 requirements.** It's the mechanism that makes R8 (identical prompts) and R11 (ReAct tools) work. Exposing the adapter's internals as composable functions is less glamorous than the type system work but equally essential.

**F11 (Library modules) is flagged ⚠️ and covers 6 requirements.** The flag is narrowed: ChainOfThought, BestOfN, and Refine have concrete designs. ReAct's tool builder and MultiChainComparison's aggregation format remain as prototyping work during implementation. We can't call the system done without at least ChainOfThought + ReAct + BestOfN working end-to-end.
