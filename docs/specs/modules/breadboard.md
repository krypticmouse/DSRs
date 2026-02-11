# DSRs Module System — Breadboard

## Current Scope Addendum (2026-02-12)

V6/dynamic graph was implemented in-repo, then intentionally deferred; the runtime code has been removed from active scope.

Canonical scope is now V1–V5 typed-only; untyped eval (`U37`) and all V6 dynamic graph/runtime surfaces are deferred.

MIPRO is intentionally instruction-only in current scope; trace-derived per-predictor demo mutation is deferred.

All content below is preserved as a historical implementation record.

> Shape F: Facet-native typed modules with dynamic graph escape hatch
> Parts: F1–F12 (see [shapes.md](./shapes.md))
> Procedure: Designing from Shaped Parts (breadboarding skill)

---

## Adaptation Notes

This breadboard applies the standard methodology to a **Rust library**, not a web UI.

| Concept | Web UI | Rust Library |
|---------|--------|--------------|
| **Place** | Screen / page / modal | Developer context — bounded API surface for a specific role |
| **UI Affordance (U)** | Button, input, display | Public API — derives, constructors, method calls, return types |
| **Code Affordance (N)** | Handler, service, subscription | Internal mechanism — macro expansion, schema derivation, Facet walking, caching |
| **Data Store (S)** | Component state, DB table | Runtime state — `OnceLock` caches, demo vectors, graph node maps |
| **Control** | click, type, scroll, render | `compile` (derive/attr — compiler gives immediate feedback), `write` (source code), `call` (method), `construct` (value), `access` (field/type) |

**Blocking test (adapted):** For a library, the blocking test is **cognitive, not physical** — everything is `pub` in Rust. A Place is separate when reaching its affordances requires a fundamentally different mental model. P1 users never need to understand prompt formatting internals (P2), Facet reflection (P3), or graph topology (P4). The "block" is that using those affordances requires knowledge the developer doesn't have and doesn't need.

**External dependencies (not breadboarded):** `BamlType` (jsonish coercion), `Facet` (reflection/shapes), LM provider (API calls), global LM configuration (`GLOBAL_SETTINGS` — existing infrastructure). These are foundations the system builds on, not interactive Places.

---

## Places

| # | Place | Developer Role | Description | Layer |
|---|-------|----------------|-------------|-------|
| **P1** | User Code | App developer | Define signatures, pick modules, call them, access results. The 90% path. | L1 |
| **P2** | Module Authoring | Library / advanced developer | Create new augmentations and modules. Generic signatures, adapter building blocks, `impl Module`. | L1 |
| **P3** | Optimization | Optimizer internals | Discover Predict leaves via Facet walker, mutate demos/instructions via DynPredictor, serialize parameter state. | L2 |
| **P4** | Dynamic Graph | Structural optimizer / interop | Construct ProgramGraph, create DynModules from strategy registry, validate edges, execute graph. Fully untyped. | L3 |

**Architectural invariants:**
- **Dependency direction is acyclic:** P1 ← P2 ← P3 ← P4. Each layer sees the one below, never above. No cycles.
- **S1 (SignatureSchema cache) is the shared backbone:** Written once (immutable after init), read by all Places. Immutable shared state across Places is coupling in name only — it's a computed property of types. If this invariant were ever violated (mutable schema), the whole Place decomposition would collapse.
- **L1/L2 share a compilation unit.** `Predict<S>` implements `DynPredictor` in the same crate (`dspy-rs`). This is intentional dependency inversion: L2 defines the interface (`DynPredictor`), L1 satisfies it. **Current mechanism:** accessor fns are resolved through a runtime shape-id registry. Dispatch is registry-only; Predict-like leaves without registration now fail explicitly with `MissingAttr` diagnostics. **Tradeoff:** stable behavior now, explicit S2 migration debt until shape-local payload extraction is available. L1 cannot be compiled without L2 type definitions. The layer separation is enforced by API design (P1 users never import L2 types), not by the crate graph.
- **"Structure IS declaration" — with bounded container support.** The walker discovers Predict leaves by reflecting on struct fields. Module authors don't annotate `#[parameter]` or implement traversal. The current implementation traverses structs plus common containers (`Option`, list/array/slice, `HashMap<String, _>`, and `Box`). Unsupported pointer-like containers (`Rc`, `Arc`, etc.) produce explicit N18 errors rather than silent skips.
- **Module combinators must be Facet-transparent.** Any wrapper that composes modules (Map, AndThen, Pipe) must expose inner modules as struct fields visible to the F6 walker (N18), not behind trait objects. `Map<M, F>` requires a manual Facet impl walking only `inner: M` (closures are opaque to Facet derive). `BestOfN<M>` has `module: M` as a concrete typed field. If a combinator hides the inner module behind `Box<dyn Module>`, the walker cannot find Predict leaves inside — optimization breaks silently. **Path namespace consequence:** Wrapping a module changes path prefixes — `predict` becomes `inner.predict`. Serialized optimizer state (U36) is tied to the module tree shape. Changing the tree (adding/removing a wrapper) invalidates saved state with a clear error, not silent misapplication.

**Boundary notes:**
- **P1 → P2 boundary:** P1 users *consume* what P2 creates. The blocking test is cognitive: P2 affordances (`#[derive(Augmentation)]`, adapter building blocks, `impl Module`) require understanding prompt pipeline internals, wrapper type mechanics, and Facet composition — a fundamentally different mental model from P1's "pick a module, call it." P2 is a valid separate Place even though nothing physically prevents a P1 user from importing P2 APIs. **Ramp:** Module combinators (U51: `.map()`, `.and_then()`) let P1 users customize output without crossing into P2. The cliff from "use a library module" to "author your own module" has an intermediate step.
- **P1 → P3 boundary:** P1 users hand their module to an optimizer. The optimizer reaches INTO their module's Predict leaves via the F6 walker + F8 DynPredictor vtable. The user's typed module is never replaced — the optimizer mutates state within it. Walker takes `&mut module` (exclusive access during optimization).
- **P3 → P4 boundary:** P3 (parameter optimization) works within existing typed modules. P4 (structural optimization) changes the module *topology* — adding, removing, or replacing nodes in a graph. P4 subsumes P3's capabilities.
- **Layer 0** (Types: `BamlType + Facet`) underpins all Places but is not a Place — it has no interactive affordances. Types are the source of truth, compiled into the binary.

**Resolved gaps:**
- ~~No LM configuration affordance~~ → **Global default with scoped override.** LM is globally scoped (existing `GLOBAL_SETTINGS` infrastructure). `dsrs::with_lm(eval_lm, || ...)` overrides per-call via scoped context. N8 checks scoped context first, falls back to global default. Global LM configuration is existing infrastructure, not breadboarded (see External dependencies).
- ~~No batching affordance~~ → **Standalone utility, not a trait method.** `dsrs::forward_all(&module, inputs, concurrency)` → `Vec<Result<Predicted<Output>, PredictError>>` (Vec-of-Results, not Result-of-Vec — individual failures don't abort batch). Module trait stays minimal (`forward` implementation hook + default `call` wrapper). Rationale: a default `forward_batch` on Module forces P2 authors to reason about concurrency composition — BestOfN already runs N concurrent calls per invocation, so default batching would produce `batch_size × N` concurrent LM requests. Standalone utility keeps this concern at P1. See U48.
- ~~Error paths underspecified~~ → `PredictError` carries raw LM response + failed field + stage + coercion detail. Error `Display` includes full LM response for iterative debugging. No separate debug API needed for V1. See U49.
- ~~Container traversal silently fails~~ → N18 now traverses supported containers (`Option`, lists, maps, `Box`) and errors on unsupported pointer-like containers (`Rc`, `Arc`, etc.) with explicit path/type diagnostics.
- ~~Strategy swap blast radius understated~~ → Updated U16 to note output type change.
- ~~N12/N13 status~~ → **Keep N13, collapse N12 into N8.** N12 (jsonish coerce) is part of the "text → BamlValue" pipeline inside N8. N13 (try_from_baml_value) is a distinct error boundary: "BamlValue → typed output." Two affordances, two error semantics (N8 failures = coercion/parsing, N13 failures = type mismatch).
- ~~Missing P1→P3 handoff~~ → Added U50 (`optimizer.compile(&mut module, trainset, metric)`). Exclusive `&mut` during optimization = no concurrent `forward()`.
- ~~P1→P2 cliff too sharp~~ → **Module combinators as P1 ramp.** Without combinators, a P1 user who wants to post-process output (e.g., derive a confidence score from reasoning) must jump to full `impl Module` — learning associated types, async plumbing, and the Module trait. With `.map()` / `.and_then()`, they write a closure. Added U51 (module combinators). This is the intermediate step between "use a library module" and "author your own module."
- ~~Calling convention undecided~~ → **Locked for V1.** N8 returns `Result<Predicted<O>, PredictError>`. `Predicted<O>` carries output + call metadata (like DSPy's `Prediction`) with `Deref<Target = O>` for direct field access and `.metadata()` for metadata access. `?` works on stable Rust without nightly `Try`. User-facing invocation is `Module::call`, while module authors implement `Module::forward` as the execution hook.

**N-affordance principle:** Keep **orchestration boundaries** (N3, N8, N17, N18, N25/N26) and **error/decision boundaries** (N13, N22, N23, N24). Collapse pure pipes/transforms into their parent. Test: "can you change the implementation without changing any wiring?" If yes, it's guts, not an affordance.

**Open (from late-stage team conversation):**
- ⚠️ **P1→P2 cliff / Module combinators:** Resolved — see U51 (`.map()`, `.and_then()`) and boundary note on P1→P2. **Remaining question:** Module combinators must be Facet-transparent for the F6 walker (N18) to see through them. `Map<M, F>` needs a manual Facet impl exposing `inner: M` as a field (closures are opaque to Facet derive). This is an architectural invariant on all future combinators: they must expose inner modules as struct fields, not trait objects.

**Deferred (acknowledged, out of scope for V1):**
- ⚠️ **Operational policy (retries, timeouts, rate limits):** Per-call execution policy — combinators around `call()`. P1 affordances that wire to U9. No new stores, no new coupling. Easy to add, no architectural impact.
- ⚠️ **Container traversal (remaining):** Common container traversal is implemented (`Option`, lists, maps, `Box`). Unsupported pointer-like containers (`Rc`, `Arc`, etc.) still error explicitly in N18; broader pointer/container strategy remains tracked in S5.

---

## UI Affordances (Public API Surface)

| # | Place | Module | Affordance | Control | Wires Out | Returns To | Part |
|---|-------|--------|------------|---------|-----------|------------|------|
| **U1** | P1 | `signature` | `#[derive(Signature)]` on struct | compile | → N1 | — | F1 |
| **U2** | P1 | `signature` | `#[input]` / `#[output]` field markers | compile | → N1 | — | F1 |
| **U3** | P1 | `signature` | Doc comment on signature struct | compile | → N1, → N2 | — | F1 |
| **U4** | P1 | `signature` | `QAInput` generated type | construct | → U9 | — | F1 |
| **U5** | P1 | `signature` | `QAOutput` / `S::Output` received type | access | → U11, → U12 | ← N13 | F1 |
| **U6** | P1 | `predict` | `Predict::<S>::new()` | construct | → S2, → S3 | — | F5 |
| **U7** | P1 | `predict` | `Predict::<S>::builder().demo(...).instruction(...).build()` | construct | → S2, → S3, → S4 | — | F5 |
| **U8** | P1 | `predict` | `Demo { input: ..., output: ... }` | construct | → U7 | — | F5 |
| **U9** | P1 | `module` | `module.call(input).await` | call | → N3 | → U10 | F4 |
| **U10** | P1 | `module` | `Result<Predicted<S::Output>, PredictError>` from `call` (`Predicted` carries output + metadata; Deref to output fields) | access | → U5 (Ok) | ← N8 | F4 |
| **U11** | P1 | — | `result.answer` — direct field access | access | — | ← U5 | F1 |
| **U12** | P1 | — | `result.reasoning` — Deref to augmented field | access | — | ← U5 | F3 |
| **U13** | P1 | `library` | `ChainOfThought::<S>::new()` | construct | → S2 (internal predict) | — | F11 |
| **U14** | P1 | `library` | `ReAct::<S>::builder().tool("name", "desc", fn).build()` | construct | → S2, → S4 | — | F11 |
| **U16** | P1 | — | Strategy swap: change type annotation (e.g. `Predict<QA>` → `ChainOfThought<QA>`). **Note:** output type also changes (`QAOutput` → `WithReasoning<QAOutput>`), breaking explicit type annotations and downstream function signatures. Compiler catches all breakage. | compile | — | — | F4 |
| **U48** | P1 | `module` | `dsrs::forward_all(&module, inputs, concurrency).await` — standalone utility. Returns `Vec<Result<Predicted<Output>, PredictError>>`. Individual failures don't abort batch. Module trait stays minimal (`forward` hook + default `call`). | call | → N8 (×N) | → Vec\<Result<Predicted<Output>, PredictError>\> | F4 |
| **U50** | P1 | `optimizer` | `optimizer.compile(&mut module, trainset, metric).await` — hands module to optimizer. Exclusive `&mut` = no concurrent forward() during optimization. This is the P1→P3 entry point. | call | → U30 (P3 entry) | → &mut module (optimized in place) | F6, F8 |
| **U51** | P1 | `module` | `module.map(\|output\| transform(output))` — output transformation combinator. Constructs `Map<M, F>` wrapping the original module. Also `.and_then()` for fallible transforms. P1 ramp to avoid `impl Module` for simple post-processing (e.g., derive confidence from reasoning). Map/AndThen must have manual Facet impls exposing `inner` field for N18 walker traversal. | construct | — | → Module\<Output=NewType\> | F4 |
| **U49** | P1 | `module` | `PredictError` variants — `Provider { source }` (retry-worthy: network, timeout, rate limit), `Parse { raw_response, field, stage, detail }` (prompt-engineering problem). `stage` distinguishes substages within N8: `SectionParsing` (missing `[[ ## field ## ]]` markers), `Coercion` (jsonish can't parse field value), `PathAssembly` (nested structure mismatch). N13 failures use stage `TypeConversion` (BamlValue→typed output mismatch). Error Display includes full LM response text. | access | — | ← N8, ← N13 | F5, F7 |
| | | | | | | | |
| **U17** | P2 | `augmentation` | `#[derive(Augmentation)]` on struct | compile | → N14 | — | F3 |
| **U18** | P2 | `augmentation` | `#[augment(output, prepend)]` attribute | compile | → N14 | — | F3 |
| **U19** | P2 | `augmentation` | `Augmented<S, A>` in type position | compile | — | — | F3 |
| **U20** | P2 | `augmentation` | `WithReasoning<O>` generated wrapper type | access | — | ← N14 | F3 |
| **U21** | P2 | `signature` | `#[derive(Signature)]` with generic type params | compile | → N15 | — | F12 |
| **U22** | P2 | `signature` | `#[flatten]` on fields | compile | → N15 | — | F12 |
| **U23** | P2 | `adapter` | `ChatAdapter::build_system(schema, override)` | call | → N3 | → Result\<String\> | F7 |
| **U24** | P2 | `adapter` | `ChatAdapter::format_input(schema, &input)` | call | → N8 (formatting internals) | → String | F7 |
| **U25** | P2 | `adapter` | `ChatAdapter::parse_sections(content)` | call | — | → IndexMap | F7 |
| **U26** | P2 | `adapter` | `ChatAdapter::parse_output::<O>(schema, &response)` | call | → N8 (coercion internals), → N13 | → Result\<O\> | F7 |
| **U27** | P2 | `module` | `impl Module for MyModule { type Input; type Output; fn forward() }` | compile | — | — | F4 |
| **U28** | P2 | — | `Predict<Augmented<S, A>>` as internal field | compile | — | — | F3, F5 |
| **U29** | P2 | — | `#[derive(Facet)]` on module struct | compile | — | — | F6 |
| | | | | | | | |
| **U30** | P3 | `discovery` | `named_parameters(&mut module)` — takes exclusive `&mut` access | call | → N18 | → U31 | F6 |
| **U31** | P3 | `discovery` | `Vec<(String, &mut dyn DynPredictor)>` return — mutable handles for optimizer mutation | access | → U32–U37 | ← N18 | F6 |
| **U32** | P3 | `dyn_predictor` | `predictor.schema()` | call | — | → &SignatureSchema | F8 |
| **U33** | P3 | `dyn_predictor` | `predictor.demos_as_examples()` | call | → N21 | → Vec\<Example\> | F8 |
| **U34** | P3 | `dyn_predictor` | `predictor.set_demos_from_examples(demos)` | call | → N22 | → Result\<()\> | F8 |
| **U35** | P3 | `dyn_predictor` | `predictor.instruction()` / `set_instruction(s)` | call | → S3 | → String | F8 |
| **U36** | P3 | `dyn_predictor` | `predictor.dump_state()` / `load_state(state)` | call | → S2, → S3 | → PredictState | F8 |
| **U37** | P3 | `dyn_predictor` | `predictor.forward_untyped(BamlValue)` | call | → N23, → N8 | → Result\<BamlValue\> | F8 |
| | | | | | | | |
| **U38** | P4 | `registry` | `registry::create(name, &schema, config)` | call | → N17, → S7 | → Box\<dyn DynModule\> | F9 |
| **U39** | P4 | `registry` | `registry::list()` | call | → S7 | → Vec\<&str\> | F9 |
| **U40** | P4 | `dyn_module` | `dyn_module.predictors()` / `predictors_mut()` | call | — | → Vec\<(&str, &dyn DynPredictor)\> | F9 |
| **U41** | P4 | `graph` | `ProgramGraph::new()` | construct | → S5, → S6 | — | F10 |
| **U42** | P4 | `graph` | `graph.add_node(name, node)` | call | → S5 | → Result | F10 |
| **U43** | P4 | `graph` | `graph.connect(from, from_field, to, to_field)` (`from == "input"` reserved for pseudo-node root wiring; user nodes cannot be named `"input"`; duplicate edges are rejected explicitly) | call | → N24, → S6 | → Result | F10 |
| **U44** | P4 | `graph` | `graph.replace_node(name, node)` | call | → S5, → N24 | → Result | F10 |
| **U45** | P4 | `graph` | `graph.execute(input).await` | call | → N25, → N26 | → Result\<BamlValue\> | F10 |
| **U46** | P4 | `graph` | `ProgramGraph::from_module(&module)` / `ProgramGraph::from_module_with_annotations(&module, annotations)` (explicit per-call annotation projection; no global annotation registry) | call | → N18 (reuses F6 walker) | → Result\<ProgramGraph\> | F10 |

---

## Code Affordances (Internal Mechanisms)

| # | Place | Module | Affordance | Control | Wires Out | Returns To | Part |
|---|-------|--------|------------|---------|-----------|------------|------|
| **N1** | P1 | `signature` (macro) | Proc macro expansion — generates `QAInput`, `QAOutput` structs + `impl Signature` | compile | → U4, → U5 | — | F1 |
| **N2** | P1 | `signature` (macro) | Extract doc comment → `fn instructions() -> &'static str` | compile | — | → N8 | F1 |
| **N3** | P1 | `schema` | `SignatureSchema::of::<S>()` — TypeId-keyed cached derivation. Internally: walk_fields (Facet shape walk, flatten-aware), build_type_ir (TypeIR from Shape), build_output_format (OutputFormatContent). Pure pipes collapsed — swapping internals changes no wiring. | cache | → S1 | → N8, → U23–U26 | F2 |
| **N8** | P1 | `adapter` | Predict call pipeline: build_system → format_demos → format_input → lm.call → parse_sections → jsonish coerce → path assembly. Internally uses format_value, navigate_path, insert_at_path, jsonish::from_str (all collapsed — pure pipes). **Error boundary for coercion:** produces `PredictError::Parse` with raw content + field name + coercion detail when LM output doesn't parse. LM resolution: scoped context (`dsrs::with_lm`) > global default (`GLOBAL_SETTINGS`). Returns `Result<Predicted<O>, PredictError>` via `Module::call` (delegating to module `forward`). | call | → N3, → S2 (read demos), → N13, → LM | → U10, → U49 (on error) | F5, F7 |
| **N13** | P1 | `adapter` | `O::try_from_baml_value()` — BamlValue → typed output. **Error boundary:** rejects structurally invalid BamlValue (constraint violations, missing fields). Distinct from N8 coercion errors: N8 = "couldn't understand LM text", N13 = "understood it but doesn't match expected type." | compute | — | → U10 | F7 |
| | | | | | | | |
| **N14** | P2 | `augmentation` (macro) | Augmentation proc macro — generates `WithX<O>` + `Deref` + `impl Augmentation`. Includes tuple composition: `impl Augmentation for (A, B)` provides `(A, B)::Wrap<T> = A::Wrap<B::Wrap<T>>` via GATs (type-level only, no code generation — collapsed from former N16). | compile | → U20 | — | F3 |
| **N15** | P2 | `signature` (macro) | Generic signature macro — `split_for_impl()`, generic param threading, flatten handling | compile | → U4, → U5 (generic variants) | — | F12 |
| **N17** | P2/P4 | `dyn_module` | Schema transformation — factory modifies `SignatureSchema` (prepend reasoning, build action schema, etc.) | compute | → N3 | → U38 | F9 |
| | | | | | | | |
| **N18** | P3 | `discovery` | `walk_value()` — recursive Facet traversal over struct fields and supported containers (`Option`, list/array/slice, `HashMap<String, _>`, `Box`). Resolves `PredictAccessorFns` through runtime shape-id registration, then casts to `&mut dyn DynPredictor` (one audited unsafe boundary). Predict-like leaves without registration fail explicitly with path diagnostics (`MissingAttr`). Unsupported pointer-like containers (`Rc`, `Arc`, etc.) error explicitly with path/type diagnostics. Target state remains shape-local typed attr payload extraction. | walk | — | → U31 | F6, F8 |
| **N21** | P3 | `dyn_predictor` | `Demo<S> → Example` — `to_baml_value()` on input + output | convert | — | → U33 | F8 |
| **N22** | P3 | `dyn_predictor` | `Example → Demo<S>` — `try_from_baml_value()` gatekeeper (type safety boundary) | convert | → N23 | → S2 | F8 |
| **N23** | P3 | `dyn_predictor` | `S::Input::try_from_baml_value(input)` — typed conversion for forward_untyped | convert | → N8 | → U37 | F8 |
| | | | | | | | |
| **N24** | P4 | `graph` | `TypeIR::is_assignable_to(&to_type)` — edge type validation at connection time | validate | — | → U43, → U44 | F10 |
| **N25** | P4 | `graph` | Topological sort — determine execution order from edges | compute | → S5, → S6 | → N26 | F10 |
| **N26** | P4 | `graph` | BamlValue piping — route output fields to input fields between nodes | compute | → each node's DynModule::forward() | → U45 | F10 |
| **N27** | P4 | `registry` | `inventory::submit!` — auto-registration of StrategyFactory at link time | generate | → S7 | — | F9 |

---

## Data Stores

| # | Place | Store | Type | Written By | Read By | Part |
|---|-------|-------|------|------------|---------|------|
| **S1** | P1 (shared) | `TypeId`-keyed schema cache | Global cache, one entry per `S: Signature`. NOT a simple `OnceLock` in a generic fn (that's one static shared across all monomorphizations — a bug). Implemented as `TypeId → &'static SignatureSchema` map behind a lock, or via `generic_once_cell` pattern. Write-once, immutable after init. | N3 (first call from any Place — P1/P2/P3/P4 can all trigger init) | N3, N8, U23–U26, U32, N17, N24 | F2 |
| **S2** | P1 | `demos: Vec<Demo<S>>` | Per-Predict demo storage | U6/U7 (init), U34/N22 (optimizer), U36 (load_state) | N8 (format demos), U33/N21 (as examples), U36 (dump_state) | F5 |
| **S3** | P1 | `instruction_override: Option<String>` | Per-Predict instruction | U7 (init), U35 (optimizer) | N8 (build_system), U35 (read), U36 (state) | F5 |
| **S4** | P1 | `tools: Vec<Arc<dyn ToolDyn>>` | Per-Predict tool set | U7/U14 (init) | N8 (lm.call) | F5 |
| **S5** | P4 | `nodes: IndexMap<String, Node>` | Graph node storage | U42, U44, U46 | N25, N26, U45 | F10 |
| **S6** | P4 | `edges: Vec<Edge>` | Graph edge storage | U43 | N25, N26 | F10 |
| **S7** | P4 (global) | Strategy registry | `name → &'static dyn StrategyFactory` | N27 (link time) | U38, U39 | F9 |

---

## Wiring Narratives

### P1 Workflow: "Define a signature, create a module, call it, access the result"

```
U1 (#[derive(Signature)]) + U2 (#[input]/#[output]) + U3 (doc comment)
  → N1 (proc macro expansion — compile time)
  → generates U4 (QAInput type) + U5 (QAOutput type)

U6 (Predict::new()) → initializes S2 (empty demos), S3 (None instruction)
  — or —
U7 (builder) + U8 (Demo) → writes S2, S3, S4

U9 (module.call(input))
  → N3 (SignatureSchema::of::<S>()) → S1 (TypeId cache: cached or init)
  → N8 (adapter pipeline)
    → reads S2 (demos), LM from scoped context or global default
    → build system prompt, format demos, format input
    → LM provider (external call)
    → parse sections, jsonish coerce, path assembly (all internal to N8)
    → N13 (try_from_baml_value — error boundary: BamlValue → typed output)
  → U10 (Result<Predicted<Output>, PredictError>)
    → on error: U49 (PredictError with raw response + stage)

U10 → U5 (typed output) → U11 (result.answer) or U12 (result.reasoning via Deref)

U48 (dsrs::forward_all(&module, inputs, concurrency))
  → N8 (×N, buffer_unordered) → Vec<Result<Predicted<Output>, PredictError>>
  Individual failures don't abort the batch.

U51 (module.map(|output| transform(output)))
  → constructs Map<M, F> wrapper (no new wiring — pure value construction)
  → the returned Module delegates call() to inner via existing U9→N8 path
  → Map<M, F> has manual Facet impl: walker sees through to inner Predict leaves
  → avoids impl Module for simple post-processing (P1→P2 ramp)
```

### P2 Workflow: "Author a new augmentation"

```
U17 (#[derive(Augmentation)]) + U18 (#[augment(output, prepend)])
  → N14 (macro expansion — compile time)
  → generates U20 (WithReasoning<O> wrapper with Deref)

Module author writes:
  U28 (Predict<Augmented<S, A>> as internal field)
  + U29 (#[derive(Facet)] on module struct)
  + U27 (impl Module for MyModule)
  (all compile — compiler validates type constraints immediately)

Inside forward(), module author calls:
  U23 (build_system) → N3 (schema)
  U24 (format_input) → N8 internals (format_value, navigate_path)
  U26 (parse_output) → N8 internals (jsonish coerce, path assembly) → N13
  — or simply delegates to internal Predict::forward() (most common path)
```

### P3 Workflow: "Discover and optimize parameters"

```
U50 (optimizer.compile(&mut module, trainset, metric))
  → exclusive &mut access — no concurrent forward() during optimization

  U30 (named_parameters(&mut module))
    → N18 (walk_value: recurse through struct fields via Facet reflection,
           resolve PredictAccessorFns via runtime shape-id registry,
           fail explicit on unregistered Predict-like leaves,
           cast to &mut dyn DynPredictor — one audited unsafe boundary)
    → U31 (Vec<(path, &mut dyn DynPredictor)>)

  For each discovered predictor:
    U32 (predictor.schema()) → S1 (understand field structure)
    U33 (demos_as_examples) → N21 (Demo→Example via to_baml_value)
    — optimizer manipulates examples —
    U34 (set_demos_from_examples) → N22 (Example→Demo via try_from_baml_value)
      → N22 is the TYPE SAFETY GATEKEEPER: mismatched schema → error, not silent data loss
      → S2 (demos overwritten on the original typed Predict)

    U35 (set_instruction) → S3 (instruction overwritten)
    U37 (forward_untyped) → N23 (BamlValue → typed input) → N8 (normal call pipeline)
      → optimizer uses this for evaluation loops
      → may use dsrs::with_lm(cheap_lm, || ...) for scoped override during eval
```

### P4 Workflow: "Build and execute a dynamic graph"

```
U41 (ProgramGraph::new()) → S5 (empty nodes), S6 (empty edges)

U38 (registry::create("chain_of_thought", &schema, config))
  → S7 (lookup factory) → N17 (schema transformation) → Box<dyn DynModule>

U42 (graph.add_node("cot", node)) → S5

U43 (graph.connect("input", "question", "cot", "question"))
  → N24 (TypeIR::is_assignable_to) → S6 (edge stored if valid)

U44 (graph.replace_node("cot", new_node)) → S5, re-validates via N24
U46 (ProgramGraph::from_module(&module))
  → N18 (reuses F6 walker) → projects S5; then uses schema/path inference to populate S6
  → multi-node projections with no resolvable edges return an explicit projection error
  or
U46 (ProgramGraph::from_module_with_annotations(&module, annotations))
  → N18 (reuses F6 walker) → applies explicit per-call annotations first
  → if `annotations` is empty, falls back to the same inference path as `from_module`
  → no global/ambient annotation registry influences projection

graph.fit(&mut module)
  → applies graph predictor state back to typed predictors by canonical path
  → enforces strict 1:1 path mapping and surfaces projection mismatch on divergence

U45 (graph.execute(input))
  → N25 (topological sort from S5 + S6)
  → N26 (pipe BamlValues between nodes following edges)
  → each node's DynModule::forward()
  → Result<BamlValue>
```

### Cross-Place Wiring

```
P1 → P3: U50 (optimizer.compile(&mut module, trainset, metric)).
  Exclusive &mut borrow — P1 cannot call forward() during optimization.
  Optimizer calls U30 (named_parameters), which uses N18 (walker)
  to reach INTO the P1 module's Predict leaves.
  N18 (walker) casts to &mut dyn DynPredictor — this is the P1→P3 boundary crossing.
  After optimization, S2/S3 are mutated but the typed module is unchanged.

P3 → P1: After optimization, &mut borrow released.
  User calls U9 (module.call()) as normal.
  The module reads from S2/S3 which now contain optimized demos/instructions.
  No code change in P1 — optimization is invisible.

P1 → P4: Typed module projected into graph.
  U46 (ProgramGraph::from_module) calls N18 (same walker as P3)
  to discover Predict leaves and create graph nodes.

P4 → P3: Graph nodes contain DynModules with internal predictors.
  U40 (dyn_module.predictors()) exposes them for P3-style optimization.

S1 (SignatureSchema cache) is the shared backbone:
  Written once by N3, read by all Places — N8 (P1 adapter pipeline),
  U23-U26 (P2 building blocks), U32 (P3 schema access),
  N17 (P4 schema transformation), N24 (P4 edge validation).
  Any Place can trigger first init — it's idempotent and immutable after.
```

---

## Vertical Slices

Each slice is a demo-able increment that demonstrates a mechanism working. Slices cut through all layers (types, logic, data) to deliver a working feature. Every affordance is assigned to the slice where it is first needed.

### Slice Summary

| # | Slice | Mechanisms | Demo |
|---|-------|-----------|------|
| **V1** | Typed call | F1, F2, F5, F7 | "Define QA, create Predict, call it, get typed result with structured output parsing." |
| **V2** | Augmentation + ChainOfThought | F3, F11(CoT) | "Use ChainOfThought, get result.reasoning and result.answer via Deref." |
| **V3** | Module authoring | F4(full), F12 | "Author a custom two-step module with generic signatures. Adapter building blocks available." |
| **V4** | ReAct + operational | F11(ReAct) | "ReAct with tool builder. Batch calls. .map() transforms output." |
| **V5** | Optimizer interface | F6, F8 | "Discover Predict leaves, mutate demos/instructions, verify effect. Dump/load state." |
| **V6** | Dynamic graph | F9, F10 | "Build graph from registry, validate edges, execute. Project typed module into graph." |

### Dependency Graph

```
V1 ← V2 ← V3 ← V4
V1 ← V2 ← V5
V1 ← V2 ← V3 ← V4 ← V6
V1 ← V5 ← V6
```

V5 (optimizer) depends on V2 (needs augmented modules to test multi-level discovery), not on V3/V4. V6 (dynamic graph) depends on V5 (graph nodes expose predictors via DynPredictor).

### Slice Details

**V1: Typed call** — F1, F2, F5, F7

| # | Affordance | Slice Role |
|---|------------|------------|
| U1, U2, U3 | Signature derive + markers + doc comment | Entry point |
| U4, U5 | Generated QAInput / QAOutput types | Compile-time output |
| U6, U7, U8 | Predict construction + builder + Demo | Module setup |
| U9, U10, U11 | forward(), Predicted<O>, field access | Call and result |
| U49 | PredictError variants | Error path |
| N1, N2 | Proc macro expansion, doc extraction | Compile-time mechanisms |
| N3 | SignatureSchema derivation | Schema cache |
| N8 | Adapter pipeline | Core call mechanism |
| N13 | try_from_baml_value | Type conversion boundary |
| S1, S2, S3 | Schema cache, demos, instruction | State |

Demo program:
```rust
#[derive(Signature, Clone, Debug)]
/// Answer questions accurately.
struct QA {
    #[input] question: String,
    #[output] answer: String,
}

let predict = Predict::<QA>::new();
let result = predict.call(QAInput { question: "What is 2+2?".into() }).await?;
println!("{}", result.answer);  // typed field access
```

**V2: Augmentation + ChainOfThought** — F3, F11(CoT)

| # | Affordance | Slice Role |
|---|------------|------------|
| U12 | Deref to augmented field | Augmented output access |
| U13 | ChainOfThought::new() | Library module |
| U16 | Strategy swap (type annotation change) | Composability demo |
| U17, U18, U19, U20 | Augmentation derive + attributes + wrapper | Augmentation system |
| U28, U29 | Predict\<Augmented\> field + derive(Facet) | Module internals |
| N14 | Augmentation macro (incl. tuple composition) | Compile-time mechanism |

Demo program:
```rust
let cot = ChainOfThought::<QA>::new();
let result = cot.call(QAInput { question: "What is 2+2?".into() }).await?;
println!("Reasoning: {}", result.reasoning);
println!("Answer: {}", result.answer);  // via Deref
```

**V3: Module authoring** — F4(full), F12

| # | Affordance | Slice Role |
|---|------------|------------|
| U21, U22 | Generic signature derive + #[flatten] | Generic type support |
| U23, U24, U25, U26 | ChatAdapter building blocks | Fine-grained adapter control |
| U27 | impl Module for MyModule | Module trait |
| N15 | Generic signature macro | Compile-time mechanism |

Demo program:
```rust
#[derive(Facet)]
struct SimpleRAG {
    retrieve: Predict<Retrieve>,
    answer: ChainOfThought<QAWithContext>,
}

impl Module for SimpleRAG {
    type Input = QAInput;
    type Output = WithReasoning<QAWithContextOutput>;
    async fn forward(&self, input: QAInput) -> Result<Predicted<Self::Output>, PredictError> {
        let ctx = self.retrieve.call(RetrieveInput { query: input.question.clone() }).await?;
        self.answer.call(QAWithContextInput { question: input.question, context: ctx.passages }).await
    }
}
```

**V4: ReAct + operational affordances** — F11(ReAct)

ReAct is the one library module beyond ChainOfThought. It exercises generic signatures (F12), tools (S4), adapter building blocks (F7), and multi-Predict composition (action + extract steps with a loop inside forward()). If ReAct works, the system handles every augmentation pattern. BestOfN/Refine/ProgramOfThought/MultiChainComparison are deferred — they're consumers of the same mechanisms ReAct proves.

| # | Affordance | Slice Role |
|---|------------|------------|
| U14 | ReAct builder with tools | Library module |
| U48 | dsrs::forward_all (batching) | Operational |
| U51 | module.map() combinator | Operational |
| S4 | Tools storage | State |

Demo program:
```rust
// ReAct with tools — exercises generic sigs, tool builder, adapter building blocks, multi-Predict
let react = ReAct::<QA>::builder()
    .tool("search", "Search the web", search_fn)
    .build();
let result = react.call(QAInput { question: "Who won the 2024 election?".into() }).await?;

// Batch 10 inputs concurrently
let results = dsrs::forward_all(&react, inputs, 5).await;

// Transform output without impl Module
let confident = cot.map(|r| ConfidentAnswer { answer: r.answer.clone(), confidence: 0.9 });
```

**V5: Optimizer interface** — F6, F8

| # | Affordance | Slice Role |
|---|------------|------------|
| U50 | optimizer.compile(&mut module, trainset, metric) | P1→P3 entry |
| U30, U31 | named_parameters, handle vec | Discovery |
| U32 | predictor.schema() | Schema access |
| U33, U34 | demos_as_examples / set_demos | Demo mutation |
| U35 | instruction / set_instruction | Instruction mutation |
| U36 | dump_state / load_state | State persistence |
| U37 | forward_untyped | Untyped evaluation |
| N18 | walk_value (Facet walker) | Discovery mechanism |
| N21, N22, N23 | Type conversions (Demo↔Example, BamlValue→typed) | Conversion boundaries |

Demo program:
```rust
let mut module = SimpleRAG::new();  // from V3

// Discover all Predict leaves — no annotations needed
let params = named_parameters(&mut module);
assert_eq!(params.len(), 2);  // retrieve.predict + answer.predict

// Mutate demos
params[0].1.set_demos_from_examples(new_demos)?;
params[1].1.set_instruction("Be concise.".into());

// Verify mutations took effect
let result = module.call(input).await?;

// Save optimized state to disk
let state = dsrs::dump_state(&module);
std::fs::write("optimized.json", serde_json::to_string(&state)?)?;
```

**V6: Dynamic graph** — F9, F10

| # | Affordance | Slice Role |
|---|------------|------------|
| U38, U39 | registry::create, registry::list | Factory interface |
| U40 | dyn_module.predictors() | Predictor access |
| U41, U42, U43, U44 | ProgramGraph construction + mutation | Graph building |
| U45 | graph.execute() | Graph execution |
| U46 | ProgramGraph::from_module() | Typed→graph projection |
| N17 | Schema transformation | Factory mechanism |
| N24 | Edge type validation | Validation mechanism |
| N25, N26 | Topological sort + BamlValue piping | Execution mechanism |
| N27 | inventory::submit! (auto-registration) | Registration mechanism |
| S5, S6, S7 | Nodes, edges, registry | State |

Demo program:
```rust
let mut graph = ProgramGraph::new();
let cot = registry::create("chain_of_thought", &schema, Default::default())?;
graph.add_node("cot", cot)?;
graph.connect("input", "question", "cot", "question")?;
let result = graph.execute(BamlValue::from_map([("question", "What is 2+2?")])).await?;
```
