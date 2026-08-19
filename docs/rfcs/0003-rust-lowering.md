# RFC 0003: Rust as a Frontend — `#[module]` Lowering

**Status:** proposed · **Date:** 2026-08-14 · **Branch:** `v1-revamp` · **Depends on:** RFC 0001 (trace format), RFC 0002 (IR) · **Amends:** RFC 0002 §4.3–4.4 (frontend story), §2.2 (`HoleNode`), §2.3 (`Route` discriminants, `Loop` init) · **Blocks:** nothing — every stage is additive until the final cleanup

---

## 0. Summary

The IR is DSRs' **internal representation**, not a sibling product. Every harness — Rust-authored or LLM-authored — passes through the same `Program`. This RFC adds the missing half of that sentence: a compiler frontend from ordinary Rust to the IR.

Decisions, in one screen:

- **`#[module]` on an async fn with a body** lowers the body to a `Program` at macro expansion. One parse emits **two projections** — the executable fn and the IR constant — so drift between code and graph is impossible *by construction* (the async/await precedent: you write straight-line code, the compiler emits the machine).
- **There is no extractor to replace.** The old trace-based graph recovery (`trace::Graph`, sequential-approximation edges, raw-pointer node identity) was deleted in `8030185` and nothing replaced it — today zero paths exist from a Rust module to a `Program`. Post-hoc extraction is not revived: it cannot see untaken branches, collapses forks to chains, and breaks on moved predictors. Lowering is **syntax-driven**, never run-driven.
- **The mappable subset of Rust** lowers to the closed node set: `let` sequences → `Seq`, `match` on enum outputs → `Route`, `if`/`else` on bool outputs → `Route` (new bool discriminant), bounded `for` with mutated locals → `Loop` (carry inferred), `join!`/`try_join!` → `ForkJoin`, recognized `dsrs::retry`/`dsrs::refine` forms → `Retry`/`Refine`, calls to `#[predict]`/`#[cot]`/`#[agent]` fns → leaves.
- **Everything else becomes a typed host hole** — a new `HoleImpl::Host` mirroring `ToolKind::Host`: the macro extracts the expression into a generated native fn, derives its signature from the Rust types, content-hashes it, and binds it at load. Program portability is tiered, reported, and explicit; `#[module(deny_holes)]` makes hole-ization a compile error.
- **Execution goes through the interpreter in v1.** The generated fn converts typed args to a `JsonMap`, reads the ambient overlay, and calls `Interpreter::run` against the emitted `Program`. One execution engine = one semantics; the monomorphized emission is `include_program!`'s job (RFC 0002 §6.1) and `#[module]` switches to that emitter transparently when it lands.
- **The builder API is demoted to plumbing.** It was never a good user surface (stringly names, panic-on-misuse combinators); it is a good assembly language. The `.dsrs` parser already lowers through it; the `#[module]` expansion becomes its second client. It leaves the documented user surface.
- **The frontends are now three, with distinct jobs:** Rust `#[module]` (humans writing harnesses), `.dsrs` text (LLMs writing harnesses; the wire form; the optimization artifact), builder (internal lowering target). All construct the same `Program`; `program.to_text()` of a Rust-authored harness is a legal, servable, optimizable `.dsrs` file.

---

## 1. Motivation: the sibling-product failure

RFC 0002 shipped the IR with two authoring surfaces — the `.dsrs` text format and the `ProgramBuilder` — and zero connection to the way DSRs users actually write harnesses (`#[predict]` fns, `fx`, `Predict<S>` composition). The result is exactly the failure the vision warned about from the DSPy side, mirrored: instead of "programs trapped in Python, parameters in the artifact," we got "programs in the artifact, users in Rust." A Rust harness cannot be optimized by structure-aware optimizers, cannot be served from an artifact, cannot be diffed, and cannot be *seen* by anything that speaks the IR — unless its author abandons the language they chose DSRs for and rewrites against a stringly builder or a text file.

The principle this RFC enforces: **closedness is an IR property; expressivity is a frontend property.** The 9-node vocabulary felt restrictive because it was the front door. It is the correct *internal* representation — AFlow's lesson stands, small operator vocabularies are what make search tractable — and the front door is Rust.

Why not extraction? The deleted `trace::Graph` path is the cautionary tale, and its three drift axes are structural, not bugs: (a) a recorded run linearizes concurrency and control flow into "the node that happened to run before me" (`dag.rs` documented its own edges as "a sequential approximation of dataflow"); (b) branches not taken in the recorded run do not exist in the recovered graph; (c) identity was the predictor's *address* (`instance_key: usize`), invalid across moves, clones, and processes. Fixing this requires a dynamo-class guard-and-specialize machine — enormous, and still a compromise. Syntax has none of these problems: every branch is visible, concurrency is explicit, and identity is a name in the source.

---

## 2. Guide level: what you write, what you get

The qa pipeline from RFC 0002 §4.3, authored as Rust. Steps are declared with the existing bodyless-fn macros (extended per §3.1); composition is an ordinary fn body:

```rust
/// Draft a thorough, factual answer.
#[cot(model = "@deep")]
fn draft(question: String) -> String;                 // → sig Draft (existing macro)

/// Verify the draft against sources. Search until every claim is
/// supported; collect URLs.
#[agent(model = "@fast", tools(search), max_turns = 6,
        budget(tokens = 40_000, on_exhausted = finalize))]
fn research(question: String, draft: String) -> Vec<String>;   // → sig Research

/// Web search; returns result snippets with URLs.
#[tool(caps("net:search"))]
fn search(query: String) -> Vec<String>;              // host tool, bound at load

#[module(caps("net:search"))]
async fn qa(question: String) -> Result<QaOut, RunError> {
    let drafter = draft(question.clone()).await?;
    let researcher = research(question, drafter.answer.clone()).await?;

    // Arbitrary Rust: not mappable → becomes a typed HOST hole named `checker`.
    let checker = QaOut {
        answer: drafter.answer,
        sources: researcher.evidence.into_iter()
            .filter(|e| e.starts_with("http")).collect(),
    };
    Ok(checker)
}
```

What the macro emits, from this single parse:

1. **`qa::program() -> &'static Program`** — assembled once (LazyLock) through the builder: `Seq[cot drafter, agent researcher, hole checker]`, with bindings read off the data flow (`question` → `$.question`, `drafter.answer` → `Out(drafter, "answer")`), leaf names taken from the `let` bindings, and the citation filter carried as `HoleImpl::Host { hash }` with signature `{draft: string, evidence: string[]} -> {answer: string, sources: string[]}` derived from the Rust types.
2. **The executable `qa()` fn** — typed boundary in, `Interpreter::run` inside, typed boundary out (§5). Ambient `with_overlay`/`with_params` apply exactly as they do to a loaded `.dsrs` program.
3. **`qa::OPACITY`** — the hole report (§6): one entry, `checker`, with the source excerpt and the reason ("closure over iterator adapters — not in the mappable subset").
4. **A generated `#[cfg(test)]` validation test** — full `Program::validate()` + round-trip under `cargo test`, the same pattern `include_program!` uses.

And because the emitted `Program` is ordinary IR: `qa::program().to_text()` prints a legal `.dsrs` artifact (the hole prints as `extern`, §4.3); GEPA mutates `drafter.instruction` through an `Overlay`; `dsrs serve` refuses it politely until the serving binary registers the host hole — or the author swaps the filter to a JS hole and the artifact becomes fully portable.

---

## 3. Reference: the lowering

### 3.1 Step declarations — metadata the frontend can see

Proc macros are syntactic; `#[module]` cannot ask the type system what `draft` is. The existing convention already solves this — `#[predict]`/`#[cot]` emit a `mod <name>` — so step macros additionally emit a well-known item:

```rust
// inside the generated `mod draft`:
pub fn __dsrs_step() -> dsrs::ir::StepDef;   // kind, &'static SignatureDef,
                                             // model ref, tools, stop/budget/caps
```

`#[module]`-generated assembly code references `<callee>::__dsrs_step()`; a call to a fn that was not declared with a step macro fails **in rustc**, at the call site, with a missing-item error — semantic checking delegated to the compiler, the standard trick. New/extended macros:

| Macro | Today | This RFC |
|---|---|---|
| `#[predict]` / `#[cot]` | bodyless fn → fx call | + `__dsrs_step()` (kind Predict/Cot, sig via `SignatureDef::of`); + optional `model = "@name"` attr. Direct calls outside `#[module]` keep the fx path — steps stay usable standalone |
| `#[agent]` | — (new) | bodyless fn + attrs `model`, `tools(...)`, `stop_tools(...)`, `max_turns`, `until_parse`, `budget(...)`, `context(...)` → `StepDef` kind Agent. Direct calls run the static-lane tool loop |
| `#[tool]` | — (new) | bodyless fn + `caps(...)` → `ToolDef` metadata (doc comment = the `ToolDesc` gene default, params/return = the tool sig). Host-bound at load, same as `ToolKind::Host` today. Sandboxed JS tools remain declared in `.dsrs` or registered on `RuntimeEnv` |
| `#[module]` | — (new) | the frontend (this RFC). Attrs: `caps(...)` (the program ceiling), `models(...)` (name → `LMConfig` expr), `deny_holes` |

### 3.2 The mappable subset

Lowering is a fold over the fn body. Constructs are matched **structurally**; anything that fails to match falls to §4 (hole) — never to an error, unless `deny_holes`.

| Rust | IR | Rules |
|---|---|---|
| `let name = step(args).await?;` | leaf node (`Predict`/`AgentLoop` per `StepDef`) | leaf name = the binding identifier (program-unique or compile error; `_` or no binding → callee name). Args must be scope inputs, prior step output fields, literals, or `.clone()`s thereof — each becomes a `Binding`. Any other arg expression is itself hole-ized into a feeder hole |
| statement sequence | `Seq` | order preserved; the fn's `Ok(expr)` / tail struct literal becomes the `out` bindings (each field must be a port; else the tail is hole-ized, as in §2) |
| `match c.field { Variant => arm, .., _ => arm }` on an enum-typed output field | `Route` | scrutinee must be a port; path patterns → variant arms, `_` → `default`. Each arm is lowered as a scope (step call or block → `Seq`); arm output shapes must agree (RFC 0002 rule, checked at build) |
| `if c.field { .. } else { .. }` on a bool-typed output port | `Route` with bool discriminant | **amendment** to RFC 0002 §2.3: `Route.on` may be `Bool`-typed with arms exactly `true`/`false` (or one + `else`). Interpreter already string-matches (`to_string()` → `"true"`/`"false"`); only `validate::route_variants` and the text grammar (BOOL as arm label) extend. Additive within format major 1 |
| `let mut x = init; for _ in 0..N { x = step(..).await?; }` (+ optional `if !cond { break }` as first or last statement) | `Loop` | `max_iters = N` (literal or const); mutated locals → `carry`; the `break` condition's port → `while_` (negated as needed); loop-body reads of `x` → `^x` |
| `let (a, b) = tokio::join!(e1, e2);` / `try_join!` | `ForkJoin` | each branch a step call or block; tuple bindings name the branches; `join` bindings read off subsequent uses |
| `dsrs::retry(attempts, backoff_ms, expr)` | `Retry` | recognized marker form; also a real fn so the standalone static lane behaves identically |
| `dsrs::refine { body, judge, threshold, max_rounds, feedback_field }` | `Refine` | ditto |
| call to another `#[module]` fn | build-time splice | §3.4 |

Two-stage lowering, mirroring `include_program!`: the macro decides **structure** (nodes, names, bindings) syntactically and emits assembly code; `SignatureDef`s, type checks, and `Program::validate()` run at first use inside the LazyLock, with the generated `#[cfg(test)]` test forcing them under `cargo test`. A binding whose type doesn't match its port fails validation with the source-mapped leaf name.

### 3.3 `Loop` iteration-0: the `init` amendment

RFC 0002's v1 shadow rule (`CarryNotScopeInput`) requires every carried name to shadow an enclosing scope *input*. Real Rust immediately violates this — `let mut draft = first(q).await?.answer;` seeds the carry from a **node output**. Amendment (additive):

```rust
pub struct LoopNode {
    // ... existing fields ...
    /// Iteration-0 values for carried names that do not shadow a scope
    /// input. Ports resolve in the enclosing scope. Empty = shadow rule.
    pub init: Box<[Binding]>,
}
```

Text form: an `init { field = port }` block in `loop`; printer omits it when empty. Validation: every carried name is either shadowed (old rule) or initialized (new rule), never both.

### 3.4 Nested modules

A call to another `#[module]` fn splices the callee's `Program` at build time: its nodes, sigs, params, tools, and caps merge into the caller's arenas, leaf names prefixed `"<binding>__<leaf>"` (`__` is reserved; `#[module]` rejects user step names containing it, consistent with the sandbox's `__dsrs` reservation). The callee's caps must be ⊆ the caller's ceiling — checked at build with a source-mapped error. Module boundaries are not preserved in the artifact in v1 (the printed `.dsrs` is flat); a first-class `sub` node is Q3.

---

## 4. Host holes

### 4.1 The IR change

`HoleNode` today is unconditionally sandboxed JS. This RFC mirrors `ToolKind` (the `interp.rs` precedent: program semantics vs host presentation):

```rust
pub struct HoleNode {
    pub name: Sym,
    pub sig: SigId,
    pub imp: HoleImpl,               // was: code: ParamId
    pub caps: CapSet,
    pub binding: Box<[Binding]>,
}

pub enum HoleImpl {
    /// Sandboxed source carried in the artifact; the Code gene is optimizable.
    Sandboxed { code: ParamId },
    /// Native fn bound by name from RuntimeEnv at load ("extern").
    /// hash = xxhash64 of the normalized Rust source tokens — integrity,
    /// lineage, and the replay/rollout-cache key. Not optimizable as a
    /// Code gene (no rustc at runtime); an optimizer MAY propose a
    /// Sandboxed replacement candidate (out of scope here, noted in Q2).
    Host { hash: u64 },
}
```

`RuntimeEnv` gains `host_holes: HashMap<String, HostHoleFn>` (`Arc<dyn Fn(JsonMap) -> BoxFuture<Result<JsonMap, String>> + Send + Sync>`); `Interpreter::load` binds by leaf name and fails with `LoadError::HostHoleUnbound` — identical shape to `HostToolUnbound`. `#[module]`-emitted code populates these from the extracted fns automatically; `dsrs serve` refuses programs with unbound host holes exactly as it refuses `ToolKind::Host` tools today.

### 4.2 Extraction and signature derivation

For an unmappable expression bound as `let name = <expr>;`, the macro: collects the free variables of `<expr>` that resolve to ports (each becomes an input field; its Rust type must map into the closed `FieldType` subset, else a compile error naming the offending type and the subset); takes the binding's type as the output shape (a struct literal or a `#[Schema]` type maps field-per-field; a scalar becomes a single `value` output — hmm no: a scalar output field named after the binding, matching `#[predict]`'s convention); wraps `<expr>` in a generated `fn __hole_<name>(input: JsonMap) -> ...` with serde at the boundary; and emits `HoleImpl::Host { hash }`. Unmappable *statements* (no binding, side-effect-only) are a compile error under `deny_holes` and otherwise hole-ized with a synthesized name `hole_<n>` — discouraged, reported.

### 4.3 Text form and portability tiers

```
checker = hole CiteCheck (draft = drafter.answer, evidence = researcher.evidence)
  caps [] extern "9f2c3a1b00d4e5f6"
```

Grammar: the `expr` production's hole arm gains `| "hole" IDENT args? "caps" "[" cap* "]" ( "js" CODE | "extern" STRING )`. Additive keyword within format major 1. The printed artifact of a Rust-authored program is thereby **honest about its portability**:

| Tier | Contents | Servable by |
|---|---|---|
| **Portable** | JS holes only | any host (`dsrs serve`, Python runner, anything) |
| **Bound** | ≥1 `extern` hole or host tool | a binary that registers every named binding with a matching hash |

`dsrs check` prints the tier and the unbound-name list. Promotion path: rewrite the hole in JS (hand or optimizer), and the artifact climbs a tier — the graph is identical either way.

### 4.4 Replay and the hole-hash fix

Two existing gaps block deterministic replay of *any* hole, and this RFC takes them as prerequisites (stage M-1):

1. **The interpreter must consult the replay scope.** `trace::replay::intercept` has one call site today (static-lane `Predict`); `eval_predict`/`eval_agent` call the LM directly. Fix: the same intercept seam at both interpreter call sites.
2. **Hole span `request_hash` is degenerate.** Hole spans open with an empty prompt and a constant `"sandbox:quickjs"` config, so *every hole span in every program hashes identically*. Fix: hole spans hash `(impl discriminant ++ code-or-source hash ++ canonical input JSON ++ sorted caps)`. The existing `ToolRun` event already records `args`/`result` verbatim — replay serves the recorded result without re-running the hole, which is precisely the contract host holes need: **native code is not assumed pure; it is recorded, and replay serves the record.** Strict-mode divergence on the new preimage catches a changed hole implementation (`hash` mismatch) before it silently changes results.

---

## 5. The execution projection

The generated fn body, in full:

```rust
pub async fn qa(question: String) -> Result<QaOut, RunError> {
    let interp = __interp();                       // LazyLock<Interpreter>: program()
                                                   // + env with host holes/tools bound,
                                                   // grants = program caps, sandbox default
    let input = __to_input(question)?;             // serde → JsonMap, sig-checked
    let overlay = dsrs::ir::bridge::current_overlay();   // ambient task-local (new accessor)
    let out = interp.run(input, overlay, Budget::default()).await?;
    __from_output(out)                             // JsonMap → QaOut via serde
}
```

Why interpret rather than emit native control flow: **one execution engine is one semantics.** A dual-engine design (macro emits native retries/loops *and* the interpreter implements them) reintroduces drift one level down — the two engines will disagree on some backoff, some budget edge, some cancellation order, and the disagreement is exactly the class of bug this RFC exists to kill. Interpreter overhead is 4–6 orders of magnitude under LM latency (RFC 0002 Q2), and the zero-cost path already has an owner: when `include_program!`'s monomorphized emitter lands (RFC 0002 §6.1, stage IR-7), `#[module]` switches its execution projection to the same emitter — callers see only the fn signature, so the switch is invisible. Compile-time-first is preserved where it pays: typed boundaries, signature derivation, hole typing, and name/structure checks all happen at compile time; what runs through the interpreter is the *validated program*, not stringly guesses.

Grants: a Rust-authored harness self-authorizes its declared ceiling (the author wrote the native code — there is nothing to withhold); sandbox caps still gate JS holes constructively. `dsrs serve` of the printed artifact re-checks grants against `--allow` as always.

---

## 6. The opacity report

Holes are legitimate — but invisible opacity rots the optimization surface, so it is always measured, never silent (the vision's "no silent caps" rule):

- `pub const OPACITY: &[HoleReport]` on every `#[module]` — `{ name, kind: Host|Sandboxed, source_excerpt, reason, input_fields, output_fields }`.
- The generated validation test prints the report; `dsrs check` on a printed artifact shows tier + hole census.
- `#[module(deny_holes)]` turns any hole-ization into a `compile_error!` carrying the excerpt and reason — for authors who want the fully-visible, fully-portable guarantee enforced.
- Rationale for no on-by-default warning: stable proc macros cannot emit warnings (`proc_macro::Diagnostic` is unstable); the report + test-time print is the reliable channel until that stabilizes (Q4).

---

## 7. Amendments to RFC 0002, and demotions

| RFC 0002 said | Now |
|---|---|
| §4.3–4.4 "dual frontends: builder / text" | **three frontends, one internal target**: `#[module]` (human surface), `.dsrs` (LLM surface + wire form), builder (plumbing: the parser's and the macro's lowering target). Builder leaves the docs/prelude; its panic-on-misuse `NodeSpec` combinators are tolerable in generated call sites and get hardened only if it ever re-surfaces |
| §2.2 `HoleNode { code: ParamId }` | `HoleNode { imp: HoleImpl }` per §4.1 (serde/text additive: `js` fence ↔ `Sandboxed`, `extern` ↔ `Host`) |
| §2.3 `Route.on` enum/literal-union only | + `Bool` with `true`/`false` arms (§3.2) |
| §2.2 `LoopNode` shadow-only carry | + `init` bindings (§3.3) |
| §3.2 interpreter table | `eval_predict`/`eval_agent` gain the replay intercept; hole spans get the non-degenerate `request_hash` preimage (§4.4) |
| — | `fx::module`/`FnModule` remain as an evaluate-only escape (whole harness opaque to the IR); documented as such. `#[module]` is the canonical composition surface |

## 8. Stages

1. **M-1 `hole-integrity`** *(no deps; prerequisite)* — `HoleImpl` split, `extern` text form, `host_holes` binding, interpreter replay intercepts, hole `request_hash` preimage. Golden tests: hole replay round-trip, strict-divergence on changed hash. **✅ Implemented 2026-08-14** (`tests/test_ir_m1_holes.rs`; hole preimage = impl discriminant ++ impl/code hash ++ sorted canonical input ++ sorted caps; host-hole spans use pseudo-model `host:extern`; interpreter `Predict`/`AgentLoop`/`Hole` all consult the replay scope, served calls reserve no budget).
2. **M-2 `stepdef`** — `StepDef` + `__dsrs_step()` on `#[predict]`/`#[cot]`; new `#[agent]`/`#[tool]` macros; `current_overlay()` accessor. Standalone (fx / static-lane) behavior unchanged. **✅ Implemented 2026-08-14** (`ir/step.rs`; `#[tool]` takes a fn *with* a body — the body is the host implementation, wrapped as a rig tool; `#[agent]` standalone calls run the static-lane tool loop; model refs via `model = "@name"`; `with_ambient_overlay`/`current_overlay` in `ir/bridge.rs`).
3. **M-3 `module-seq`** — `#[module]` for straight-line bodies only (let-sequences, tail out, feeder holes, host holes, OPACITY, generated validation test, `deny_holes`). This already delivers the headline: Rust harnesses print, serve, and optimize. **✅ Implemented 2026-08-14** (`ir/module_build.rs` is the linker — the macro emits a structure-only `ModuleSpec`, every field type resolves from step signatures at first use; holes require a simple-type ascription (`let x: Vec<String> = …;`) and extract port-shaped subexpressions as typed inputs; generated `env()` self-grants the declared ceiling and binds `default` from global settings; the generated fn runs `Interpreter::run` with the ambient overlay; step sigs are renamed to their fn names in the artifact. Tests: `test_module_macro.rs`, `test_module_agent.rs` — incl. strict replay *through* a module fn and overlay-mutated instructions).
4. **M-4 `module-flow`** — `match`/`if` → `Route` (+ bool discriminant), `for` → `Loop` (+ `init` amendment), `join!` → `ForkJoin`, `dsrs::retry`/`refine` markers.
5. **M-5 `module-compose`** — nested-module splice with `__` prefixing and cap-ceiling checks.
6. **M-6 `surface-cleanup`** — builder out of docs/prelude; frontend guide rewritten around `#[module]` + `.dsrs`; kitchen-sink golden: a Rust-authored kitchen program whose printed text round-trips against a hand-written `.dsrs` twin.

## 9. Open questions

1. **Should `#[module]` accept `impl Module` structs too?** — *Recommended: no, v1.* The fn form covers composition; struct modules keep the static lane. Revisit if real harnesses need stateful construction that fns can't express.
2. **Optimizer-proposed hole promotion (Host → Sandboxed candidates)?** — high leverage (the "evolve tool implementations" lane) but needs the judge/equivalence story; defer to the optimizer RFC.
3. **First-class `sub` node vs flat splice?** — flat keeps the enum at 9 and the grammar small; a `sub` node would preserve module boundaries in artifacts and enable per-submodule overlays. Defer until artifacts from M-5 usage argue for it.
4. **Warnings channel** — adopt `proc_macro::Diagnostic` for hole-ization warnings the day it stabilizes; until then OPACITY + test print.
5. **`while` loops with non-literal bounds** — v1 rejects (hole-izes) non-const `max_iters`; a `bound(expr)` marker form could lower a runtime bound into a `Lit` port if real programs need it. The IR stays bounded-by-construction either way.
