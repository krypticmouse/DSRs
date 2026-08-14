# DSRs v1: Harness Engineering — Vision & Research Report

**Date:** 2026-08-14 · **Branch:** `v1-revamp` · **Status:** direction-setting report, pre-RFC

---

## 1. Executive Summary

DSRs' thesis: **no model is bad — harnesses are the bottleneck.** Given a good enough harness with good enough tools, any capable model can accomplish nearly anything; the friction is that we can't find (or build, or optimize) the right harness for a given problem. DSRs should become the library that builds *harnesses for harnesses*: the best harness engineering, harness optimization, and harness generation library in existence.

Five pillars, plus two additions this research pass surfaced:

1. **An IR** that LLMs can generate to express harnesses — modules *are* harnesses
2. **Fast tool execution** — LLMs create their own tools at runtime, executed in microseconds
3. **Harness optimization** — GEPA-class optimizers that work over *any* harness
4. **De-bloat** — collapse today's over-expressive abstractions without losing declarative power
5. **Compilation, defined** — program → raw executable/servable artifact; the artifact is the IR
6. *(added)* **The trace as a co-equal artifact** — one format feeding optimizers, replay, RL export, and debugging
7. *(added)* **The eval engine as shared infrastructure** — every optimizer is a thin strategy over it

Core architectural decisions this report argues for:

- **IR shape:** closed typed graph core (small combinator vocabulary incl. a first-class `agent_loop` node) + typed-hole escape nodes; every mutable thing is a named, addressable parameter; dual frontends (Rust builder + serialized IR) constructing the same runtime object.
- **Compile-time-first internals:** rustc/Cranelift-style data structures — arenas, newtyped dense indices, interned symbols, closed enums. String-keyed maps exist *only* at the serde boundary. Never "force a map."
- **Optimizer contract:** candidate = dense data overlay over a fixed skeleton (never object mutation), task-local trace capture, metrics with textual feedback.
- **Tools:** JS-first two-tier — in-process QuickJS (~300µs) for ephemeral LLM-written tools, Wasmtime components (~5µs instantiation) for graduated ones.
- **Compilation:** lowering — interpreter for the inner loop, IR→Rust codegen (`include_program!`) for shipping.

---

## 2. The Vision

- An **intermediate representation** LLMs can generate to express themselves as a harness, or express a harness as a harness. Modules are the harness abstraction.
- **Tool execution**: LLMs come up with and build their own tools, and they must be fast.
- **Harness optimization**: whatever module/harness a user brings, optimizers (GEPA and beyond) work around it.
- **De-bloat**: current abstractions are over-expressive; minimize while keeping DSPy-style declarative expressibility.
- **Compilation**: taking a program and producing a raw form that can be executed or served as-is — plausibly the IR itself.

The ambition: an abstraction of a fundamental building block — different in kind from anything that exists.

---

## 3. Where DSRs Stands Today (Codebase Audit)

~26k LOC across `crates/dspy-rs` and `crates/dsrs-macros`. Full audit details live in the v1-revamp working session; the load-bearing findings:

### 3.1 Bloat is real and quantified

| Redundancy | Today | Target |
|---|---|---|
| Data carriers (`Example<S>`, `RawExample`, `Prediction`, `Predicted<O>`, `mipro::Trace`) | 5 | 2 |
| Trace concepts (`trace::Graph`, `mipro::Trace`, `evaluate::ExecutionTrace`) | 3 | 1 |
| Ways to do chain-of-thought | 4 | 1 |
| Call paths per predictor | 8 | ~2 |
| Instruction/demo mutation paths | 5 | 1 |
| Tool-execution paths (LM auto-loop, ReAct text protocol, `CallerManaged`) | 3 | 1 |
| Authoring styles (struct `Module`, `ModuleExt`, `fx`, macros) | 4 | 1 canonical |
| Concepts for a 2-step pipeline | ~13–15 (~22 with opt + CoT) | ~5–6 (Python DSPy parity) |

Dead weight to delete: the `Adapter` trait (empty marker; `Settings` discards it and `Predict` hardcodes `ChatAdapter`), `core/specials.rs` placeholders (`dspy_rs::ToolCall` is an empty struct shadowing the real `rig` type), `BamlValue` alias, `DummyLM` (superseded by `TestCompletionModel`), the `field!` macro + `schemars` dependency, generated `XAll` structs, empty `evaluate/metrics.rs`, the non-functional `trace::Executor`.

Worst UX bug: users must derive `facet::Facet` on their own module structs or the optimizer **silently finds no predictors**.

### 3.2 Nothing program-level serializes

`ModuleState` (dotted-path → instruction override + demos) is the only serialization that exists — a *parameter* blob presupposing an already-constructed, structurally identical module. Not serializable today: module topology (lives in hand-written `forward` bodies), signatures (`&'static` macro statics; no value-level constructor exists), LM config (no serde; live client embedded), adapter choice, tools (`Arc<dyn ToolDyn>`, no registry), augmentations (type-level GATs), closures in `Map`/`AndThen`, loop limits.

### 3.3 Structural traps for an IR

1. **Signature-as-type** — no runtime construction path anywhere.
2. **Three identities per predictor** — facet dotted path (optimizer/state), fx name (`Params`), raw pointer address (trace `instance_key`). Incompatible with each other and with process boundaries.
3. **Four `Box::leak`'d `TypeId` caches** — fine for a closed static type set, fatal once IR loading constructs signatures dynamically (unbounded leak per load).
4. **`&mut module` optimization** — all three optimizers serially set→eval→restore with duplicated save/restore blocks; no concurrent candidate evaluation possible.
5. **`Module` is not dyn-safe** (unboxed `async fn` in trait) — no heterogeneous runtime nodes.

### 3.4 The nucleus already exists

The `fx` module (commit `071d20a`) — task-local `Params` injection + name-addressed predictors + `#[predict]`/`#[cot]` macros making the function name the canonical slot ID — is the closest thing in the repo to the serializable-harness model, and it's architecturally *correct*: overlay-injection rather than mutation. Known gap: `FnModule` doesn't derive `Facet`, so fx harnesses currently can't be optimized by any existing optimizer. The fix is not bolting `Facet` on — it's making the overlay the contract (§5.4).

---

## 4. Research Findings

### 4.1 IRs for LLM programs (prior art)

**DSPy's failure mode is precise:** its compile artifact is a parameter blob (instructions + demos as JSON; whole-program save = cloudpickle). All control flow stays as opaque Python — not inspectable, not diffable, not portable, not LLM-writable, not structurally optimizable. GEPA can rewrite instruction text but cannot touch topology.

**The spectrum and its failure modes:**

| Representation | Exemplars | Fails because |
|---|---|---|
| Open code | DSPy `forward()`, ADAS, Temporal, OpenAI Agents SDK | Pickle-class serialization; statically unoptimizable; needs sandboxing |
| Closed graph | AWS Step Functions ASL, ONNX | Real control flow doesn't fit → escape hatches proliferate → IR rots into lossy export (teams maintain two versions) |
| Config/data | TensorZero, ADK YAML, CrewAI | Parameters only; blind to structure |

**The pattern that works** (AFlow, LACUNA, Microsoft Agent Framework, WASM component model):

- Closed typed graph core with a **small combinator vocabulary** — `predict`, `tool`, `seq`, `fork/join`, `route`, `retry/refine`, bounded `loop`, and a first-class **`agent_loop`** node (model + tools + stop conditions + budgets), since the LLM+tool loop is the unit everyone actually serves. AFlow's result: constraining generation to a small operator library is what made search tractable.
- **Typed-hole escape nodes** (LACUNA): embedded sandboxed code with a declared signature; the optimizer treats holes as opaque-but-typed, like SQL treats UDFs. WIT maps ~1:1 onto Signature.
- **Every mutable thing is a named, addressable parameter** — the serialized program *is* the optimization surface.
- **Dual frontends, one runtime object** (Microsoft Agent Framework): builder API and IR construct the same in-memory value; `to_ir()` is total, never a lossy export. If a construct can't round-trip, it doesn't belong in the convenience API except behind an explicit hole. This rule is what prevents the cloudpickle trap.
- **Syntax designed for LLM generation**: BAML's compact type syntax cuts 50–80% of schema tokens; minimal grammars + constrained decoding give validity by construction. Capability declarations belong in the syntax (a harness that never declares `network` cannot be lowered into one that has it).
- **Separate the program artifact from the execution-state artifact** (Temporal lesson): durability = determinism boundary + event log; the trace format doubles as the optimizer's input.

Key precedents: [DSPy saving](https://dspy.ai/tutorials/saving/) · [GEPA](https://arxiv.org/abs/2507.19457) · [AFlow](https://arxiv.org/abs/2410.10762) · [ADAS](https://arxiv.org/abs/2408.08435) · [Trace/OPTO](https://arxiv.org/pdf/2406.16218) · [LACUNA](https://arxiv.org/pdf/2605.28617) · [BAML](https://github.com/BoundaryML/baml) · [TensorZero](https://github.com/tensorzero/tensorzero) · [Letta .af](https://github.com/letta-ai/agent-file) · [Agent Framework Declarative Workflows](https://devblogs.microsoft.com/agent-framework/move-agent-orchestration-workflows-out-of-code-with-agent-framework-declarative-workflows-1-0/) · [WIT](https://github.com/WebAssembly/component-model/blob/main/design/mvp/WIT.md) · [MLIR](https://rcs.uwaterloo.ca/~ali/cs842-s23/papers/mlir.pdf)

### 4.2 Fast tool execution

**Code-as-action is decisively better than JSON tool calls:** CodeAct +20.7pp success across 17 LLMs, ~30% fewer turns ([ICML 2024](https://arxiv.org/abs/2402.01030)); Anthropic programmatic tool calling −38% billed tokens ([docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)); Cloudflare Code Mode collapsed 1.17M tokens of tool schemas to ~1K ([blog](https://blog.cloudflare.com/code-mode/)). LLMs are simply better at writing TypeScript than emitting tool-call tokens.

**Language verdict: JavaScript.** Rust→WASM mid-run is a dead end (seconds of compile, toolchain required). Python cannot be sandboxed in-process (PyO3 explicitly not for sandboxing; RustPython lacks limits). Rhai/Starlark lose on LLM fluency. JS uniquely wins fluency + latency + sandboxability, at two isolation levels.

**Recommended two-tier architecture:**

- **Tier 1 — ephemeral tools:** in-process QuickJS-ng via `rquickjs`: **<300µs** full runtime lifecycle, per-call memory limit + interrupt-handler deadline, no ambient authority — host capabilities injected as explicit functions, *including existing DSRs tools as an async JS API* (Code Mode falls out for free). Registration follows the LATM/Voyager lifecycle: generate tool + generate tests → parse-check → schema-validate → self-test in sandbox → only then register (wrapping `rig::tool::ToolDyn`, which DSRs already uses with `schemars`). QuickJS bytecode cached by content hash.
- **Tier 2 — graduated tools:** Wasmtime components: **~5µs** instantiation (pooling allocator + CoW), epoch interruption for timeouts, `StoreLimits`, WIT-typed interfaces projected to JSON Schema (the Wassette pattern), `.cwasm` artifacts content-hash cached and OCI-distributable. `Send+Sync` — safe for concurrent optimizer eval paths.
- **Escape hatch:** an `Executor` trait for subprocess (+seccomp/landlock via `extrasafe`/`birdcage`), Hyperlight (~1–2ms microVM), or remote executors.

Key sources: [Wasmtime fast instantiation](https://docs.wasmtime.dev/examples-fast-instantiation.html) · [rquickjs](https://github.com/DelSkayn/rquickjs) · [Wassette](https://github.com/microsoft/wassette) · [Extism](https://github.com/extism/extism) · [script-bench-rs](https://github.com/khvzak/script-bench-rs) · [smolagents secure execution](https://huggingface.co/docs/smolagents/tutorials/secure_code_execution) · [Hyperlight](https://opensource.microsoft.com/blog/2025/03/26/hyperlight-wasm-fast-secure-and-os-free/)

### 4.3 Optimization state of the art

- **GEPA** (ICLR 2026 oral): reflective mutation over per-component traces + per-instance Pareto frontier + minibatch acceptance + system-aware merge. Beats MIPROv2 by >10% avg; beats GRPO by up to +20% with **35x fewer rollouts** (~78x to match GRPO's best). Requires: named text components, per-component sub-traces, metric with optional textual feedback.
- **optimize_anything** (GEPA team's universal API): optimize *any* text artifact — code, agent architectures, configs. Central empirical lesson: **actionable side-information (error messages, profiler output, judge critiques) converges dramatically faster than scores alone.** Discovered agent architectures taking Gemini Flash 32.5%→89.5% on ARC-AGI.
- **SIMBA**: cheap minibatch introspective ascent (append-demo + append-rule); good agentic default. **MIPROv2**: Bayesian search over instruction × demo sets; dominated by GEPA — build later, if at all.
- **Structure optimization**: ADAS (meta-agent writes whole agent code), AFlow (MCTS over operator graphs), MASS (key finding: **interleave** — per-agent prompt opt → pruned topology search → global prompt opt; topology search over unoptimized prompts is misleading), MaAS (per-query architecture distribution with cost as first-class objective), RoboPhD (Elo tournament selection beats Pareto on complex code-heavy tasks under tight budgets). Free-form code evolution (a whole-module `Code` gene, expensively evaluated) beats constrained graph search on hard tasks — structure search needs no extra IR machinery.
- **Tool descriptions are optimizable parameters that matter**: a single doc-field change shifts success avg 6.34pp (DocsChisel); PLAY2PROMPT optimizes docs from execution "play." Context/memory policies are converging with prompt optimization (ACE playbooks, Dynamic Cheatsheet, Training-Free GRPO): *optimize the context artifact, not the model.*
- **RL**: layer, don't lead. Prompt/context optimization needs 10²–10³ rollouts; RL needs 10⁴⁺ and verifiable rewards. Every RL framework (Agent Lightning, verifiers, ART/RULER, Arbor) converged on the same contract as GEPA — structured traces of LM sub-calls + reward — differing only in update target. DSRs should emit rollouts in a standard format (message lists + reward + spans) and let external trainers do weights.

**The distilled optimizer contract** — a generic optimizer needs exactly five things: (1) enumerable named parameters (typed genes: instruction, demos, tool description, model choice, code, config); (2) per-rollout execution traces attributable per component, surviving loops; (3) a metric returning score + optional textual feedback; (4) cheap cloning — candidates as data, never object graphs; (5) budgeted, cached, parallel evaluation.

Key sources: [GEPA](https://arxiv.org/abs/2507.19457) · [optimize_anything](https://arxiv.org/abs/2605.19633) · [dspy GEPA in depth](https://dspy.ai/diving-deeper/gepa-in-depth/) · [SIMBA](https://dspy.ai/api/optimizers/SIMBA/) · [MASS](https://arxiv.org/pdf/2502.02533) · [MaAS](https://arxiv.org/abs/2502.04180) · [RoboPhD](https://arxiv.org/pdf/2604.04347) · [ShinkaEvolve](https://arxiv.org/pdf/2509.19349) · [DocsChisel](https://arxiv.org/abs/2608.10037) · [ACE](https://arxiv.org/abs/2510.04618) · [Agent Lightning](https://arxiv.org/pdf/2508.03680) · [verifiers](https://github.com/willccbb/verifiers) · [ART/RULER](https://github.com/openpipe/art) · [gepars (Rust GEPA)](https://github.com/Epistates/gepars)

---

## 5. Design Direction

### 5.1 The IR

Closed typed graph core (`predict`, `tool`, `seq`, `fork/join`, `route`, `retry/refine`, bounded `loop`, first-class `agent_loop`) + typed-hole escape nodes carrying sandboxed code with declared signatures. Every mutable thing — instruction, demos, tool description, model ref, context policy, topology gene — is a named, addressable parameter. Dual frontends (Rust builder API, serialized IR text) construct the same runtime value; `to_ir()` is total. Surface syntax designed for LLM generation: compact, small closed keyword set, grammar-constrainable, capability declarations in the syntax. Program artifact and execution-state artifact are separate, both specified; no code-executing deserialization; secrets nulled on export.

### 5.2 Compile-time-first internals — the data-structure principle

**Design principle: deeply designed, compile-time-first Rust. Never "force a map."** String-keyed maps exist only at the serde boundary.

**Two lanes, one contract:**

- **Static lane (default, zero-cost):** Rust-authored harnesses stay fully monomorphized. Kill facet runtime reflection — the derive macros emit parameter-enumeration code directly (they know every field at expansion time). This removes the user-facing `facet` dependency, the raw-pointer accessor hack, and the silent-discovery bug, and is *more* compile-time-first than reflection.
- **Dynamic lane (loaded IR):** rustc/Cranelift internals —

```rust
struct Program {
    nodes:  PrimaryMap<NodeId, Node>,        // closed enum: Predict, AgentLoop, Seq, Route, Hole…
    sigs:   PrimaryMap<SigId, SignatureDef>, // owned, interned — replaces Box::leak'd &'static statics
    params: PrimaryMap<ParamId, ParamSlot>,  // kind-typed: Instruction | Demos | ToolDesc | ModelRef | …
    syms:   Interner,
}
struct Overlay(SecondaryMap<ParamId, ParamValue>); // a candidate; Clone ≈ memcpy
```

Newtyped dense indices, closed enums, arenas, exhaustive matches. `ParamPath` (human/serde form) resolves to `ParamId(u32)` once at load; traces, overlays, optimizers, and serving all speak ids. This collapses the three-identities problem by construction. Typed param kinds (`Slot<Instruction>`, `Slot<Demos>`) make optimizer-side mutation mistakes compile errors. Replace the four leaked `TypeId` caches with a runtime-owned interner.

**The third leg — IR→Rust codegen:** `dsrs::include_program!("qa.dsrs")` validates the artifact at build time (signatures type-checked, capabilities verified) and emits monomorphized code — the sqlx/prost precedent. Lifecycle: LLM generates IR → optimizer evolves it through the interpreter (no rustc in the loop) → winning artifact checked in → build produces the zero-cost version.

### 5.3 What "compilation" means

**Lowering.** Optimizer passes over the IR produce a lowered, servable artifact; the interpreter serves it directly (mid-run generation, optimizer inner loops), codegen compiles it into the static lane (shipping). The IR is the compile target in both directions: Rust → IR (macro-emitted introspection) and IR → Rust (codegen).

### 5.4 The optimizer contract in DSRs

- **Candidate = `Overlay`** applied at render time — never mutate the module tree. Cloning free, serialization trivial, parallel evaluation of many candidates over one skeleton.
- **Task-local trace capture** (`task_local!`, not thread-local): every LM sub-call records `(ParamId, rendered prompt, raw output, parsed output, model, usage, error?)`; loops produce multiple entries; `Trace::for_component(id)` sub-slicing is exactly GEPA's `pred_trace`.
- **Metric:** `fn(example, prediction, Option<&Trace>) -> Eval { score: f64, feedback: Option<String> }` + per-component variant. Textual feedback ships day one — highest-ROI feature in the literature.
- **Shared eval engine** (the actual core): bounded-concurrency async fan-out, rollout caching keyed on (candidate hash, example id, sampling params), budget metering (metric calls + tokens + $), minibatch gating, per-instance Pareto bookkeeping (candidates × examples score matrix), checkpoint/resume. Every optimizer is a thin strategy over this. Rust's concurrency story is the moat Python DSPy cannot match.
- **Build order:** BootstrapFewShot (minimal end-to-end contract exercise, ~200 lines on the engine) → **GEPA** (flagship) → SIMBA (cheap agentic default). Skip MIPROv2 initially. Structure search = an opt-in whole-module `Code` gene, expensively evaluated. RL = export bridge, not trainer.

### 5.5 Tool runtime

As §4.2: JS-first, Tier 1 rquickjs in-process, Tier 2 Wasmtime graduation, LATM validate-then-register lifecycle, content-hash caching, capability-scoped host APIs, existing DSRs tools injected as a JS API (Code Mode for free). One tool execution path replaces today's three.

### 5.6 De-bloat hit list (phase 0)

Collapse per the table in §3.1; delete the dead weight listed there; converge on **fx + `#[predict]`/`#[cot]`** as the canonical authoring path (name-addressed slots + injected params *is* the IR/overlay model); make `ParamId`/`ParamPath` the single identity; split `LM` into serializable config + live client.

---

## 6. Identified Gaps (beyond the original five pillars)

Ranked by retrofit cost:

1. **Streaming** — the most dangerous omission. Typed signatures need macro-derived partial output types (BAML's semantic-streaming precedent). Whether a `predict` node emits a value or a stream changes the execution model, trace format, and serving — must be decided before the IR freezes.
2. **Execution-state artifact** — checkpoints, resumability, interrupts. Per-step snapshots + deterministic replay (LM/tool calls are the nondeterministic activities). Human-in-the-loop as a first-class node (`await_approval`). Falls out of the trace format only if designed now.
3. **The serving host** — `dsrs serve program.dsrs`: loads an artifact, exposes typed endpoints derived from signature schemas, streams, meters cost, exports OTel. The forcing function that keeps the IR honest.
4. **Python (and TS) frontends over the IR** — the polars/pydantic-core play: Rust core owns execution/optimization/tools; thin SDKs author and load the same IR. The difference between "great Rust library" and "the substrate everyone builds on." One more reason to keep the IR language-neutral.
5. **Traces as the data flywheel** — (a) record/replay tests: capture a run once, replay as a deterministic zero-API-call fixture; (b) production traces → trainsets harvested from serving.
6. **Context policy as a named parameter** — history truncation, compaction, tool-result summarization, playbook injection as typed, *optimizable* param slots. No framework has made context policy first-class-optimizable; open lane.

Smaller: artifact registry/sharing (OCI, `.af`-style import), IR semver from day one, and a **dogfood harness** — one real, hard, end-to-end harness (SWE-agent or legal-domain) built and optimized with DSRs as standing proof and design pressure (point `benchmarks/` at it, not just microbenches).

---

## 7. Making Harness Optimization a Reality

The optimizers are the easy 20%. What makes it real:

1. **Solve the metric bottleneck** — the actual blocker. Users arrive with ~20 examples and vibes, not trainsets and metrics. Make eval authoring a guided workflow: judge distillation from a handful of 👍/👎-rated traces (RULER-style relative judging needs zero reward engineering), metric-free single-instance mode (optimize against the live task with side-information as signal), synthetic trainset expansion filtered by the judge. The library that makes "20 examples and an opinion" sufficient wins the category.
2. **Counterfactual replay** — the sample-efficiency cheat code nobody has shipped. Checkpointed traces + overlay candidates = replay a recorded run from step *k* with one mutated parameter; the prefix replays free, only the divergent suffix costs money. Near-causal attribution at a fraction of full-rollout cost, multiplying GEPA's already-35x efficiency. Only possible if trace format and deterministic replay are designed for it (§6.2).
3. **Environments** — agentic harnesses act on worlds (repos, browsers, APIs). Without a resettable, seedable `Environment` abstraction (dataset + world + rollout protocol), harness optimization silently degrades to prompt optimization on stateless pipelines. The tool sandbox doubles as the environment sandbox — which enables the unique DSRs move: **evolving tool *implementations*** with real execution feedback, safely, because tools are already sandboxed. Descriptions move metrics ~6pp; implementations are an open lane no one can touch safely in-process.
4. **Trust machinery** — adoptable, not just good: diffable IR artifacts, holdout regression gates before promotion, artifact lineage stamped into the IR (optimizer, data, budget), canary candidate-vs-incumbent serving with auto-rollback. Turns "the optimizer says it's better" into "provably not worse."
5. **Optimization as a background process** — not a 3-hour blocking script: a daemon harvesting production traces, running GEPA within a nightly budget, checkpointing across runs, surfacing a Pareto frontier of diffed candidates, promoting on approval. Rust makes hundreds of concurrent rollouts on one machine routine.

Differentiator bets: **#1 and #2** — everyone will eventually have a GEPA; almost nobody is making evals authorable, and nobody has counterfactual replay because nobody designed traces and candidates to compose.

---

## 8. Roadmap

| Phase | Work | Why this order |
|---|---|---|
| 0 | **De-bloat** (§5.6 hit list) | Shrink the surface before freezing any of it into an IR |
| 1 | **Value-level signatures, single addressing scheme (`ParamId`), serializable LM config, interner** | Signatures are `&'static` statics today; nothing else can happen until a signature is a value |
| 2 | **Trace format + eval engine + feedback metric**, then Bootstrap → GEPA → SIMBA on the overlay contract | Optimizers are thin once the engine exists; trace format must precede replay/flywheel |
| 3 | **IR graph core + dual frontends + `include_program!` codegen + serving host** | Grown out of `fx` + `ModuleState`, which already round-trip |
| 4 | **Tool runtime** (rquickjs tier → Wasmtime graduation), Code Mode, environments | Rides on the capability model and registry from phase 3 |
| 5 | **Flywheel**: record/replay tests, counterfactual replay, background optimization daemon, Python SDK | Each composes artifacts from phases 2–4 |

Cross-cutting, decided early even if built late: streaming semantics, execution-state artifact, context-policy params, IR semver.

---

*Compiled from a four-agent research pass (codebase audit; IR prior art; tool-execution runtimes; optimization SOTA) plus design discussion, 2026-08-13/14. Full source links inline per section.*
