# S4 Spike: Refine Scoped Context for Hint Injection

## Context

S4 asks for a concrete scoped-context mechanism for Refine hint injection and explicitly calls out three options: `tokio::task_local!`, `thread_local!` with `RefCell`, or an explicit context parameter (`docs/specs/modules/shapes.md:243`).

Refine is currently defined as BestOfN with feedback injection that adds a `hint` field to predictor prompts on retries, but this mechanism is still unresolved (`docs/specs/modules/design_reference.md:942`). The shaping doc also marks this as one of the F11 prototyping gaps (`docs/specs/modules/shapes.md:68`, `docs/specs/modules/shapes.md:72`).

## Goal

Select a first-pass mechanism for scoped hint propagation in Refine that is safe under async/concurrency, works with nested module composition, and minimizes user-facing API churn.

## Questions

| ID | Question |
|---|---|
| **S4-Q1** | What scoped-context pattern already exists in runtime code that we can reuse? |
| **S4-Q2** | How do current settings/instruction paths constrain hint injection design? |
| **S4-Q3** | What concurrency behavior in module evaluation affects context isolation? |
| **S4-Q4** | What are the concrete tradeoffs between task-local, thread-local, and explicit-parameter approaches? |
| **S4-Q5** | What first-pass implementation path gives low risk now while preserving an escape hatch for harder async cases? |

## Findings

1. **There is already a production task-local scoping pattern in tracing.**
   - `tokio::task_local!` is used to hold scoped trace state (`crates/dspy-rs/src/trace/context.rs:4`, `crates/dspy-rs/src/trace/context.rs:7`).
   - Scope lifecycle uses `CURRENT_TRACE.scope(...).await` (`crates/dspy-rs/src/trace/context.rs:19`).
   - Reads/writes use `try_with(...)`, making scope optional from call sites (`crates/dspy-rs/src/trace/context.rs:35`, `crates/dspy-rs/src/trace/context.rs:47`, `crates/dspy-rs/src/trace/context.rs:63`).
   - The file explicitly documents orphaned-task/leak caveats when scope owners outlive expected lifetime (`crates/dspy-rs/src/trace/context.rs:21`).

2. **Current settings are process-global, not scoped per evaluation/attempt.**
   - Global settings are a `LazyLock<RwLock<Option<Settings>>>` singleton (`crates/dspy-rs/src/core/settings.rs:20`).
   - `configure()` replaces the singleton for the whole process (`crates/dspy-rs/src/core/settings.rs:27`).
   - Typed `Predict` reads LM from global settings each call, and `LegacyPredict` reads both adapter + LM from the same singleton (`crates/dspy-rs/src/predictors/predict.rs:67`, `crates/dspy-rs/src/predictors/predict.rs:582`).
   - Implication: hint context should not be placed in process-global settings.

3. **Instruction injection exists today, but as mutable predictor state, not per-call scoped state.**
   - Prompt system text accepts a per-call optional override (`crates/dspy-rs/src/adapter/chat.rs:438`).
   - Typed predict path currently passes `self.instruction_override` into that override channel (`crates/dspy-rs/src/predictors/predict.rs:74`).
   - Optimizers mutate predictor instruction in place before evaluation (`crates/dspy-rs/src/optimizer/mipro.rs:421`, `crates/dspy-rs/src/optimizer/copro.rs:223`, `crates/dspy-rs/src/optimizer/gepa.rs:444`).
   - Implication: reusing mutable instruction state for retry hints risks cross-attempt/cross-example contamination.

4. **Evaluation paths are concurrent, so context isolation is mandatory.**
   - `Module::batch()` runs item futures with `buffer_unordered` (`crates/dspy-rs/src/core/module.rs:53`).
   - `Evaluator::evaluate()` also computes metrics with `buffer_unordered` (`crates/dspy-rs/src/evaluate/evaluator.rs:51`).
   - GEPA candidate evaluation runs example futures via `join_all` (`crates/dspy-rs/src/optimizer/gepa.rs:263`, `crates/dspy-rs/src/optimizer/gepa.rs:274`).
   - Implication: shared mutable hint channels (especially thread/process global) are high-risk; attempt-scoped isolation must hold while these futures are in flight.

5. **Explicit-context style exists in evaluation helpers, but only as local function API, not module-wide contract.**
   - Retrieval helper takes optional `context_docs` explicitly (`crates/dspy-rs/src/evaluate/feedback_helpers.rs:33`).
   - This is viable ergonomically for narrow helpers, but not currently applied to the core `Module` trait.

6. **Changing to explicit context on `Module::forward` would be a broad, invasive API shift.**
   - Core `Module` trait currently accepts only `inputs` (`crates/dspy-rs/src/core/module.rs:11`).
   - The call shape is used broadly (e.g., tracing example wraps `module.forward(example)` directly, `crates/dspy-rs/examples/12-tracing.rs:84`).

7. **Design-level evidence: Facet is positioned as type metadata source, not request-scoped mutable runtime state.**
   - Shaping frames Facet as the single source of truth for type metadata (`docs/specs/modules/shapes.md:24`).
   - Design reference reiterates Facet-derived schema as metadata/plumbing, with runtime execution behavior owned by module flow (`docs/specs/modules/design_reference.md:31`, `docs/specs/modules/design_reference.md:39`).
   - Implication: Refine hint scoping belongs in runtime execution context, not Facet metadata.

## Mechanism Options (task-local/thread-local/explicit parameter) + Tradeoffs

| Option | How it would work | Pros | Cons / Risks |
|---|---|---|---|
| **Task-local (`tokio::task_local!`)** | Enter scope once per Refine attempt and read hint during prompt construction | Reuses existing runtime pattern; no `Module` trait break; per-attempt isolation for async work that stays in the same task | Does not auto-inherit across `tokio::spawn` / thread hops; child tasks must receive payload explicitly and re-enter scope; detached/orphaned child tasks need strict lifetime discipline (`crates/dspy-rs/src/trace/context.rs:21`) |
| **Thread-local (`thread_local!` + `RefCell`)** | Store current hint per worker thread and clear/reset around attempts | Simple primitive in synchronous code | Async evaluation uses concurrent futures (`buffer_unordered`, `join_all`); thread affinity is not request affinity, so this is brittle for attempt isolation and easy to leak between logical flows |
| **Explicit parameter** | Pass `RefineAttemptContext` through call APIs (full `Module::forward` change or parallel path) | Fully explicit dataflow; naturally safe across spawn/thread boundaries | Broad API migration cost across trait, impls, wrappers, optimizers, and examples (`crates/dspy-rs/src/core/module.rs:11`, `crates/dspy-rs/examples/12-tracing.rs:84`) |

## Decision

**Deferred.** The scoping mechanism will be determined when Refine is actually built. The spike findings and tradeoff analysis are preserved below for when that happens.

The three viable options remain `tokio::task_local!` (pragmatic, spawn footgun), explicit `Module::forward` parameter (correct, invasive), and `thread_local!` (rejected — brittle under async concurrency).

## Original Recommendation (not adopted)

The spike originally recommended `tokio::task_local!`:

- **Primary:** `tokio::task_local!` attempt scope for Refine retry hints.
- **Boundary rule:** whenever execution crosses task/thread boundaries (`tokio::spawn`, threadpool work), propagate hint payload explicitly and re-enter scope in the child task.
- **Reject for now:** `thread_local!` for hint state.
- **Defer:** full explicit context parameter on `Module::forward` unless spawn-heavy usage proves task-local + boundary propagation insufficient.

Reasoning:
- It matches an existing, working runtime idiom (`trace/context.rs`) and avoids a system-wide API break.
- It gives per-attempt isolation without reusing mutable predictor instruction state.
- It keeps the typed primary path ergonomic while making cross-task behavior explicit and testable.

## Concrete Implementation Steps

1. Add a new Refine context module with `tokio::task_local!` state for retry hint payload (for example, `hint`, `attempt_idx`).
2. Provide helpers: `with_refine_hint_scope(...)`, `current_refine_hint()`, and a boundary helper that re-scopes explicitly in spawned child tasks.
3. In Refine retry loop, execute each attempt future within `with_refine_hint_scope(...)` at the attempt boundary.
4. In typed `Predict` call path, compute an effective instruction by composing:
   - explicit instruction override (if any), and
   - scoped Refine hint (if present).
5. Pass the effective instruction through existing adapter override channel (`format_system_message_typed_with_instruction`) without mutating persistent predictor state.
6. Keep hint out of optimizer state serialization (`dump_state`/`load_state`) so retry hints do not persist.
7. Add tests:
   - nested module call sees hint only inside attempt scope,
   - concurrent Refine runs (`buffer_unordered`/`join_all` paths) do not leak hints across attempts,
   - hint is absent outside scope,
   - `tokio::spawn` child task does not inherit hint by default,
   - explicit propagation helper correctly restores hint inside spawned child task.
8. Document scope boundary semantics and spawned-task caveat in Refine docs.

## Open Risks

- If a future module path introduces background tasks without using the boundary helper, hint visibility can silently diverge from parent attempt scope.
- If Refine evolves toward deeper cross-task orchestration, the deferred explicit-parameter design may become necessary to preserve clarity.

## Acceptance

S4 is complete when:

- We can describe and justify one concrete mechanism for Refine hint scoping with source-backed tradeoffs.
- Hint injection is per-attempt scoped and does not leak across concurrent evaluations.
- Base module APIs remain unchanged for this first pass.
- Spawned-task behavior is explicitly documented, including the required explicit propagation strategy.
- All S4 questions are answered with references to current code paths.
