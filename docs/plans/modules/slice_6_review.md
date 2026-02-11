## Current Scope Addendum (2026-02-11)

V6/dynamic graph was implemented in-repo, then intentionally deferred; the runtime code has been removed from active scope.

Canonical scope is now V1–V5 typed-only; untyped eval (`U37`) and all V6 dynamic graph/runtime surfaces are deferred.

All content below is preserved as a historical implementation record.

### Findings

#### Finding 1
- Severity: high
- Category: Spec fidelity / Shape compliance
- Location: `crates/dspy-rs/src/core/dyn_factories.rs:241`, `crates/dspy-rs/src/core/dyn_factories.rs:267`, `crates/dspy-rs/src/core/dyn_factories.rs:275`, `crates/dspy-rs/src/core/dyn_factories.rs:359`
- Issue: The dynamic `react` strategy is implemented as a single generic predictor pass-through, not ReAct orchestration. `ReActDynModule` exposes only one predictor and `forward()` delegates directly to `SchemaPredictor::forward_untyped`, while `max_steps` and `tools` are unused. This does not match the ground-truth F9 expectation that ReAct factory logic builds ReAct-specific schemas/behavior (action + extract, tool-driven loop). Spec refs: `docs/specs/modules/shapes.md:65`, `docs/specs/modules/design_reference.md:813`.
- Suggestion: Implement dynamic ReAct as a true multi-step `DynModule`: construct action/extract schemas from base schema + tool definitions, run iterative action/tool/extract flow, and expose both internal predictors via `predictors()`/`predictors_mut()`.

#### Finding 2
- Severity: high
- Category: Breadboard consistency / Spec fidelity
- Location: `crates/dspy-rs/src/core/program_graph.rs:347`, `crates/dspy-rs/src/core/program_graph.rs:350`, `crates/dspy-rs/src/core/program_graph.rs:512`
- Issue: The graph cannot realize the breadboard’s external-input wiring model. `connect()` requires both endpoints to be existing nodes (so `"input"` is rejected), and `execute()` only passes the root input to nodes with zero incoming edges; nodes with incoming edges get an input map built only from edges. That makes the documented `graph.connect("input", "question", "cot", "question")` flow impossible and prevents mixing root input fields with piped fields on downstream nodes. Spec refs: `docs/specs/modules/breadboard.md:263`, `docs/specs/modules/breadboard.md:490`.
- Suggestion: Add explicit graph input/output pseudo-node semantics, or merge root input into each node input before applying edge overwrites. Align API behavior with the breadboard examples.

#### Finding 3
- Severity: medium
- Category: Spec fidelity / Cleanliness
- Location: `crates/dspy-rs/src/core/program_graph.rs:18`, `crates/dspy-rs/src/core/program_graph.rs:168`, `crates/dspy-rs/src/core/program_graph.rs:206`, `crates/dspy-rs/src/core/dyn_module.rs:78`
- Issue: Registry-created modules do not flow directly into graph mutation APIs as specified. `registry::create()` returns `Box<dyn DynModule>`, but `add_node`/`replace_node` require a `Node` wrapper with duplicated `schema` and `module`. This diverges from the spec examples and introduces drift risk between `node.schema` and `node.module.schema()`. Spec refs: `docs/specs/modules/design_reference.md:1062`, `docs/specs/modules/design_reference.md:1063`, `docs/specs/modules/breadboard.md:488`.
- Suggestion: Accept `Box<dyn DynModule>` in `add_node`/`replace_node`, derive node schema from `module.schema()`, and make `Node` construction internal to keep schema/module consistent.

#### Finding 4
- Severity: medium
- Category: Breadboard consistency
- Location: `crates/dspy-rs/src/core/program_graph.rs:456`, `crates/dspy-rs/src/core/program_graph.rs:463`, `crates/dspy-rs/tests/test_program_graph_annotations.rs:67`
- Issue: `ProgramGraph::from_module()` only adds edges from pre-registered annotations; plain projections produce zero edges. Ground truth says projection auto-populates nodes/edges (S5/S6) and design allows inferred edges (trace or annotation). Current behavior means multi-predictor modules project into disconnected graphs unless callers manually register annotations first. Spec refs: `docs/specs/modules/breadboard.md:267`, `docs/specs/modules/design_reference.md:885`.
- Suggestion: Implement at least one automatic edge inference path (trace-derived or schema/path-based heuristic), and return a clear projection error when a multi-node projection has no resolvable edges.

#### Finding 5
- Severity: medium
- Category: Maintainability / Correctness
- Location: `crates/dspy-rs/src/core/program_graph.rs:270`, `crates/dspy-rs/src/core/program_graph.rs:272`, `crates/dspy-rs/src/core/program_graph.rs:278`, `crates/dspy-rs/src/core/program_graph.rs:289`
- Issue: `insert_between()` is not failure-atomic. It inserts the node and removes the original edge before validating inserted-node input/output fields. If validation fails (`?` on missing input/output field), the function returns with partially mutated graph state.
- Suggestion: Pre-validate inserted-node schema before mutating graph, or use transactional rollback on every failure path (including missing input/output field errors).

#### Finding 6
- Severity: low
- Category: Simplicity / Maintainability
- Location: `crates/dspy-rs/src/core/dyn_module.rs:15`, `crates/dspy-rs/src/core/dyn_module.rs:20`, `crates/dspy-rs/src/core/dyn_factories.rs:364`
- Issue: Config validation/error plumbing exists but is mostly unused. `StrategyError::InvalidConfig` and `BuildFailed` are defined but factories do not use them; `ReActFactory` silently defaults when config is malformed. This weakens debuggability and makes runtime behavior less predictable.
- Suggestion: Validate incoming config against `config_schema()` and return `InvalidConfig` on mismatch; reserve defaulting for explicit optional fields.

### Summary
- High: 2
- Medium: 3
- Low: 1

Overall assessment: Slice 6 has solid scaffolding (registry, graph data structures, edge validation hooks, and tests for core happy paths), but it is not yet ground-truth compliant for dynamic-graph semantics in two critical areas: true ReAct dynamic behavior and breadboard-consistent input wiring. The current API shape also adds avoidable friction and drift risk around node/schema handling, and projection/mutation behavior has correctness gaps that should be tightened before considering V6 complete.
