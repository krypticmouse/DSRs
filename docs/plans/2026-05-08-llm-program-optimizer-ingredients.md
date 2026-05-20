# Key ingredients for an LLM program optimizer

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [Executive design thesis](#executive-design-thesis)
- [Taxonomy of optimizer families](#taxonomy-of-optimizer-families)
- [Key ingredients with evidence and citations](#key-ingredients-with-evidence-and-citations)
  - [Search space design](#search-space-design)
  - [Proposal mechanisms](#proposal-mechanisms)
  - [Feedback signals](#feedback-signals)
  - [Trace and blame assignment](#trace-and-blame-assignment)
  - [Pareto and cost tradeoffs](#pareto-and-cost-tradeoffs)
  - [Typed contracts and deterministic boundaries](#typed-contracts-and-deterministic-boundaries)
  - [Offline compilation and bounded online adaptation](#offline-compilation-and-bounded-online-adaptation)
- [Practical architecture requirements for DSRs and Rust](#practical-architecture-requirements-for-dsrs-and-rust)
- [What to build first versus leave as scaffolding](#what-to-build-first-versus-leave-as-scaffolding)
  - [Build first](#build-first)
  - [Build next](#build-next)
  - [Leave as scaffolding](#leave-as-scaffolding)
- [Evaluation and benchmarking protocol](#evaluation-and-benchmarking-protocol)
  - [Core benchmark matrix](#core-benchmark-matrix)
  - [Reported metrics](#reported-metrics)
  - [Protocol details](#protocol-details)
- [Risks and open research gaps](#risks-and-open-research-gaps)
- [Annotated bibliography of the most important papers](#annotated-bibliography-of-the-most-important-papers)
- [References](#references)

Key Ingredients for an LLM Program Optimizer

A strong LLM program optimizer is best treated as a compiler over a typed, editable program representation rather than as a prompt tuner. The core path should be offline and batch-oriented: optimize instructions, demonstrations, tool schemas, routing rules, validators, and a small set of graph edits against train and dev sets with explicit cost budgets, rich traces, and full artifact logging. The second layer should be bounded online adaptation: reversible updates to memory, retrieval context, tool descriptions, and thresholds, gated by validators and rollback. This design is the common denominator behind the most useful recent systems, even when they differ in search algorithm or target artifact (Agrawal et al., 2025; He et al., 2025; Lee et al., 2026; Opsahl-Ong et al., 2024; Wang et al., 2025; Q. Zhang et al., 2025).

The DSPy line provides the anchor abstraction: declarative LM programs compiled against downstream metrics (Khattab et al., 2023, 2024). Recent work then adds the missing optimizer ingredients that matter in practice: richer search spaces, trace-based diagnostics, block or node blame assignment, Pareto-aware selection, typed contracts, and explicit separation between generative planning and deterministic execution (Cheng et al., 2024; Ghoshal et al., 2026; Harikumar, 2026; Ma et al., 2026; J. Zhang et al., 2025). For a Rust DSPy and DSRs-like system, the right architecture is therefore a typed IR, deterministic evaluator, trace store, optimizer kernel, and a narrow online adaptation controller.

## Executive design thesis

The optimizer should optimize a layered program IR with three rings of mutability.

| Ring | Editable artifacts | Default status | Why it matters |
|:---|:---|:---|:---|
| Core | module instructions, demonstrations, tool docs, decoding knobs, validator thresholds | build first | highest evidence and lowest risk (Agrawal et al., 2025; Ghoshal et al., 2026; Opsahl-Ong et al., 2024) |
| Structural | decomposition, routing, graph topology, verifier placement, retrieval plan, harness policies | add after core | fixes failure modes prompt tuning cannot reach (Lee et al., 2026; Wang et al., 2025; J. Zhang et al., 2025; Zhou et al., 2025) |
| Online | memory entries, playbooks, tool-local descriptions, routing thresholds, exemplar caches | bounded and reversible only | supports safe post-deployment adaptation (Singhvi et al., 2023; Q. Zhang et al., 2025) |

The default optimizer loop should be:

1.  Compile a typed program into an executable graph with deterministic validators.
2.  Run the graph on a train split while capturing full traces.
3.  Convert traces into module, block, and node diagnostics.
4.  Propose local edits first, then small structural edits only when local edits saturate.
5.  Select candidates on a quality, cost, latency Pareto frontier.
6.  Re-evaluate promoted candidates on a larger dev slice, then on the full dev set.
7.  Emit a frozen artifact bundle for deployment.
8.  Allow online updates only to explicitly whitelisted state, with canarying and rollback (Harikumar, 2026; He et al., 2025; Opsahl-Ong et al., 2024; Wang et al., 2025; Q. Zhang et al., 2025).

The key design bet is that trace quality and blame assignment matter more than exotic search. MIPRO shows that even prompt and demo optimization becomes much stronger when proposals are grounded and evaluated with a surrogate over minibatches (Opsahl-Ong et al., 2024). GEPA, CE-Graph, JudgeFlow, and Maestro all show the same pattern from a different angle: once traces expose where and why failures occur, the optimizer can spend budget on targeted edits instead of blind global search (Agrawal et al., 2025; Ma et al., 2026; Wang et al., 2025; J. Zhang et al., 2025).

## Taxonomy of optimizer families

| Family | Search space | Proposal mechanism | Feedback | Best use |
|:---|:---|:---|:---|:---|
| Modular prompt and demo compilers | instructions, demos per module | grounded LM proposals plus Bayesian or random search | downstream metric on minibatches and full dev | offline compile for fixed graphs (Khattab et al., 2024; Opsahl-Ong et al., 2024) |
| Reflective prompt evolution | instructions, sometimes module-local text | trace-conditioned reflection, mutation, merge, Pareto selection | scalar score plus textual critique and trajectories | low-rollout offline optimization (Agrawal et al., 2025) |
| Textual gradient methods | arbitrary text and code variables in a graph | backward LLM generates local critiques and rewrites | textual gradients over graph edges | local module updates and rapid prototyping (Cheng et al., 2024; Yuksekgonul et al., 2024) |
| Structured prompt program search | prompt sections, formats, examples, symbolic prompt structure | symbolic mutators plus beam or evolutionary search | compile-time objective | prompt programs with explicit sections (Schnabel & Neville, 2024; Spiess et al., 2025) |
| Workflow and topology optimizers | nodes, edges, control flow, config | staged or alternating graph plus config edits | traces, scores, evaluator rationale | agentic systems with structural failure modes (Ma et al., 2026; Wang et al., 2025; J. Zhang et al., 2025; Zhou et al., 2025) |
| Harness and context optimizers | retrieval policy, memory policy, orchestration code, context playbooks | coding agents or structured context evolution | full logs, code diffs, execution traces | long-horizon agents and context-heavy systems (Lee et al., 2026; Q. Zhang et al., 2025) |
| Typed and deterministic compilers | plan schemas, node registry, validators, typed interfaces | planner emits typed plan, compiler validates and assembles | structural validity and task success | high-reliability structured workflows (Harikumar, 2026; Lin et al., 2025; Singhvi et al., 2023) |
| Online adaptation systems | memory, playbooks, tool docs, thresholds, exemplars | bounded reflection, replay, retrieval updates | production traces and delayed reward | post-deployment improvement under guardrails (Banerjee et al., 2026; Hu et al., 2025; Q. Zhang et al., 2025) |

Three practical conclusions follow from this taxonomy.

- Prompt and demo optimization is the foundation, not the whole optimizer (Khattab et al., 2024; Opsahl-Ong et al., 2024).
- Structural search should be sparse, constrained, and trace-driven rather than always-on (Wang et al., 2025; J. Zhang et al., 2025; Zhou et al., 2025).
- Online learning should mutate context and routing long before it mutates topology or code (Lee et al., 2026; Singhvi et al., 2023; Q. Zhang et al., 2025).

## Key ingredients with evidence and citations

### Search space design

The search space should be explicit, typed, and factorized. The highest value editable axes today are module instructions, demonstration sets, tool descriptions, retrieval and routing policies, validators, and a narrow set of graph edits (Ghoshal et al., 2026; Opsahl-Ong et al., 2024; Wang et al., 2025). MIPRO gives the strongest base case for fixed graphs by jointly searching instructions and few-shot demonstrations per module with grounded proposal generation and Bayesian selection (Opsahl-Ong et al., 2024). GEPA then shows that natural language rule updates can outperform rollout-heavy RL when the prompt is the dominant artifact (Agrawal et al., 2025).

Graph and harness search matter, but only after the local text space is mature. Maestro reports consistent gains from joint graph and config optimization over config-only tuning, especially on workflows with missing intermediate nodes or poor information flow (Wang et al., 2025). CE-Graph reaches a similar conclusion by restricting structure search to operator-constrained edits such as revise prompt, insert node, and delete node, targeted at the densest failure mode rather than broadcast over the whole graph (J. Zhang et al., 2025). Meta-Harness pushes the editable boundary outward to retrieval and memory orchestration code, which is important for production systems but too open-ended for a first release (Lee et al., 2026).

A practical Rust IR should therefore expose these parameter classes:

| Parameter class | Examples | Type discipline | First release |
|:---|:---|:---|:---|
| Prompt text | system instruction, rubric, tool-use policy | structured text with role tags | yes |
| Demonstrations | module-local exemplars, trajectories | typed input and output records | yes |
| Tool metadata | tool descriptions, slot hints, examples | schema plus editable doc strings | yes |
| Control knobs | model choice, temperature, retry budget, top k retrieval | numeric or enum | yes |
| Contracts | regex checks, JSON schemas, custom validators | executable predicates | yes |
| Routing | fallback order, verifier gating, abstain threshold | policy objects | yes |
| Structure | insert verifier, split module, add retrieval hop, reroute edge | graph edits over typed nodes | later |
| Harness policy | memory write rules, context assembly, retrieval cache policy | code or declarative policy | later |

### Proposal mechanisms

Proposal quality matters as much as search strategy. MIPRO grounds proposals in program summaries, data summaries, successful traces, and bootstrapped demonstrations, then uses a Tree-structured Parzen Estimator to search combinations under minibatch evaluation (Opsahl-Ong et al., 2024). This is the right offline starting point because it combines cheap proposal generation with a robust selector.

Reflective methods are the next ingredient. GEPA creates prompt mutations from execution traces and textual feedback, then preserves diversity through Pareto frontier maintenance and system-aware merge (Agrawal et al., 2025). TextGrad and OPTO generalize this idea by turning feedback into local textual gradients over a computation graph, which is useful when different parameter types need different update prompts (Cheng et al., 2024; Yuksekgonul et al., 2024). For structural search, staged optimization works better than fully joint search when budgets are limited. MASS warms up blocks, then searches topologies, then retunes globally (Zhou et al., 2025). Cognify adapts the same principle with hierarchical layers and budget reallocation across architecture, step, and prompt changes (He et al., 2025).

The implementation implication is simple. The optimizer should support multiple proposal engines behind one interface:

- grounded proposer for prompt and demo candidates
- reflective proposer for trace-conditioned rewrites
- symbolic mutator for section and template edits
- graph proposer for small typed structure edits
- code or harness proposer kept behind a feature gate (Agrawal et al., 2025; Lee et al., 2026; Opsahl-Ong et al., 2024; Schnabel & Neville, 2024; Wang et al., 2025)

### Feedback signals

Recent papers converge on one lesson: scalar reward alone is not enough. OPTO argues that execution traces play the role that gradients play in differentiable systems, because they expose the causal path from parameter to failure (Cheng et al., 2024). CE-Graph names the scalar-only problem directly as information collapse and replaces it with failure signatures that encode both where a failure occurred and what semantic error occurred (J. Zhang et al., 2025). JudgeFlow further shows that ranking block responsibility across failed traces yields more stable local optimization than trying to infer blame from a single end-to-end score (Ma et al., 2026).

A serious optimizer should therefore capture four feedback layers for every run:

| Layer | Contents | Use |
|:---|:---|:---|
| Outcome | task metric, pass or fail, quality rubric | promotion and Pareto ranking |
| Cost | tokens, dollars, latency, tool count, retries | Pareto ranking and budget gating |
| Trace | per-node inputs, outputs, tool calls, retrieved docs, exceptions | diagnosis and blame assignment |
| Judgment | evaluator rationale, LLM critique, human preference, validator messages | proposal grounding |

Tool-using systems need an additional tool layer. JTPRO shows that tool selection accuracy, slot filling accuracy, and overall success should be measured separately because tool choice and argument correctness fail for different reasons (Ghoshal et al., 2026). That directly argues for separate blame channels for tool selection, argument shaping, and downstream answer quality.

### Trace and blame assignment

A Rust optimizer should treat traces as first-class data, not logging afterthoughts. Each module boundary should emit a typed record with input values, output values, prompt version, demonstrations used, retrieved evidence, tool arguments, validator outcomes, and parent edge identifiers. This is the minimum needed to support the three strongest blame strategies in the literature.

| Blame strategy | Mechanism | Transferable lesson |
|:---|:---|:---|
| Surrogate sensitivity | learn which parameter combinations raise downstream score | useful for prompt and demo compilers (Opsahl-Ong et al., 2024) |
| Reflective local diagnosis | ask an LM to inspect failed trajectories and rewrite a targeted module | useful when errors are semantic and legible in traces (Agrawal et al., 2025; Ghoshal et al., 2026) |
| Structural failure attribution | convert traces into block or node failure signatures and rank responsibility | needed for workflow edits (Ma et al., 2026; J. Zhang et al., 2025) |

The implementation choice is to keep blame assignment layered. First use deterministic localization when a validator or parser fails. Second use structural heuristics such as first failing node, repeated retry boundary, or wrong tool choice. Third use an LLM judge only when deterministic and heuristic signals do not isolate the cause. JudgeFlow supports this hierarchy indirectly by showing that judge signals become much more useful when the workflow is already segmented into meaningful blocks (Ma et al., 2026).

### Pareto and cost tradeoffs

No single objective is sufficient. LangProBe shows large optimizer by architecture interactions and a strong quality-cost Pareto story for optimized language programs, but not a universal winner across tasks and models (Tan et al., 2025). GEPA shows sample efficiency in rollouts (Agrawal et al., 2025). Cognify shows quality, cost, and latency can all be improved when the optimizer can change different layers and reallocate budget adaptively (He et al., 2025). A production optimizer should thus maintain a live Pareto frontier over at least quality, dollar cost, and latency, with optional robustness as a fourth axis (Agrawal et al., 2025; He et al., 2025; Tan et al., 2025).

This implies two separate frontiers.

- Training frontier over optimization cost versus candidate quality
- Deployment frontier over runtime quality, runtime cost, and latency

These frontiers should not be collapsed. A candidate that is expensive to discover may still be cheap and strong at runtime. Compile-time search methods such as SAMMO and MIPRO assume this amortization explicitly (Opsahl-Ong et al., 2024; Schnabel & Neville, 2024).

### Typed contracts and deterministic boundaries

Typed contracts are not optional in a Rust system. DSPy Assertions shows that soft suggestions and hard assertions can be compiled into both demonstration filtering and bounded retry logic (Singhvi et al., 2023). TACs formalize type compliance with parse and canonicalization steps between modules, which is useful even if the full probabilistic training scheme is not adopted (Lin et al., 2025). PlanCompiler shows the strongest deterministic version of the same idea: fixed node registry, static graph validation, typed plan schema, and code generation only after validation passes (Harikumar, 2026).

The transferable architecture is:

1.  Every module declares input type, output type, schema, and validator set.
2.  Every LM output passes through parse, canonicalize, and validate steps before downstream use.
3.  Hard contract failures stop candidate promotion.
4.  Soft contract failures can trigger bounded repair or route to a verifier path (Harikumar, 2026; Lin et al., 2025; Singhvi et al., 2023).

### Offline compilation and bounded online adaptation

Offline compilation should be the first-class path. Most of the strongest results come from repeated evaluation on train and dev sets with minibatching, surrogate ranking, or staged halving (He et al., 2025; Opsahl-Ong et al., 2024; Spiess et al., 2025). This is where structural changes, prompt set search, verifier insertion, and routing changes belong.

Online adaptation should be deliberately smaller in scope. ACE gives the clearest design for safe online evolution: contexts are represented as granular playbook items with deterministic merge and pruning, which avoids monolithic rewrite and context collapse (Q. Zhang et al., 2025). Assertions add bounded retry and repair as a second safe online primitive (Singhvi et al., 2023). The lesson is to whitelist only reversible state:

- memory or playbook entries
- exemplar caches
- tool-local descriptions
- routing thresholds
- verifier enable or disable flags (Ghoshal et al., 2026; Singhvi et al., 2023; Q. Zhang et al., 2025)

Unrestricted online graph mutation should stay out of scope for an initial Rust optimizer.

## Practical architecture requirements for DSRs and Rust

The implementation target should be a typed optimizer runtime with six core subsystems.

| Subsystem | Responsibilities | Rust shape | Build priority |
|:---|:---|:---|:---|
| Program IR | typed nodes, edges, contracts, parameter handles | enums, traits, serde structs, graph crate | first |
| Executor and tracer | run graph, record per-node trace, collect costs | async executor plus append-only event log | first |
| Evaluator | task metrics, judges, validators, Pareto scorer | trait objects over metrics and judges | first |
| Optimizer kernel | candidate pool, proposal engines, selection, promotion | scheduler plus pluggable proposer traits | first |
| Compiler | freeze artifact bundle for deployment | deterministic serializer and manifest writer | first |
| Online controller | canary, rollback, memory updates, threshold tuning | separate state machine with audit log | second |

A good IR is more important than a clever search loop. It should separate immutable structure from mutable parameters and expose provenance for every parameter value.

``` rust
struct Program {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    contracts: Vec<Contract>,
    params: ParamStore,
}

enum Param {
    Instruction(TextParam),
    DemoSet(DemoParam),
    ToolDoc(ToolDocParam),
    Decode(DecodeParam),
    Route(RouteParam),
    Threshold(ThresholdParam),
}

struct TraceRecord {
    node_id: NodeId,
    param_version: ParamVersion,
    input: Value,
    output: Value,
    retrieved: Vec<ArtifactRef>,
    tool_call: Option<ToolCall>,
    validators: Vec<ValidatorResult>,
    latency_ms: u64,
    token_cost: TokenCost,
}
```

Three architecture requirements are non-negotiable.

- Deterministic artifact bundles. A compiled candidate should be a frozen manifest of graph version, parameter versions, validators, and evaluation results (Harikumar, 2026; Khattab et al., 2024).
- Full artifact logging. Meta-Harness shows that optimizer quality rises when prior code, scores, and traces remain inspectable rather than compressed into short summaries (Lee et al., 2026). The same principle should hold for prompt and graph candidates.
- Small, typed edit operators. Each proposer should emit typed patches rather than free-form rewritten programs. CE-Graph and Maestro both benefit from edit libraries and trust regions over graph changes (Wang et al., 2025; J. Zhang et al., 2025).

For DSRs-like ergonomics, expose a declarative user surface and keep optimizer internals out of the authoring API. Users should define modules, signatures, contracts, and metrics. The system should own the search, trace capture, and promotion policy (Khattab et al., 2023, 2024; Opsahl-Ong et al., 2024).

## What to build first versus leave as scaffolding

### Build first

| Component | Why first | Evidence |
|:---|:---|:---|
| Typed graph IR with contracts | everything else depends on safe composition and localized edits | (Harikumar, 2026; Lin et al., 2025; Singhvi et al., 2023) |
| Executor with rich traces | trace quality is the main lever for later optimization | (Cheng et al., 2024; Ma et al., 2026; J. Zhang et al., 2025) |
| MIPRO-like prompt and demo compiler | strongest validated baseline for modular offline optimization | (Opsahl-Ong et al., 2024) |
| Candidate pool with Pareto ranking | avoids greedy collapse and supports cost-aware selection | (Agrawal et al., 2025; Tan et al., 2025) |
| Deterministic validators and retry wrappers | enables safe compile-time filtering and bounded repair | (Harikumar, 2026; Singhvi et al., 2023) |
| Benchmark harness and artifact store | necessary to avoid chasing anecdotes | (Lee et al., 2026; Tan et al., 2025) |

### Build next

| Component | Why next | Evidence |
|:---|:---|:---|
| Reflective rewrite engine | improves low-budget search using traces and critique | (Agrawal et al., 2025; Ghoshal et al., 2026) |
| Hierarchical budget allocator | important once the search space spans multiple edit layers | (He et al., 2025; Spiess et al., 2025) |
| Tool-doc optimizer | high leverage for agents with many tools | (Ghoshal et al., 2026) |
| Block judge | stabilizes blame for workflow-local edits | (Ma et al., 2026) |
| Playbook memory updater | safest online adaptation primitive | (Q. Zhang et al., 2025) |

### Leave as scaffolding

| Component | Reason to delay | Evidence |
|:---|:---|:---|
| Open-ended topology search | high payoff but large search explosion and evaluation cost | (Wang et al., 2025; Zhou et al., 2025) |
| Harness code synthesis | powerful but operationally heavy and hard to sandbox in v1 | (Lee et al., 2026) |
| RL-heavy online optimization | weaker evidence than reflective and trace-based methods at equal budget | (Agrawal et al., 2025) |
| Fully probabilistic cascade training | promising but a bigger systems and training commitment than needed for v1 | (Lin et al., 2025) |
| Unbounded self-modifying agents | safety and observability burden is too high for early deployment | (Lee et al., 2026; Q. Zhang et al., 2025) |

The best first version is therefore not a universal optimizer. It is a strong offline compiler with rich traces, typed validators, and enough reflective local repair to avoid wasting search budget on obvious repeats.

## Evaluation and benchmarking protocol

Evaluation should separate optimizer quality from model quality, architecture quality, and deployment cost. LangProBe is the best benchmark anchor because it exposes optimizer by architecture interactions instead of reporting single best cases (Tan et al., 2025).

### Core benchmark matrix

| Axis | Minimum design |
|:---|:---|
| Tasks | include classification, extraction, reasoning, RAG, tool use, multi-step agent tasks |
| Models | at least one frontier API model and two smaller open models |
| Architectures | single call, fixed modular pipeline, verifier-augmented pipeline, one agentic workflow |
| Optimizers | no optimization, random or few-shot search, MIPRO-like, reflective, hierarchical |
| Budgets | fixed optimizer token budget and fixed rollout budget |

### Reported metrics

| Metric family | What to report |
|:---|:---|
| Quality | task score, robustness under seed variation, held-out test score |
| Runtime cost | input and output tokens, tool calls, dollars, latency percentile |
| Optimization cost | total rollouts, optimizer tokens, wall-clock compile time |
| Generalization | transfer across models, across nearby tasks, and across prompt seeds |
| Reliability | validator pass rate, parse success, tool selection accuracy, slot accuracy |

### Protocol details

- Use separate train, dev, and test splits. Do not promote candidates on test.
- Plot budgeted optimization curves, not only final best score.
- Report Pareto frontiers at runtime and compile time.
- Include ablations for trace richness, blame method, proposal engine, and search space width.
- Re-run with multiple seeds because prompt optimization variance is real (Tan et al., 2025; X. Zhang et al., 2026).
- Evaluate both fixed-model and cross-model transfer because prompt quality is model-specific more often than many systems assume (Schnabel & Neville, 2024; Tan et al., 2025).
- For tool agents, break out tool selection accuracy, slot filling accuracy, and overall success (Ghoshal et al., 2026).
- For online adaptation, require canary traffic, delayed promotion, and rollback statistics (Q. Zhang et al., 2025).

A practical internal benchmark suite should include at least one task where structure search matters. Otherwise the optimizer will look stronger than it is by succeeding only on prompt-local problems. Maestro and CE-Graph both show that some failures are structural and remain invisible to prompt-only compilers (Wang et al., 2025; J. Zhang et al., 2025).

## Risks and open research gaps

The first risk is optimizer overfitting. LangProBe shows that some optimizers, especially rule-induction styles, can improve dev while losing on test (Tan et al., 2025). The second risk is blame error. Rich traces help, but judge-based attribution is still noisy and can send the optimizer to the wrong block (Ma et al., 2026). The third risk is search-space inflation. Once topology, harness code, and online state are all mutable, evaluation cost can dominate model cost (He et al., 2025; Lee et al., 2026).

The deeper research gaps are more important than any single algorithmic choice.

| Gap | Why unsolved | Implication |
|:---|:---|:---|
| Stable cross-model optimization | prompts and structures transfer poorly across models | keep optimizer model-specific by default (Schnabel & Neville, 2024; Tan et al., 2025) |
| Reliable structural blame | localizing root cause in long agent traces is still noisy | structural search needs stronger diagnostics (Ma et al., 2026; J. Zhang et al., 2025) |
| Multi-objective selection under drift | Pareto fronts shift with model price and latency changes | deployment policy must be recalibrated continuously (He et al., 2025; Tan et al., 2025) |
| Safe online learning | context updates help, but long-horizon credit remains weak | keep online edits local and reversible (Q. Zhang et al., 2025) |
| Typed generation under semantic constraints | syntax can be enforced more easily than semantic correctness | combine type checks with domain validators (Harikumar, 2026; Lin et al., 2025) |
| Benchmark realism | current benchmarks only partly capture production harness behavior | internal harness benchmarks remain necessary (Lee et al., 2026; Tan et al., 2025) |

One more open point is optimizer self-reference. Meta-optimizers such as metaTextGrad and Meta-Harness suggest gains from optimizing the optimizer or the harness around the model, but they also increase system complexity fast (Lee et al., 2026; Xu et al., 2025). For a Rust v1, that complexity should remain out of the critical path.

## Annotated bibliography of the most important papers

| Paper | Why it matters for this build |
|:---|:---|
| (Opsahl-Ong et al., 2024) | Best current anchor for offline compilation of modular LM programs. Defines the practical baseline for joint instruction and demo search with grounded proposal generation, minibatch evaluation, and surrogate-guided selection. |
| (Khattab et al., 2024) | Canonical DSPy anchor for declarative LM programs compiled against downstream metrics. Useful for the user-facing abstraction and compiler framing even though the optimizer details are less implementation-specific here. |
| (Khattab et al., 2023) | Earlier DSPy paper that makes the self-improving pipeline idea explicit and remains useful for the design philosophy behind program-level optimization. |
| (Agrawal et al., 2025) | Strong evidence that reflective prompt evolution with trace-conditioned natural language feedback can beat rollout-heavy RL while using far fewer rollouts. Important for local rewrite engines and Pareto candidate management. |
| (Cheng et al., 2024) | Provides the cleanest conceptual model for traces as optimizer inputs. Useful for optimizer APIs, trace representation, and the idea of a minimal relevant subgraph for updates. |
| (He et al., 2025) | Best source for hierarchical budget allocation across architecture, step, and prompt changes. Important once the optimizer spans more than prompts. |
| (Wang et al., 2025) | Best evidence that joint graph plus config search fixes failures prompt-only methods cannot. Important for future structure search, trust regions, and graph edit libraries. |
| (J. Zhang et al., 2025) | Sharpest treatment of failure distribution modeling and operator-constrained workflow repair. Important for failure signatures, clustering, and targeted graph edits. |
| (Ma et al., 2026) | Best current paper for block-level blame assignment in complex workflows. Important for ranking responsibility and limiting edits to one block at a time. |
| (Ghoshal et al., 2026) | Most concrete recent treatment of tool-description optimization. Important for splitting tool selection from slot filling and for co-optimizing global instructions with tool-local schema text. |
| (Q. Zhang et al., 2025) | Best source for bounded online adaptation through structured playbooks, deterministic merge, and pruning. Important for safe memory evolution after deployment. |
| (Lin et al., 2025) | Strong typed-systems paper. Even without adopting its full training method, its parse and canonicalize discipline and type-compliance framing are directly valuable. |
| (Singhvi et al., 2023) | Best bridge between compile-time optimization and runtime repair through hard and soft assertions. Important for validators, retries, and demonstration filtering. |
| (Harikumar, 2026) | Strong deterministic systems anchor. Important for separating generative planning from deterministic compilation and for static validation before execution. |
| (Tan et al., 2025) | Best benchmark anchor for optimizer evaluation because it studies tasks, models, programs, and optimizers jointly rather than in isolation. |
| (Lee et al., 2026) | Most compelling evidence that harness logic itself is a major optimization surface. Important longer-term, but should stay out of the critical path for a first Rust release. |

The implementation-first reading of this literature is straightforward. Build a typed offline compiler first, make traces and contracts first-class, keep candidate selection Pareto-aware, add reflective local rewrite before broad structure search, and constrain online learning to reversible context updates. That is the smallest design that matches where the 2025 and 2026 literature is actually strongest (Agrawal et al., 2025; He et al., 2025; Opsahl-Ong et al., 2024; Tan et al., 2025; Q. Zhang et al., 2025).

---

## References

Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Dimakis, A., Stoica, I., Klein, D., Zaharia, M. A., & Khattab, O. (2025). GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. *ArXiv*, *abs/2507.19457*.

Banerjee, P., Moshtaghi, M., & Chadha, A. (2026). *APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay*.

Cheng, C.-A., Nie, A., & Swaminathan, A. (2024). Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs. *Advances in Neural Information Processing Systems 37*. <https://doi.org/10.52202/079017-2287>

Ghoshal, S., Mittal, A., Singh, J., Ballesteros, M., Sun, W., Tu, F., Singh, S., Benajiba, Y., Shah, F., Bharadwaj, S., Ravi, S., & Roth, D. (2026). *JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents*.

Harikumar, P. (2026). *PlanCompiler: A Deterministic Compilation Architecture for Structured Multi-Step LLM Pipelines*.

He, Z., Abhyankar, R., Srivatsa, V., & Zhang, Y. (2025). Cognify: Supercharging Gen-AI Workflows With Hierarchical Autotuning. *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2*. <https://doi.org/10.1145/3711896.3736884>

Hu, M., Durme, B. V., Andreas, J., & Jhamtani, H. (2025). Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting. *ArXiv*, *abs/2510.10304*. <https://doi.org/10.48550/arXiv.2510.10304>

Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., Miller, H., Zaharia, M., & Potts, C. (2023). DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. *ArXiv*, *abs/2310.03714*.

Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., Miller, H., Zaharia, M., & Potts, C. (2024). DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines. *International Conference on Learning Representations*.

Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., & Finn, C. (2026). *Meta-Harness: End-to-End Optimization of Model Harnesses*.

Lin, C., Peng, D., Lu, Y., Zhang, M., & Ie, E. (2025). Type-Compliant Adaptation Cascades: Adapting Programmatic LM Workflows to Data. *ArXiv*, *abs/2508.18244*. <https://doi.org/10.48550/arXiv.2508.18244>

Ma, Z., Zhao, Z., Hua, C., Berto, F., & Park, J. (2026). JudgeFlow: Agentic Workflow Optimization via Block Judge. *ArXiv*, *abs/2601.07477*. <https://doi.org/10.48550/arXiv.2601.07477>

Opsahl-Ong, K., Ryan, M. J., Purtell, J., Broman, D., Potts, C., Zaharia, M., & Khattab, O. (2024). Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs. *ArXiv*, *abs/2406.11695*. <https://doi.org/10.18653/v1/2024.emnlp-main.525>

Schnabel, T., & Neville, J. (2024). Symbolic Prompt Program Search: A Structure-Aware Approach to Efficient Compile-Time Prompt Optimization. *Conference on Empirical Methods in Natural Language Processing*, 670–686. <https://doi.org/10.18653/v1/2024.findings-emnlp.37>

Singhvi, A., Shetty, M., Tan, S., Potts, C., Sen, K., Zaharia, M., & Khattab, O. (2023). DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines. *ArXiv*, *abs/2312.13382*. <https://doi.org/10.48550/arXiv.2312.13382>

Spiess, C., Vaziri, M., Mandel, L., & Hirzel, M. (2025). AutoPDL: Automatic Prompt Optimization for LLM Agents. *ArXiv*, *abs/2504.04365*. <https://doi.org/10.48550/arXiv.2504.04365>

Tan, S., Agrawal, L. A., Singhvi, A., Lai, L., Ryan, M. J., Klein, D., Khattab, O., Sen, K., & Zaharia, M. (2025). LangProBe: a Language Programs Benchmark. *ArXiv*, *abs/2502.20315*. <https://doi.org/10.48550/arXiv.2502.20315>

Wang, W., Kattakinda, P., & Feizi, S. (2025). Maestro: Joint Graph & Config Optimization for Reliable AI Agents. *ArXiv*, *abs/2509.04642*. <https://doi.org/10.48550/arXiv.2509.04642>

Xu, G., Yuksekgonul, M., Guestrin, C., & Zou, J. (2025). metaTextGrad: Automatically optimizing language model optimizers. *ArXiv*, *abs/2505.18524*. <https://doi.org/10.48550/arXiv.2505.18524>

Yuksekgonul, M., Bianchi, F., Boen, J., Liu, S., Huang, Z., Guestrin, C., & Zou, J. (2024). TextGrad: Automatic “Differentiation” via Text. *ArXiv*, *abs/2406.07496*. <https://doi.org/10.48550/arXiv.2406.07496>

Zhang, J., Cai, K., Zeng, Q., Liu, N., Fan, S., Chen, Z., & Wang, K. (2025). Failure-Driven Workflow Refinement. *ArXiv*, *abs/2510.10035*. <https://doi.org/10.48550/arXiv.2510.10035>

Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., Thakker, U., Zou, J., & Olukotun, K. (2025). Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. *ArXiv*, *abs/2510.04618*. <https://doi.org/10.48550/arXiv.2510.04618>

Zhang, X., Wang, G., Cui, Y., Qiu, W., Li, Z., Zhu, B., & He, P.-G. (2026). *Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems*.

Zhou, H., Wan, X., Wan, X., Sun, R., Palangi, H., Iqbal, S., Vuli’c, I., Korhonen, A., & Arik, S. Ö. (2025). Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies. *ArXiv*, *abs/2502.02533*. <https://doi.org/10.48550/arXiv.2502.02533>
