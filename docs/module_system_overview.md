# DSRs Module System — What Changed, What It Enables

A quick overview of the module system as it stands after the v1 program unification. The typed core from the earlier redesign is unchanged; the graph-optimization story ("Section 1.3") is now concrete in the IR rather than a planned `ProgramGraph` layer.

---

## What's changed

| Before | Now |
|--------|-----|
| `Example` / `Prediction` as primary I/O | Typed `S::Input` / `Predicted<S::Output>`; trainset rows are plain structs projected via `ToInput`/`ToOutput` |
| `#[Signature(cot)]` applies CoT at signature level | `ChainOfThought::<S>::new()` — strategy is the module, not the signature |
| Reflection-based leaf discovery (facet walker, `DynPredictor` handles) | Explicit declaration: the `Predictors` trait, one `predictors!(MyModule { field_a, field_b })` line per module |
| Optimizers mutate the module during search | Candidates are data injected ambiently per rollout (`fx::with_params`); the single mutation is the final install of the winner |
| Per-optimizer engines (`EvalEngine` / `ProgramEvalEngine`) | One shared `Engine` over an `OptimizeTarget` (typed module lane or loaded-program lane) |
| `ReAct<S>` module, `ModuleExt::map`/`and_then` combinators | Deleted. Tool loops are IR `AgentLoop` nodes (`#[agent]`, or tools on a `Predict`); output transforms are plain Rust in `forward` |
| `Predict` renders/parses on its own LM path | `Predict<S>` executes as a 1-node IR program through the `Interpreter`; instance state is an `ir::Overlay` |
| Graph optimization as future work | The IR edit calculus: `Program::edited(&[Edit])`, `legal_edits`, `migrate_overlay` |

---

## What users write

```rust
#[derive(Signature, Clone)]
/// Answer questions accurately.
struct QA {
    #[input] question: String,
    #[output] answer: String,
}

// Pick a strategy by changing the type — everything else stays the same
let module = ChainOfThought::<QA>::new();
let result = module.call(QAInput { question: "2+2?".into() }).await?;
result.reasoning  // augmented field — direct access
result.answer     // original field — via Deref

// Batch without changing the module
let results = dspy_rs::forward_all(&module, inputs, 5).await;
```

---

## What writing a new library module looks like

A new augmentation (like adding confidence scoring to any output):
```rust
#[derive(Augmentation)]
#[augment(output, append)]
struct Confidence {
    /// Model's self-assessed confidence
    #[output] confidence: f64,
}
// Done — WithConfidence<O> now exists and composes with any signature
// Users write: Predict<Augmented<QA, Confidence>>
// They get: result.answer + result.confidence
```

A new composite module is a struct with predictor fields, a `predictors!` line, and a `forward` body of ordinary Rust:

```rust
struct TwoStepQA {
    retrieve: Predict<RetrieveSig>,
    answer: ChainOfThought<AnswerSig>,
}

dspy_rs::predictors!(TwoStepQA { retrieve, answer });

impl Module for TwoStepQA {
    type Input = RetrieveInput;
    type Output = WithReasoning<AnswerOutput>;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let ctx = self.retrieve.call(input).await?;
        self.answer.call(AnswerInput { context: ctx.passages.clone() }).await
    }
}
```

The `predictors!` line is the whole discovery story: each field identifier becomes the leaf's canonical name — its trace-span component, its optimizer-candidate key, and its `ModuleState` persistence key. No derive magic, no traversal code, no pointer casts.

---

## What optimizers see

```rust
optimizer.compile_module(&mut module, &trainset, &metric).await?;
// internally:
let mut target = OptimizeTarget::module(&mut module, &trainset, &metric);
//   — snapshots each declared leaf as a LeafInfo (schema, instruction, demos)
//   — stamps each leaf's trace name once (the naming pass)
let mut engine = Engine::new(optimizer.engine_config());
optimizer.compile(&mut target, &mut engine).await?;
//   — candidates are name-keyed `Candidate`s, injected ambiently per rollout;
//     evaluation never mutates the module, so candidates fan out concurrently
//   — the winner is installed exactly once via PredictorInfo::load_state
// after compile returns, module.call() uses optimized params — no code change
```

The `Optimizer` trait is object-safe: `Box<dyn Optimizer>` pipelines can share one `Engine` — one budget, one rollout cache, one score matrix — across stages. The same trait drives the program lane (`OptimizeTarget::program`): an interpreter-loaded `.dsrs` program, JSON examples, and an overlay winner for `Program::bake`.

---

## Structural optimization (Section 1.3 made concrete)

The paper's "Dynamic Workflow Optimization" landed as the IR **edit calculus**, not a separate graph layer. A `#[module]` function lowers to a `Program` — a validated node tree with named leaves and addressable parameter slots — and structural moves are plain serde values applied purely:

```rust
use dspy_rs::ir::{Edit, migrate_overlay};

let leaf = program.leaf_id("drafter").unwrap();
let menu = program.legal_edits(leaf);        // the proposer menu (LLM-promptable)
let child = program.edited(&[Edit::SwapLeaf { leaf, to: swap_target }])?;
let carried = migrate_overlay(&program, &tuned, &child);  // value progress survives
```

`edited` clones, applies, re-validates with the loader's own rules, and seals a new content hash; the parent program and every hash-bound artifact minted against it stay coherent. **Split**, **fuse**, wrap-in-retry, predict↔agent swaps, and tool add/remove are all expressible as `Edit` batches; data-flow legality stays with the one validator both the builder and the loader use.

The key architectural property is unchanged: the typed path and the program path share one rendering pipeline (`SignatureDef` → `ChatAdapter` → prompt). A `Predict<QA>` — which itself executes as a 1-node program — and a loaded `predict` leaf over the same signature produce identical prompts, so restructuring cannot cause prompt divergence.

---

## Layer stack

```
You're here          What you touch                          What's invisible to you
────────────────────────────────────────────────────────────────────────────────────
App developer        Signature, module.call()                Everything below
Module author        predictors!, forward()                  IR lowering, interpreter
Optimizer dev        Optimizer::compile, OptimizeTarget,     IR internals
                     Engine, Candidate
Structural optimizer Program, Edit, legal_edits,             validator internals
                     migrate_overlay
```

Each layer only exists if you need it. Simple usage never touches the IR directly.
