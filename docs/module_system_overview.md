# DSRs Module System — What Changed, What It Enables

This is a quick overview of the module system redesign. It builds on everything from the paper but adds a typed core and makes Section 1.3 (graph optimization) concrete.

---

## What's changed

| Before | Now |
|--------|-----|
| `Example` / `Prediction` as primary I/O | Typed `S::Input` / `Predicted<S::Output>` for the typed path; `Example` still used at optimizer/dynamic boundary |
| `#[Signature(cot)]` applies CoT at signature level | `ChainOfThought::<S>::new()` — strategy is the module, not the signature |
| `predict.forward(example).await` | `module.call(input).await?` on the typed path |
| Manual `#[derive(Optimizable)]` + `#[parameter]` | Automatic discovery from struct shape |
| Static `FieldSpec` arrays from macros | `SignatureSchema` derived from types at runtime |
| `CallOutcome` with `.into_result()?` | `Result<Predicted<O>, PredictError>` — `?` works on stable |
| Section 1.3 graph optimization (future work) | `ProgramGraph` being built now (V6) — walker foundation landed in V5 |

> **TODO:** Nail down the long-term role of `Example`. It's still load-bearing at the DynPredictor boundary (demo conversion, optimizer manipulation, DataLoader). The typed path doesn't kill it — but its scope and future API need a decision.

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

// Swap to ReAct — same call site
let module = ReAct::<QA>::builder()
    .tool("search", "Search the web", search_fn)
    .build();

// Batch without changing the module
let results = dsrs::forward_all(&module, inputs, 5).await;

// Simple transform without impl Module
let confident = module.map(|r| Confident { answer: r.answer, confidence: 0.9 });
```

---

## What writing a new library module looks like

A new augmentation (like adding confidence scoring to any output):
```rust
#[derive(Augmentation)]
#[augment(output, append)]
struct Confidence {
    /// Model's self-assessed confidence
    confidence: f64,
}
// Done — WithConfidence<O> now exists and composes with any signature
// Users write: Predict<Augmented<QA, Confidence>>
// They get: result.answer + result.confidence
```

A new module (like BestOfN — runs N times, picks best):
```rust
#[derive(Module)]
struct BestOfN<M: Module> {
    module: M,              // walker sees through — finds all Predict leaves inside
    #[skip] n: usize,
    #[skip] reward_fn: Box<dyn Fn(&M::Input, &M::Output) -> f64 + Send + Sync>,
}

impl<M: Module> Module for BestOfN<M> where M::Input: Clone {
    type Input = M::Input;
    type Output = M::Output;

    async fn forward(&self, input: M::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let mut best = None;
        let mut best_score = f64::NEG_INFINITY;
        for _ in 0..self.n {
            let result = self.module.call(input.clone()).await?;
            let score = (self.reward_fn)(&input, &result);
            if score > best_score { best_score = score; best = Some(result); }
        }
        best.ok_or(PredictError::AllAttemptsFailed)
    }
}
```

`#[derive(Module)]` makes `module: M` discoverable — optimizers automatically find and tune the Predict leaves inside whatever `M` is. `#[skip]` fields (closures, config) are invisible to the walker. No traversal code, no schema construction.

---

## What optimizers see

```rust
optimizer.compile(&mut module, trainset, metric).await;
// internally:
visit_named_predictors_mut(&mut module, |path, predictor| {
    // mutate demos, instructions, dump/load state — all through DynPredictor handles
    ControlFlow::Continue(())
})?;
// after compile returns, module.call() uses optimized params — no code change
```

---

## What ProgramGraph enables (Section 1.3 made concrete)

This is the paper's "Dynamic Workflow Optimization" — pipelines as executable graphs that can restructure themselves.

**Current state:** the V5 walker (`visit_named_predictors_mut`) enumerates all Predict leaves in a typed module through callback traversal. Everything else — `ProgramGraph`, `DynModule`, `StrategyFactory`, registry, type-validated edges, topological execution — is being built now in V6.

```rust
// Project a typed module into a mutable graph (snapshot — original untouched)
let graph = ProgramGraph::from_module(&module);

// Or build from scratch via registry
let mut graph = ProgramGraph::new();
let cot = registry::create("chain_of_thought", &schema, Default::default())?;
graph.add_node("cot", cot)?;
graph.connect("input", "question", "cot", "question")?;  // edges type-validated
let result = graph.execute(input).await?;

// After optimization, fit back to the typed module
graph.fit(&mut module);
```

**Split** from the paper: a meta planner decides a complex signature should be two steps. It calls `graph.add_node` twice with simpler schemas from `registry::create`, rewires edges with `graph.connect`, removes the original with `graph.replace_node`. Edge type validation catches wiring errors immediately.

**Fuse**: two adjacent nodes with compatible schemas get replaced by a single node with a merged signature. Same mutation APIs.

**The key architectural property**: both the typed path and the graph path use the same `SignatureSchema` → `ChatAdapter` → prompt format pipeline. A `Predict<QA>` and a `registry::create("predict", &qa_schema, ...)` produce identical prompts. The meta planner can restructure the graph without worrying about prompt divergence.

**The cycle**: project → optimize (parameter and/or structural) → fit-back → evaluate → repeat. The graph is the optimizer's scratch space; the user's typed module is the stable interface.

---

## Layer stack

```
You're here          What you touch                What's invisible to you
─────────────────────────────────────────────────────────────────────────
App developer        Signature, module.call()       Everything below
Module author        #[derive(Module)], forward()   Discovery, graph
Optimizer dev        Optimizer::compile internals (`visit_named_predictors_mut`, DynPredictor)  Graph, registry
Meta planner         ProgramGraph, registry          (bottom layer — Section 1.3)
```

Each layer only exists if you need it. Simple usage never instantiates the graph layer.
