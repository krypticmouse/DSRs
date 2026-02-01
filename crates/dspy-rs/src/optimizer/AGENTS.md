# AGENTS.md - optimizer

## Boundary

Prompt optimization algorithms that tune instructions/demos via `compile()`.

**Depends on:** `core::{Module, Optimizable, MetaSignature}`, `evaluate::{Evaluator, FeedbackEvaluator}`, `data::Example`
**Depended on by:** User modules implementing `Module + Optimizable + Evaluator`
**NEVER:** Access LM providers directly; bypass `Optimizable` trait; mutate module state outside `compile()`

---

## How to work here

### The compile() Pattern

All optimizers implement `Optimizer::compile(&self, module: &mut M, trainset: Vec<Example>)` where
`M: Module + Optimizable + Evaluator`. The flow:

1. Call `module.parameters()` to get `HashMap<String, &mut dyn Optimizable>`
2. Read current instructions via `predictor.get_signature().instruction()`
3. Generate/evaluate candidate instructions (optimizer-specific)
4. Apply best via `predictor.update_signature_instruction(new_instruction)?`

### #[parameter] Marking

User modules use `#[derive(Optimizable)]` with `#[parameter]` on predictor fields:

```rust
#[derive(Optimizable)]
struct MyModule {
    #[parameter]          // Exposed to optimizer
    answerer: LegacyPredict,
    helper: SomeOther,    // Not optimized (no #[parameter])
}
```

### Golden Examples

- `examples/04-optimize-hotpotqa.rs` - COPRO on HotpotQA (simple QA optimization)
- `examples/08-optimize-mipro.rs` - MIPROv2 with trace generation and prompting tips

### Key Files

- `mod.rs` - `Optimizer` trait definition
- `copro.rs` - Breadth/depth search with iterative refinement
- `mipro.rs` - Three-stage: traces -> description -> candidates
- `gepa.rs` - Pareto-based evolutionary with LLM reflection
- `pareto.rs` - Per-example dominance tracking for GEPA

---

## Verification

```bash
cargo test -p dspy-rs test_miprov2   # MIPROv2 unit tests (no API)
cargo test -p dspy-rs test_gepa      # GEPA candidate/mutation tests
cargo build --example 04-optimize-hotpotqa -p dspy-rs --features dataloaders
cargo build --example 08-optimize-mipro -p dspy-rs --features dataloaders
```

---

## Don't do this

- **Calling `forward()` without updating instruction first** - Results are meaningless
- **Modifying module outside parameters()** - Use the trait methods
- **Ignoring async boundaries** - All evaluation is async; use `join_all` for batching
- **Hardcoding LM calls** - Use `self.prompt_model` or `get_lm()` pattern

---

## Gotchas

1. **MIPROv2 needs traces** - Stage 1 runs `forward()` on trainset; empty trainset = no candidates
2. **GEPA requires `FeedbackEvaluator`** - Not `Evaluator`. Use `compile_with_feedback()` method
3. **GEPA is gradient-free** - Uses LLM reflection, not backprop; mutation = LLM-generated rewrites
4. **COPRO breadth must be > 1** - Fails fast otherwise (needs candidates to compare)
5. **Optimizers use LegacyPredict internally** - Meta-prompts for instruction generation are legacy
6. **`parameters()` returns mutable refs** - Borrow ends when you drop the HashMap
