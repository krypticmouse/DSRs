# DSPy Module System: Complete Architecture Reference

> Written for the oxide Rust rewrite. Self-contained -- no DSPy source access required.

## What DSPy Is (In One Paragraph)

DSPy is a framework for programming with language models where you declare *what* you want (via typed signatures), not *how* to prompt. The framework handles prompt construction, output parsing, and -- critically -- automatic optimization of prompts and few-shot examples. The module system is the backbone that makes all of this possible.

## The Core Insight

Everything in DSPy is built on a single primitive: **`Predict`**. A `Predict` takes a typed signature (input fields -> output fields), formats it into a prompt via an adapter, calls an LM, and parses the response back into typed outputs. Every higher-level module (ChainOfThought, ReAct, ProgramOfThought) is just orchestration on top of one or more `Predict` instances.

Optimizers work by discovering all `Predict` instances in a module tree, then modifying their **demos** (few-shot examples) and **signature instructions** (the task description). This is the entire optimization surface.

## Architecture Diagram

```
User Program (a Module subclass)
  |
  |-- Module.__call__()
  |     |-- callbacks, usage tracking, caller stack
  |     |-- self.forward(**kwargs)
  |
  |-- Contains Predict instances (the leaf parameters)
  |     |-- Each Predict has:
  |     |     signature  (Signature class -- typed I/O contract)
  |     |     demos      (list[Example] -- few-shot examples)
  |     |     lm         (optional per-predictor LM override)
  |     |     config     (LM kwargs: temperature, n, etc.)
  |     |
  |     |-- Predict.forward():
  |     |     1. _forward_preprocess: resolve LM, merge config, get demos
  |     |     2. adapter(lm, signature, demos, inputs)
  |     |     3. _forward_postprocess: build Prediction, append to trace
  |     |
  |     |-- Adapter pipeline:
  |           format(signature, demos, inputs) -> messages
  |           lm(messages, **kwargs) -> completions
  |           parse(signature, completion) -> dict of output fields
  |
  |-- named_parameters() walks the tree, finds all Predict instances
  |-- Optimizers modify demos/instructions on discovered Predicts
  |-- save()/load() serializes the optimized state
```

## Document Index

| Document | What It Covers |
|----------|---------------|
| [01_module_system.md](01_module_system.md) | `BaseModule`, `Module`, `Parameter` -- the tree structure, traversal, serialization, copy mechanics, the `_compiled` freeze flag |
| [02_signatures.md](02_signatures.md) | `Signature`, `SignatureMeta`, `InputField`/`OutputField` -- DSPy's type system, string parsing, Pydantic integration, manipulation methods |
| [03_predict.md](03_predict.md) | `Predict` -- the foundation primitive, forward pipeline, preprocessing, tracing, state management |
| [04_augmentation_patterns.md](04_augmentation_patterns.md) | How ChainOfThought, ReAct, ProgramOfThought, MultiChainComparison, BestOfN, Refine build on Predict |
| [05_adapters.md](05_adapters.md) | Adapter base class, ChatAdapter, JSONAdapter -- how signatures become prompts and responses become Predictions |
| [06_optimizers.md](06_optimizers.md) | How optimizers discover modules, what they modify, BootstrapFewShot, MIPRO, COPRO, BootstrapFinetune, the compile() contract, tracing |
| [07_rust_implications.md](07_rust_implications.md) | What all of this means for a Rust implementation -- trait design, type-state patterns, the hard problems |

## Key Terminology

| Term | Meaning |
|------|---------|
| **Module** | A composable unit of computation. Has `__call__` -> `forward()`. Can contain other Modules. |
| **Parameter** | Marker trait. Only `Predict` implements it. Makes a module discoverable by optimizers. |
| **Predict** | The leaf parameter. Holds a signature, demos, and LM config. Calls adapter -> LM -> parse. |
| **Signature** | A typed contract: named input fields -> named output fields, with instructions. Implemented as a Pydantic BaseModel *class* (not instance). |
| **Adapter** | Converts (signature, demos, inputs) -> LM messages and parses responses back. ChatAdapter uses `[[ ## field ## ]]` delimiters. |
| **Demo** | A few-shot example (an `Example` dict with input+output field values). Stored on `Predict.demos`. |
| **Trace** | A list of `(predictor, inputs, prediction)` tuples recorded during execution. Used by optimizers to attribute outputs to predictors. |
| **Compiled** | `module._compiled = True` means optimizers won't recurse into it. Freezes the optimized state. |
| **Teleprompter** | DSPy's name for an optimizer. `compile(student, trainset)` returns an optimized copy. |
| **Example** | Dict-like data container with `.inputs()` / `.labels()` separation. Training data and demos are Examples. |
| **Prediction** | Subclass of Example returned by all modules. Carries completions and LM usage info. |
