<div align='center'>
<img width="768" alt="logo" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs
<em>A DSPy rewrite(not a port) with Rust in mind.</em>
</div>

## Project Status

⚠️ **This project is in early development** ⚠️

## TODOs for v0

### Core Framework
- [ ] Implement core DSPy abstractions
  - [x] `Signature` trait and basic implementations (file created: `src/premitives/signature.rs`)
  - [ ] `Module` trait system (file created: `src/premitives/module.rs`)
  - [x] `Example` type (implemented in `src/data/example.rs`)
  - [x] `Prediction` type with `LmUsage` tracking (implemented in `src/data/prediction.rs`)
  - [x] Field types and validation (file created: `src/premitives/field.rs`)
  - [x] Serialization support (serde integration)
- [ ] Implement language model abstractions
  - [x] LM trait (file created: `src/premitives/lm.rs`)
  - [ ] RM (Retrieval Model) trait (file created: `src/premitives/rm.rs`)
- [ ] Implement basic modules (files created but empty)
  - [ ] `Predict` module (`src/programs/predict.rs`)
  - [ ] `ChainOfThought` module (`src/programs/cot.rs`)
  - [ ] `ReAct` module (`src/programs/react.rs`)
  - [ ] `Retry` module (`src/programs/retry.rs`)
  - [ ] `Refine` module (`src/programs/refine.rs`)
  - [ ] `BestOfN` module (`src/programs/best_of_n.rs`)
  - [ ] `Parallel` module (`src/programs/parallel.rs`)

### LM Integrations
- [x] Litellm-like client[Going with open router]

### Data Management
- [x] Basic data structures (`Example`, `Prediction`)
- [x] Dataset loading and management
- [ ] Caching layer for LM calls
- [ ] Metrics and evaluation framework

### Performance Optimizations
- [ ] Implement zero-copy parsing where possible[optim clean up]
- [x] Rayon dependency added for parallel execution
- [ ] Batch processing for LM calls
