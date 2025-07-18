<div align='center'>
<img width="768" alt="logo" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs
<em>A DSPy rewrite(not a port) with Rust in mind.</em>
</div>

## Project Status

⚠️ **This project is in early development** ⚠️

## TODOs for v0

### Core Framework
- [x] Implement core DSPy abstractions
  - [x] `Signature` trait and basic implementations (file created: `src/premitives/signature.rs`)
  - [x] `Module` trait system (file created: `src/premitives/module.rs`)
  - [x] `Example` type (implemented in `src/data/example.rs`)
  - [x] `Prediction` type with `LmUsage` tracking (implemented in `src/data/prediction.rs`)
  - [x] Field types and validation (file created: `src/premitives/field.rs`)
  - [x] Serialization support (serde integration)
- [x] Implement language model abstractions
  - [x] LM trait (file created: `src/premitives/lm.rs`)
- [x] Implement basic modules (files created but empty)
  - [x] `Predict` module (`src/programs/predict.rs`)
  - [x] `ChainOfThought` module (`src/programs/cot.rs`)

### LM Integrations
- [x] Litellm-like client[Going with open router]

### Data Management
- [x] Basic data structures (`Example`, `Prediction`)
- [x] Dataset loading and management

### Performance Optimizations
- [ ] Implement zero-copy parsing where possible[optim clean up]
- [x] Rayon dependency added for parallel execution
- [ ] Batch processing for LM calls
