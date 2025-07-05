# DSRs
A DSPy rewrite in Rust - not a direct port, but a reimplementation with Rust in mind.

## Project Status

⚠️ **This project is in early development** ⚠️

## TODOs

### Core Framework
- [ ] Implement core DSPy abstractions
  - [ ] `Signature` trait and basic implementations (file created: `src/premitives/signature.rs`)
  - [ ] `Module` trait system (file created: `src/premitives/module.rs`)
  - [x] `Example` type (implemented in `src/data/example.rs`)
  - [x] `Prediction` type with `LmUsage` tracking (implemented in `src/data/prediction.rs`)
  - [ ] Field types and validation (file created: `src/premitives/field.rs`)
  - [x] Serialization support (serde integration)
- [ ] Implement language model abstractions
  - [ ] LM trait (file created: `src/premitives/lm.rs`)
  - [ ] RM (Retrieval Model) trait (file created: `src/premitives/rm.rs`)
- [ ] Implement basic modules (files created but empty)
  - [ ] `Predict` module (`src/programs/predict.rs`)
  - [ ] `ChainOfThought` module (`src/programs/cot.rs`)
  - [ ] `ReAct` module (`src/programs/react.rs`)
  - [ ] `Retry` module (`src/programs/retry.rs`)
  - [ ] `Refine` module (`src/programs/refine.rs`)
  - [ ] `BestOfN` module (`src/programs/best_of_n.rs`)
  - [ ] `Parallel` module (`src/programs/parallel.rs`)
- [ ] Implement optimizers
  - [ ] `BootstrapFewShot`
  - [ ] `BootstrapFewShotWithRandomSearch`
  - [ ] `COPRO`
  - [ ] `MIPRO`
- [ ] Implement retrievers
  - [ ] Vector database integrations

### LM Integrations
- [ ] Litellm-like client

### Data Management
- [x] Basic data structures (`Example`, `Prediction`)
- [x] Dataset loading and management
- [ ] Caching layer for LM calls
- [ ] Metrics and evaluation framework

### Developer Experience
- [ ] Fix typo: rename `premitives` to `primitives`
- [ ] Comprehensive documentation
  - [ ] API documentation
  - [ ] Tutorial/guide book
  - [ ] Migration guide from DSPy
- [ ] Examples
  - [ ] Basic usage examples
  - [ ] Complex pipeline examples
  - [ ] Benchmarks vs Python DSPy
- [x] Testing foundation
  - [x] Tests for `Example` type
  - [x] Tests for `Prediction` type
  - [ ] Integration tests
  - [ ] Property-based tests
- [ ] CI/CD
  - [ ] GitHub Actions setup
  - [ ] Code coverage reporting
  - [ ] Automated benchmarks

### Performance Optimizations
- [ ] Implement zero-copy parsing where possible
- [x] Rayon dependency added for parallel execution
- [ ] Batch processing for LM calls
- [ ] Memory pool for frequent allocations
- [ ] SIMD optimizations for embeddings

### Advanced Features
- [ ] Distributed execution support
- [ ] Streaming/async interfaces throughout
- [ ] Plugin system for custom modules
- [ ] Telemetry and observability
- [ ] WebAssembly support

## Current Project Structure

```
dsrs/
├── src/
│   ├── data/            # Data structures (Example, Prediction)
│   ├── premitives/      # Core traits (empty files)
│   ├── programs/        # DSPy modules (empty files)
│   └── lib.rs
├── tests/               # Unit tests
├── Cargo.toml           # Dependencies: rayon, rstest, serde
└── README.md
```
