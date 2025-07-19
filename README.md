<div align='center'>
<img width="768" alt="logo" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs
<em>A high-performance DSPy rewrite in Rust for building LLM-powered applications</em>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](#)

</div>

---

## 🚀 Overview

**DSRs** (DSPy Rust) is a ground-up rewrite of the DSPy framework in Rust, designed for building robust, high-performance applications powered by Language Models. Unlike a simple port, DSRs leverages Rust's type system, memory safety, and concurrency features to provide a more efficient and reliable foundation for LLM applications.

## 📦 Installation

Add DSRs to your `Cargo.toml`:

```toml
[dependencies]
dspy-rs = "0.0.1-beta"
```

Or use cargo:

```bash
cargo add dspy-rs
```

## 🔧 Quick Start

Here's a simple example to get you started:

```rust
use dspy_rs::prelude::*;
use std::collections::HashMap;
use indexmap::IndexMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a signature for Q&A
    let signature = Signature::builder()
        .name("QASignature".to_string())
        .instruction("Answer the question concisely and accurately.".to_string())
        .input_fields(IndexMap::from([(
            "question".to_string(),
            Field::InputField {
                prefix: "Question:".to_string(),
                desc: "The question to answer".to_string(),
                format: None,
                output_type: "String".to_string(),
            },
        )]))
        .output_fields(IndexMap::from([(
            "answer".to_string(),
            Field::OutputField {
                prefix: "Answer:".to_string(),
                desc: "The answer to the question".to_string(),
                format: None,
                output_type: "String".to_string(),
            },
        )]))
        .build()?;

    // Create a predictor
    let predictor = Predict { 
        signature: &signature 
    };

    // Prepare input
    let inputs = HashMap::from([
        ("question".to_string(), "What is the capital of France?".to_string())
    ]);

    // Execute prediction
    let result = predictor.forward(inputs, None, None).await?;
    
    println!("Answer: {}", result.get("answer", None));

    Ok(())
}
```

## 🧪 Testing

Run the test suite:

```bash
# All tests
cargo test

# Specific test
cargo test test_predictors

# With output
cargo test -- --nocapture
```

---

## 📈 Project Status

⚠️ **Early Development** - DSRs is actively being developed. The API may change between versions.

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

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original [DSPy](https://github.com/stanfordnlp/dspy) framework
- Built with the amazing Rust ecosystem

<div align="center">

**[Documentation](https://docs.rs/dsrs) • [Examples](examples/) • [Issues](https://github.com/yourusername/dsrs/issues) • [Discussions](https://github.com/yourusername/dsrs/discussions)**

</div>
