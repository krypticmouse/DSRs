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
# Option 1: Use the shorter alias (recommended)
dsrs = { package = "dspy-rs", version = "0.0.2-beta" }

# Option 2: Use the full name
dspy-rs = "0.0.2-beta"
```

Or use cargo:

```bash
# Option 1: Add with alias (recommended)
cargo add dsrs --package dspy-rs

# Option 2: Add with full name
cargo add dspy-rs
```

## 🔧 Quick Start

Here's a simple example to get you started:

```rust
use dsrs::prelude::*;
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
- [x] Metrics traits and Evaluators
- [x] Signature Macros
- [x] Structured Output Parsing
- [ ] Adding More Predictors
    - [ ] dsrs.Refine
    - [ ] dsrs.BestOfN
    - [ ] dsrs.Retry
- [ ] Retriever Module Support

### LM Integrations
- [ ] Ability to use native provider keys for popular clients

### Data Management
- [ ] Example Macros
- [ ] Support for data loading from sources

### Performance Optimizations
- [ ] Batch processing for LM calls
- [ ] Batch Input support for Module
- [ ] Memory Analysis and Optimization
- [ ] Caching Support for Providers
- [ ] Performance benchmarking b/w DSPy and DSRs

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original [DSPy](https://github.com/stanfordnlp/dspy) framework
- Built with the amazing Rust ecosystem

<div align="center">

**[Documentation](https://docs.rs/dspy-rs) • [Examples](examples/) • [Issues](https://github.com/krypticmouse/dsrs/issues) • [Discussions](https://github.com/krypticmouse/dsrs/discussions)**

</div>
