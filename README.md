<div align='center'>
<img width="768" alt="logo" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs
<em>A high-performance DSPy rewrite in Rust for building LLM-powered applications</em>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Crates.io](https://img.shields.io/crates/v/dspy-rs)](https://crates.io/crates/dspy-rs)
[![Documentation](https://docs.rs/dspy-rs/badge.svg)](https://docs.rs/dspy-rs)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](#)

[Documentation](https://dsrs.herumbshandilya.com) • [API Reference](https://docs.rs/dspy-rs) • [Examples](crates/dspy-rs/examples/) • [Issues](https://github.com/krypticmouse/dsrs/issues) • [Discord](https://discord.com/invite/ZAEGgxjPUe)

</div>

---

## 🚀 Overview

**DSRs** (DSPy Rust) is a ground-up rewrite of the [DSPy framework](https://github.com/stanfordnlp/dspy) in Rust, designed for building robust, high-performance applications powered by Language Models. Unlike a simple port, DSRs leverages Rust's type system, memory safety, and concurrency features to provide a more efficient and reliable foundation for LLM applications.

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
use anyhow::Result;

#[Signature]
struct QASignature {
    /// You are a helpful assistant that answers questions accurately.
    
    #[input]
    pub question: String,
    
    #[output]
    pub answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Configure your LM (Language Model)
    configure(
        LM::builder()
            .api_key(SecretString::from(std::env::var("OPENAI_API_KEY")?))
            .build(),
        ChatAdapter {},
    );
    
    // Create a predictor
    let predictor = Predict::new(QASignature::new());
    
    // Prepare input
    let example = example! {
        "question": "input" => "What is the capital of France?",
    };
    
    // Execute prediction
    let result = predictor.forward(example).await?;
    
    println!("Answer: {}", result.get("answer", None));
    
    Ok(())
}
```

## 🏗️ Architecture

DSRs follows a modular architecture with clear separation of concerns:

```
dsrs/
├── core/           # Core abstractions (LM, Module, Signature)
├── adapter/        # LLM provider adapters (OpenAI, etc.)
├── data/           # Data structures (Example, Prediction)
├── predictors/     # Built-in predictors (Predict, Chain, etc.)
├── evaluate/       # Evaluation framework and metrics
└── macros/         # Derive macros for signatures
```

### Core Components

#### 1. **Signatures** - Define Input/Output Specifications
```rust
#[Signature(cot)]  // Enable chain-of-thought reasoning
struct TranslationSignature {
    /// Translate the text accurately while preserving meaning
    
    #[input]
    pub text: String,
    
    #[input]
    pub target_language: String,
    
    #[output]
    pub translation: String,
}
```

#### 2. **Modules** - Composable Pipeline Components
```rust
#[derive(Builder)]
pub struct CustomModule {
    predictor: Predict,
}

impl Module for CustomModule {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        // Your custom logic here
        self.predictor.forward(inputs).await
    }
}
```

#### 3. **Predictors** - Pre-built LLM Interaction Patterns
```rust
// Simple prediction
let predict = Predict::new(MySignature::new());

// Chain of thought
let cot_predict = Predict::new(MySignature::new().with_cot());

// Future: More predictors coming
// let chain = Chain::new(vec![step1, step2]);
// let retry = Retry::new(predictor, max_attempts: 3);
```

#### 4. **Language Models** - Configurable LLM Backends
```rust
// Configure with OpenAI
let lm = LM::builder()
    .api_key(secret_key)
    .model("gpt-4")
    .temperature(0.7)
    .max_tokens(1000)
    .build();

// Future: Support for other providers
// .provider(Provider::Anthropic)
// .provider(Provider::Local(model_path))
```

## 📚 Examples

### Example 1: Multi-Step Reasoning Pipeline

```rust
use dsrs::prelude::*;

#[Signature]
struct AnalyzeSignature {
    #[input]
    pub text: String,
    
    #[output]
    pub sentiment: String,
    
    #[output]
    pub key_points: String,
}

#[Signature]
struct SummarizeSignature {
    #[input]
    pub key_points: String,
    
    #[output]
    pub summary: String,
}

#[derive(Builder)]
pub struct AnalysisPipeline {
    analyzer: Predict,
    summarizer: Predict,
}

impl Module for AnalysisPipeline {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        // Step 1: Analyze the text
        let analysis = self.analyzer.forward(inputs).await?;
        
        // Step 2: Summarize key points
        let summary_input = example! {
            "key_points": "input" => analysis.get("key_points", None),
        };
        let summary = self.summarizer.forward(summary_input).await?;
        
        // Combine results
        Ok(prediction! {
            "sentiment" => analysis.get("sentiment", None),
            "key_points" => analysis.get("key_points", None),
            "summary" => summary.get("summary", None),
        })
    }
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

# Run examples
cargo run --example 01-simple
```

## 🛠️ Other Features

### Chain of Thought (CoT) Reasoning
```rust
#[Signature(cot)]  // Enable CoT with attribute
struct ComplexReasoningSignature {
    #[input(desc="Question")
    pub problem: String,
    
    #[output]
    pub solution: String,
}
```

---

## 📈 Project Status

⚠️ **Beta Release** - DSRs is in active development. The API is stabilizing but may have breaking changes.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/krypticmouse/dsrs.git
cd dsrs

# Build the project
cargo build

# Run tests
cargo test

# Run with examples
cargo run --example 01-simple

# Check formatting
cargo fmt -- --check

# Run clippy
cargo clippy -- -D warnings
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original [DSPy](https://github.com/stanfordnlp/dspy) framework
- Built with the amazing Rust ecosystem
- Special thanks to the DSPy community for the discussion and ideas

## 🔗 Resources

- [Documentation](https://dsrs.herumbshandilya.com)
- [API Reference](https://docs.rs/dspy-rs)
- [Examples](crates/dspy-rs/examples/)
- [GitHub Issues](https://github.com/krypticmouse/dsrs/issues)
- [Discord Community](https://discord.com/invite/ZAEGgxjPUe)
- [Original DSPy Paper](https://arxiv.org/abs/2310.03714)

---

<div align="center">
<strong>Built with 🦀 by the DSPy x Rust community</strong>
<br>
<sub>Star ⭐ this repo if you find it useful!</sub>
</div>
