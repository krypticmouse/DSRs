<div align='center'>
<img width="768" alt="logo" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs
<em>A high-performance Rust runtime for building typed LM-powered applications</em>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Crates.io](https://img.shields.io/badge/crates-dsrs--core%20%7C%20dsrs--predict-orange)](#crates)
[![Documentation](https://img.shields.io/badge/docs-DSRs-blue)](https://dsrs.herumbshandilya.com)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](#)

[Documentation](https://dsrs.herumbshandilya.com) • [Crates](#crates) • [Examples](crates/dsrs-predict/examples/) • [Issues](https://github.com/krypticmouse/dsrs/issues) • [Discord](https://discord.com/invite/ZAEGgxjPUe)

</div>

---

## Overview

**DSRs** is a ground-up Rust runtime for building robust, high-performance applications powered by language models. It uses Rust's type system, memory safety, and concurrency features to provide a reliable foundation for typed LM pipelines.

## Installation

Depend on the crates you use:

```toml
[dependencies]
dsrs-core = "0.7"
dsrs-lm = "0.7"
dsrs-predict = "0.7"
dsrs-trace = "0.7"
```

Or use cargo:

```bash
cargo add dsrs-core dsrs-lm dsrs-predict dsrs-trace
```

## Quick Start

Here's a simple example to get you started:

```rust
use anyhow::Result;
use dsrs_lm::{configure, ChatAdapter, LM};
use dsrs_macros::Signature;
use dsrs_predict::Predict;
use dsrs_trace::init_tracing;

#[derive(Signature, Clone)]
struct SentimentAnalyzer {
    /// Predict the sentiment of the given text 'Positive', 'Negative', or 'Neutral'.

    #[input]
    pub text: String,

    #[output]
    pub sentiment: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    // API key automatically read from OPENAI_API_KEY env var
    configure(
        LM::builder()
            .model("gpt-4o-mini".to_string())
            .temperature(0.5)
            .build()
            .await?,
        ChatAdapter,
    );

    // Create a predictor
    let predictor = Predict::<SentimentAnalyzer>::new();

    // Prepare typed input
    let input = SentimentAnalyzerInput {
        text: "Acme is a great company with excellent customer service.".to_string(),
    };

    // Execute prediction
    let result = predictor.call(input).await?;

    println!("Answer: {}", result.sentiment);

    Ok(())
}

```
Result:
```
Answer: "Positive"
```

## Crates

DSRs is split into layer-aligned crates. There is no facade crate; depend on the leaf crates directly.

| Crate | Purpose |
|-------|---------|
| `dsrs-core` | Signatures, modules, schema, errors, typed data, and abstract bridge traits. |
| `dsrs-lm` | LM client, client registry, usage accounting, and `ChatAdapter`. |
| `dsrs-predict` | `Predict`, `ChainOfThought`, and ReAct predictors. |
| `dsrs-evaluate` | Evaluation framework, typed metrics, and feedback helpers. |
| `dsrs-gepa` | GEPA optimizer. |
| `dsrs-data` | DataLoader with feature-gated CSV, Parquet, and Hugging Face support. |
| `dsrs-trace` | Execution graph recording and tracing helpers. |
| `dsrs-cache` | Foyer-backed LM cache. |
| `dsrs-leaven` | Leaven integration scaffold. |
| `dsrs-macros` | Derive macros for signatures and field metadata. |

### Core Components

#### 1. **Signatures** - Define Input/Output Specifications
```rust
#[derive(Signature, Clone)]
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
#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
pub struct CustomModule {
    predictor: Predict<TranslationSignature>,
}

impl Module for CustomModule {
    type Input = TranslationSignatureInput;
    type Output = TranslationSignatureOutput;

    async fn forward(&self, input: TranslationSignatureInput) -> Result<Predicted<TranslationSignatureOutput>, PredictError> {
        self.predictor.call(input).await
    }
}
```

#### 3. **Predictors** - Pre-built LM Interaction Patterns
```rust
// Get prediction
let predict = Predict::<MySignature>::new();
```

#### 4. **Language Models** - Configurable LM Backends
```rust
// Configure with OpenAI (API key read from OPENAI_API_KEY env var)
let lm = LM::builder()
    .model("gpt-4o-mini".to_string())
    .temperature(0.7)
    .max_tokens(1000)
    .build()
    .await?;

// For local models (e.g., vLLM, Ollama)
let lm = LM::builder()
    .base_url("http://localhost:11434".to_string())
    .model("llama3".to_string())
    .build()
    .await?;
```

#### 5. **Evaluation** - Evaluating your Modules

```rust
struct ExactMatchMetric;

impl TypedMetric<MySignature, MyModule> for ExactMatchMetric {
    async fn evaluate(
        &self,
        example: &Example<MySignature>,
        prediction: &Predicted<MySignatureOutput>,
    ) -> Result<MetricOutcome> {
        let expected = example.output.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();
        Ok(MetricOutcome::score((expected == actual) as u8 as f32))
    }
}

// Evaluate your module
let test_examples = load_test_data();
let module = MyModule::new();
let metric = ExactMatchMetric;

// Automatically runs predictions and computes average metric
let outcomes = evaluate_trainset(&module, &test_examples, &metric).await?;
let score = average_score(&outcomes);
println!("Average score: {}", score);
```

#### 6. **Optimization** - Optimize your Modules

DSRs keeps GEPA as the active optimizer crate while the Leaven integration is being built out. COPRO and MIPROv2 were deleted with the crate split.

```rust
use dsrs_gepa::GEPAOptimizer;

let optimizer = GEPAOptimizer::builder().build();
optimizer.compile(&mut module, train_examples, &metric).await?;
```

#### 7. **Typed Data Loading** - Ingest Directly Into `Example<S>`

`DataLoader` now provides typed loaders that return `Vec<Example<S>>` directly.
Default behavior is:
- Unknown source fields are ignored.
- Missing signature-required fields return an error with row + field context.

```rust
use dsrs_data::{DataLoader, TypedLoadOptions};
use dsrs_macros::Signature;

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

let trainset = DataLoader::load_csv::<QA>(
    "data/train.csv",
    ',',
    true,
    TypedLoadOptions::default(),
)?;
```

For custom source schemas, use mapper overloads:

```rust
let trainset = DataLoader::load_csv_with::<QA, _>(
    "data/train.csv",
    ',',
    true,
    TypedLoadOptions::default(),
    |row| {
        Ok(dsrs_core::Example::new(
            QAInput {
                question: row.get::<String>("prompt")?,
            },
            QAOutput {
                answer: row.get::<String>("completion")?,
            },
        ))
    },
)?;
```

Migration note:
- Removed legacy raw signatures that required `input_keys` / `output_keys`.
- `save_json` / `save_csv` were removed from `DataLoader`.
- Use typed `load_*` / `load_*_with` APIs.

See the `dsrs-data` crate tests and examples for complete loader coverage.

**Component Discovery:**
```rust
#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
pub struct ComplexPipeline {
    analyzer: Predict<AnalyzeSignature>,
    
    // Additional Predict leaves are also optimizer-visible
    summarizer: Predict<SummarizeSignature>,
    
    // Non-predict fields are ignored by optimizers
    config: Config,
}

let visible = named_parameters_ref(&pipeline)?
    .into_iter()
    .map(|(path, _)| path)
    .collect::<Vec<_>>();
println!("optimizer-visible leaves: {:?}", visible);
```

## 📚 Examples

### Example 1: Multi-Step Pipeline

```rust
#[derive(Signature, Clone, Debug)]
/// Analyze text for sentiment and key points.
struct Analyze {
    #[input] text: String,
    #[output] sentiment: String,
    #[output] key_points: String,
}

#[derive(Signature, Clone, Debug)]
/// Summarize the given key points.
struct Summarize {
    #[input] key_points: String,
    #[output] summary: String,
}

// Chain predictors with typed inputs/outputs
let analyzer = Predict::<Analyze>::new();
let summarizer = Predict::<Summarize>::new();

let analysis = analyzer.call(AnalyzeInput { text: document.into() }).await?;
let summary = summarizer.call(SummarizeInput {
    key_points: analysis.key_points.clone()
}).await?;

println!("Sentiment: {}", analysis.sentiment);
println!("Summary: {}", summary.summary);
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
use dsrs_predict::ChainOfThought;

// ChainOfThought wraps any signature, adding a `reasoning` field
let cot = ChainOfThought::<QA>::new();
let result = cot.call(QAInput {
    question: "What is 2+2?".into(),
}).await?;

println!("Reasoning: {}", result.reasoning);
println!("Answer: {}", result.answer);
```

### Tracing System

DSRs includes a tracing system that captures the dataflow through modules as a Directed Acyclic Graph (DAG). Wrap any execution in `trace::trace()` to capture the graph, then inspect nodes, replay with new inputs via `trace::Executor`, or modify the graph structure.

See `examples/12-tracing.rs` for a complete example.

### GEPA

**When to use GEPA:**
- Tasks where score alone doesn't explain what went wrong
- Need an LLM judge to provide actionable feedback
- Want Pareto-optimal exploration of the instruction space

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
- [Crates](#crates)
- [Examples](crates/dsrs-predict/examples/)
- [GitHub Issues](https://github.com/krypticmouse/dsrs/issues)
- [Discord Community](https://discord.com/invite/ZAEGgxjPUe)
- [Original DSPy Paper](https://arxiv.org/abs/2310.03714)

---

<div align="center">
<strong>Built with 🦀 by the DSPy x Rust community</strong>
<br>
<sub>Star ⭐ this repo if you find it useful!</sub>
</div>
