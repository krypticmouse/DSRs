<div align='center'>
<img width="768" alt="logo" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs
<em>A high-performance DSPy rewrite in Rust for building LM-powered applications</em>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Crates.io](https://img.shields.io/crates/v/dspy-rs)](https://crates.io/crates/dspy-rs)
[![Documentation](https://docs.rs/dspy-rs/badge.svg)](https://docs.rs/dspy-rs)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](#)

[Documentation](https://dsrs.herumbshandilya.com) • [API Reference](https://docs.rs/dspy-rs) • [Examples](crates/dspy-rs/examples/) • [Issues](https://github.com/krypticmouse/dsrs/issues) • [Discord](https://discord.com/invite/ZAEGgxjPUe)

</div>

---

## 🚀 Overview

**DSRs** (DSPy Rust) is a ground-up rewrite of the [DSPy framework](https://github.com/stanfordnlp/dspy) in Rust, designed for building robust, high-performance applications powered by Language Models. Unlike a simple port, DSRs leverages Rust's type system, memory safety, and concurrency features to provide a more efficient and reliable foundation for LM applications.

## 📦 Installation

Add DSRs to your `Cargo.toml`:

```toml
[dependencies]
# Option 1: Use the shorter alias (recommended)
dsrs = { package = "dspy-rs", version = "0.7.3" }

# Option 2: Use the full name
dspy-rs = "0.7.3"
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
use anyhow::Result;
use dspy_rs::{configure, init_tracing, ChatAdapter, LM, Predict, Signature};

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

## 🏗️ Architecture

DSRs follows a modular architecture with clear separation of concerns:

```
dsrs/
├── core/           # Core abstractions (LM, Module, Signature)
├── adapter/        # LM provider adapters (OpenAI, etc.)
├── data/           # Data structures (Example, Prediction)
├── predictors/     # Built-in predictors (Predict, Chain, etc.)
├── evaluate/       # Evaluation framework and metrics
└── macros/         # Derive macros for signatures
```

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

DSRs provides two powerful optimizers:

**COPRO (Collaborative Prompt Optimization)**
```rust
#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
pub struct MyModule {
    predictor: Predict<MySignature>,
}

// Create and configure the optimizer
let optimizer = COPRO::builder()
    .breadth(10)  // Number of candidates per iteration
    .depth(3)     // Number of refinement iterations
    .build();

// Prepare training data
let train_examples = load_training_data();
let metric = ExactMatchMetric;

// Compile optimizes the module in-place
let mut module = MyModule::new();
optimizer.compile(&mut module, train_examples, &metric).await?;
```

**MIPROv2 (Multi-prompt Instruction Proposal Optimizer v2)** - Advanced optimizer using LLMs
```rust
// MIPROv2 uses a 3-stage process:
// 1. Generate execution traces
// 2. LLM generates candidate prompts with best practices
// 3. Evaluate and select the best prompt

let optimizer = MIPROv2::builder()
    .num_candidates(10)    // Number of candidate prompts to generate
    .num_trials(20)        // Number of evaluation trials
    .minibatch_size(25)    // Examples per evaluation
    .temperature(1.0)      // Temperature for prompt generation
    .build();

optimizer.compile(&mut module, train_examples, &metric).await?;
```

#### 7. **Typed Data Loading** - Ingest Directly Into `Example<S>`

`DataLoader` now provides typed loaders that return `Vec<Example<S>>` directly.
Default behavior is:
- Unknown source fields are ignored.
- Missing signature-required fields return an error with row + field context.

```rust
use dspy_rs::{DataLoader, Signature, TypedLoadOptions};

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
        Ok(dspy_rs::Example::new(
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

See `examples/08-optimize-mipro.rs` for a complete example (requires `parquet` feature).

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
use dspy_rs::ChainOfThought;

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

### Optimizer Comparison

| Feature | COPRO | MIPROv2 | GEPA |
|---------|-------|---------|------|
| **Approach** | Iterative refinement | LLM-guided generation | Evolutionary search with textual feedback |
| **Complexity** | Simple | Advanced | Advanced |
| **Best For** | Quick optimization | Best results | Complex tasks with subtle failure modes |
| **Training Data** | Uses scores | Uses traces & descriptions | Uses rich textual feedback |
| **Prompting Tips** | No | Yes (15+ best practices) | No |
| **Program Understanding** | Basic | LLM-generated descriptions | LLM-judge feedback |
| **Few-shot Examples** | No | Yes (auto-selected) | No |

**When to use COPRO:**
- Fast iteration needed
- Simple tasks
- Limited compute budget

**When to use MIPROv2:**
- Best possible results needed
- Complex reasoning tasks
- Have good training data (15+ examples recommended)

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
- MIPROv2 implementation

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
