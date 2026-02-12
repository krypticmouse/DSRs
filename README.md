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
#[derive(Builder)]
pub struct CustomModule {
    predictor: Predict<TranslationSignature>,
}

impl Module for CustomModule {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        // Your custom logic here
        self.predictor.forward(inputs).await
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

### Tracing System

The tracing system allows you to capture the dataflow through modules and build a Directed Acyclic Graph (DAG) representation of the execution flow.

#### Overview

The tracing system consists of:

1. **Graph**: A DAG structure representing nodes (modules/predictors) and edges (data dependencies)
2. **Trace Context**: Captures execution traces and builds the DAG using `tokio::task_local`
3. **Executor**: Executes captured graphs with new inputs

#### Basic Usage

Use `trace::trace()` to wrap your execution and capture the DAG:

```rust
use dspy_rs::{trace, example, Predict, Signature};

#[Signature]
struct QASignature {
    #[input]
    pub question: String,
    #[output]
    pub answer: String,
}

let predictor = Predict::new(QASignature::new());
let example = example! {
    "question": "input" => "Hello",
};

// Trace the execution
let (result, graph) = trace::trace(|| async {
    predictor.forward(example).await
}).await;

// Inspect the graph
println!("Graph Nodes: {}", graph.nodes.len());
for node in &graph.nodes {
    println!("Node {}: Type={:?}, Inputs={:?}", node.id, node.node_type, node.inputs);
}

// Execute the graph with new input
let executor = trace::Executor::new(graph);
let new_input = example! {
    "question": "input" => "What is the capital of France?",
};
let predictions = executor.execute(new_input).await?;
```

#### Tracked Values

When building pipelines, use `get_tracked()` to preserve data lineage:

```rust
let prediction = predictor.forward(inputs).await?;
let answer = prediction.get_tracked("answer"); // Preserves source node info

// The example! macro automatically detects tracked values and records Map nodes
let next_input = example! {
    "answer": "input" => answer.clone(),
};
```

#### Graph Structure

**Node**: Represents a single execution step:
- `id`: Unique identifier
- `node_type`: Type of node (`Root`, `Predict`, `Map`, `Operator`)
- `inputs`: IDs of parent nodes
- `output`: Output Prediction
- `input_data`: Input Example (for root nodes)

**Graph**: Contains all nodes and provides execution capabilities:
- `nodes`: Vector of all nodes
- `Executor`: Can execute the graph with new inputs

#### Modifying the Graph

The graph is fully modifiable - you can:
- Split nodes (add intermediate steps)
- Remove nodes
- Fuse nodes (combine operations)
- Insert nodes between existing ones
- Modify node configurations (signatures, instructions)

```rust
// Example: Modify a node's signature
if let Some(node) = graph.nodes.get_mut(1) {
    if let NodeType::Predict { signature, .. } = &mut node.node_type {
        // Modify signature instruction, demos, etc.
    }
}
```

#### Example

See `examples/12-tracing.rs` for a complete example demonstrating:
- Tracing module execution
- Inspecting the DAG
- Executing graphs with new inputs
- Modifying graph structure

### Optimizer Comparison

| Feature | COPRO | MIPROv2 |
|---------|-------|---------|
| **Approach** | Iterative refinement | LLM-guided generation |
| **Complexity** | Simple | Advanced |
| **Best For** | Quick optimization | Best results |
| **Training Data** | Uses scores | Uses traces & descriptions |
| **Prompting Tips** | No | Yes (15+ best practices) |
| **Program Understanding** | Basic | LLM-generated descriptions |
| **Few-shot Examples** | No | Yes (auto-selected) |

**When to use COPRO:**
- Fast iteration needed
- Simple tasks
- Limited compute budget

**When to use MIPROv2:**
- Best possible results needed
- Complex reasoning tasks
- Have good training data (15+ examples recommended)

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
