<div align='center'>
<img width="768" alt="logo" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs
<em>A high-performance DSPy rewrite in Rust for building LM-powered applications</em>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/dspy-rs)](https://crates.io/crates/dspy-rs)
[![Documentation](https://docs.rs/dspy-rs/badge.svg)](https://docs.rs/dspy-rs)

[Documentation](https://dsrs.herumbshandilya.com) • [API Reference](https://docs.rs/dspy-rs) • [Examples](crates/dspy-rs/examples/) • [Issues](https://github.com/krypticmouse/dsrs/issues) • [Discord](https://discord.com/invite/ZAEGgxjPUe)

</div>

---

## 🚀 Overview

**DSRs** (DSPy Rust) is a ground-up rewrite of the [DSPy framework](https://github.com/stanfordnlp/dspy) in Rust, designed for building robust, high-performance applications powered by Language Models. Unlike a simple port, DSRs leverages Rust's type system, memory safety, and concurrency features to provide a more efficient and reliable foundation for LM applications.

## 📦 Installation

Add DSRs and the dependencies used by the complete example to your `Cargo.toml`:

```toml
[dependencies]
dspy-rs = "0.7.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
anyhow = "1"
serde_json = "1"
```

Or add the DSRs dependency via Cargo:

```bash
cargo add dspy-rs
```

For the shorter import name, use `cargo add dspy-rs --rename dsrs`, or replace the dependency with `dsrs = { package = "dspy-rs", version = "0.7.3" }`. Examples below use `dspy_rs`; change that import to `dsrs` when using the alias.

This README describes the repository checkout. A published release may not yet contain every documented API. To use a local checkout, replace the version dependency with `dspy-rs = { path = "../DSRs/crates/dspy-rs" }`. See the [public export migration notes](docs/docs/api/index.mdx) when upgrading.

The default Cargo features are `data` (CSV, Parquet and Hugging Face ingestion) and `cache` (optional disk response caching). JSON/JSONL loading and bounded execution history also work with `default-features = false`. Response caching is off per LM unless `.cache(true)` is set; that setting requires the `cache` feature.

## 🔧 Quick Start

Define typed inputs and outputs, inspect the prompt, then send that same chat to the LM:

```rust
use dspy_rs::{LM, Predict, Signature};

/// Answer questions accurately and concisely.
#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let inspect_only = std::env::args().any(|arg| arg == "--inspect-only");
    let api_key = if inspect_only {
        "unused-for-prompt-inspection".to_string()
    } else {
        std::env::var("OPENAI_API_KEY")?
    };
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .api_key(api_key)
        .build()
        .await?;
    let predict = Predict::<QA>::builder().lm(lm.clone()).build();
    let input = QAInput {
        question: "What is the capital of France?".into(),
    };

    // Building a chat formats the prompt without sending a provider request.
    let chat = predict.build_chat(&input).await?;
    println!("{}", serde_json::to_string_pretty(&chat.to_json())?);
    if inspect_only {
        return Ok(());
    }

    let (prediction, _conversation) = predict.call_and_parse(chat).await?;
    println!("Answer: {}", prediction.answer);
    for entry in lm.inspect_history(1).await {
        println!("Raw output: {:?}", entry.raw_output);
        println!("Parsed output: {:?}", entry.output);
    }
    Ok(())
}
```

Set `OPENAI_API_KEY` before running the [complete example](crates/dspy-rs/examples/00-quickstart.rs):

```bash
cargo run -p dspy-rs --no-default-features --example 00-quickstart
```

For an offline preview, append `-- --inspect-only`. It formats the prompt and exits without credentials or a provider request.

The derive generates `QAInput` and `QAOutput`. Calls return `Predicted<QAOutput>`: output fields through `Deref`, with raw response, usage and field metadata through `.metadata()`. Use `.into_inner()` to take ownership of the output.

`build_chat` alone does not create a history entry. Executed calls, cache hits and failures do. History is bounded and independent of response caching; see the [LM guide](docs/docs/components/lm.mdx) for capacities and module scopes.

The following snippets reuse the `QA` signature above. Run asynchronous fragments inside an async function returning `anyhow::Result<()>`, and configure a default LM with `configure(lm)` or attach an LM to each predictor.

## 🏗️ Architecture

DSRs follows a modular architecture with clear separation of concerns:

```
crates/
├── dspy-rs/src/
│   ├── core/       # LM clients, Module, Signature, history and errors
│   ├── adapter/    # Prompt rendering and response parsing
│   ├── data/       # Typed dataset loading
│   ├── predictors/ # Predict and few-shot demos
│   ├── modules/    # ChainOfThought and validation wrappers
│   ├── evaluate/   # Evaluation and metrics
│   ├── optimizer/  # Instruction, demo and structural optimization
│   ├── ir/         # Program graph and interpreter
│   └── trace/      # Trace capture, serialization and replay
└── dsrs-macros/     # Signature, schema and program macros
```

### Core Components

#### 1. **Signatures** - Define Input/Output Specifications
```rust
/// Translate the text accurately while preserving meaning.
#[derive(Signature, Clone, Debug)]
struct TranslationSignature {
    #[input]
    pub text: String,

    #[input]
    pub target_language: String,

    #[output]
    pub translation: String,
}
```

#### 2. **Modules** - Composable Pipeline Components

A `Predict<S>` is already a `Module`. Implement `Module::forward` when composing tasks, and use `Module::call` as the caller entry point. Declare optimizer-visible leaves with `predictors!`:

```rust
use dspy_rs::{Module, PredictError, Predicted};

struct QAModule {
    predictor: Predict<QA>,
}

dspy_rs::predictors!(QAModule { predictor });

impl Module for QAModule {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.predictor.call(input).await
    }
}
```

The `predictors!` declaration provides explicit leaf discovery; deriving `Facet` alone does not discover predictor fields. See the [module guide](docs/docs/components/modules.mdx) for a complete composed module.

#### 3. **Predictors** - Pre-built LM Interaction Patterns
```rust
// Get prediction
let predict = Predict::<QA>::new();
```

#### 4. **Language Models** - Configurable LM Backends
```rust
// Configure with OpenAI (API key read from OPENAI_API_KEY env var)
let lm = LM::builder()
    .model("openai:gpt-4o-mini".to_string())
    .temperature(0.7)
    .max_tokens(1000)
    .build()
    .await?;

// Native Ollama (download and serve the model separately).
let lm = LM::builder()
    .model("ollama:llama3.2".to_string())
    .build()
    .await?;
```

For an OpenAI-compatible server, use its API base, such as `http://localhost:8000/v1`, with the server's model name. See the [LM guide](docs/docs/components/lm.mdx) for provider configuration, explicit call overrides and scoped defaults.

#### 5. **Evaluation** - Evaluating your Modules

A metric receives the dataset row, typed prediction and optional execution trace. Here, each row is an `(input, expected output)` tuple:

```rust
use dspy_rs::{Eval, Trace, TypedMetric};

struct ExactMatchMetric;

impl TypedMetric<(QAInput, QAOutput), QAModule> for ExactMatchMetric {
    async fn evaluate(
        &self,
        example: &(QAInput, QAOutput),
        prediction: &Predicted<QAOutput>,
        _trace: Option<&Trace>,
    ) -> anyhow::Result<Eval> {
        let expected = example.1.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();
        Ok(Eval::score((expected == actual) as u8 as f64))
    }
}
```

```rust
use dspy_rs::{average_score, evaluate_trainset};

let trainset = vec![
    (
        QAInput { question: "What is 2+2?".into() },
        QAOutput { answer: "4".into() },
    ),
    (
        QAInput { question: "What is the capital of France?".into() },
        QAOutput { answer: "Paris".into() },
    ),
];
let mut module = QAModule { predictor: Predict::new() };
let metric = ExactMatchMetric;
let outcomes = evaluate_trainset(&module, &trainset, &metric).await?;
println!("Average score: {}", average_score(&outcomes));
```

Evaluation propagates failed module calls and returns results in dataset order. Use `evaluate_trainset_with_concurrency` to set the number of calls in flight; see [Evaluation](docs/docs/components/evaluation.mdx).

#### 6. **Optimization** - Optimize your Modules

Reuse the module, trainset and metric above. Optimizers evaluate candidates and install the selected instructions/demos through `compile_module`.

**COPRO (Collaborative Prompt Optimization)**

```rust
use dspy_rs::COPRO;

let optimizer = COPRO::builder()
    .breadth(4)
    .depth(1)
    .build();
optimizer.compile_module(&mut module, &trainset, &metric).await?;
```

**MIPROv2 (Multi-prompt Instruction Proposal Optimizer v2)**

MIPROv2 collects traces, bootstraps demonstrations, generates instruction candidates and evaluates them:

```rust
use dspy_rs::MIPROv2;

let optimizer = MIPROv2::builder()
    .num_candidates(4)
    .num_trials(4)
    .minibatch_size(2)
    .max_bootstrapped_demos(2)
    .min_demo_score(1.0)
    .seed(42)
    .build();
optimizer.compile_module(&mut module, &trainset, &metric).await?;
```

See [02-module-iteration-and-updation.rs](crates/dspy-rs/examples/02-module-iteration-and-updation.rs) for a complete COPRO workflow and [08-optimize-mipro.rs](crates/dspy-rs/examples/08-optimize-mipro.rs) for MIPROv2 on loaded data. Optimization makes additional LM calls; configure its search size and evaluation concurrency for your workload.

#### 7. **Typed Data Loading** - Load Typed Dataset Rows

`DataLoader` loads row structs, rather than a signature-specific `Example<S>`. Extra source fields are ignored by default; missing required fields return a row/field error. CSV, Parquet and Hugging Face loaders require the `data` feature; JSON/JSONL loading is always available.

For a flat row struct, add `facet = "0.43"` and `serde = { version = "1", features = ["derive"] }` to your dependencies:

```rust
use dspy_rs::{DataLoader, Example, TypedLoadOptions};

#[derive(Example, Clone, Debug, facet::Facet, serde::Serialize, serde::Deserialize)]
#[facet(crate = facet)]
struct QARow {
    question: String,
    answer: String,
}

let rows = DataLoader::load_csv::<QARow>(
    "data/train.csv",
    ',',
    true,
    TypedLoadOptions::default(),
)?;
```

`#[derive(Example)]` projects the row into a signature's generated input/output by field name. Metrics can read the gold fields directly from the row.

For a custom source schema, map rows into the tuples used by `ExactMatchMetric` above:

```rust
let trainset: Vec<(QAInput, QAOutput)> = DataLoader::load_csv_with(
    "data/train.csv",
    ',',
    true,
    TypedLoadOptions::default(),
    |row| {
        Ok((
            QAInput { question: row.get::<String>("prompt")? },
            QAOutput { answer: row.get::<String>("completion")? },
        ))
    },
)?;
```

Mapper overloads bypass shape-driven conversion; their `TypedLoadOptions` argument is accepted for API parity but is not applied. Legacy loaders taking `input_keys`/`output_keys` and `DataLoader::save_json`/`save_csv` are removed. See [Data](docs/docs/components/data.mdx) for mapping, schema checks and errors.

**Component Discovery:**

The following example uses the `Analyze` and `Summarize` signatures defined in the next section. The ordinary label field is not an optimizer leaf:

```rust
use dspy_rs::Predictors;

struct ComplexPipeline {
    analyzer: Predict<Analyze>,
    summarizer: Predict<Summarize>,
    label: String,
}

dspy_rs::predictors!(ComplexPipeline { analyzer, summarizer });

let pipeline = ComplexPipeline {
    analyzer: Predict::new(),
    summarizer: Predict::new(),
    label: "analysis".into(),
};
let visible = pipeline.predictors()
    .into_iter()
    .map(|(name, _)| name)
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

let document = "Acme has friendly staff and quick service.";
let analysis = analyzer.call(AnalyzeInput { text: document.into() }).await?.into_inner();
let summary = summarizer.call(SummarizeInput {
    key_points: analysis.key_points
}).await?;

println!("Sentiment: {}", analysis.sentiment);
println!("Summary: {}", summary.summary);
```

See the [example index](docs/docs/guides/examples.mdx) for runnable programs. Examples 03, 04 and 08 require `data`; example 97 requires `cache`. Both are default features. For example, `cargo run -p dspy-rs --no-default-features --features data --example 03-evaluate-hotpotqa`.

## 🧪 Testing

Run the test suite:

```bash
# All tests
cargo test

# Specific test
cargo test -p dspy-rs --test test_predict_conversation

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

Capture a composed module call to record its predictor spans, prompts, inputs, outputs and usage, then save the trace as JSONL:

```rust
let (result, trace) = dspy_rs::capture(|| {
    module.call(QAInput { question: "What is 2+2?".into() })
}).await;
let prediction = result?;
std::fs::write("run.jsonl", trace.to_jsonl()?)?;
println!("Answer: {}", prediction.answer);
```

See [12-tracing.rs](crates/dspy-rs/examples/12-tracing.rs) and the [trace guide](docs/docs/components/traces.mdx) for capture, per-component inspection and replay. History inspection is a separate bounded record of LM invocations; neither feature is a persistent interaction database.

### Custom validation with feedback

Use `Predict::validated()` for Rust field/cross-field rules that return corrective text. Configure total attempts with `max_attempts`; the default is one. See [Predict](docs/docs/components/predict.mdx) for callback examples, optional soft-check repair and restrictions on retrying attached tools.

### Optimizer Comparison

| Feature | COPRO | MIPROv2 | GEPA |
|---------|-------|---------|------|
| **Search** | Breadth-first instruction search | Trace-guided instruction and demo search | Feedback-guided genetic/Pareto search |
| **Metric** | Numerical scores | Numerical scores | Scores and textual feedback |
| **Few-shot demos** | Instruction tuning only | Bootstraps demos from traced runs | Instruction tuning only |
| **Useful starting point** | Exploring instruction variants | Joint instruction/demo tuning | Using feedback that explains failures |

**When to use COPRO:** Start with a small breadth/depth when exploring instructions. Evaluate the resulting module against a held-out set.

**When to use MIPROv2:** Use representative examples when tuning both demonstrations and instructions. Its candidate and trial counts control the search.

**When to use GEPA:** Supply metric feedback that explains failures. A configured reflection LM uses that feedback to propose instructions; the no-prompt-model path uses a bounded deterministic fallback.

SIMBA and BootstrapFewShot also support typed modules. Structural optimization operates on interpreter-loaded programs through `compile_program`. See [Optimizers](docs/docs/components/optimizers.mdx) for the distinct configuration and report contracts.

Performance harnesses measure their stated workloads. Mocked calls measure framework overhead rather than live inference or task quality; see [cross-framework benchmarks](benchmarks/cross-framework/README.md).

---

## 📈 Project Status

⚠️ **Beta Release** - DSRs is in active development. The API is stabilizing but may have breaking changes.

## 🤝 Contributing

We welcome contributions! Check existing issues and pull requests before starting work. See [documentation development](docs/README.md) for local docs previews and API generation.

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
