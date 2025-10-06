# GEPA: Reflective Prompt Optimizer for DSRs

**GEPA** (Genetic-Pareto) is a state-of-the-art reflective prompt optimizer that uses rich textual feedback and evolutionary algorithms to improve LM-powered applications.

> Reference: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025, [arxiv:2507.19457](https://arxiv.org/abs/2507.19457))

## What is GEPA?

GEPA is a reflective optimizer that adaptively evolves textual components (such as prompts) of arbitrary systems. In addition to scalar scores returned by metrics, users can also provide GEPA with text feedback to guide the optimization process. Such textual feedback provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. This allows GEPA to propose high performing prompts in very few rollouts.

## What Makes GEPA Unique?

Unlike traditional optimizers (COPRO, MIPROv2), GEPA introduces several key innovations:

### 1. **Rich Textual Feedback**
Instead of just scalar scores (0.8, 0.9), GEPA leverages detailed explanations:
```
✗ Incorrect classification
  Expected: "positive"
  Predicted: "negative"
  Input text: "Great product but shipping was slow"
  ✗ May have misunderstood mixed sentiment
```

### 2. **Pareto-based Selection**
GEPA maintains a diverse set of candidates that excel on different examples, preventing premature convergence:
- Candidate A: Best on examples 1, 3, 5
- Candidate B: Best on examples 2, 4, 6
- Both stay in the population (complementary strengths)

### 3. **LLM-driven Reflection**
Uses LLMs to analyze execution traces and propose targeted improvements:
```
"The current instruction doesn't handle mixed sentiments well. 
Suggest modifying to explicitly consider both positive and negative aspects..."
```

### 4. **Inference-Time Search**
Can optimize at test time, not just training time.

---

## Quick Start

### 1. Implement FeedbackEvaluator

```rust
use dspy_rs::*;

#[derive(Builder, Optimizable)]
struct MyModule {
    #[parameter]
    predictor: Predict,
}

impl Module for MyModule {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        self.predictor.forward(inputs).await
    }
}

// Implement regular Evaluator for non-GEPA optimizers
impl Evaluator for MyModule {
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        let feedback = self.feedback_metric(example, prediction).await;
        feedback.score
    }
}

// Implement FeedbackEvaluator for GEPA
impl FeedbackEvaluator for MyModule {
    async fn feedback_metric(&self, example: &Example, prediction: &Prediction) 
        -> FeedbackMetric 
    {
        let predicted = prediction.get("answer", None).as_str().unwrap_or("");
        let expected = example.get("expected", None).as_str().unwrap_or("");
        
        let correct = predicted == expected;
        let score = if correct { 1.0 } else { 0.0 };
        
        let feedback = if correct {
            format!("✓ Correct answer: {}", predicted)
        } else {
            format!("✗ Incorrect\n  Expected: {}\n  Predicted: {}", expected, predicted)
        };
        
        FeedbackMetric::new(score, feedback)
    }
}
```

### 2. Configure and Run GEPA

```rust
let gepa = GEPA::builder()
    .num_iterations(20)
    .minibatch_size(25)
    .num_trials(10)
    .temperature(0.9)
    .track_stats(true)
    .max_rollouts(Some(500))  // Budget control
    .build();

let result = gepa.compile_with_feedback(&mut module, trainset).await?;

println!("Best score: {:.3}", result.best_candidate.average_score());
println!("Best instruction: {}", result.best_candidate.instruction);
```

---

## Feedback Helpers

DSRs provides utilities for common feedback patterns:

### Document Retrieval
```rust
use dspy_rs::feedback_helpers::retrieval_feedback;

let feedback = retrieval_feedback(
    &retrieved_docs,
    &expected_docs,
    Some(&all_available_docs)
);

// Output:
// Retrieved 3/5 correct documents (Precision: 0.6, Recall: 0.6, F1: 0.6)
// ✓ Correctly retrieved: doc1, doc2, doc3
// ✗ Missed: doc4, doc5
```

### Code Generation
```rust
use dspy_rs::feedback_helpers::{code_pipeline_feedback, CodeStage, StageResult};

let stages = vec![
    (CodeStage::Parse, StageResult::Success),
    (CodeStage::Compile, StageResult::Success),
    (CodeStage::Execute, StageResult::Failure {
        error: "Division by zero on line 42".to_string(),
    }),
];

let feedback = code_pipeline_feedback(&stages, 0.6);

// Output:
// ✓ Parse: Success
// ✓ Compile: Success
// ✗ Execute: Division by zero on line 42
```

### Multi-Objective Optimization
```rust
use dspy_rs::feedback_helpers::multi_objective_feedback;

let mut objectives = HashMap::new();
objectives.insert("accuracy".to_string(), (0.9, "High accuracy".to_string()));
objectives.insert("latency".to_string(), (0.7, "Acceptable latency".to_string()));
objectives.insert("cost".to_string(), (0.8, "Within budget".to_string()));

let mut weights = HashMap::new();
weights.insert("accuracy".to_string(), 2.0);  // Double weight for accuracy
weights.insert("latency".to_string(), 1.0);
weights.insert("cost".to_string(), 1.0);

let feedback = multi_objective_feedback(&objectives, Some(&weights));

// Output:
// [accuracy] Score: 0.9 - High accuracy
// [cost] Score: 0.8 - Within budget
// [latency] Score: 0.7 - Acceptable latency
// Overall: 0.85 (weighted average)
```

### String Similarity
```rust
use dspy_rs::feedback_helpers::string_similarity_feedback;

let feedback = string_similarity_feedback("Hello world", "Hello World");

// Output:
// ✓ Match ignoring case (minor formatting difference)
```

### Classification
```rust
use dspy_rs::feedback_helpers::classification_feedback;

let feedback = classification_feedback("positive", "negative", Some(0.85));

// Output:
// ✗ Incorrect classification
//   Expected: "negative"
//   Predicted: "positive"
//   Confidence: 0.850
```

---

## Configuration Options

```rust
GEPA::builder()
    .num_iterations(20)          // Number of evolutionary iterations
    .minibatch_size(25)          // Examples per rollout
    .num_trials(10)              // Trials per candidate evaluation
    .temperature(0.9)            // LLM temperature for mutations
    .track_stats(true)           // Track detailed statistics
    .track_best_outputs(false)   // Store best outputs per example
    .max_rollouts(Some(500))     // Budget: max rollouts
    .max_lm_calls(Some(1000))    // Budget: max LM calls
    .prompt_model(Some(lm))      // Separate LM for meta-prompting
    .valset(Some(examples))      // Validation set (default: trainset)
    .build()
```

---

## Understanding GEPA Results

```rust
let result = gepa.compile_with_feedback(&mut module, trainset).await?;

// Best candidate found
println!("Best instruction: {}", result.best_candidate.instruction);
println!("Average score: {:.3}", result.best_candidate.average_score());
println!("Generation: {}", result.best_candidate.generation);

// Resource usage
println!("Total rollouts: {}", result.total_rollouts);
println!("Total LM calls: {}", result.total_lm_calls);

// Evolution over time
for (generation, score) in &result.evolution_history {
    println!("Gen {}: {:.3}", generation, score);
}

// Pareto frontier statistics
for (i, stats) in result.frontier_history.iter().enumerate() {
    println!("Iteration {}: {} candidates, coverage: {:.1}", 
        i, stats.num_candidates, stats.avg_coverage);
}
```

---

## 🏗️ Architecture Deep Dive

### Core Components

#### 1. **FeedbackMetric**
```rust
pub struct FeedbackMetric {
    pub score: f32,                                    // Numerical score
    pub feedback: String,                              // Rich explanation
    pub metadata: HashMap<String, serde_json::Value>,  // Structured data
}
```

#### 2. **ExecutionTrace**
```rust
pub struct ExecutionTrace {
    pub inputs: Example,
    pub outputs: Option<Prediction>,
    pub feedback: Option<FeedbackMetric>,
    pub intermediate_steps: Vec<(String, serde_json::Value)>,
    pub errors: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}
```

#### 3. **ParetoFrontier**
Maintains candidates using per-example dominance:
- Each candidate tracks which examples it wins on
- Sampling is proportional to coverage (# examples won)
- Automatically prunes dominated candidates

#### 4. **GEPACandidate**
```rust
pub struct GEPACandidate {
    pub id: usize,
    pub instruction: String,
    pub module_name: String,
    pub example_scores: Vec<f32>,
    pub parent_id: Option<usize>,
    pub generation: usize,
}
```

### Evolutionary Algorithm

The GEPA algorithm follows these steps:

1. **Initialize** the candidate pool with the unoptimized program
2. **Iterate**:
   - **Sample a candidate** from Pareto frontier (proportional to coverage)
   - **Sample a minibatch** from the training set
   - **Collect execution traces + feedback** for module rollout on minibatch
   - **Select a module** of the candidate for targeted improvement
   - **LLM Reflection**: Propose a new instruction/prompt for the targeted module using reflective meta-prompting and the gathered feedback
   - **Roll out the new candidate** on the minibatch; if improved, evaluate on Pareto validation set
   - **Update the candidate pool/Pareto frontier**
   - **[Optional] System-aware merge/crossover**: Combine best-performing modules from distinct lineages
3. **Continue** until rollout or metric budget is exhausted
4. **Return** candidate with best aggregate performance on validation

---

## Implementing Feedback Metrics

A well-designed metric is central to GEPA's sample efficiency and learning signal richness. The DSRs implementation expects the metric to return a `FeedbackMetric` struct with both a score and rich textual feedback. GEPA leverages natural language traces from LM-based workflows for optimization, preserving intermediate trajectories and errors in plain text rather than reducing them to numerical rewards. This mirrors human diagnostic processes, enabling clearer identification of system behaviors and bottlenecks.

### Practical Recipe for GEPA-Friendly Feedback

- **Leverage Existing Artifacts**: Use logs, unit tests, evaluation scripts, and profiler outputs; surfacing these often suffices
- **Decompose Outcomes**: Break scores into per-objective components (e.g., correctness, latency, cost, safety) and attribute errors to steps
- **Expose Trajectories**: Label pipeline stages, reporting pass/fail with salient errors (e.g., in code generation pipelines)
- **Ground in Checks**: Employ automatic validators (unit tests, schemas, simulators) or LLM-as-a-judge for non-verifiable tasks
- **Prioritize Clarity**: Focus on error coverage and decision points over technical complexity

### Feedback Examples by Domain

- **Document Retrieval** (e.g., HotpotQA): List correctly retrieved, incorrect, or missed documents, beyond mere Recall/F1 scores
- **Multi-Objective Tasks** (e.g., PUPA): Decompose aggregate scores to reveal contributions from each objective, highlighting tradeoffs (e.g., quality vs. privacy)
- **Stacked Pipelines** (e.g., code generation: parse → compile → run → profile → evaluate): Expose stage-specific failures; natural-language traces often suffice for LLM self-correction

---

## Best Practices

### 1. **Design Feedback for Actionability**
```rust
// BAD: Vague feedback
FeedbackMetric::new(0.5, "Wrong answer")

// GOOD: Specific, actionable feedback
FeedbackMetric::new(0.5, 
    "✗ Incorrect answer\n\
     Expected: 'Paris'\n\
     Predicted: 'France'\n\
     Issue: Returned country instead of city")
```

### 2. **Leverage Domain Knowledge**
```rust
// Code generation: Show stage-specific failures
// Retrieval: List specific documents missed
// QA: Explain reasoning errors
```

### 3. **Balance Feedback Detail**
- Too brief: Not actionable
- Too verbose: Drowns out signal
- Sweet spot: 2-5 lines per issue

### 4. **Use Metadata for Structured Analysis**
```rust
FeedbackMetric::new(score, feedback)
    .add_metadata("error_type", json!("parsing"))
    .add_metadata("line_number", json!(42))
    .add_metadata("latency_ms", json!(250))
```

### 5. **Set Realistic Budgets**
```rust
// For development/testing
GEPA::builder()
    .num_iterations(5)
    .max_rollouts(Some(100))
    .build()

// For production optimization
GEPA::builder()
    .num_iterations(20)
    .max_rollouts(Some(1000))
    .build()
```

---

## 📦 Examples

- **[09-gepa-sentiment.rs](../crates/dspy-rs/examples/09-gepa-sentiment.rs)**: Sentiment analysis with rich feedback
- See [GEPA.md](../GEPA.md) for paper details and advanced features

---

## 🔬 Comparison with Other Optimizers

| Feature              | COPRO | MIPROv2 | GEPA |
|----------------------|-------|---------|------|
| **Feedback Type**    | Score | Score   | Score + Text |
| **Selection Strategy** | Best | Batch | Pareto |
| **Diversity**        | Low   | Medium  | High |
| **Actionability**    | Low   | Medium  | High |
| **Compute Cost**     | Low   | Medium  | Medium-High |
| **Sample Efficiency** | Medium | High | Very High |

### When to Use GEPA

- Complex tasks with subtle failure modes
- When you can provide rich feedback
- Multi-objective optimization
- Need for diverse solutions
- Inference-time search

### When to Use Alternatives

- **COPRO**: Simple tasks, quick iteration
- **MIPROv2**: Best prompting practices, single objective

---

## 🐛 Troubleshooting

### Issue: "GEPA requires FeedbackEvaluator trait"
```rust
// Solution: Implement both Evaluator and FeedbackEvaluator
impl Evaluator for MyModule {
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        self.feedback_metric(example, prediction).await.score
    }
}

impl FeedbackEvaluator for MyModule {
    async fn feedback_metric(&self, example: &Example, prediction: &Prediction) 
        -> FeedbackMetric { ... }
}
```

### Issue: Slow convergence
```rust
// Increase minibatch size for better gradient
GEPA::builder().minibatch_size(50).build()

// Increase temperature for more exploration
GEPA::builder().temperature(1.2).build()
```

### Issue: Running out of budget
```rust
// Reduce iterations or increase budget
GEPA::builder()
    .num_iterations(10)           // Fewer iterations
    .max_rollouts(Some(2000))     // Higher budget
    .build()
```

---

## 🚦 Next Steps

1. Read the [Quick Start](#quick-start)
2. Run the sentiment analysis example
3. Implement FeedbackEvaluator for your use case
4. Use feedback helpers for common patterns
5. Experiment with configuration
6. Monitor evolution history and Pareto statistics

---

## Implementation Details

### Statistics

| Component | Lines of Code | Tests | Status |
|-----------|---------------|-------|--------|
| Core Data Structures | 301 | 4 | Complete |
| Pareto Frontier | 361 | 5 | Complete |
| GEPA Optimizer | 563 | 2 | Complete |
| Feedback Helpers | 458 | 7 | Complete |
| Example | 240 | - | Complete |
| Documentation | ~900 | - | Complete |
| **Total** | **~2800** | **18** | **Complete** |

### Files Created

1. `crates/dspy-rs/src/evaluate/feedback.rs` - Rich feedback structures
2. `crates/dspy-rs/src/evaluate/feedback_helpers.rs` - Helper utilities
3. `crates/dspy-rs/src/optimizer/pareto.rs` - Pareto frontier implementation
4. `crates/dspy-rs/src/optimizer/gepa.rs` - GEPA optimizer
5. `crates/dspy-rs/examples/09-gepa-sentiment.rs` - Example usage
6. `crates/dspy-rs/tests/test_gepa.rs` - GEPA tests
7. `crates/dspy-rs/tests/test_pareto.rs` - Pareto frontier tests

### Dependencies Added

```toml
rand = "0.8.5"  # For coverage-weighted sampling
```

### Key Features

- Per-example Pareto frontier (not simplified aggregate)
- LLM reflection on execution traces
- Budget controls (max rollouts, max LM calls)
- Module selection support for multi-module programs
- Evolution history tracking
- Inference-time search capability
- Parallel evaluation for speed
- Comprehensive feedback helper library

---

## Inference-Time Search

GEPA can act as a test-time/inference search mechanism. By setting your `valset` to your evaluation batch and using `track_best_outputs=True`, GEPA produces for each batch element the highest-scoring outputs found during the evolutionary search.

```rust
let gepa = GEPA::builder()
    .track_stats(true)
    .track_best_outputs(true)
    .valset(Some(my_tasks.clone()))
    .build();

let result = gepa.compile_with_feedback(&mut module, my_tasks).await?;

// Access per-task best scores and outputs
let best_scores = result.highest_score_achieved_per_val_task;
let best_outputs = result.best_outputs_valset;
```

---

## Additional Resources

- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [GEPA GitHub](https://github.com/gepa-ai/gepa) - Core GEPA evolution pipeline
- [DSRs Documentation](https://dsrs.herumbshandilya.com)
- [API Reference](https://docs.rs/dspy-rs)
- [Examples Directory](../crates/dspy-rs/examples/)

---

Built with Rust for the DSPy x Rust community
