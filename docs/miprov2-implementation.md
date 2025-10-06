# MIPROv2 Implementation

## Overview

MIPROv2 (Multi-prompt Instruction Proposal Optimizer v2) is an optimizer that uses LLMs to generate and refine prompts automatically. It differs from COPRO by using an LLM to understand the program and apply prompting best practices, rather than just iterating on existing instructions.

The approach:
- LLM generates descriptions of what your program does
- Incorporates prompting tips and techniques  
- Evaluates candidates systematically

## How it Works

### Stage 1: Trace Generation
```rust
async fn generate_traces<M>(
    &self,
    module: &M,
    examples: &[Example],
) -> Result<Vec<Trace>>
```

- Runs the existing program with training examples
- Captures input/output pairs along with evaluation scores
- Creates a dataset of execution traces that show current behavior

### Stage 2: Candidate Prompt Generation
```rust
async fn generate_candidate_instructions(
    &self,
    program_description: &str,
    traces: &[Trace],
    num_candidates: usize,
) -> Result<Vec<String>>
```

This stage has two sub-steps:

First, an LLM analyzes the signature and traces to generate a program description. Then it uses that description along with prompting tips to generate candidate instructions.

The prompting tips library includes:
- Use clear, specific language
- Consider chain-of-thought for complex tasks
- Specify output formats
- Use role-playing when appropriate
- Handle edge cases explicitly
- Request structured outputs when needed

### Stage 3: Evaluation and Selection
```rust
async fn evaluate_and_select_best<M>(
    &self,
    module: &mut M,
    candidates: Vec<PromptCandidate>,
    eval_examples: &[Example],
    predictor_name: &str,
) -> Result<PromptCandidate>
```

- Evaluates each candidate on a minibatch of examples
- Computes performance scores
- Selects the best performing candidate
- Applies it to the module

## Configuration

Default settings:
```rust
let optimizer = MIPROv2::builder()
    .num_candidates(10)
    .minibatch_size(25)
    .temperature(1.0)
    .track_stats(true)
    .build();
```

You can adjust:
- Number of candidates to generate
- Minibatch size for evaluation
- Temperature for generation diversity
- Whether to display progress stats

## Implementation Notes

The code follows standard Rust practices:
- No unsafe blocks
- Results for error handling with context via anyhow
- Strong types (Trace, PromptCandidate, PromptingTips)
- Builder pattern for configuration
- Async throughout, no blocking calls

Key types:
- `Trace` - Input/output pair with evaluation score
- `PromptCandidate` - Instruction text with score
- `PromptingTips` - Library of best practices

Main methods:
- `generate_traces()` - Run module and capture traces
- `generate_program_description()` - LLM describes the program
- `generate_candidate_instructions()` - LLM generates prompts using tips
- `evaluate_and_select_best()` - Test candidates and pick winner

## Usage Example

```rust
use dspy_rs::{MIPROv2, Optimizer};

// Create optimizer
let optimizer = MIPROv2::builder()
    .num_candidates(10)
    .num_trials(20)
    .minibatch_size(25)
    .build();

// Optimize your module
optimizer.compile(&mut module, train_examples).await?;
```

## COPRO vs MIPROv2

| | COPRO | MIPROv2 |
|---|---|---|
| Approach | Iterative refinement | LLM-guided generation |
| LLM calls | Moderate | High |
| Speed | Faster | Slower |
| Use for | Quick iteration, simple tasks | Best results, complex tasks |

Use MIPROv2 when:
- You have decent training data (15+ examples recommended)
- Quality matters more than speed
- Task benefits from prompting best practices

Use COPRO when:
- You need fast iteration
- Compute budget is limited
- Task is straightforward

## Future Work

Potential additions:

**Few-shot demos** - The `PromptCandidate` struct has a `demos` field that's currently unused. Could integrate demo selection into the optimization.

**Iterative refinement** - Run multiple rounds where good candidates inform the next generation.

**Custom tips** - Let users provide domain-specific prompting tips alongside the defaults.

## Testing

Run tests:
```bash
cargo test optimizer::mipro::tests
```

There are 29 test cases covering trace generation, candidate selection, configuration, and edge cases.

## Example Usage

See `examples/08-optimize-mipro.rs` for a working example. It loads HuggingFace data, measures baseline performance, runs optimization, and shows the improvement.

## References

- [DSPy Framework](https://github.com/stanfordnlp/dspy)
- [DSPy Paper](https://arxiv.org/abs/2310.03714)
