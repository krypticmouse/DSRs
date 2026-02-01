# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT:** If ANY information in this file is out of date or misleading, update it immediately as part of your current task. This file must stay accurate.

See also: `@AGENTS.md` files throughout the codebase for directory-specific guidance.

## Overview

DSRs (DSPy Rust) is a ground-up rewrite of the [DSPy framework](https://github.com/stanfordnlp/dspy) in Rust for building LM-powered applications. It uses Rust's type system and proc macros to provide compile-time safety for LM signatures.

## Commands

```bash
# Build
cargo build
cargo build --features rlm  # With Python interop (RLM feature)

# Run tests
cargo test                           # All tests
cargo test -p dspy-rs                # Main crate tests only
cargo test -p baml-bridge            # BAML bridge tests
cargo test test_signature            # Specific test by name
cargo test -- --nocapture            # With output

# Run examples
cargo run --example 01-simple
cargo run --example rlm_trajectory --features rlm  # RLM examples require feature

# Lint and format
cargo fmt -- --check
cargo clippy -- -D warnings
```

## Architecture

### Workspace Structure

```
crates/
├── dspy-rs/          # Main library - signatures, predictors, optimizers, tracing
├── dsrs-macros/      # Proc macros: #[derive(Signature)], #[derive(Optimizable)]
├── baml-bridge/      # LLM output parsing via BAML's jsonish parser + type registration
├── baml-bridge-derive/  # #[derive(BamlType)] for output format generation
├── rlm-core/         # Core traits for RLM type description (RlmDescribe, RlmVariable)
├── rlm-derive/       # #[derive(RlmDescribe)] for Python interop
vendor/baml/          # Vendored BAML crates (jsonish, jinja, types)
```

### Core Flow

1. **Signatures** define input/output specifications via `#[derive(Signature)]`:
   - Fields marked `#[input]` or `#[output]` with optional descriptions, aliases, constraints
   - Generates `{Name}Input` and `__{Name}Output` structs automatically
   - Implements `BamlType` for output parsing and format rendering

2. **Predictors** (`Predict<S>`) execute signatures against LMs:
   - `call(input)` → typed `CallResult` with parsed output
   - `forward(Example)` → untyped `Prediction` (Module trait compatibility)
   - Uses `ChatAdapter` to format prompts and parse responses via BAML

3. **Modules** compose predictors into pipelines:
   - Implement `Module` trait with `async fn forward(&self, Example) -> Result<Prediction>`
   - Mark optimizable components with `#[derive(Optimizable)]` + `#[parameter]`

4. **Optimizers** tune module prompts:
   - `COPRO`: Iterative refinement
   - `MIPROv2`: LLM-guided generation with traces
   - `GEPA`: Gradient-free evolutionary optimization

5. **Tracing** captures execution DAGs for introspection/replay via `trace::trace()`.

### Key Types

- `LM` - Language model client (configured globally via `configure()`)
- `Signature` trait - Compile-time signature metadata
- `BamlType` - Output format generation + parsing
- `Predict<S>` - Generic typed predictor
- `CallResult` - Contains parsed output, constraints, raw fields
- `Example`/`Prediction` - Untyped data containers (for Module trait)

### Feature Flags

- `rlm` - Enables Python interop via PyO3 for REPL-based LM interaction (RlmDescribe, TypedRlm)

## Code Patterns

### Defining Signatures

```rust
#[derive(dspy_rs::Signature, Clone)]
pub struct QA {
    #[input]
    pub question: String,

    /// Description becomes the field prompt hint
    #[output]
    pub reasoning: String,

    #[output]
    #[check("len(this) < 100", label = "short_answer")]
    pub answer: String,
}
```

### Using Predictors

```rust
// Typed call (preferred)
let predict = Predict::<QA>::builder().instruction("Answer concisely.").build();
let result = predict.call(QAInput { question: "...".into() }).await?;
println!("{}", result.output.answer);

// Untyped forward (for Module composition)
let prediction = predict.forward(example).await?;
```

### Constraints

- `#[check("expr", label = "name")]` - Soft validation, result available in `CallResult.checks`
- `#[assert("expr")]` - Hard validation, fails parsing if false

Expressions use BAML's jinja-like syntax with `this` as the field value.

## Testing Notes

- Tests in `crates/*/tests/` use `rstest` for parameterization
- UI tests in `tests/ui/` verify compile-time macro errors via `trybuild`
- Integration tests that call LLMs should be marked `#[ignore]` or gated

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI models
- Model format: `"openai:gpt-4o-mini"` or `"anthropic:claude-3-haiku"` (provider prefix)
