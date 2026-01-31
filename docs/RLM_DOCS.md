# RLM (Recursive Language Model) - Comprehensive WIP Documentation

> **Status**: WIP - Work in Progress
> **Last Updated**: 2026-01-31

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Concepts](#core-concepts)
4. [Usage Guide](#usage-guide)
5. [Type System](#type-system)
6. [Serialization & Storage](#serialization--storage)
7. [Configuration](#configuration)
8. [Integration Points](#integration-points)
9. [Examples](#examples)

---

## Overview

RLM (Recursive Language Model) is an agentic execution module that enables LLMs to iteratively solve problems through a Python REPL environment. Instead of generating answers in one shot, the LLM:

1. **Explores** input data by writing Python code
2. **Iterates** based on observed outputs
3. **Uses sub-LLM calls** for semantic analysis when needed
4. **SUBMITs** the final answer when ready

This approach allows for complex data processing, multi-step reasoning, and verified outputs with constraint checking.

### Key Features

- **Typed inputs/outputs** via DSRs Signatures
- **Python REPL execution** with PyO3
- **Sub-LLM queries** from Python code (`llm_query`, `llm_query_batched`)
- **Constraint validation** (soft checks + hard assertions)
- **Extraction fallback** when max iterations reached
- **Full trajectory serialization** with UUIDs and timestamps
- **Rich type descriptions** via `#[rlm_type]` derive macro

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Rlm<S>                                │
│  (Generic over Signature S with typed Input/Output)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ Predict<        │     │ Predict<        │                    │
│  │  RlmActionSig>  │     │  RlmExtractSig  │                    │
│  │                 │     │     <S>>        │                    │
│  │ "What code      │     │ "Extract final  │                    │
│  │  next?"         │     │  outputs from   │                    │
│  └────────┬────────┘     │  trajectory"    │                    │
│           │              └────────┬────────┘                    │
│           │                       │                             │
│           ▼                       │                             │
│  ┌─────────────────┐              │                             │
│  │  Python REPL    │              │                             │
│  │  (PyO3)         │              │                             │
│  │                 │              │                             │
│  │  - Input vars   │              │                             │
│  │  - llm_query()  │              │                             │
│  │  - SUBMIT()     │              │                             │
│  └────────┬────────┘              │                             │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    RlmResult<S>                             ││
│  │  - input: S::Input                                          ││
│  │  - output: S::Output                                        ││
│  │  - trajectory: REPLHistory (id, created_at, entries)        ││
│  │  - field_metas: IndexMap<String, FieldMeta>                 ││
│  │  - iterations, llm_calls, extraction_fallback               ││
│  │  - constraint_summary                                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
crates/
├── rlm-core/           # Core traits (no PyO3 dependency)
│   ├── describe.rs     # RlmDescribe trait for type introspection
│   ├── variable.rs     # RlmVariable for prompt descriptions
│   └── input.rs        # RlmInputFields trait (PyO3 integration)
│
├── rlm-derive/         # Proc macros
│   ├── lib.rs          # #[rlm_type] and #[derive(RlmType)]
│   └── generators/     # Code generation for getters, iter, etc.
│
└── dspy-rs/src/rlm/    # Main implementation
    ├── mod.rs          # Public exports
    ├── rlm.rs          # Rlm struct and RlmBuilder
    ├── config.rs       # RlmConfig and RlmResult
    ├── history.rs      # REPLHistory and REPLEntry
    ├── storage.rs      # StorableRlmResult for serialization
    ├── exec.rs         # Python code execution
    ├── submit.rs       # SUBMIT handler and validation
    ├── tools.rs        # LlmTools (llm_query from Python)
    ├── adapter.rs      # RlmAdapter for prompt generation
    ├── prompt.rs       # Prompt templates
    ├── signatures.rs   # RlmActionSig, RlmExtractSig<S>
    └── error.rs        # RlmError enum
```

---

## Core Concepts

### 1. The RLM Loop

```rust
async fn call(&self, input: S::Input) -> Result<RlmResult<S>, RlmError> {
    // 1. Setup Python globals with input variables + tools
    let globals = setup_globals::<S>(&input, &tools, &submit_handler)?;
    let mut history = REPLHistory::new();  // Gets UUID + timestamp

    while iterations < max_iterations {
        // 2. Ask LLM: "What code should I run next?"
        let action = self.generate_action.call(RlmActionSigInput {
            variables_info,      // Type descriptions of inputs
            repl_history,        // Previous code + outputs
            iteration: "3/20",   // Current iteration
        }).await?;

        // 3. Execute the code in Python
        let output = execute_repl_code(&globals, &action.code, max_chars)?;

        // 4. Check if SUBMIT was called
        if let Some(result) = take_submit_result(&submit_rx) {
            // Validation passed! Return typed output
            return Ok(RlmResult::new(input, typed_output, ...));
        }

        // 5. Append to history and continue
        history = history.append_with_reasoning(reasoning, code, output);
    }

    // 6. Fallback: extract outputs from trajectory
    self.extraction_fallback(input, context).await
}
```

### 2. SUBMIT Handler

The `SUBMIT()` function is injected into Python and validates outputs:

```python
# In the REPL, the LLM calls:
SUBMIT(summary="...", count=42)
```

This triggers:
1. **Field validation** - checks all required fields are present
2. **Type conversion** - converts Python values to Rust types via BamlValue
3. **Constraint evaluation** - runs `#[check(...)]` and `#[assert(...)]` expressions
4. **Result signaling** - signals success/error back to Rust

### 3. Trajectory Identity

Each RLM execution has a unique identity preserved throughout:

```rust
pub struct REPLHistory {
    pub id: Uuid,              // Unique trajectory ID (generated once)
    pub created_at: DateTime<Utc>,  // When trajectory started
    pub entries: Vec<REPLEntry>,
}

pub struct REPLEntry {
    pub reasoning: String,
    pub code: String,
    pub output: String,
    pub timestamp: DateTime<Utc>,  // When this step executed
}
```

The `append()` method preserves `id` and `created_at`:

```rust
impl REPLHistory {
    pub fn append(&self, code: String, output: String) -> Self {
        Self {
            id: self.id,           // Preserved!
            created_at: self.created_at,  // Preserved!
            entries: ...,
        }
    }
}
```

---

## Usage Guide

### Basic Usage

```rust
use dspy_rs::{configure, Signature, LM, ChatAdapter};
use dspy_rs::rlm::Rlm;

// 1. Define a signature
#[derive(Signature, Clone, Debug)]
struct Summarize {
    #[input]
    text: String,

    #[output]
    #[check("len(this) >= 10", label = "min_length")]
    summary: String,
}

// 2. Configure DSRs
let lm = LM::builder()
    .model("openai:gpt-4o")
    .build()
    .await?;
configure(lm, ChatAdapter);

// 3. Create and call RLM
let rlm = Rlm::<Summarize>::new();
let result = rlm.call(SummarizeInput {
    text: "Long text here...".into(),
}).await?;

// 4. Access results
println!("Summary: {}", result.output.summary);
println!("Iterations: {}", result.iterations);
println!("Is fallback: {}", result.is_fallback());
```

### Builder Pattern

```rust
let rlm = Rlm::<Summarize>::builder()
    .max_iterations(10)
    .max_llm_calls(20)
    .enable_extraction_fallback(true)
    .strict_assertions(false)
    .max_output_chars(50_000)
    .with_lm(custom_lm)  // Override global LM
    .build();
```

### Using RlmResult

```rust
let result: RlmResult<Summarize> = rlm.call(input).await?;

// Access typed output
let summary: &String = &result.output.summary;

// Check execution stats
println!("Iterations: {}", result.iterations);
println!("LLM calls: {}", result.llm_calls);
println!("Fallback: {}", result.extraction_fallback);

// Check constraint results
if result.has_constraint_warnings() {
    for check in result.failed_checks() {
        println!("Warning: {} - {}", check.label, check.expression);
    }
}

// Convert to full signature if needed
let full: Summarize = result.to_signature();

// Serialize for storage
let storable = result.to_storable()?;
```

---

## Type System

### #[rlm_type] Macro

The `#[rlm_type]` attribute macro generates:
- `#[pyclass]` for PyO3 integration
- `#[derive(BamlType, RlmType)]` for serialization and introspection
- Field getters for Python access
- `__repr__`, `__len__`, `__iter__`, `__getitem__` as configured

```rust
#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(
    repr = "Trajectory({len(self.steps)} steps)",
    iter = "steps",      // Enables __len__ and __iter__
    index = "steps",     // Enables __getitem__
)]
pub struct Trajectory {
    pub session_id: String,

    #[rlm(
        desc = "Conversation steps",
        filter_property = "user_steps",  // Generates .user_steps property
        filter_value = "user",
        filter_field = "source"
    )]
    pub steps: Vec<Step>,
}
```

### RlmDescribe Trait

Types implementing `RlmDescribe` can describe themselves for prompts:

```rust
pub trait RlmDescribe {
    fn type_name() -> &'static str;
    fn fields() -> Vec<RlmFieldDesc>;
    fn properties() -> Vec<RlmPropertyDesc>;
    fn is_iterable() -> bool;
    fn is_indexable() -> bool;
    fn describe_value(&self) -> String;
    fn describe_type() -> String;
}
```

This generates rich variable descriptions in prompts:

```
Variable: `trajectories` (access it in your code)
Type: Trajectory
Count: 50
Description: Trajectories to analyze
Total length: 142,000 characters
Usage:
  - len(trajectories)
  - trajectories[i]
  - trajectories.user_steps
Shape:
  Trajectory {
    session_id: string
    steps: list[Step]
  }
Preview:
  - Trajectory(45 steps, session=sess-001...)
  - Trajectory(32 steps, session=sess-002...)
  - ...
```

### RlmInputFields Trait

Enables input types to be injected into Python:

```rust
pub trait RlmInputFields {
    fn rlm_py_fields(&self, py: Python<'_>) -> Vec<(String, Py<PyAny>)>;
    fn rlm_variables(&self) -> Vec<RlmVariable>;
    fn inject_into_python(&self, py: Python<'_>, globals: &Bound<'_, PyDict>) -> PyResult<()>;
}
```

---

## Serialization & Storage

### StorableRlmResult

For storing/analyzing RLM executions:

```rust
#[derive(Serialize, Deserialize)]
pub struct StorableRlmResult {
    pub id: Uuid,                              // Trajectory identity
    pub created_at: DateTime<Utc>,
    pub input_json: Value,                     // Serialized input
    pub output_json: Value,                    // Serialized output
    pub trajectory: REPLHistory,               // Full REPL history
    pub field_metas: IndexMap<String, StorableFieldMeta>,
    pub iterations: usize,
    pub llm_calls: usize,
    pub extraction_fallback: bool,
    pub constraint_summary: ConstraintSummary,
    pub metadata: HashMap<String, Value>,      // User-defined metadata
}
```

### Usage

```rust
// Convert to storable format
let storable = result.to_storable()?;

// Or with custom metadata
let mut metadata = HashMap::new();
metadata.insert("task_id".into(), json!("task-123"));
metadata.insert("experiment".into(), json!("v2"));
let storable = result.to_storable_with_metadata(metadata)?;

// Serialize
let json = storable.to_json_pretty()?;

// Deserialize
let restored = StorableRlmResult::from_json(&json)?;

// Access preserved identity
assert_eq!(storable.id, restored.id);
assert_eq!(storable.trajectory.id, restored.trajectory.id);
```

### Trajectory Serialization

```json
{
  "id": "5b9f24d8-e46e-4de6-8558-0aff449fd4a9",
  "created_at": "2026-01-31T15:33:36.400665Z",
  "entries": [
    {
      "reasoning": "First, let me analyze the text",
      "code": "text = input_data['text']\nprint(len(text))",
      "output": "142",
      "timestamp": "2026-01-31T15:33:36.401132Z"
    },
    {
      "reasoning": "Now I'll summarize",
      "code": "summary = '...'\nSUBMIT(summary=summary)",
      "output": "SUBMIT successful!",
      "timestamp": "2026-01-31T15:33:37.234567Z"
    }
  ]
}
```

---

## Configuration

### RlmConfig

```rust
pub struct RlmConfig {
    /// Maximum REPL iterations before extraction fallback (default: 20)
    pub max_iterations: usize,

    /// Maximum sub-LLM calls from Python code (default: 50)
    pub max_llm_calls: usize,

    /// Whether to attempt extraction on max iterations (default: true)
    pub enable_extraction_fallback: bool,

    /// Whether assertion failures are fatal (default: true)
    pub strict_assertions: bool,

    /// Max chars from Python output per step (default: 100_000)
    pub max_output_chars: usize,

    /// Max chars when rendering history in prompts (default: 5_000)
    pub max_history_output_chars: usize,
}
```

### Constraint Types

```rust
// Soft check - violations noted but output accepted
#[check("len(this) >= 10", label = "min_length")]

// Hard assert - violations block completion
#[assert("this > 0", label = "positive")]
```

---

## Integration Points

### With Signatures (Deep Dive)

RLM is parameterized over any type implementing `Signature`:

**How it works:**

1. **`Signature` trait as the contract** - Defines:
   - `type Input` / `type Output` - Type-safe I/O
   - `input_fields()` / `output_fields()` - Static metadata (names, types, descriptions, constraints)
   - `instruction()` - Task description
   - `output_format_content()` - Schema for parsing

2. **`#[signature(rlm = true)]` (default)** - Generates `RlmInputFields` impl:
   ```rust
   impl RlmInputFields for MySignatureInput {
       fn rlm_py_fields(&self, py: Python) -> Vec<(String, Py<PyAny>)> {
           // Converts each input field to Python object
       }

       fn rlm_variables(&self) -> Vec<RlmVariable> {
           // Creates prompt-ready descriptions with constraints
       }
   }
   ```

3. **Variable injection** - During RLM setup:
   ```rust
   fn setup_globals<S: Signature>(input: &S::Input, ...) {
       input.inject_into_python(py, &globals)?;  // Sets Python globals
       // Now LLM-generated code can access `trajectories`, `query`, etc.
   }
   ```

4. **Prompt generation** - `RlmAdapter` reads signature metadata:
   - `S::instruction()` → Task description
   - `S::input_fields()` → Variable info blocks
   - `S::output_fields()` → Output schema + constraints

RLM is parameterized over any type implementing `Signature`:

```rust
pub struct Rlm<S: Signature> {
    config: RlmConfig,
    generate_action: Predict<RlmActionSig>,
    extract: Predict<RlmExtractSig<S>>,  // Uses S::Output
    // ...
}
```

The `#[signature(rlm = true)]` (default) generates `RlmInputFields` implementation.

### With Predict (Deep Dive)

RLM uses two internal Predict modules:

1. **`Predict<RlmActionSig>`** - Generates next code action
2. **`Predict<RlmExtractSig<S>>`** - Extracts final outputs (fallback)

**Internal architecture:**

```rust
pub struct Rlm<S: Signature> {
    config: RlmConfig,
    lm_override: Option<Arc<LM>>,
    generate_action: Predict<RlmActionSig>,     // "What code next?"
    extract: Predict<RlmExtractSig<S>>,         // Fallback extraction
}
```

**RLM loop uses Predict:**

```rust
// Each iteration calls generate_action.call()
let action = self.generate_action.call(RlmActionSigInput {
    variables_info,
    repl_history,
    iteration: "3/20",
}).await?;

// LLM returns {reasoning, code}
execute_repl_code(&globals, &action.code)?;
```

### With LM (Deep Dive)

Uses global LM settings or explicit override:

```rust
// Uses global settings
let rlm = Rlm::<MySig>::new();

// Uses explicit LM
let rlm = Rlm::<MySig>::with_lm(my_lm);
```

**Two-tier LM resolution:**

```rust
// In Predict::call()
let lm = match &self.lm_override {
    Some(lm) => Arc::clone(lm),           // Explicit
    None => {
        let settings = GLOBAL_SETTINGS.read()?;
        Arc::clone(&settings.lm)          // Global fallback
    }
};
```

**The `llm_query` tool from Python:**

```rust
#[pymethods]
impl LlmTools {
    fn llm_query(&self, prompt: String) -> PyResult<String> {
        self.reserve_calls(1)?;  // Check quota
        self.runtime.block_on(async {
            self.lm.prompt(&prompt).await  // Same LM as RLM
        })
    }

    fn llm_query_batched(&self, prompts: Vec<String>) -> PyResult<Vec<String>> {
        // Concurrent execution via futures::join_all
    }
}
```

The same `Arc<LM>` is shared between RLM's Predict calls and the Python-accessible `llm_query`.

### With baml-bridge (Deep Dive)

baml-bridge is the **type system engine** powering all RLM I/O:

**1. Type System Foundation:**

- **`BamlType` trait** - Defines type identity, TypeIR, output format, conversion
- **`BamlValue` enum** - JSON-like intermediate (String, Int, Float, List, Map, Class, Enum, etc.)
- **`ToBamlValue` trait** - Bidirectional conversion for Rust types

**2. Parsing Pipeline:**

```
Raw LLM output → jsonish::from_str() → BamlValueWithFlags
  → BamlValue (flags stripped) → T (via try_from_baml_value)
```

**3. SUBMIT Processing:**

```rust
// In submit.rs
let baml_value = py::kwargs_to_baml_value::<S>(py, kwargs)?;  // Python → BamlValue
let checks = py::collect_checks_for_output::<S>(&baml_value)?;  // Run constraints
// Returns (BamlValue, FieldMeta)
```

**4. Constraint Evaluation:**

- **Check** (soft) - Violations noted but output accepted
- **Assert** (hard) - Violations block completion, raise `BamlParseError::ConstraintAssertsFailed`

```rust
// run_user_checks() evaluates constraints
for (constraint, ok) in results {
    if constraint.level == ConstraintLevel::Assert && !ok {
        failed.push(ResponseCheck { ... });
    }
}
if !failed.is_empty() {
    return Err(BamlParseError::ConstraintAssertsFailed { failed });
}
```

**5. REPLHistory Rendering:**

Uses `#[render(default = "...")]` Jinja templates:

```rust
#[derive(BamlType)]
#[render(default = r#"
{%- for entry in value.entries -%}
=== Step {{ loop.index }} ===
{% if entry.reasoning %}Reasoning: {{ entry.reasoning }}{% endif %}
Code: ```python
{{ entry.code }}
```
Output ({{ entry.output | length }} chars):
{% if entry.output | length > ctx.max_output_chars %}
{{ entry.output | slice_chars(ctx.max_output_chars) }}
... (truncated)
{% else %}
{{ entry.output }}
{% endif %}
{% endfor -%}
"#)]
pub struct REPLHistory { ... }
```

**6. Python ↔ Rust Bridge:**

```rust
// Python → BamlValue
py_to_baml_value(py, &obj, &type_ir, output_format)
  // Calls __baml__() if present (RlmType objects)
  // Normalizes via model_dump(), dict(), etc.

// BamlValue → Python
baml_value_to_py(py, &value)
  // Maps to Python primitives/collections
```

---

## Data Structures Reference

### Core Execution Types

#### `Rlm<S: Signature>`

The main RLM executor, generic over your signature:

```rust
pub struct Rlm<S: Signature> {
    config: RlmConfig,                      // Execution settings
    lm_override: Option<Arc<LM>>,           // Optional explicit LM
    generate_action: Predict<RlmActionSig>, // Internal: "what code next?"
    extract: Predict<RlmExtractSig<S>>,     // Internal: fallback extraction
}
```

**Key methods:**
- `Rlm::new()` - Default config, global LM
- `Rlm::with_config(config)` - Custom config
- `Rlm::with_lm(lm)` - Explicit LM override
- `Rlm::builder()` - Fluent configuration
- `async fn call(input) -> Result<RlmResult<S>, RlmError>` - Execute

---

#### `RlmConfig`

Execution parameters:

```rust
pub struct RlmConfig {
    pub max_iterations: usize,           // Default: 20
    pub max_llm_calls: usize,            // Default: 50 (from Python code)
    pub enable_extraction_fallback: bool, // Default: true
    pub strict_assertions: bool,          // Default: true
    pub max_output_chars: usize,          // Default: 100_000 (per step)
    pub max_history_output_chars: usize,  // Default: 5_000 (in prompts)
}
```

---

#### `RlmResult<S: Signature>`

Execution result with full context:

```rust
pub struct RlmResult<S: Signature> {
    pub input: S::Input,                              // Original input
    pub output: S::Output,                            // Typed output
    pub trajectory: REPLHistory,                      // Full execution trace
    pub field_metas: IndexMap<String, FieldMeta>,     // Per-field metadata
    pub iterations: usize,                            // Steps taken
    pub llm_calls: usize,                             // Total LLM calls
    pub extraction_fallback: bool,                    // Was fallback used?
    pub constraint_summary: ConstraintSummary,        // Check/assert counts
}
```

**Key methods:**
- `to_signature() -> S` - Reconstruct full signature
- `failed_checks() -> Vec<&ConstraintResult>` - Get failed soft checks
- `has_constraint_warnings() -> bool` - Any soft failures?
- `is_fallback() -> bool` - Was extraction fallback used?
- `to_storable() -> Result<StorableRlmResult>` - Serialize for storage
- `to_storable_with_metadata(HashMap<String, Value>)` - With custom metadata

---

### Trajectory Types

#### `REPLHistory`

Immutable trajectory container with identity:

```rust
pub struct REPLHistory {
    pub id: Uuid,                    // Unique trajectory ID (preserved through appends)
    pub created_at: DateTime<Utc>,   // When trajectory started
    pub entries: Vec<REPLEntry>,     // Execution steps
}
```

**Key methods:**
- `REPLHistory::new()` - Create with fresh UUID + timestamp
- `append(code, output) -> Self` - Immutable append (preserves id/created_at)
- `append_with_reasoning(reasoning, code, output) -> Self` - With reasoning
- `len() -> usize` / `is_empty() -> bool`
- `entries() -> &[REPLEntry]`

**Serialization:** Implements `Serialize`/`Deserialize`. UUID and timestamps are preserved through JSON roundtrips.

---

#### `REPLEntry`

A single execution step:

```rust
pub struct REPLEntry {
    pub reasoning: String,           // LLM's thinking (may be empty)
    pub code: String,                // Python code executed
    pub output: String,              // Execution output/result
    pub timestamp: DateTime<Utc>,    // When this step ran
}
```

---

### Constraint Types

#### `ConstraintResult`

Result of evaluating a constraint:

```rust
#[derive(Serialize, Deserialize)]
pub struct ConstraintResult {
    pub label: String,       // e.g., "min_length"
    pub expression: String,  // e.g., "len(this) >= 10"
    pub passed: bool,
}
```

---

#### `ConstraintSummary`

Aggregate constraint statistics:

```rust
#[derive(Serialize, Deserialize)]
pub struct ConstraintSummary {
    pub checks_passed: usize,
    pub checks_failed: usize,
    pub assertions_passed: usize,  // Assertions that passed (failures = error)
}
```

---

#### `FieldMeta`

Per-field parsing metadata:

```rust
pub struct FieldMeta {
    pub raw_text: String,            // Raw extracted text
    pub flags: Vec<Flag>,            // Parsing hints (internal)
    pub checks: Vec<ConstraintResult>, // Constraint results
}
```

---

### Storage Types

#### `StorableRlmResult`

Fully serializable execution record:

```rust
#[derive(Serialize, Deserialize)]
pub struct StorableRlmResult {
    pub id: Uuid,                                    // Trajectory ID
    pub created_at: DateTime<Utc>,                   // Start time
    pub input_json: serde_json::Value,               // Serialized input
    pub output_json: serde_json::Value,              // Serialized output
    pub trajectory: REPLHistory,                     // Full REPL trace
    pub field_metas: IndexMap<String, StorableFieldMeta>,
    pub iterations: usize,
    pub llm_calls: usize,
    pub extraction_fallback: bool,
    pub constraint_summary: ConstraintSummary,
    pub metadata: HashMap<String, serde_json::Value>, // Your custom data
}
```

**Key methods:**
- `to_json() -> Result<String>` - Compact JSON
- `to_json_pretty() -> Result<String>` - Pretty-printed
- `from_json(&str) -> Result<Self>` - Deserialize

---

#### `StorableFieldMeta`

Serializable field metadata (without internal flags):

```rust
#[derive(Serialize, Deserialize)]
pub struct StorableFieldMeta {
    pub raw_text: String,
    pub checks: Vec<ConstraintResult>,
}
```

---

### Type Description Types

#### `RlmVariable`

Variable description for prompts:

```rust
pub struct RlmVariable {
    pub name: String,                    // Variable name in Python
    pub type_desc: String,               // Type description
    pub description: String,             // Human description
    pub constraints: Vec<String>,        // Constraint expressions
    pub total_length: usize,             // Character count
    pub preview: String,                 // Truncated preview
    pub properties: Vec<(String, String)>, // Computed properties
}
```

---

#### `RlmDescribe` Trait

Types that can describe themselves:

```rust
pub trait RlmDescribe {
    fn type_name() -> &'static str;
    fn fields() -> Vec<RlmFieldDesc>;
    fn properties() -> Vec<RlmPropertyDesc>;
    fn is_iterable() -> bool;
    fn is_indexable() -> bool;
    fn describe_value(&self) -> String;
    fn describe_type() -> String;
}
```

---

## Building on Top of RLM

### Example 1: Experiment Tracking

Store and analyze RLM executions across experiments:

```rust
use std::collections::HashMap;
use std::fs;
use dspy_rs::rlm::{Rlm, StorableRlmResult};

struct ExperimentTracker {
    experiment_name: String,
    results_dir: PathBuf,
}

impl ExperimentTracker {
    /// Run RLM and store the result with experiment metadata
    pub async fn run_and_track<S: Signature>(
        &self,
        rlm: &Rlm<S>,
        input: S::Input,
        run_id: &str,
    ) -> Result<RlmResult<S>>
    where
        S::Input: Serialize + RlmInputFields,
        S::Output: Clone + ToBamlValue + Serialize,
    {
        let result = rlm.call(input).await?;

        // Store with experiment metadata
        let mut metadata = HashMap::new();
        metadata.insert("experiment".into(), json!(self.experiment_name));
        metadata.insert("run_id".into(), json!(run_id));
        metadata.insert("timestamp".into(), json!(Utc::now().to_rfc3339()));

        let storable = result.to_storable_with_metadata(metadata)?;
        let path = self.results_dir.join(format!("{}.json", storable.id));
        fs::write(&path, storable.to_json_pretty()?)?;

        Ok(result)
    }

    /// Load all results for analysis
    pub fn load_all_results(&self) -> Vec<StorableRlmResult> {
        fs::read_dir(&self.results_dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let json = fs::read_to_string(&path).ok()?;
                StorableRlmResult::from_json(&json).ok()
            })
            .collect()
    }

    /// Analyze iteration distribution
    pub fn iteration_stats(&self) -> IterationStats {
        let results = self.load_all_results();
        let iterations: Vec<usize> = results.iter().map(|r| r.iterations).collect();
        IterationStats {
            count: iterations.len(),
            mean: iterations.iter().sum::<usize>() as f64 / iterations.len() as f64,
            max: iterations.iter().copied().max().unwrap_or(0),
            fallback_rate: results.iter().filter(|r| r.extraction_fallback).count() as f64
                           / results.len() as f64,
        }
    }
}
```

---

### Example 2: Trajectory Analysis Pipeline

Analyze execution patterns across trajectories:

```rust
use dspy_rs::rlm::{REPLHistory, REPLEntry, StorableRlmResult};

/// Analyze code patterns in trajectories
pub struct TrajectoryAnalyzer;

impl TrajectoryAnalyzer {
    /// Extract all unique tool calls from trajectories
    pub fn extract_llm_query_calls(trajectory: &REPLHistory) -> Vec<String> {
        trajectory.entries.iter()
            .filter_map(|entry| {
                // Find llm_query calls in code
                let re = regex::Regex::new(r#"llm_query\(([^)]+)\)"#).ok()?;
                re.captures(&entry.code).map(|c| c[1].to_string())
            })
            .collect()
    }

    /// Calculate reasoning-to-code ratio
    pub fn reasoning_density(trajectory: &REPLHistory) -> f64 {
        let total_reasoning: usize = trajectory.entries.iter()
            .map(|e| e.reasoning.len())
            .sum();
        let total_code: usize = trajectory.entries.iter()
            .map(|e| e.code.len())
            .sum();

        if total_code == 0 { 0.0 } else { total_reasoning as f64 / total_code as f64 }
    }

    /// Find error patterns
    pub fn error_steps(trajectory: &REPLHistory) -> Vec<(usize, &REPLEntry)> {
        trajectory.entries.iter()
            .enumerate()
            .filter(|(_, entry)| entry.output.contains("[Error]"))
            .collect()
    }

    /// Time between steps (if timestamps available)
    pub fn step_durations(trajectory: &REPLHistory) -> Vec<chrono::Duration> {
        trajectory.entries.windows(2)
            .map(|window| window[1].timestamp - window[0].timestamp)
            .collect()
    }
}

/// Batch analysis across many trajectories
pub fn analyze_corpus(results: &[StorableRlmResult]) -> CorpusAnalysis {
    let mut total_iterations = 0;
    let mut total_llm_calls = 0;
    let mut error_count = 0;
    let mut fallback_count = 0;

    for result in results {
        total_iterations += result.iterations;
        total_llm_calls += result.llm_calls;
        error_count += TrajectoryAnalyzer::error_steps(&result.trajectory).len();
        if result.extraction_fallback {
            fallback_count += 1;
        }
    }

    CorpusAnalysis {
        total_runs: results.len(),
        avg_iterations: total_iterations as f64 / results.len() as f64,
        avg_llm_calls: total_llm_calls as f64 / results.len() as f64,
        error_rate: error_count as f64 / total_iterations as f64,
        fallback_rate: fallback_count as f64 / results.len() as f64,
    }
}
```

---

### Example 3: Custom RLM Wrapper with Retry Logic

Build a wrapper that handles failures gracefully:

```rust
use dspy_rs::rlm::{Rlm, RlmConfig, RlmError, RlmResult};

pub struct RobustRlm<S: Signature> {
    rlm: Rlm<S>,
    max_retries: usize,
    backoff_ms: u64,
}

impl<S: Signature> RobustRlm<S> {
    pub fn new(max_retries: usize) -> Self {
        Self {
            rlm: Rlm::new(),
            max_retries,
            backoff_ms: 1000,
        }
    }

    pub async fn call_with_retry(&self, input: S::Input) -> Result<RlmResult<S>, RlmError>
    where
        S::Input: Clone + RlmInputFields,
        S::Output: Clone + ToBamlValue,
    {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(self.backoff_ms * attempt as u64)).await;
            }

            match self.rlm.call(input.clone()).await {
                Ok(result) => {
                    // Check if result has too many constraint warnings
                    if result.constraint_summary.checks_failed > 3 {
                        eprintln!("Warning: {} constraints failed", result.constraint_summary.checks_failed);
                    }
                    return Ok(result);
                }
                Err(RlmError::MaxIterationsReached { .. }) if attempt < self.max_retries => {
                    // Retry with more iterations
                    eprintln!("Max iterations reached, retrying...");
                    last_error = Some(RlmError::MaxIterationsReached { max: 20 });
                    continue;
                }
                Err(e) => {
                    last_error = Some(e);
                    break;
                }
            }
        }

        Err(last_error.unwrap())
    }
}
```

---

### Example 4: Streaming Trajectory Updates

Build a system that streams trajectory updates:

```rust
use tokio::sync::mpsc;
use dspy_rs::rlm::REPLEntry;

/// Event emitted during RLM execution
#[derive(Clone)]
pub enum TrajectoryEvent {
    Started { id: Uuid, created_at: DateTime<Utc> },
    StepCompleted { entry: REPLEntry, iteration: usize },
    SubmitAttempt { success: bool, message: String },
    Completed { iterations: usize, llm_calls: usize },
    Error { message: String },
}

/// Observer that receives trajectory updates
pub struct TrajectoryObserver {
    tx: mpsc::Sender<TrajectoryEvent>,
}

impl TrajectoryObserver {
    pub fn new() -> (Self, mpsc::Receiver<TrajectoryEvent>) {
        let (tx, rx) = mpsc::channel(100);
        (Self { tx }, rx)
    }

    pub async fn emit(&self, event: TrajectoryEvent) {
        let _ = self.tx.send(event).await;
    }
}

// Usage: integrate with RLM by wrapping the result
pub async fn run_with_streaming<S: Signature>(
    rlm: &Rlm<S>,
    input: S::Input,
    observer: &TrajectoryObserver,
) -> Result<RlmResult<S>, RlmError>
where
    S::Input: RlmInputFields,
    S::Output: Clone + ToBamlValue,
{
    let result = rlm.call(input).await;

    match &result {
        Ok(r) => {
            observer.emit(TrajectoryEvent::Started {
                id: r.trajectory.id,
                created_at: r.trajectory.created_at,
            }).await;

            for (i, entry) in r.trajectory.entries.iter().enumerate() {
                observer.emit(TrajectoryEvent::StepCompleted {
                    entry: entry.clone(),
                    iteration: i + 1,
                }).await;
            }

            observer.emit(TrajectoryEvent::Completed {
                iterations: r.iterations,
                llm_calls: r.llm_calls,
            }).await;
        }
        Err(e) => {
            observer.emit(TrajectoryEvent::Error {
                message: e.to_string(),
            }).await;
        }
    }

    result
}
```

---

### Example 5: Database Storage with SQLite

Store trajectories in a database for querying:

```rust
use rusqlite::{Connection, params};
use dspy_rs::rlm::StorableRlmResult;

pub struct TrajectoryDB {
    conn: Connection,
}

impl TrajectoryDB {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute(r#"
            CREATE TABLE IF NOT EXISTS trajectories (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                llm_calls INTEGER NOT NULL,
                extraction_fallback BOOLEAN NOT NULL,
                checks_passed INTEGER NOT NULL,
                checks_failed INTEGER NOT NULL,
                experiment TEXT,
                data JSON NOT NULL
            )
        "#, [])?;
        Ok(Self { conn })
    }

    pub fn insert(&self, result: &StorableRlmResult) -> Result<()> {
        let experiment = result.metadata.get("experiment")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.conn.execute(r#"
            INSERT OR REPLACE INTO trajectories
            (id, created_at, iterations, llm_calls, extraction_fallback,
             checks_passed, checks_failed, experiment, data)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#, params![
            result.id.to_string(),
            result.created_at.to_rfc3339(),
            result.iterations,
            result.llm_calls,
            result.extraction_fallback,
            result.constraint_summary.checks_passed,
            result.constraint_summary.checks_failed,
            experiment,
            result.to_json()?,
        ])?;
        Ok(())
    }

    pub fn query_by_experiment(&self, experiment: &str) -> Vec<StorableRlmResult> {
        let mut stmt = self.conn.prepare(
            "SELECT data FROM trajectories WHERE experiment = ?1 ORDER BY created_at DESC"
        ).unwrap();

        stmt.query_map([experiment], |row| {
            let json: String = row.get(0)?;
            Ok(StorableRlmResult::from_json(&json).unwrap())
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    pub fn high_iteration_runs(&self, threshold: usize) -> Vec<StorableRlmResult> {
        let mut stmt = self.conn.prepare(
            "SELECT data FROM trajectories WHERE iterations > ?1"
        ).unwrap();

        stmt.query_map([threshold], |row| {
            let json: String = row.get(0)?;
            Ok(StorableRlmResult::from_json(&json).unwrap())
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }
}
```

---

## Examples

### Example 1: Simple Summarization

```rust
#[derive(Signature)]
struct Summarize {
    #[input]
    text: String,
    #[output]
    summary: String,
}

let result = Rlm::<Summarize>::new()
    .call(SummarizeInput { text: long_text })
    .await?;
```

### Example 2: Complex Data Analysis

```rust
#[rlm_type]
#[derive(Clone)]
#[rlm(iter = "items", index = "items")]
struct Dataset {
    #[rlm(desc = "Data items to analyze")]
    items: Vec<DataItem>,
}

#[derive(Signature)]
struct AnalyzeData {
    #[input]
    dataset: Dataset,

    #[output]
    #[check("len(this) > 0")]
    insights: Vec<String>,

    #[output]
    #[assert("this >= 0")]
    score: i32,
}
```

### Example 3: Storage and Analysis

```rust
let result = rlm.call(input).await?;

// Store with experiment metadata
let storable = result.to_storable_with_metadata(hashmap!{
    "experiment" => json!("baseline_v1"),
    "model" => json!("gpt-4o"),
})?;

// Save to file/database
let json = storable.to_json_pretty()?;
std::fs::write("result.json", json)?;

// Later: load and analyze
let restored = StorableRlmResult::from_json(&std::fs::read_to_string("result.json")?)?;
println!("Trajectory {} took {} iterations", restored.id, restored.iterations);

for entry in &restored.trajectory.entries {
    println!("[{}] {}", entry.timestamp, entry.code);
}
```

---

## Error Handling

```rust
pub enum RlmError {
    LlmError { source: LmError },
    AssertionFailed { label: String, expression: String },
    ConversionError { source: BamlConvertError, value: BamlValue },
    ExtractionFailed { source: ParseError, raw_response: String },
    PredictError { stage: &'static str, source: PredictError },
    MaxIterationsReached { max: usize },
    MaxLlmCallsExceeded { max: usize },
    PythonError { message: String },
    RuntimeUnavailable { message: String },
    ConfigurationError { message: String },
}
```

---

## Feature Flag

RLM requires the `rlm` feature:

```toml
[dependencies]
dspy-rs = { version = "0.7", features = ["rlm"] }
```

This enables:
- Python REPL execution (requires Python installation)
- PyO3 integration for type conversion
- All RLM-specific modules

---

## Summary

RLM provides:

1. **Agentic execution** - LLM iterates with code execution
2. **Typed interfaces** - Signature-based inputs/outputs
3. **Rich introspection** - `#[rlm_type]` generates Python-compatible types with descriptions
4. **Constraint validation** - Soft checks and hard assertions
5. **Sub-LLM access** - `llm_query()` from Python for semantic tasks
6. **Full traceability** - UUID-identified trajectories with timestamps
7. **Serialization** - StorableRlmResult for analysis/storage
8. **Fallback extraction** - Recovers outputs even on iteration limits
