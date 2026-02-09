# DSRs Module System — Technical Design Reference

> Companion to the Shaping Document. The shaping doc says **what** we want (R's) and **what parts** we need (F's). This document captures **how each part works**: the concrete types, traits, data flow, code sketches, and design decisions from the shaping process.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Signature: The Typed Contract (F1, F12)](#2-signature)
3. [SignatureSchema: Facet-Derived Metadata (F2)](#3-signatureschema)
4. [Augmentation: Typed Signature Extension (F3)](#4-augmentation)
5. [Module Trait: The Composition Interface (F4)](#5-module-trait)
6. [Predict: The Leaf Parameter (F5)](#6-predict)
7. [Facet-Powered Parameter Discovery (F6)](#7-parameter-discovery)
8. [Adapter Building Blocks (F7)](#8-adapter)
9. [DynPredictor: The Optimizer Bridge (F8)](#9-dynpredictor)
10. [DynModule + StrategyFactory (F9)](#10-dynmodule)
11. [ProgramGraph (F10)](#11-programgraph)
12. [Library Modules (F11)](#12-library-modules)
13. [Layer Architecture](#13-layers)
14. [Key Design Decisions](#14-decisions)
15. [Resolved Spikes](#15-spikes)
16. [Typed vs Dynamic Path Summary](#16-typed-vs-dynamic)

---

## 1. Design Principles

These emerged through the conversation and should guide all implementation choices:

**Facet Shapes are the source of truth.** Field names, types, order, docs, constraints — all derived from Facet at runtime. No parallel schema systems. Macro-emitted static `FieldSpec` arrays are replaced by `SignatureSchema::of::<S>()`.

**Parse, don't validate.** A `ProgramGraph` with edges has already been type-checked at connection time. An augmented output type with `#[flatten]` encodes the field expansion in the type — the adapter doesn't need to know about augmentation, it just walks the Shape.

**If it requires vigilance, the design is leaking.** Module authors don't annotate `#[parameter]` on predictors. They don't implement traversal. They `#[derive(Facet)]` and the walker finds everything. The structure IS the declaration.

**Modules are prompting strategies.** ChainOfThought doesn't modify your code — it modifies the prompt. The Signature type changes (augmented output), but the user's data contract stays the same. `result.answer` works whether you used Predict or ChainOfThought.

**Typed path is primary, dynamic path is escape hatch.** Layer 1 (typed modules) is where 90% of programs live. Layer 3 (dynamic graph) exists for structural optimization and interop. Both paths share the same adapter and prompt format.

**One adapter, one prompt format.** Both the typed path and the dynamic graph path use `SignatureSchema` → ChatAdapter building blocks → `[[ ## field ## ]]` delimited prompts. Identical prompts for identical logical signatures regardless of which path produced them.

---

## 2. Signature: The Typed Contract (F1, F12)

### The trait

```rust
pub trait Signature: Send + Sync + 'static {
    type Input: BamlType + Facet + Send + Sync;
    type Output: BamlType + Facet + Send + Sync;

    fn instructions() -> &'static str { "" }
}
```

Bounds: `BamlType` for jsonish coercion and value conversion. `Facet` for schema derivation. Both are derived, not manual.

Note: `from_parts`/`into_parts` were removed from the trait (S7). The current codebase uses them to combine input+output into one struct and split back apart, but with demos stored as `Demo<S> { input: S::Input, output: S::Output }` pairs and `Predict::call()` returning `S::Output` directly, the round-trip is unnecessary. The user's `#[derive(Signature)]` still generates the combined struct for ergonomic field access, but that's a convenience on the user's type, not a trait requirement.

### User-facing derive

```rust
/// Answer questions accurately and concisely.
#[derive(Signature, Clone, Debug)]
struct QA {
    /// The question to answer
    #[input]
    question: String,

    /// A clear, direct answer
    #[output]
    answer: String,
}
```

### What the derive generates

```rust
// Public input type — users construct this to call the module
#[derive(Clone, Debug, Facet, BamlType)]
pub struct QAInput {
    /// The question to answer
    pub question: String,
}

// Output type — returned from the module
#[derive(Clone, Debug, Facet, BamlType)]
pub struct QAOutput {
    /// A clear, direct answer
    pub answer: String,
}

// Signature impl
impl Signature for QA {
    type Input = QAInput;
    type Output = QAOutput;

    fn instructions() -> &'static str {
        "Answer questions accurately and concisely."  // from struct doc comment
    }
}
```

Key change from current: the derive does NOT emit static `FieldSpec` arrays or `OutputFormatContent`. Those are derived from Facet Shapes at runtime (F2).

### Generic signatures (F12)

Module authors define signatures with type parameters for reuse:

```rust
#[derive(Signature, Clone)]
#[instruction = "Write Python code that produces the answer."]
struct CodeGen<I: BamlType + Clone> {
    #[input, flatten] base: I,
    #[output] generated_code: String,
}
```

The derive generates `CodeGenInput<I>` with I's fields flattened inline, plus `generated_code` as output. S1 confirmed this is feasible — the proc macro needs `split_for_impl()` to thread type parameters and bounds through generated types. Decision: Option C (full replacement) — `SignatureSchema` derived from Facet replaces `FieldSpec` entirely, no incremental migration (see `S1-generic-signature-derive.md`).

### Attributes (moving to Facet long-term)

Current: `#[input]`, `#[output]`, `#[alias = "..."]`, `#[check("...")]`, `#[assert("...")]`, `#[format("...")]`.

Long-term direction: these become Facet attributes via `define_attr_grammar!`:

```rust
define_attr_grammar! {
    pub mod dsrs {
        input,
        output,
        flatten,
        alias(String),
        format(String),
        check(expr: String, label: String),
        assert_constraint(expr: String, label: Option<String>),
        parameter,  // marks Predict for discovery
    }
}
```

This is a migration, not a blocker. Current custom attributes work. Facet attributes are the end state.

---

## 3. SignatureSchema: Facet-Derived Metadata (F2)

### The type

```rust
pub struct SignatureSchema {
    pub instruction: String,
    pub input_fields: Vec<FieldSchema>,
    pub output_fields: Vec<FieldSchema>,
    pub output_format: Arc<OutputFormatContent>,
}

pub struct FieldSchema {
    pub name: String,           // LM-facing name (alias or field name)
    pub rust_name: String,      // Rust field identifier
    pub type_ir: TypeIR,        // for jsonish coercion
    pub facet_shape: fn() -> &'static Shape,  // full Rust type info
    pub desc: String,           // from doc comment
    pub path: FieldPath,        // e.g. ["inner", "answer"] for flattened fields
    pub constraints: Vec<ConstraintSpec>,
    pub format: Option<String>, // input formatting hint
}

pub struct FieldPath(pub Vec<String>);
```

### Derivation

```rust
impl SignatureSchema {
    pub fn of<S: Signature>() -> &'static Self {
        static CACHE: OnceLock<SignatureSchema> = OnceLock::new();
        CACHE.get_or_init(|| {
            Self::derive::<S>()
        })
    }

    fn derive<S: Signature>() -> Self {
        SignatureSchema {
            instruction: S::instructions().to_string(),
            input_fields: walk_fields::<S::Input>(/* dsrs::input */),
            output_fields: walk_fields::<S::Output>(/* dsrs::output */),
            output_format: Arc::new(build_output_format::<S::Output>()),
        }
    }
}
```

### The Facet walk (flatten-aware)

```rust
fn walk_fields<T: Facet>(prefix_path: &[String]) -> Vec<FieldSchema> {
    let shape = T::SHAPE;
    let mut fields = Vec::new();

    // shape.def → StructType → iterate fields in declaration order
    for field in struct_fields(shape) {
        let path = [prefix_path, &[field.name.to_string()]].concat();

        // S8: Facet exposes flatten as field.is_flattened() — O(1) flag check.
        // field.shape() returns the inner type's Shape for recursion.
        if field.is_flattened() {
            // Recurse into the flattened type — inline its fields at this level
            let inner_shape = field.shape.get();  // ShapeRef → Shape
            fields.extend(walk_fields_from_shape(inner_shape, &path));
        } else {
            fields.push(FieldSchema {
                name: effective_name(field),  // alias or field name
                rust_name: field.name.to_string(),
                type_ir: build_type_ir(field.shape.get()),
                facet_shape: field.shape,
                desc: field.doc.join(" "),
                path: FieldPath(path),
                constraints: extract_constraints(field),
                format: extract_format(field),
            });
        }
    }

    fields
}
```

**The flatten walk is the core trick.** When the schema builder encounters `#[flatten]`, it doesn't emit one field — it recurses and emits the inner type's fields with extended paths. This is how `WithReasoning<QAOutput>` produces fields `[reasoning, answer]` instead of `[reasoning, inner]`.

### What replaces current FieldSpec arrays

| Current | New |
|---------|-----|
| `static __QA_INPUT_FIELDS: &[FieldSpec]` | `SignatureSchema::of::<QA>().input_fields` |
| `static __QA_OUTPUT_FIELDS: &[FieldSpec]` | `SignatureSchema::of::<QA>().output_fields` |
| `FieldSpec { type_ir: fn() -> TypeIR }` | `FieldSchema { type_ir: TypeIR, facet_shape: fn() -> &Shape }` |
| `S::output_format_content()` | `SignatureSchema::of::<S>().output_format` |

The `Signature` trait no longer exposes field-level methods. It provides types + instruction. `SignatureSchema` provides everything else, derived from those types via Facet.

---

## 4. Augmentation: Typed Signature Extension (F3)

### The augmentation trait

```rust
pub trait Augmentation: 'static {
    type Wrap<T: BamlType + Facet>: BamlType + Facet + Deref<Target = T>;
}
```

An Augmentation knows how to wrap any output type. `Deref` is the key — it makes the inner type's fields accessible through the wrapper.

### The derive

```rust
#[derive(Augmentation)]
#[augment(output, prepend)]
pub struct Reasoning {
    /// Step-by-step reasoning
    reasoning: String,
}
```

Generates:

```rust
// The wrapper type
#[derive(Clone, Debug, Facet, BamlType)]
pub struct WithReasoning<O: BamlType + Facet> {
    /// Step-by-step reasoning
    pub reasoning: String,

    #[facet(dsrs::flatten)]
    pub inner: O,
}

impl<O: BamlType + Facet> Deref for WithReasoning<O> {
    type Target = O;
    fn deref(&self) -> &O { &self.inner }
}

impl<O: BamlType + Facet> WithReasoning<O> {
    pub fn into_inner(self) -> O { self.inner }
}

// The augmentation trait impl
impl Augmentation for Reasoning {
    type Wrap<T: BamlType + Facet> = WithReasoning<T>;
}
```

### The signature combinator

```rust
pub struct Augmented<S: Signature, A: Augmentation>(PhantomData<(S, A)>);

impl<S: Signature, A: Augmentation> Signature for Augmented<S, A> {
    type Input = S::Input;
    type Output = A::Wrap<S::Output>;

    fn instructions() -> &'static str { S::instructions() }
}
```

`Augmented` is a zero-sized type-level combinator. It exists purely to map `S::Input → A::Wrap<S::Output>` at the type level. Modules hold `Predict<Augmented<S, A>>` where demos are stored as `Demo { input: S::Input, output: A::Wrap<S::Output> }` pairs and `call()` returns `A::Wrap<S::Output>` directly. No `from_parts`/`into_parts` needed (S7).

### How BamlType works for flatten

`WithReasoning<O>` implements `BamlType` with flatten semantics. Facet drives both directions:

**Serialization (for demos, tracing):** `to_baml_value()` produces a flat `BamlValue::Class` with reasoning + O's fields merged. The Facet shape walk on `WithReasoning<O>` expands flatten, so the schema builder already knows the flat field list.

**Deserialization (from LM output):** The adapter parses flat sections `{reasoning: "...", answer: "..."}`. Construction uses the `FieldPath` from `SignatureSchema`:
- `reasoning` → path `["reasoning"]` → set on WithReasoning directly
- `answer` → path `["inner", "answer"]` → navigate into inner, set on O

This could use Facet's `Partial` for progressive construction, or the simpler BamlValue intermediary:
1. Build nested `BamlValue` following paths
2. Call `BamlType::try_from_baml_value()` on WithReasoning<O>

Both work. BamlValue intermediary is simpler; Partial is more efficient for streaming. Decision deferred.

### User-facing field access

```rust
let cot = ChainOfThought::<QA>::new();
let result = cot.call(QAInput { question: "2+2?".into() }).await?;

result.reasoning   // String — direct field on WithReasoning
result.answer      // String — Deref to QAOutput → .answer
result.question    // NOT available (that's on the input, not output)
```

Type inference handles everything. The user never writes `WithReasoning<QAOutput>` unless they're naming a return type explicitly. Even then, it reads as English: "QA output, with reasoning."

### Composition (R13, S3 resolved)

```rust
// Tuple augmentation
impl<A: Augmentation, B: Augmentation> Augmentation for (A, B) {
    type Wrap<T: BamlType + Facet> = A::Wrap<B::Wrap<T>>;
}

// Usage:
type CotWithConfidence<S> = Augmented<S, (Reasoning, Confidence)>;
// Output: WithReasoning<WithConfidence<QAOutput>>
```

Field access: `result.reasoning` works directly. `result.confidence` requires Deref through WithReasoning to WithConfidence. `result.answer` requires double Deref. S3 confirmed: Rust auto-Deref resolves field access and method calls through multiple wrapper layers. Pattern matching does NOT auto-deref — it requires explicit layer-by-layer destructuring, which is an acceptable documented limitation. Mutability contract: `Deref`-only unless `DerefMut` is proven necessary (see `S3-augmentation-deref-composition.md`).

---

## 5. Module Trait: The Composition Interface (F4)

```rust
#[async_trait]
pub trait Module: Send + Sync {
    type Input: BamlType + Facet + Send + Sync;
    type Output: BamlType + Facet + Send + Sync;

    async fn forward(&self, input: Self::Input) -> Result<Self::Output, PredictError>;
}
```

Every prompting strategy implements this. The associated types make composition type-safe:

```rust
// The compiler rejects this at compile time:
struct Bad {
    step1: Predict<QA>,
    step2: Predict<Summarize>,  // Summarize expects SummarizeInput, not QAOutput
}
// step2.forward(step1_output) → type mismatch → compile error
```

### Swapping strategies

```rust
// These all implement Module<Input = QAInput, Output = _>
let direct: Predict<QA>                          // Output = QA
let cot: ChainOfThought<QA>                      // Output = WithReasoning<QA>
let react: ReAct<QA>                              // Output = QA (or WithTrajectory<QA>)
let bon: BestOfN<ChainOfThought<QA>>              // Output = WithReasoning<QA>

// Generic over strategy:
struct RAG<A: Module<Input = AnswerInput>> {
    retrieve: Predict<Retrieve>,
    answer: A,
}
```

---

## 6. Predict: The Leaf Parameter (F5)

```rust
#[derive(Facet)]
#[facet(dsrs::parameter)]  // marks for discovery by F6 walker
pub struct Predict<S: Signature> {
    demos: Vec<Demo<S>>,
    instruction_override: Option<String>,
    tools: Vec<Arc<dyn ToolDyn>>,
}

/// Demos are input+output pairs — not combined signature structs.
/// This is what makes Augmented work: Demo<Augmented<QA, Reasoning>>
/// stores (QAInput, WithReasoning<QAOutput>) without needing from_parts.
pub struct Demo<S: Signature> {
    pub input: S::Input,
    pub output: S::Output,
}
```

### Key change from current

Demos are `Vec<Demo<S>>` (typed input + output pair), NOT `Vec<S>` (the full combined signature struct). This makes augmentation work: `Predict<Augmented<QA, Reasoning>>` stores demos where output is `WithReasoning<QAOutput>`, so demo assistant messages include reasoning.

### Builder

```rust
let predict = Predict::<QA>::builder()
    .demo(Demo {
        input: QAInput { question: "What is 2+2?".into() },
        output: QAOutput { answer: "4".into() },
    })
    .instruction("Answer with exactly one word.")
    .build();
```

### The call pipeline

```rust
impl<S: Signature> Predict<S> {
    pub async fn call(&self, input: S::Input) -> Result<S::Output, PredictError> {
        let schema = SignatureSchema::of::<S>();  // F2: Facet-derived, cached
        let lm = get_global_lm();
        let adapter = ChatAdapter;

        // Build prompt
        let system = adapter.build_system(schema, self.instruction_override.as_deref());
        let mut chat = Chat::new(vec![Message::system(system)]);

        // Format demos
        for demo in &self.demos {
            let user = adapter.format_input(schema, &demo.input);
            let assistant = adapter.format_output(schema, &demo.output);
            chat.push_message(Message::user(user));
            chat.push_message(Message::assistant(assistant));
        }

        // Format current input
        let user = adapter.format_input(schema, &input);
        chat.push_message(Message::user(user));

        // Call LM
        let response = lm.call(chat, self.tools.clone()).await?;

        // Parse response
        let output = adapter.parse_output::<S::Output>(schema, &response)?;

        Ok(output)
    }
}
```

### State serialization (R10)

```rust
impl<S: Signature> Predict<S> {
    pub fn dump_state(&self) -> PredictState {
        PredictState {
            demos: self.demos.iter().map(|d| demo_to_example(d)).collect(),
            instruction_override: self.instruction_override.clone(),
            // signature structure comes from code, not serialized
        }
    }

    pub fn load_state(&mut self, state: PredictState) -> Result<()> {
        self.demos = state.demos.iter()
            .map(|e| example_to_demo::<S>(e))
            .collect::<Result<_>>()?;
        self.instruction_override = state.instruction_override;
        Ok(())
    }
}
```

---

## 7. Facet-Powered Parameter Discovery (F6)

### The walker

```rust
pub fn named_parameters<'a>(
    root: &'a dyn Reflect,  // or: root with known Facet Shape
) -> Vec<(String, /* handle to predictor */)> {
    let mut results = Vec::new();
    walk_value(root, "", &mut results);
    results
}

fn walk_value(value: /* Peek or similar */, path: &str, results: &mut Vec<...>) {
    let shape = value.shape();

    // Check: is this a parameter leaf?
    if has_dsrs_parameter(shape) {
        results.push((path.to_string(), /* extract DynPredictor handle */));
        return;  // don't recurse into Predict's internals
    }

    // Recurse based on shape.def
    // V1: struct-field recursion only. Container traversal (Option/Vec/HashMap/Box)
    // deferred (S5) — all V1 library modules use struct fields.
    match shape.def {
        Def::Struct(struct_type) => {
            for field in struct_type.fields {
                let child = value.field(field.name);
                let child_path = format!("{}.{}", path, field.name);
                walk_value(child, &child_path, results);
            }
        }
        _ => {} // containers, primitives, enums — skip for V1
    }
}
```

### What the user writes

```rust
#[derive(Facet)]
pub struct RAG {
    retrieve: Predict<Retrieve>,
    answer: ChainOfThought<Answer>,
}
```

### What the walker produces

```
[
    ("retrieve", handle to Predict<Retrieve>),
    ("answer.predict", handle to Predict<Augmented<Answer, Reasoning>>),
]
```

The walker recurses into `ChainOfThought<Answer>` (a struct with a `predict` field), finds the Predict inside, and reports the dotted path. Identical to DSPy's `named_parameters()` output.

### How the handle works (S2 resolved: Mechanism A)

The walker finds a value whose Shape has `dsrs::parameter`. It needs to hand back something the optimizer can call `get_demos()`, `set_demos()`, `set_instruction()` on.

S2 evaluated three mechanisms and selected **Mechanism A: shape-local accessor payload**. `Predict<S>` carries a `PredictAccessorFns` payload as a typed Facet attribute (fn-pointer based, `'static + Copy`). The walker extracts it via `attr.get_as::<PredictAccessorFns>()` — the same pattern already used by `WithAdapterFns` in `bamltype/src/facet_ext.rs`. The payload provides a direct cast to `&mut dyn DynPredictor` at the leaf, with one audited unsafe boundary.

Global registry (Mechanism B) is deferred — only needed if cross-crate runtime loading is later required. Interior dyn-handle state (Mechanism C) was rejected for V1 (see `S2-dynpredictor-handle-discovery.md`).

---

## 8. Adapter Building Blocks (F7)

### Public API for module authors

```rust
impl ChatAdapter {
    /// Build system message from a SignatureSchema
    pub fn build_system(
        schema: &SignatureSchema,
        instruction_override: Option<&str>,
    ) -> String;

    /// Format a typed input value as user message fields
    /// Uses Facet Peek to walk the value generically
    pub fn format_input<I: BamlType + Facet>(
        schema: &SignatureSchema,
        input: &I,
    ) -> String;

    /// Format a typed output value as assistant message fields
    pub fn format_output<O: BamlType + Facet>(
        schema: &SignatureSchema,
        output: &O,
    ) -> String;

    /// Parse [[ ## field ## ]] sections from raw LM response
    pub fn parse_sections(content: &str) -> IndexMap<String, String>;

    /// Parse typed output from sections using SignatureSchema
    pub fn parse_output<O: BamlType + Facet>(
        schema: &SignatureSchema,
        response: &Message,
    ) -> Result<O, ParseError>;
}
```

### How format_input uses Facet

```rust
pub fn format_input<I: BamlType + Facet>(
    schema: &SignatureSchema,
    input: &I,
) -> String {
    let baml_value = input.to_baml_value();
    let fields = baml_value_fields(&baml_value);

    let mut result = String::new();
    for field_schema in &schema.input_fields {
        // Navigate the BamlValue using the field's path
        if let Some(value) = navigate_path(fields, &field_schema.path) {
            result.push_str(&format!("[[ ## {} ## ]]\n", field_schema.name));
            result.push_str(&format_value(value, field_schema.format.as_deref()));
            result.push_str("\n\n");
        }
    }

    // Append response instructions
    result.push_str(&format_response_instructions(&schema.output_fields));
    result
}
```

The `field_schema.path` handles flatten navigation. For a flat field like `question`, path is `["question"]`. For a flattened field like `answer` inside `WithReasoning<QAOutput>`, path is `["inner", "answer"]`. `navigate_path` follows the path through the BamlValue tree.

### How parse_output uses field paths

```rust
pub fn parse_output<O: BamlType + Facet>(
    schema: &SignatureSchema,
    response: &Message,
) -> Result<O, ParseError> {
    let sections = parse_sections(&response.content());

    // Build a nested BamlValue following field paths
    let mut root = BamlMap::new();
    for field_schema in &schema.output_fields {
        let raw = sections.get(&field_schema.name)
            .ok_or(ParseError::MissingField { field: field_schema.name.clone() })?;

        // Coerce raw text to typed value via jsonish
        let value = jsonish::from_str(
            &schema.output_format,
            &field_schema.type_ir,
            raw,
            true,
        )?;

        // Insert at path
        insert_at_path(&mut root, &field_schema.path, value.into());
    }

    // Convert nested BamlValue to typed output
    O::try_from_baml_value(BamlValue::Class(
        O::baml_internal_name().to_string(),
        root,
    )).map_err(|e| ParseError::ExtractionFailed { reason: e.to_string() })
}
```

The `insert_at_path` function creates nested BamlValue::Class entries as needed. For `path = ["inner", "answer"]`, it ensures `root["inner"]` exists as a Class, then inserts `answer` into it. This is how flat LM output maps back to nested Rust types.

---

## 9. DynPredictor: The Optimizer Bridge (F8)

```rust
pub trait DynPredictor: Send + Sync {
    /// The Facet-derived schema for this predictor
    fn schema(&self) -> &SignatureSchema;

    /// Current instruction (override or default)
    fn instruction(&self) -> String;
    fn set_instruction(&mut self, instruction: String);

    /// Demos as untyped Examples (for optimizer manipulation)
    fn demos_as_examples(&self) -> Vec<Example>;
    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()>;

    /// Parameter state serialization
    fn dump_state(&self) -> PredictState;
    fn load_state(&mut self, state: PredictState) -> Result<()>;

    /// Untyped forward (for dynamic graph execution)
    async fn forward_untyped(&self, input: BamlValue) -> Result<BamlValue, PredictError>;
}
```

Every `Predict<S>` implements this. The implementation converts between typed and untyped representations:

```rust
impl<S: Signature> DynPredictor for Predict<S>
where S::Input: BamlType, S::Output: BamlType
{
    fn demos_as_examples(&self) -> Vec<Example> {
        self.demos.iter().map(|d| {
            let input_value = d.input.to_baml_value();
            let output_value = d.output.to_baml_value();
            // Merge into Example with input/output key separation
            Example::from_baml_values(input_value, output_value)
        }).collect()
    }

    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()> {
        self.demos = demos.into_iter().map(|e| {
            Ok(Demo {
                input: S::Input::try_from_baml_value(e.input_as_baml_value()?)?,
                output: S::Output::try_from_baml_value(e.output_as_baml_value()?)?,
            })
        }).collect::<Result<_>>()?;
        Ok(())
    }

    async fn forward_untyped(&self, input: BamlValue) -> Result<BamlValue, PredictError> {
        let typed_input = S::Input::try_from_baml_value(input)?;
        let output = self.call(typed_input).await?;
        Ok(output.to_baml_value())
    }
}
```

**How the Facet walker obtains a `&dyn DynPredictor`** — S2 Mechanism A. The walker detects `dsrs::parameter` on the Shape, extracts the `PredictAccessorFns` payload via typed attr decoding, and uses it to cast the value to `&dyn DynPredictor` (or `&mut dyn DynPredictor` for mutation). See section 7 for walker details.

**Type safety through the dynamic boundary:** The optimizer manipulates demos as untyped `Example` values, but `DynPredictor` is always backed by a concrete `Predict<S>` that knows its types at compile time. `set_demos_from_examples` converts `Example → Demo<S>` via `S::Input::try_from_baml_value()` / `S::Output::try_from_baml_value()` — if the data doesn't match the schema, this fails with an error, never silent data loss. The typed module is never replaced or wrapped by the optimizer; it reaches IN to the existing `Predict<S>` and mutates state. When the optimizer is done, the user's module still has correctly typed demos because the conversion gatekeeper enforced the schema at every write.

---

## 10. DynModule + StrategyFactory (F9)

### DynModule

```rust
#[async_trait]
pub trait DynModule: Send + Sync {
    /// The external schema (what callers see)
    fn schema(&self) -> &SignatureSchema;

    /// All internal predictors for optimizer discovery
    fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)>;
    fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)>;

    /// Execute with untyped values
    async fn forward(&self, input: BamlValue) -> Result<BamlValue>;
}
```

### StrategyFactory

```rust
pub trait StrategyFactory: Send + Sync {
    fn name(&self) -> &str;
    fn config_schema(&self) -> StrategyConfigSchema;
    fn create(
        &self,
        base_schema: &SignatureSchema,
        config: StrategyConfig,
    ) -> Result<Box<dyn DynModule>>;
}
```

Factories perform **schema transformations**. `ChainOfThoughtFactory::create(schema)` prepends a `reasoning` field to `schema.output_fields`. `ReActFactory::create(schema, {tools})` builds an action schema and extract schema from the base schema + tool definitions.

### Registry

```rust
pub mod registry {
    pub fn get(name: &str) -> Result<&'static dyn StrategyFactory>;
    pub fn create(name: &str, schema: &SignatureSchema, config: StrategyConfig) -> Result<Box<dyn DynModule>>;
    pub fn list() -> Vec<&'static str>;
}
```

Module types register factories via `inventory::submit!` or similar distributed static init.

---

## 11. ProgramGraph (F10)

```rust
pub struct ProgramGraph {
    nodes: IndexMap<String, Node>,
    edges: Vec<Edge>,
}

pub struct Node {
    schema: SignatureSchema,
    module: Box<dyn DynModule>,
}

pub struct Edge {
    from_node: String,
    from_field: String,
    to_node: String,
    to_field: String,
}
```

### Edge validation at insertion time

```rust
impl ProgramGraph {
    pub fn connect(&mut self, from: &str, from_field: &str, to: &str, to_field: &str) -> Result<(), GraphError> {
        let from_type = self.nodes[from].schema.output_field(from_field)?.type_ir;
        let to_type = self.nodes[to].schema.input_field(to_field)?.type_ir;

        if !from_type.is_assignable_to(&to_type) {
            return Err(GraphError::TypeMismatch { /* ... */ });
        }

        self.edges.push(Edge { from_node: from.into(), from_field: from_field.into(), to_node: to.into(), to_field: to_field.into() });
        Ok(())
    }
}
```

### Execution

Topological sort → pipe BamlValues between nodes following edges. Each node's `DynModule::forward()` handles its internal orchestration (single LM call for Predict, multi-step loop for ReAct, etc.).

### Projection from typed modules

```rust
impl ProgramGraph {
    pub fn from_module<M: Facet>(module: &M) -> Self {
        let params = named_parameters(module);  // F6 walker
        let mut graph = ProgramGraph::new();
        for (path, predictor_handle) in params {
            graph.add_node(path, Node {
                schema: predictor_handle.schema().clone(),
                module: /* wrap predictor as DynModule */,
            });
        }
        // Edges: inferred from trace or explicit annotation
        graph
    }
}
```

---

## 12. Library Modules (F11)

### ChainOfThought

```rust
#[derive(Augmentation)]
#[augment(output, prepend)]
pub struct Reasoning {
    /// Step-by-step reasoning
    reasoning: String,
}

#[derive(Facet)]
pub struct ChainOfThought<S: Signature> {
    predict: Predict<Augmented<S, Reasoning>>,
}

#[async_trait]
impl<S: Signature> Module for ChainOfThought<S> {
    type Input = S::Input;
    type Output = WithReasoning<S::Output>;

    async fn forward(&self, input: S::Input) -> Result<WithReasoning<S::Output>, PredictError> {
        self.predict.call(input).await
    }
}
```

### BestOfN

```rust
#[derive(Facet)]
pub struct BestOfN<M: Module> {
    module: M,
    n: usize,
    threshold: f64,
    reward_fn: Box<dyn Fn(&M::Input, &M::Output) -> f64 + Send + Sync>,
}

#[async_trait]
impl<M: Module> Module for BestOfN<M>
where M::Input: Clone
{
    type Input = M::Input;
    type Output = M::Output;

    async fn forward(&self, input: M::Input) -> Result<M::Output, PredictError> {
        let mut best = None;
        let mut best_score = f64::NEG_INFINITY;
        for _ in 0..self.n {
            let output = self.module.forward(input.clone()).await?;
            let score = (self.reward_fn)(&input, &output);
            if score >= self.threshold { return Ok(output); }
            if score > best_score { best_score = score; best = Some(output); }
        }
        best.ok_or(PredictError::AllAttemptsFailed)
    }
}
```

### ReAct

Uses generic signature derives (F12) for action and extract steps. Builder API for tools. Action loop uses adapter building blocks (F7) for dynamic trajectory formatting. Extraction step uses `ChainOfThought<Extract<S::Input, S::Output>>`.

Two Predict leaves: `action` and `extract.predict`. Both discoverable by F6 walker.

### ProgramOfThought

Three ChainOfThought modules with custom generic signatures (`CodeGen<I>`, `CodeRegen<I>`, `OutputInterp<I, O>`). Code execution loop with retry. Three Predict leaves discoverable by walker.

### Refine

BestOfN with feedback injection. Scoped context mechanism deferred (S4) — will be determined when Refine is built. Options investigated: `tokio::task_local!` (pragmatic, spawn footgun), explicit `Module::forward` context parameter (correct, invasive), `thread_local!` (rejected — brittle under async concurrency). See `S4-refine-scoped-context.md` for findings.

### MultiChainComparison

M source modules (typically ChainOfThought) run in parallel. Results aggregated into a comparison prompt via either a `Vec<String>` field or dynamic schema builder (F7). Comparison predictor produces synthesized output.

---

## 13. Layer Architecture

```
Layer 0: Types (always present)
  ┌──────────────────────────────────────────┐
  │ Rust types + Facet + BamlType            │
  │ Source of truth. In the binary.          │
  └──────────────────────────────────────────┘

Layer 1: Typed Modules (default experience)
  ┌──────────────────────────────────────────┐
  │ Signature (F1) → SignatureSchema (F2)    │
  │ Augmentation (F3) → Module trait (F4)    │
  │ Predict (F5) → Library modules (F11)     │
  │ Generic Signature derive (F12)           │
  └──────────────────────────────────────────┘

Layer 2: Optimization Bridge (when optimizing)
  ┌──────────────────────────────────────────┐
  │ Facet parameter discovery (F6)           │
  │ DynPredictor vtable (F8)                 │
  │ Adapter building blocks (F7)             │
  └──────────────────────────────────────────┘

Layer 3: Dynamic Graph (opt-in)
  ┌──────────────────────────────────────────┐
  │ DynModule + StrategyFactory (F9)         │
  │ ProgramGraph (F10)                       │
  │ Strategy registry                        │
  └──────────────────────────────────────────┘
```

Each layer only instantiated if needed. Simple usage: Layers 0-1. Optimization: add Layer 2. Structural optimization / interop: add Layer 3.

---

## 14. Key Design Decisions

| # | Decision | Rationale | Alternatives Rejected |
|---|----------|-----------|----------------------|
| **D1** | Augmentation via wrapper types + `#[flatten]` + `Deref`, not runtime signature manipulation | Type-safe, compile-time checked, Facet does the work | Runtime string-based field manipulation (DSPy-style) — gives up Rust's type system |
| **D2** | `SignatureSchema` derived from Facet at runtime, not emitted by macro as static arrays | Single source of truth (types), no drift between schema and type, flatten works generically | Static `FieldSpec` arrays (current) — can't handle generic augmentation |
| **D3** | Module trait with associated Input/Output types, not trait objects | Compile-time type safety for composition, no runtime type errors | `dyn Module` everywhere — loses all type checking |
| **D4** | Predict is the ONLY leaf parameter, discovered by Facet attribute | Same as DSPy: one optimization surface. Discovery is automatic | Multiple parameter types, manual registration |
| **D5** | Demos stored as typed `Demo<S>` not `Vec<S>` (combined signature struct) | Separates input/output cleanly, augmented demos naturally include strategy metadata | `Vec<S>` (current) — blocks augmented signatures where S is a phantom type |
| **D6** | Dynamic graph (Layer 3) shares `SignatureSchema` and adapter with typed path | One prompt format, validated edges, no divergence between paths | Separate dynamic system — double the code, divergent behavior |
| **D7** | Module authoring: `#[derive(Facet)]` on struct, `impl Module`, done | Zero framework tax. Structure is the declaration. | `#[derive(Optimizable)]` + `#[parameter]` — requires vigilance |
| **D8** | `#[derive(Augmentation)]` generates wrapper + Deref + trait impl | 5 lines for a new augmentation. Reusable across any Signature. | Manual wrapper + manual Signature impl per augmentation (verbose) |
| **D9** | StrategyFactory creates DynModules from name + schema + config | Optimizer can propose arbitrary strategies for graph nodes | Fixed set of typed strategies only — blocks structural optimization |

---

## 15. Spikes

All spikes have been investigated. Full findings in `spikes/S{n}-*.md`.

| # | Question | Decision | Spike doc |
|---|----------|----------|-----------|
| **S1** | Generic `#[derive(Signature)]` with `#[flatten]` type parameters | **Option C: full replacement.** Build `SignatureSchema` from Facet, replace `FieldSpec` everywhere, delete the old system. | `S1-generic-signature-derive.md` |
| **S2** | Mechanism for Facet walker to obtain `&dyn DynPredictor` | **Mechanism A**: shape-local accessor payload (`PredictAccessorFns` as typed Facet attr). | `S2-dynpredictor-handle-discovery.md` |
| **S3** | Rust auto-Deref chain for nested wrapper field access | **Works for reads/methods.** Pattern matching not ergonomic (don't care). `Deref`-only. | `S3-augmentation-deref-composition.md` |
| **S4** | Scoped context mechanism for Refine hint injection | **Deferred.** Determined when Refine is built. | `S4-refine-scoped-context.md` |
| **S5** | Facet walker behavior for Option/Vec/HashMap/Box containers | **Deferred.** Struct-field recursion covers V1 library modules. | `S5-facet-walker-containers.md` |
| **S6** | Migration from FieldSpec/MetaSignature to SignatureSchema | **Subsumed by S1 → Option C.** No migration — full replacement. | `S6-migration-fieldspec-to-signatureschema.md` |
| **S7** | `#[derive(Augmentation)]` feasibility + `Augmented` phantom type | **Feasible.** `from_parts`/`into_parts` removed from Signature. `Augmented` is a clean type-level combinator. | `S7-augmentation-derive-feasibility.md` |
| **S8** | Facet flatten metadata behavior | **`field.is_flattened()` + `field.shape()` recurse.** Direct mapping to design pseudocode. | `S8-facet-flatten-metadata.md` |

---

## 16. Typed vs Dynamic Path Summary

**The typed path (what users and module authors interact with):**

```rust
// User defines signature
#[derive(Signature, Clone)]
struct QA { ... }

// User picks a module
let cot = ChainOfThought::<QA>::new();
let result = cot.call(input).await?;
```

No factory. No registry. No DynModule. Pure typed Rust.

**The dynamic path (what the optimizer's internals use when doing structural optimization):**

```rust
// Inside the optimizer's own code, never user-facing:
let node = registry::create("chain_of_thought", &schema, config)?;
graph.replace_node("answer", node)?;
```

The factory exists so that the **optimizer** (an LM or search algorithm) can say "make me a ChainOfThought for this schema" without having the concrete Rust type `ChainOfThought<QA>` available. It's the bridge between "an optimizer proposes a strategy by name" and "a concrete module gets instantiated."

Nobody hand-writes a factory. Each library module (ChainOfThought, ReAct, etc.) **auto-registers** its factory as a side effect of existing. The registration could be:

```rust
// Inside the ChainOfThought module definition — library code, not user code
inventory::submit! {
    StrategyRegistration {
        name: "chain_of_thought",
        factory: &ChainOfThoughtFactory,
    }
}
```

Or it could be generated by whatever derive/macro defines the module.

**The hierarchy of who touches what:**

| Layer | Who interacts | What they see |
|---|---|---|
| Signature, Module, Predict | **Users** | Typed structs, `.call()`, field access |
| Augmentation derive | **Module authors** (library or advanced users) | 5-line struct with `#[derive(Augmentation)]` |
| Generic Signature derive | **Module authors** | `#[derive(Signature)]` with generics + flatten |
| SignatureSchema | **Nobody directly** — adapter uses it internally | Invisible |
| Facet walker + DynPredictor | **Nobody directly** — optimizer uses it internally | Invisible |
| StrategyFactory + registry | **Nobody directly** — structural optimizer uses it internally | Invisible |
| ProgramGraph | **Structural optimizer internals** OR advanced interop code | Rare, opt-in |

Layers below the line are pure plumbing. They exist so the system can optimize itself. The user never imports a Factory, never calls a registry, never constructs a DynModule.
