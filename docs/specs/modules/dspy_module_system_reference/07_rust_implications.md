# Rust Rewrite Implications

## What DSPy's Module System Actually Is

Strip away the Python dynamism and DSPy's module system is:

1. **A tree of composable nodes** where leaf nodes (Predict) hold optimizable state
2. **A typed I/O contract** (Signature) that describes what goes in and what comes out
3. **A formatting/parsing layer** (Adapter) that converts typed contracts to LM prompts and back
4. **A tree traversal** that lets optimizers discover and modify leaf nodes
5. **A tracing mechanism** that records execution for optimizer feedback

That's it. Everything else is orchestration (how modules compose Predicts) and strategy (how optimizers search the space).

---

## The Hard Problems

### 1. Dynamic Signature Manipulation

In Python, signatures are *classes* created at runtime via metaclass magic. Modules like ChainOfThought do `signature.prepend("reasoning", OutputField(...))` which creates a new type at runtime.

**In Rust**: Signatures are data, not types. Model them as:

```rust
struct Signature {
    name: String,
    instructions: String,
    fields: IndexMap<String, Field>,  // Ordered map (insertion order matters)
}

struct Field {
    direction: FieldDirection,  // Input | Output
    type_annotation: TypeAnnotation,
    prefix: String,
    desc: String,
    format: Option<Box<dyn Fn(&str) -> String>>,
    constraints: Option<String>,
}

enum FieldDirection {
    Input,
    Output,
}

enum TypeAnnotation {
    Str,
    Int,
    Float,
    Bool,
    List(Box<TypeAnnotation>),
    Dict(Box<TypeAnnotation>, Box<TypeAnnotation>),
    Optional(Box<TypeAnnotation>),
    Enum(Vec<String>),
    Literal(Vec<String>),
    Json(serde_json::Value),  // For complex types, store JSON schema
}
```

All manipulation methods (`with_instructions`, `prepend`, `append`, `delete`, `with_updated_fields`) return new `Signature` values. This maps cleanly to Rust's ownership model -- signatures are cheap to clone and manipulate.

### 2. The Parameter Tree Walk

Python does this by walking `__dict__` and checking `isinstance`. Rust doesn't have runtime reflection.

**Options**:

**Option A: Explicit children** (recommended)
```rust
trait Module {
    fn forward(&self, inputs: HashMap<String, Value>) -> Result<Prediction>;
    fn named_parameters(&self) -> Vec<(String, &dyn Parameter)>;
    fn named_sub_modules(&self) -> Vec<(String, &dyn Module)>;
}

trait Parameter: Module {
    fn demos(&self) -> &[Example];
    fn set_demos(&mut self, demos: Vec<Example>);
    fn signature(&self) -> &Signature;
    fn set_signature(&mut self, sig: Signature);
    fn dump_state(&self) -> serde_json::Value;
    fn load_state(&mut self, state: &serde_json::Value);
    fn reset(&mut self);
}
```

Each module explicitly returns its children. ChainOfThought returns `[("predict", &self.predict)]`. ReAct returns `[("react", &self.react), ("extract.predict", &self.extract.predict)]`.

**Option B: Derive macro**
```rust
#[derive(DspyModule)]
struct ChainOfThought {
    #[parameter]
    predict: Predict,
}
```

A proc macro generates `named_parameters()` by inspecting fields marked with `#[parameter]`.

**Option C: Inventory/registry** -- each module registers itself. More complex, probably overkill.

**Recommendation**: Start with Option A (explicit). It's simple, correct, and makes the tree structure obvious. Add a derive macro later if the boilerplate becomes painful.

### 3. The `_compiled` Freeze Flag

In Python, `_compiled = True` makes `named_parameters()` skip a sub-module. In Rust:

**Simple approach**: A boolean flag on every module, checked in `named_parameters()`.

**Type-state approach** (more Rusty):
```rust
struct CompiledModule<M: Module> {
    inner: M,
    // named_parameters() returns empty vec
    // Cannot be modified without explicitly un-compiling
}

impl<M: Module> Module for CompiledModule<M> {
    fn named_parameters(&self) -> Vec<(String, &dyn Parameter)> {
        vec![]  // Frozen -- parameters are not exposed
    }
    fn forward(&self, inputs: HashMap<String, Value>) -> Result<Prediction> {
        self.inner.forward(inputs)
    }
}
```

### 4. The Adapter System

Adapters are the most straightforward part to port. They're essentially:
- Template formatting (building message strings from signature + demos + inputs)
- Regex-based parsing (splitting LM output by `[[ ## field ## ]]` markers)
- Type coercion (parsing strings into typed values)

```rust
trait Adapter {
    fn format(&self, sig: &Signature, demos: &[Example], inputs: &HashMap<String, Value>) -> Vec<Message>;
    fn parse(&self, sig: &Signature, completion: &str) -> Result<HashMap<String, Value>>;
}

struct ChatAdapter;
struct JsonAdapter;
```

The fallback pattern (ChatAdapter -> JSONAdapter on parse failure) is just:
```rust
impl Adapter for ChatAdapter {
    fn call(&self, lm: &LM, sig: &Signature, demos: &[Example], inputs: &HashMap<String, Value>) -> Result<Vec<HashMap<String, Value>>> {
        match self.try_call(lm, sig, demos, inputs) {
            Ok(result) => Ok(result),
            Err(e) if !e.is_context_window_error() => {
                JsonAdapter.call(lm, sig, demos, inputs)
            }
            Err(e) => Err(e),
        }
    }
}
```

### 5. Tracing

Python uses a global thread-local list that Predicts append to. In Rust:

```rust
// Thread-local trace context
thread_local! {
    static TRACE: RefCell<Option<Vec<TraceEntry>>> = RefCell::new(None);
}

struct TraceEntry {
    predictor_id: PredictorId,  // Not a reference -- an ID for lookup
    inputs: HashMap<String, Value>,
    prediction: Prediction,
}

// In Predict::forward:
TRACE.with(|trace| {
    if let Some(ref mut trace) = *trace.borrow_mut() {
        trace.push(TraceEntry { predictor_id: self.id, inputs, prediction });
    }
});

// In optimizer:
let trace = with_trace(|| teacher.forward(example.inputs()));
```

Use IDs instead of references. Python uses `id(predictor)` (memory address); Rust should use a stable identifier (UUID, path string, or index).

### 6. Value Types and Parsing

DSPy uses Python's dynamic typing + Pydantic for validation. In Rust, you need a value type:

```rust
enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<Value>),
    Dict(HashMap<String, Value>),
    Null,
    Json(serde_json::Value),  // For complex/unknown types
}
```

Parsing (`parse_value` equivalent):
```rust
fn parse_value(raw: &str, annotation: &TypeAnnotation) -> Result<Value> {
    match annotation {
        TypeAnnotation::Str => Ok(Value::Str(raw.to_string())),
        TypeAnnotation::Int => raw.parse::<i64>().map(Value::Int),
        TypeAnnotation::Bool => parse_bool(raw),
        TypeAnnotation::Enum(variants) => parse_enum(raw, variants),
        TypeAnnotation::Literal(allowed) => parse_literal(raw, allowed),
        TypeAnnotation::Json(schema) => {
            let v: serde_json::Value = serde_json::from_str(raw)?;
            // Validate against schema
            Ok(Value::Json(v))
        }
        // ...
    }
}
```

---

## What to Build First

### Phase 1: Core Primitives
1. `Signature` struct with manipulation methods
2. `Field` and `TypeAnnotation`
3. `Value` enum for dynamic values
4. `Example` and `Prediction` data containers

### Phase 2: Module System
1. `Module` trait with `forward()` and `named_parameters()`
2. `Parameter` trait extending Module
3. `Predict` implementing both
4. `BaseModule` trait for tree traversal, serialization

### Phase 3: Adapter Layer
1. `Adapter` trait
2. `ChatAdapter` (formatting and parsing)
3. `JsonAdapter`
4. `parse_value` for type coercion

### Phase 4: Composition Modules
1. `ChainOfThought` (signature extension pattern)
2. `ReAct` (multi-signature orchestration pattern)
3. `BestOfN` / `Refine` (module wrapping pattern)

### Phase 5: Optimization
1. Tracing infrastructure
2. `Evaluate`
3. `BootstrapFewShot`
4. `LabeledFewShot`
5. More complex optimizers as needed

---

## Design Decisions to Make Early

### 1. Static vs Dynamic Signatures

Python signatures carry Python types (Pydantic models, etc.). Rust signatures will need to decide:
- **Fully dynamic** (`TypeAnnotation` enum + `Value` enum) -- flexible, similar to Python, but loses Rust's type safety
- **Partially typed** (generics for common cases, `Value` for complex) -- more Rusty but more complex
- **Schema-driven** (JSON Schema as the universal type description) -- pragmatic, works with any LM

**Recommendation**: Start fully dynamic. The type safety that matters here is at the *LM boundary* (parsing), not at compile time. You're dealing with strings from an LM no matter what.

### 2. Ownership of Demos and Signatures

In Python, optimizers freely mutate `predictor.demos` and `predictor.signature`. In Rust:
- **Mutable references**: Optimizers take `&mut` references to the program
- **Interior mutability**: Use `RefCell<Vec<Example>>` for demos
- **Clone + replace**: Clone the whole program, modify the clone, return it (matches Python's `reset_copy()` pattern)

**Recommendation**: Clone + replace. It matches the Python pattern where optimizers always copy the student first, and it avoids fighting the borrow checker.

### 3. Async vs Sync

LM calls are inherently async (HTTP requests). The question is whether `forward()` should be async.

**Recommendation**: Make it async from the start. `async fn forward(&self, ...) -> Result<Prediction>`. Easier than retrofitting later.

### 4. Error Types

DSPy uses `AdapterParseError`, `ContextWindowExceededError`, and generic exceptions. Design a clean error enum:

```rust
enum DspyError {
    ParseError { adapter: String, raw: String, partial: HashMap<String, Value> },
    ContextWindowExceeded { model: String, token_count: usize },
    MissingInput { field: String },
    LmError(Box<dyn std::error::Error>),
    // ...
}
```

---

## What NOT to Port

1. **The metaclass machinery** (`ProgramMeta`, `SignatureMeta`). These exist to paper over Python's limitations. Rust structs with derive macros are cleaner.

2. **`magicattr`** (AST-based nested attribute access). In Rust, named_parameters returns paths; use them to index directly.

3. **`__getattribute__` forward-call guard**. In Rust, make `forward()` private and only expose `call()`.

4. **Dynamic `__dict__` walking**. Replace with explicit trait implementations.

5. **`cloudpickle` serialization**. Use `serde` with JSON/MessagePack. The "save whole program" feature is Python-specific.

6. **The Settings singleton**. Use explicit context passing or a structured configuration type.
