# DSRs + BAML Integration Specification

**Version:** 0.1.0  
**Status:** Draft  
**Last Updated:** 2026-01-08

> Status Update (2026-02-08):
> Legacy bridge crates are removed from the workspace.
> Current typed and optimizer contracts remain unchanged in Phase 1.
> Phase 2 next: compat-trait removal from typed paths plus signature/optimizer API redesign for facet-native runtime.
>
> Planning note:
> The “Implementation Order” section in this document is historical rollout guidance.
> Current execution status and cleanup-phase decision tracking are maintained in:
> - `docs/plans/modules/tracker.md`
> - `docs/plans/modules/slices_closure_audit.md`
> - `docs/plans/modules/phase_4_5_cleanup_kickoff.md`

---

## 1. Overview

### 1.1 Goals

1. **Type-native API**: Write Rust types, get Rust types back. No `HashMap<String, Value>` in user-facing code.
2. **Expressive constraints**: Jinja constraint expressions via attributes, evaluated at parse time.
3. **Robust parsing**: Replace `serde_json::from_str` with BAML's jsonish parser (handles malformed JSON, markdown fences, type coercion).
4. **Rust-idiomatic feel**: Docstrings for descriptions, derive macros, standard Rust patterns.
5. **Zero information loss**: Parse metadata (flags, coercions, constraint results) accessible when needed.
6. **Optimizer compatibility**: Internal `BamlValue` interchange allows optimizers to work generically.

### 1.2 Non-Goals (This Spec)

- Streaming API
- Custom render options configuration
- Breaking optimizer internals
- Crate restructuring decisions

### 1.3 Migration Impact

| Aspect | Current | New |
|--------|---------|-----|
| Macro | `#[Signature]` | `#[derive(Signature)]` |
| Construction | `Predict::new(QA::new())` | `Predict::<QA>::new()` |
| Input | `example!{ "k": "input" => v }` | `QAInput { k: v }` |
| Output access | `pred["field"].as_str().unwrap()` | `output.field` |
| Parsing | serde_json | jsonish (transparent) |
| Concepts | Same | Same |

---

## 2. Type System

### 2.1 Primitive Types

Built-in, no derive needed:

```rust
String
i8, i16, i32, i64, i128
u8, u16, u32, u64, u128
f32, f64
bool
Option<T>  // where T: BamlType
Vec<T>     // where T: BamlType
HashMap<String, T>  // where T: BamlType
```

### 2.2 `#[derive(BamlType)]`

For custom structs and enums used within signatures.

#### 2.2.1 Struct Syntax

```rust
/// Type-level description (appears in hoisted schema)
#[derive(BamlType)]
pub struct Answer {
    /// Field-level description
    pub text: String,
    
    /// Optional field - renders as `reasoning: string?`
    pub reasoning: Option<String>,
    
    /// With soft constraint
    #[check("this >= 0.0 && this <= 1.0", label = "range")]
    pub confidence: f32,
    
    /// With hard constraint
    #[assert("this.len() > 0")]
    pub required_text: String,
    
    /// With alias (LLM sees "output_name", Rust uses "rust_name")
    #[alias = "output_name"]
    pub rust_name: String,
}
```

#### 2.2.2 Enum Syntax

```rust
/// Enum-level description
#[derive(BamlType)]
pub enum Sentiment {
    /// Variant description
    Positive,
    
    Negative,
    
    /// With alias
    #[alias = "meh"]
    Neutral,
}
```

#### 2.2.3 Tagged Union Syntax

```rust
#[derive(BamlType)]
#[baml(tag = "type")]
pub enum Response {
    Text { content: String },
    Error { code: i32, message: String },
}
```

Renders as:
```
Response {
  type: "Text" | "Error"
  content?: string  // present when type = "Text"
  code?: int        // present when type = "Error"
  message?: string  // present when type = "Error"
}
```

#### 2.2.4 What Cannot Derive BamlType

These produce **compile errors**:

| Pattern | Error |
|---------|-------|
| Tuple struct `struct Foo(A, B)` | "tuple structs not supported, use named fields" |
| Unit struct `struct Foo;` | "unit structs not supported" |
| Tuple variant `enum E { V(A) }` | "tuple variants not supported, use `V { field: A }`" |
| `serde_json::Value` field | "dynamic JSON not supported, use concrete types" |
| `Box<dyn Trait>` field | "trait objects not supported" |

### 2.3 `#[derive(Signature)]`

Defines a complete signature with input and output fields.

#### 2.3.1 Syntax

```rust
/// Instruction text (becomes the signature instruction)
#[derive(Signature)]
pub struct QA {
    /// Input field description
    #[input]
    pub question: String,
    
    /// Another input (optional)
    #[input]
    pub context: Option<String>,
    
    /// Output field description
    #[output]
    pub answer: String,
    
    /// Output with constraint
    #[output]
    #[check("this >= 0.0 && this <= 1.0")]
    pub confidence: f32,
}
```

#### 2.3.2 Generated Code

For the above, the macro generates:

```rust
// Input struct - only #[input] fields
#[derive(Debug, Clone, BamlType)]
pub struct QAInput {
    /// Input field description
    pub question: String,
    /// Another input (optional)
    pub context: Option<String>,
}

// Original struct unchanged, but with BamlType impl
impl BamlType for QA { ... }

// Signature trait impl
impl Signature for QA {
    type Input = QAInput;
    
    fn instruction() -> &'static str {
        "Instruction text"
    }
    
    fn input_type_ir() -> TypeIR { ... }
    fn output_type_ir() -> TypeIR { ... }
    fn output_format_content() -> OutputFormatContent { ... }
}
```

#### 2.3.3 Field Rules

| Rule | Violation | Error Message |
|------|-----------|---------------|
| Every field must be `#[input]` or `#[output]` | Bare field | "field `{name}` must be marked `#[input]` or `#[output]`" |
| Not both | `#[input] #[output]` | "field `{name}` cannot be both `#[input]` and `#[output]`" |
| At least one output | All inputs | "signature `{name}` must have at least one `#[output]` field" |
| At least one input | All outputs | "signature `{name}` must have at least one `#[input]` field" |

---

## 3. Attributes

### 3.1 `#[input]`

Marks a field as an input to the signature.

```rust
#[input]
pub question: String,
```

No parameters. Description comes from docstring.

### 3.2 `#[output]`

Marks a field as an output from the signature.

```rust
#[output]
pub answer: String,
```

No parameters. Description comes from docstring.

### 3.3 `#[check("expr", label = "name")]`

Soft constraint. Evaluated at parse time. Failure produces metadata, not error.

```rust
#[check("this.len() < 1000", label = "length")]
pub text: String,
```

- `expr`: Jinja expression. `this` refers to the field value.
- `label`: Required. Identifies the constraint in metadata.

If `label` omitted:
```
error: #[check] requires a label
  --> src/lib.rs:4:5
   |
 4 |     #[check("this.len() < 1000")]
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = help: add label: #[check("this.len() < 1000", label = "length")]
```

### 3.4 `#[assert("expr")]`

Hard constraint. Evaluated at parse time. Failure produces `ParseError`.

```rust
#[assert("this >= 0.0 && this <= 1.0")]
pub confidence: f32,
```

- `expr`: Jinja expression. `this` refers to the field value.
- Label optional (defaults to field name + expression hash for identification).

```rust
// explicit label
#[assert("this > 0", label = "positive")]
pub count: i32,
```

### 3.5 `#[alias = "name"]`

LLM sees alias, Rust code uses field name.

```rust
#[alias = "user_query"]
pub question: String,
```

Prompt renders: `user_query: string`  
Rust access: `output.question`

### 3.6 Docstrings

Triple-slash comments become descriptions.

```rust
/// This appears as field description in prompt
#[input]
pub question: String,
```

For types:
```rust
/// This appears in hoisted schema definition
#[derive(BamlType)]
pub struct Answer { ... }
```

---

## 4. Constraint Expression Language

### 4.1 Scope

Within constraint expressions:

| Identifier | Meaning |
|------------|---------|
| `this` | The field value being validated |

### 4.2 Supported Operations

Jinja subset:

```python
# Comparison
this > 0
this >= 0
this < 100
this <= 100
this == "expected"
this != "bad"

# Boolean
this and other
this or other
not this

# String
this.len()
this.lower()
this.upper()
this.startswith("prefix")
this.endswith("suffix")
this.contains("sub")

# Numeric
this + 1
this - 1
this * 2
this / 2

# Collection
this.len()
this[0]
this["key"]

# Ternary
"yes" if this > 0 else "no"
```

### 4.3 Invalid Expressions

Compile-time error for malformed expressions:

```rust
#[check("this.len() < ")]  // incomplete
```
```
error: invalid constraint expression
  --> src/lib.rs:4:13
   |
 4 |     #[check("this.len() < ")]
   |             ^^^^^^^^^^^^^^^^
   |
   = note: unexpected end of expression
   = help: complete the comparison, e.g., "this.len() < 100"
```

---

## 5. Traits

### 5.1 `BamlType`

Implemented by all types usable in signatures.

```rust
pub trait BamlType: Sized + Send + Sync {
    /// TypeIR representation for parsing/rendering
    fn type_ir() -> TypeIR;
    
    /// Convert to BamlValue (for optimizer interchange)
    fn to_baml_value(&self) -> BamlValue;
    
    /// Convert from BamlValue
    fn from_baml_value(value: BamlValue) -> Result<Self, ConversionError>;
    
    /// Type name for error messages
    fn type_name() -> &'static str;
}
```

Implemented for:
- All primitives (String, i32, f32, bool, etc.)
- `Option<T: BamlType>`
- `Vec<T: BamlType>`
- `HashMap<String, T: BamlType>`
- Any `#[derive(BamlType)]` struct/enum

### 5.2 `Signature`

Marker trait with associated types for typed signatures.

```rust
pub trait Signature: BamlType + Sized + Send + Sync + 'static {
    type Input: BamlType;
    
    /// Instruction text
    fn instruction() -> &'static str;
    
    /// TypeIR for input fields
    fn input_type_ir() -> TypeIR;
    
    /// TypeIR for output fields  
    fn output_type_ir() -> TypeIR;
    
    /// Precomputed output format for prompt rendering
    fn output_format_content() -> OutputFormatContent;
    
    /// Field specs for ChatAdapter
    fn input_fields() -> &'static [FieldSpec];
    fn output_fields() -> &'static [FieldSpec];
}
```

### 5.3 `FieldSpec`

Static field metadata.

```rust
pub struct FieldSpec {
    pub name: &'static str,
    pub rust_name: &'static str,  // if different from name (alias)
    pub description: &'static str,
    pub type_ir: TypeIR,
    pub constraints: &'static [ConstraintSpec],
}

pub struct ConstraintSpec {
    pub kind: ConstraintKind,
    pub label: &'static str,
    pub expression: &'static str,
}

pub enum ConstraintKind {
    Check,   // soft
    Assert,  // hard
}
```

---

## 6. Predictor API

### 6.1 Construction

```rust
let predict = Predict::<QA>::new();
```

With configuration:
```rust
let predict = Predict::<QA>::builder()
    .temperature(0.7)
    .max_tokens(1024)
    .build();
```

### 6.2 Basic Call

```rust
let output: QA = predict.call(QAInput {
    question: "Why is the sky blue?".into(),
    context: None,
}).await?;

// Access typed fields
println!("{}", output.question);  // input preserved
println!("{}", output.answer);    // from LLM
println!("{}", output.confidence);
```

### 6.3 Call with Metadata

```rust
let result: CallResult<QA> = predict.call_with_meta(QAInput {
    question: "Why is the sky blue?".into(),
    context: None,
}).await?;

// Typed output
let output: &QA = &result.output;

// Metadata access
let raw: &str = &result.raw_response;
let usage: &LmUsage = &result.lm_usage;

// Per-field flags
let confidence_flags: &[Flag] = result.field_flags("confidence");
let confidence_checks: &[ConstraintResult] = result.field_checks("confidence");
```

### 6.4 `CallResult<O>` Definition

```rust
pub struct CallResult<O> {
    /// Typed output
    pub output: O,
    
    /// Raw LLM response text
    pub raw_response: String,
    
    /// Token usage
    pub lm_usage: LmUsage,
    
    /// Per-field metadata
    fields: IndexMap<String, FieldMeta>,
    
    /// Tool calls if any
    pub tool_calls: Vec<ToolCall>,
    pub tool_executions: Vec<String>,
    
    /// Trace node ID
    pub node_id: Option<usize>,
}

impl<O> CallResult<O> {
    /// Get flags for a specific field
    pub fn field_flags(&self, field: &str) -> &[Flag] {
        self.fields.get(field)
            .map(|m| m.flags.as_slice())
            .unwrap_or(&[])
    }
    
    /// Get constraint results for a specific field
    pub fn field_checks(&self, field: &str) -> &[ConstraintResult] {
        self.fields.get(field)
            .map(|m| m.checks.as_slice())
            .unwrap_or(&[])
    }
    
    /// Get raw text extracted for a field
    pub fn field_raw(&self, field: &str) -> Option<&str> {
        self.fields.get(field).map(|m| m.raw_text.as_str())
    }
}
```

### 6.5 `FieldMeta` Definition

```rust
pub struct FieldMeta {
    /// Raw text extracted from response for this field
    pub raw_text: String,
    
    /// Parse/coercion flags
    pub flags: Vec<Flag>,
    
    /// Constraint check results (soft constraints only)
    pub checks: Vec<ConstraintResult>,
}

pub struct ConstraintResult {
    pub label: String,
    pub expression: String,
    pub passed: bool,
}
```

### 6.6 Demos

```rust
// Builder pattern
let predict = Predict::<QA>::new()
    .demo(QA {
        question: "What is 2+2?".into(),
        context: None,
        answer: "4".into(),
        confidence: 1.0,
    })
    .demo(QA {
        question: "What is the capital of France?".into(),
        context: None,
        answer: "Paris".into(),
        confidence: 0.95,
    });

// Or batch
let predict = Predict::<QA>::new()
    .with_demos(vec![demo1, demo2, demo3]);
```

Demos are full `QA` structs (both input and output fields populated).

---

## 7. Error Handling

### 7.1 Error Type Hierarchy

```
PredictError
├── Lm { source: LmError }
├── Parse { source: ParseError, raw_response, lm_usage }
└── Conversion { source: ConversionError, parsed: BamlValue }

ParseError
├── MissingField { field, raw_response }
├── ExtractionFailed { field, raw_response, reason }
├── CoercionFailed { field, expected_type, raw_text, source: JsonishError }
├── AssertFailed { field, label, expression, value }
└── Multiple { errors: Vec<ParseError>, partial: Option<BamlValue> }

ConversionError
├── TypeMismatch { expected, actual }
├── MissingField { class, field }
└── UnknownVariant { enum_name, got }

LmError
├── Network { endpoint, source: io::Error }
├── RateLimit { retry_after: Option<Duration> }
├── InvalidResponse { status, body }
├── Timeout { after: Duration }
└── Provider { provider, message, source: Box<dyn Error> }
```

### 7.2 `PredictError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum PredictError {
    #[error("LLM call failed")]
    Lm {
        #[source]
        source: LmError,
    },
    
    #[error("failed to parse LLM response")]
    Parse {
        #[source]
        source: ParseError,
        /// The raw response we tried to parse
        raw_response: String,
        /// Usage still counts even on parse failure
        lm_usage: LmUsage,
    },
    
    #[error("failed to convert parsed value to output type")]
    Conversion {
        #[source]
        source: ConversionError,
        /// What jsonish successfully parsed
        parsed: BamlValue,
    },
}
```

**Classification:**

```rust
impl PredictError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Lm { source } => source.class(),
            Self::Parse { .. } => ErrorClass::BadResponse,
            Self::Conversion { .. } => ErrorClass::Internal,
        }
    }
    
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Lm { source } => source.is_retryable(),
            Self::Parse { .. } => true,  // different response might parse
            Self::Conversion { .. } => false,  // type mismatch won't fix itself
        }
    }
}
```

### 7.3 `ParseError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("field `{field}` not found in response")]
    MissingField {
        field: String,
        raw_response: String,
    },
    
    #[error("could not extract field `{field}` from response")]
    ExtractionFailed {
        field: String,
        raw_response: String,
        reason: String,  // e.g., "no closing marker found"
    },
    
    #[error("field `{field}` could not be parsed as {expected_type}")]
    CoercionFailed {
        field: String,
        expected_type: String,  // TypeIR display string
        raw_text: String,
        #[source]
        source: JsonishError,
    },
    
    #[error("assertion `{label}` failed on field `{field}`")]
    AssertFailed {
        field: String,
        label: String,
        expression: String,
        value: BamlValue,
    },
    
    #[error("{} field(s) failed to parse", errors.len())]
    Multiple {
        errors: Vec<ParseError>,
        /// Partial result if some fields succeeded
        partial: Option<BamlValue>,
    },
}
```

**Field accessors:**

```rust
impl ParseError {
    /// Which field failed (if single field error)
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::MissingField { field, .. } => Some(field),
            Self::ExtractionFailed { field, .. } => Some(field),
            Self::CoercionFailed { field, .. } => Some(field),
            Self::AssertFailed { field, .. } => Some(field),
            Self::Multiple { .. } => None,
        }
    }
    
    /// All fields that failed
    pub fn fields(&self) -> Vec<&str> {
        match self {
            Self::Multiple { errors, .. } => {
                errors.iter().filter_map(|e| e.field()).collect()
            }
            other => other.field().into_iter().collect(),
        }
    }
}
```

### 7.4 `ConversionError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: String,
    },
    
    #[error("missing required field `{field}` in class `{class}`")]
    MissingField {
        class: String,
        field: String,
    },
    
    #[error("enum `{enum_name}` has no variant `{got}`")]
    UnknownVariant {
        enum_name: String,
        got: String,
        valid_variants: Vec<String>,
    },
}
```

### 7.5 `LmError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum LmError {
    #[error("could not reach {endpoint}")]
    Network {
        endpoint: String,
        #[source]
        source: std::io::Error,
    },
    
    #[error("rate limited by provider")]
    RateLimit {
        retry_after: Option<Duration>,
    },
    
    #[error("invalid response from provider: HTTP {status}")]
    InvalidResponse {
        status: u16,
        body: String,
    },
    
    #[error("request timed out after {after:?}")]
    Timeout {
        after: Duration,
    },
    
    #[error("provider error from {provider}: {message}")]
    Provider {
        provider: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl LmError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Network { .. } => ErrorClass::Temporary,
            Self::RateLimit { .. } => ErrorClass::Temporary,
            Self::InvalidResponse { status, .. } if *status >= 500 => ErrorClass::Temporary,
            Self::InvalidResponse { .. } => ErrorClass::BadRequest,
            Self::Timeout { .. } => ErrorClass::Temporary,
            Self::Provider { .. } => ErrorClass::Internal,
        }
    }
    
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network { .. }
                | Self::RateLimit { .. }
                | Self::Timeout { .. }
                | Self::InvalidResponse { status, .. } if *status >= 500
        )
    }
}
```

### 7.6 `ErrorClass`

```rust
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ErrorClass {
    /// Bad input from caller
    BadRequest,
    /// Valid request, but resource not found
    NotFound,
    /// Authentication/authorization failure
    Forbidden,
    /// Temporary failure, retry may help
    Temporary,
    /// Invalid response from LLM
    BadResponse,
    /// Internal error (bug or unexpected state)
    Internal,
}
```

### 7.7 Error Display Examples

**MissingField:**
```
PredictError: failed to parse LLM response

Caused by:
    field `confidence` not found in response
    
    raw response:
      [[ ## answer ## ]]
      The sky is blue because of Rayleigh scattering.
      [[ ## completed ## ]]
```

**CoercionFailed:**
```
PredictError: failed to parse LLM response

Caused by:
    field `confidence` could not be parsed as float
    
    expected: float
    raw text: "very confident"
    
    jsonish error: expected numeric value, got string "very confident"
```

**AssertFailed:**
```
PredictError: failed to parse LLM response

Caused by:
    assertion `range` failed on field `confidence`
    
    expression: this >= 0.0 && this <= 1.0
    value: 1.5
```

**Multiple:**
```
PredictError: failed to parse LLM response

Caused by:
    2 field(s) failed to parse
    
    1. field `confidence` could not be parsed as float
       raw text: "high"
       
    2. assertion `non_empty` failed on field `answer`
       expression: this.len() > 0
       value: ""
```

---

## 8. Flags (Parse Metadata)

### 8.1 `Flag` Enum

```rust
#[derive(Debug, Clone)]
pub enum Flag {
    // Source format normalization
    ObjectFromMarkdown,
    ObjectFromFixedJson { fixes: Vec<JsonFix> },
    
    // Type coercions
    StringToBool { original: String },
    StringToInt { original: String },
    StringToFloat { original: String },
    FloatToInt { original: f64 },
    
    // Structure coercions
    SingleToArray,
    ObjectToString { original: BamlValue },
    ImpliedKey { key: String },
    
    // Value normalization
    StrippedNonAlphaNumeric { original: String },
    SubstringMatch { original: String },
    
    // Constraint results (checks only, not asserts)
    CheckPassed { label: String, expression: String },
    CheckFailed { label: String, expression: String },
    
    // Defaults
    DefaultFromNoValue,
    OptionalDefaultFromNoValue,
}
```

### 8.2 `JsonFix` Enum

```rust
#[derive(Debug, Clone)]
pub enum JsonFix {
    AddedMissingQuotes { around: String },
    AddedMissingComma { after: String },
    AddedMissingBrace { kind: BraceKind },
    RemovedTrailingComma,
    UnescapedString { original: String },
}
```

### 8.3 Accessing Flags

```rust
let result = predict.call_with_meta(input).await?;

// Per-field flags
for flag in result.field_flags("confidence") {
    match flag {
        Flag::StringToFloat { original } => {
            println!("coerced '{}' to float", original);
        }
        Flag::CheckFailed { label, .. } => {
            println!("soft constraint '{}' failed", label);
        }
        _ => {}
    }
}

// Check if any coercion happened
let had_coercion = result.field_flags("confidence")
    .iter()
    .any(|f| matches!(f, 
        Flag::StringToFloat { .. } | 
        Flag::StringToInt { .. } |
        Flag::FloatToInt { .. }
    ));
```

---

## 9. ChatAdapter Changes

### 9.1 Current Flow

```
format() → Chat (prompt)
         ↓
      LM::call()
         ↓
parse_response() → HashMap<String, Value>
```

### 9.2 New Flow

```
format() → Chat (prompt with BAML-rendered schema)
         ↓
      LM::call()
         ↓
parse_response() → ParsedResponse { fields: IndexMap<String, FieldParseResult> }
```

### 9.3 `format()` Changes

**Before:**
```rust
fn format(&self, sig: &dyn MetaSignature, inputs: Example) -> Chat {
    // ... build field descriptions from JSON schema ...
    // ... embed schemars schema in prompt ...
}
```

**After:**
```rust
fn format<S: Signature>(&self, inputs: &S::Input) -> Chat {
    let schema = S::output_format_content()
        .render(RenderOptions::default())
        .expect("schema render");
    
    let system = format!(
        "{field_descriptions}\n\n\
         All interactions will be structured in the following way:\n\n\
         {input_structure}\
         {output_structure}\
         [[ ## completed ## ]]\n\n\
         Answer in this schema:\n{schema}\n\n\
         {instruction}",
        field_descriptions = self.format_field_descriptions::<S>(),
        input_structure = self.format_input_structure::<S>(),
        output_structure = self.format_output_structure::<S>(),
        schema = schema,
        instruction = S::instruction(),
    );
    
    let user = self.format_user_message::<S>(inputs);
    
    let mut chat = Chat::new(vec![]);
    chat.push("system", &system);
    // ... demos ...
    chat.push("user", &user);
    chat
}
```

### 9.4 `parse_response()` Changes

**Before:**
```rust
fn parse_response(&self, sig: &dyn MetaSignature, response: Message) -> HashMap<String, Value> {
    for (field_name, field) in get_iter_from_value(&sig.output_fields()) {
        let raw = extract_between_markers(&response.content(), &field_name);
        let value = serde_json::from_str(raw).unwrap();
        output.insert(field_name, value);
    }
    output
}
```

**After:**
```rust
fn parse_response<S: Signature>(
    &self,
    response: &Message,
) -> Result<ParsedResponse, ParseError> {
    let content = response.content();
    let mut fields = IndexMap::new();
    let mut errors = Vec::new();
    
    for field_spec in S::output_fields() {
        let field_name = field_spec.name;
        
        // Extract raw text for this field
        let raw_text = match self.extract_field(&content, field_name) {
            Ok(text) => text,
            Err(reason) => {
                errors.push(ParseError::ExtractionFailed {
                    field: field_name.to_string(),
                    raw_response: content.clone(),
                    reason,
                });
                continue;
            }
        };
        
        // Build OutputFormatContent for this field's type
        let output_format = OutputFormatContent::from_type_ir(
            field_spec.type_ir.clone()
        );
        
        // Parse with jsonish
        let parse_result = jsonish::from_str(
            &output_format,
            &field_spec.type_ir,
            &raw_text,
            true,  // is_complete
        );
        
        match parse_result {
            Ok(baml_value_with_flags) => {
                // Evaluate constraints
                let constraint_results = self.evaluate_constraints(
                    &baml_value_with_flags.value(),
                    field_spec.constraints,
                )?;
                
                // Check for assert failures
                for result in &constraint_results {
                    if result.kind == ConstraintKind::Assert && !result.passed {
                        errors.push(ParseError::AssertFailed {
                            field: field_name.to_string(),
                            label: result.label.clone(),
                            expression: result.expression.clone(),
                            value: baml_value_with_flags.value().clone(),
                        });
                    }
                }
                
                fields.insert(field_name.to_string(), FieldParseResult {
                    raw_text,
                    value: baml_value_with_flags.value().clone(),
                    flags: baml_value_with_flags.flags().to_vec(),
                    checks: constraint_results
                        .into_iter()
                        .filter(|r| r.kind == ConstraintKind::Check)
                        .collect(),
                });
            }
            Err(jsonish_err) => {
                errors.push(ParseError::CoercionFailed {
                    field: field_name.to_string(),
                    expected_type: field_spec.type_ir.to_string(),
                    raw_text,
                    source: jsonish_err,
                });
            }
        }
    }
    
    if errors.is_empty() {
        Ok(ParsedResponse { fields })
    } else if errors.len() == 1 {
        Err(errors.pop().unwrap())
    } else {
        Err(ParseError::Multiple {
            errors,
            partial: Some(self.build_partial_value(&fields)),
        })
    }
}
```

### 9.5 Field Extraction

Keeps existing `[[ ## field ## ]]` protocol:

```rust
fn extract_field(&self, content: &str, field_name: &str) -> Result<String, String> {
    let start_marker = format!("[[ ## {} ## ]]", field_name);
    let end_marker = "[[ ## ";
    
    let start = content.find(&start_marker)
        .ok_or_else(|| format!("start marker not found for field '{}'", field_name))?;
    
    let after_marker = start + start_marker.len();
    let rest = &content[after_marker..];
    
    let end = rest.find(end_marker)
        .unwrap_or(rest.len());
    
    Ok(rest[..end].trim().to_string())
}
```

---

## 10. Internal Interchange (BamlValue)

### 10.1 Purpose

Optimizers work at `BamlValue` level, not typed structs. This keeps them generic across all signatures.

### 10.2 Conversion Path

```
User types ←→ BamlValue ←→ Parsed response
     ↑              ↑
     |              |
  BamlType      jsonish
   trait        parser
```

### 10.3 Module Trait (Untyped)

```rust
pub trait Module: Send + Sync {
    /// Untyped forward pass for optimizers
    fn forward_untyped(
        &self,
        input: BamlValue,
    ) -> impl Future<Output = Result<BamlValue, PredictError>> + Send;
    
    /// Signature spec for introspection
    fn signature_spec(&self) -> &dyn SignatureSpec;
}
```

### 10.4 Predict Implements Both

```rust
impl<S: Signature> Predict<S> {
    /// Typed API
    pub async fn call(&self, input: S::Input) -> Result<S, PredictError> {
        // ... typed implementation ...
    }
}

impl<S: Signature> Module for Predict<S> {
    async fn forward_untyped(&self, input: BamlValue) -> Result<BamlValue, PredictError> {
        let typed_input = S::Input::from_baml_value(input)?;
        let typed_output = self.call(typed_input).await?;
        Ok(typed_output.to_baml_value())
    }
    
    fn signature_spec(&self) -> &dyn SignatureSpec {
        &S::SPEC  // static
    }
}
```

---

## 11. Compile-Time Error Messages

### 11.1 Missing Field Marker

```rust
#[derive(Signature)]
pub struct QA {
    question: String,  // no #[input] or #[output]
}
```
```
error: field `question` must be marked #[input] or #[output]
  --> src/lib.rs:4:5
   |
 4 |     question: String,
   |     ^^^^^^^^
   |
   = help: add #[input] for input fields or #[output] for output fields

error: could not compile `myproject` due to previous error
```

### 11.2 Both Input and Output

```rust
#[derive(Signature)]
pub struct QA {
    #[input]
    #[output]
    question: String,
}
```
```
error: field `question` cannot be both #[input] and #[output]
  --> src/lib.rs:4:5
   |
 3 |     #[input]
   |     -------- first marker here
 4 |     #[output]
   |     ^^^^^^^^^ second marker here
 5 |     question: String,
   |
   = help: remove one of the markers
```

### 11.3 No Outputs

```rust
#[derive(Signature)]
pub struct QA {
    #[input]
    question: String,
}
```
```
error: signature `QA` must have at least one #[output] field
  --> src/lib.rs:1:1
   |
 1 | pub struct QA {
   | ^^^^^^^^^^^^^
   |
   = help: add #[output] to at least one field
```

### 11.4 No Inputs

```rust
#[derive(Signature)]
pub struct QA {
    #[output]
    answer: String,
}
```
```
error: signature `QA` must have at least one #[input] field
  --> src/lib.rs:1:1
   |
 1 | pub struct QA {
   | ^^^^^^^^^^^^^
   |
   = help: add #[input] to at least one field
```

### 11.5 Invalid Constraint

```rust
#[derive(Signature)]
pub struct QA {
    #[input]
    question: String,
    
    #[output]
    #[check("this.len() < ")]
    answer: String,
}
```
```
error: invalid constraint expression in #[check]
  --> src/lib.rs:7:13
   |
 7 |     #[check("this.len() < ")]
   |             ^^^^^^^^^^^^^^^^
   |
   = note: unexpected end of expression after `<`
   = help: complete the comparison, e.g., #[check("this.len() < 100")]
```

### 11.6 Check Missing Label

```rust
#[check("this.len() < 100")]
```
```
error: #[check] requires a label parameter
  --> src/lib.rs:7:5
   |
 7 |     #[check("this.len() < 100")]
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = help: add label: #[check("this.len() < 100", label = "length")]
   = note: labels identify constraints in error messages and metadata
```

### 11.7 Type Does Not Implement BamlType

```rust
struct Custom { x: i32 }  // no derive

#[derive(Signature)]
pub struct QA {
    #[input]
    question: String,
    
    #[output]
    custom: Custom,
}
```
```
error[E0277]: the trait bound `Custom: BamlType` is not satisfied
  --> src/lib.rs:9:13
   |
 9 |     custom: Custom,
   |             ^^^^^^ the trait `BamlType` is not implemented for `Custom`
   |
   = help: the following other types implement trait `BamlType`:
             String
             i32
             f32
             bool
             Vec<T>
             Option<T>
             ...
   = help: consider adding #[derive(BamlType)] to `Custom`
```

### 11.8 Unsupported Type Pattern (BamlType)

```rust
#[derive(BamlType)]
pub struct Bad(String, i32);  // tuple struct
```
```
error: tuple structs are not supported
  --> src/lib.rs:2:1
   |
 2 | pub struct Bad(String, i32);
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = help: use named fields instead: `pub struct Bad { field1: String, field2: i32 }`
```

```rust
#[derive(BamlType)]
pub enum Bad {
    Variant(String),  // tuple variant
}
```
```
error: tuple variants are not supported
  --> src/lib.rs:3:5
   |
 3 |     Variant(String),
   |     ^^^^^^^^^^^^^^^
   |
   = help: use named fields instead: `Variant { value: String }`
```

---

## 12. Runtime Error Display

### 12.1 PredictError Display Format

```
PredictError: {summary}

Caused by:
    {cause chain, indented}

Context:
    {relevant fields}
```

### 12.2 Example: LM Network Error

```
PredictError: LLM call failed

Caused by:
    could not reach api.openai.com:443
    
    Connection refused (os error 111)

Context:
    provider: openai
    model: gpt-4o-mini
```

### 12.3 Example: Parse Coercion Failed

```
PredictError: failed to parse LLM response

Caused by:
    field `confidence` could not be parsed as float
    
    expected: float
    raw text: "I'm very confident about this answer"
    
    jsonish error: cannot coerce string to float - no numeric content found

Context:
    raw response:
      [[ ## answer ## ]]
      The sky is blue due to Rayleigh scattering of sunlight.
      [[ ## confidence ## ]]
      I'm very confident about this answer
      [[ ## completed ## ]]
    
    lm usage:
      prompt tokens: 127
      completion tokens: 42
```

### 12.4 Example: Assert Failed

```
PredictError: failed to parse LLM response

Caused by:
    assertion `range` failed on field `confidence`
    
    expression: this >= 0.0 && this <= 1.0
    value: 1.5

Context:
    raw response:
      [[ ## answer ## ]]
      The answer is 42.
      [[ ## confidence ## ]]
      1.5
      [[ ## completed ## ]]
```

### 12.5 Example: Multiple Failures

```
PredictError: failed to parse LLM response

Caused by:
    2 field(s) failed to parse
    
    [1] field `confidence` could not be parsed as float
        raw text: "high"
    
    [2] assertion `non_empty` failed on field `answer`
        expression: this.len() > 0
        value: ""

Context:
    raw response:
      [[ ## answer ## ]]
      
      [[ ## confidence ## ]]
      high
      [[ ## completed ## ]]
    
    partial parse:
      answer: "" (assertion failed)
      confidence: (coercion failed)
```

---

## 13. Open Questions (Deferred)

### 13.1 Streaming

- How does partial parsing surface to users?
- Does `call()` support streaming or separate method?
- How do flags/checks work incrementally?

### 13.2 Render Options

- Should users be able to configure schema rendering?
- Hoisting behavior?
- Or/union syntax?

### 13.3 Crate Structure

- Fold legacy bridge crates into dsrs?
- Keep as dependency?
- Public API surface considerations?

### 13.4 Backwards Compatibility

- Deprecation path for `example!` macro?
- Deprecation path for `HashMap` prediction access?
- Migration guide?

---

## 14. Implementation Order

### Phase 1: Foundation
1. Add legacy bridge crates as dependencies (or vendor)
2. Implement `BamlType` trait and primitive impls
3. Implement `#[derive(BamlType)]` macro

### Phase 2: Parsing
4. Modify `ChatAdapter::parse_response` to use jsonish
5. Add `ParseError` and related types
6. Test with existing signatures

### Phase 3: Typed API
7. Implement `Signature` trait
8. Implement `#[derive(Signature)]` macro (generates Input struct)
9. Implement `Predict::<S>` typed wrapper
10. Add `CallResult<O>` and metadata access

### Phase 4: Polish
11. Add `call_with_meta()`
12. Implement demo builder
13. Documentation and examples
14. Deprecation warnings for old API

---

## 15. Acceptance Criteria

### 15.1 Types Work

```rust
#[derive(BamlType)]
pub struct Answer {
    pub text: String,
    #[check("this >= 0.0 && this <= 1.0", label = "range")]
    pub confidence: f32,
}

#[derive(Signature)]
/// Answer questions accurately
pub struct QA {
    #[input]
    pub question: String,
    #[output]
    pub answer: Answer,
}

let predict = Predict::<QA>::new();
let output: QA = predict.call(QAInput { 
    question: "Why is the sky blue?".into() 
}).await?;

assert!(!output.question.is_empty());  // input preserved
assert!(!output.answer.text.is_empty());  // output populated
```

### 15.2 Errors Are Informative

```rust
// This should produce a clear compile error
#[derive(Signature)]
pub struct Bad {
    question: String,  // missing marker
}

// This should produce a clear runtime error with context
let result = predict.call(input).await;
if let Err(e) = result {
    assert!(e.to_string().contains("field"));
    assert!(e.to_string().contains("raw"));
}
```

### 15.3 Metadata Accessible

```rust
let result = predict.call_with_meta(input).await?;
let flags = result.field_flags("confidence");
let checks = result.field_checks("confidence");
let raw = result.field_raw("confidence");
```

### 15.4 Optimizers Still Work

```rust
// Existing optimizer code should compile and run
let optimizer = GEPA::builder().build();
optimizer.compile(&mut predict, trainset).await?;
```

---

*End of specification.*
