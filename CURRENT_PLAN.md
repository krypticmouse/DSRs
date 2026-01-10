Below is a “walk the codebase” integration plan that’s detailed enough to be used as a checklist while you implement. I’m going to treat `CURRENT_SPEC.md` as the source of truth, and I’ll point out the few places where the spec implies machinery you don’t currently have (notably: serializing typed demo values and prompting inputs without `serde_json::Value`).

I’ll organize this into phases, but each phase is still broken down by file, exact insertion points, and what each piece does.

---

## Big picture: what changes and why

### Today (in `crates/dspy-rs`)

* **User-facing:** `Example` (HashMap<String, serde_json::Value>) in, `Prediction` (HashMap<String, Value>) out.
* **Signature definition:** `MetaSignature` trait + `#[Signature]` *attribute macro* (dsrs-macros) generates JSON schema-ish metadata.
* **Prompting:** `ChatAdapter` renders field list + `[[ ## field ## ]]` protocol.
* **Parsing:** `ChatAdapter::parse_response` extracts between markers and uses `serde_json::from_str` for non-string key outputs.

### Target (per `CURRENT_SPEC.md`)

* **User-facing:** typed `QAInput { … }` in, typed `QA` out (where `QA` includes input fields preserved + output fields parsed).
* **Signature definition:** `#[derive(Signature)]` generates `QAInput` and implements a new typed `Signature` trait.
* **Prompting:** system message embeds BAML-rendered schema (`OutputFormatContent::render`) not schemars JSON fragments.
* **Parsing:** use BAML **jsonish** parser per-field, evaluate constraints (checks/ asserts), return structured errors, and expose metadata via `CallResult<O>`.

---

## Phase 0: Add dependencies and decide compatibility mode

### 0.1 Add baml-bridge dependency to dspy-rs

**File:** `crates/dspy-rs/Cargo.toml`

Add:

* A path dependency to the bridge crate (crate name is likely `baml-bridge` in Cargo, imported as `baml_bridge` in code).
* Enable the derive feature so users can `#[derive(BamlType)]`.

Example:

```toml
[dependencies]
baml-bridge = { path = "../baml-bridge", features = ["derive"] }
indexmap = "..." # already present
```

(Adjust path relative to `crates/dspy-rs`.)

**Side effects / gotchas**

* You’ll pull in `minijinja` and related crates through baml-bridge. This should be fine, but expect compile times to rise.

### 0.2 Decide whether to keep the legacy API during migration

You have two workable approaches:

1. **Parallel APIs (recommended for sanity):**

   * Keep `MetaSignature`, old `Predict`, old `ChatAdapter` as “legacy”.
   * Add new typed APIs under new names or modules (`typed::Predict`, `typed::Signature`).
   * Slowly migrate internal code and docs, then flip defaults later.

2. **Hard switch now:**

   * Replace `#[Signature]` attribute macro with `#[derive(Signature)]`.
   * Replace `Predict` with generic `Predict<S>`.
   * Update optimizers/tracing/evaluator accordingly.

Given you asked for “implement the spec as written”, I’ll plan for the **hard switch** but I’ll also show where to keep legacy behind a feature flag if you want.

---

## Phase 1: Introduce the new typed core API in dspy-rs

This phase is purely internal scaffolding: new traits, new errors, new result types. No macro yet.

### 1.1 Re-export BAML types so macros can refer to `dspy_rs::TypeIR`, etc.

**File:** `crates/dspy-rs/src/lib.rs`

Add re-exports near the existing `pub use ...` block.

You want dsrs-macros to generate code that references `dspy_rs::TypeIR`, `dspy_rs::Constraint`, etc without depending on the baml-bridge crate directly.

Recommended re-exports:

```rust
pub use baml_bridge; // optional: re-export the crate for power-users

pub use baml_bridge::BamlType; // derive macro (when feature derive enabled)
pub use baml_bridge::baml_types::{
    BamlValue, Constraint, ConstraintLevel, ResponseCheck, TypeIR, StreamingMode, TypeValue,
};
pub use baml_bridge::internal_baml_jinja::types::{OutputFormatContent, RenderOptions};
pub use baml_bridge::jsonish::deserializer::deserialize_flags::Flag;
pub use baml_bridge::convert::BamlConvertError; // if not already public via bridge
```

**Why:** your new `#[derive(Signature)]` macro will generate code that constructs `Constraint`, `TypeIR`, etc. Having them in `dspy_rs` avoids fragile cross-crate paths.

---

### 1.2 Define the new typed `Signature` trait and static metadata types

**File:** `crates/dspy-rs/src/core/signature.rs` (currently contains `MetaSignature`)

You have two options:

* Replace `MetaSignature` entirely.
* Or move `MetaSignature` to `core/meta_signature.rs` and put new typed `Signature` in `core/signature.rs`.

To avoid confusion, I’d do:

* **Create:** `crates/dspy-rs/src/core/meta_signature.rs` and move the old trait there (if you want legacy).
* **Rewrite:** `core/signature.rs` to define the new typed API.

#### New types to add (per spec)

```rust
pub struct FieldSpec {
    pub name: &'static str,       // LLM-facing name (alias or original)
    pub rust_name: &'static str,  // Rust field ident as string
    pub description: &'static str,
    pub type_ir: fn() -> TypeIR,  // fn ptr because TypeIR isn't const
    pub constraints: &'static [ConstraintSpec],
}

pub struct ConstraintSpec {
    pub kind: ConstraintKind,
    pub label: &'static str,
    pub expression: &'static str,
}

pub enum ConstraintKind { Check, Assert }
```

#### New trait (typed signature)

The spec’s trait includes `type Input`, plus instruction + TypeIRs + output format.

In practice, you also need a way to build the final returned struct (`QA`) out of `(QAInput, parsed_outputs)` and to split demos `QA -> (QAInput, outputs)`.

So you want:

```rust
pub trait Signature: Send + Sync + 'static {
    type Input: baml_bridge::BamlType;   // from baml-bridge
    type Output: baml_bridge::BamlType;  // output-only internal struct

    fn instruction() -> &'static str;

    fn input_fields() -> &'static [FieldSpec];
    fn output_fields() -> &'static [FieldSpec];

    fn output_format_content() -> &'static OutputFormatContent;

    fn from_parts(input: Self::Input, output: Self::Output) -> Self;
    fn into_parts(self) -> (Self::Input, Self::Output);
}
```

**Important reasoning**

* `Self` here is the signature struct (like `QA`) and also the return type of `Predict::<QA>::call`, as in the spec examples.
* `Self::Output` exists even though the spec doesn’t explicitly name it. It’s the cleanest way to produce an output-only BAML schema without needing a hand-built Registry path.

**Side effects**

* Your optimizers currently rely on `MetaSignature`. If you hard-switch, they’ll need to be updated to use `Signature` + `FieldSpec` instead.

---

### 1.3 Add the new error types exactly as spec requires

**File:** create `crates/dspy-rs/src/core/errors.rs` and re-export from `core/mod.rs` and `lib.rs`.

Implement the hierarchy from the spec:

* `PredictError`
* `ParseError`
* `ConversionError`
* `LmError`
* `ErrorClass`

You can keep `LmError` coarse initially (wrap provider errors), but structure the enum now so you can refine later.

Key implementation details:

#### `PredictError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum PredictError {
    #[error("LLM call failed")]
    Lm { #[source] source: LmError },

    #[error("failed to parse LLM response")]
    Parse {
        #[source] source: ParseError,
        raw_response: String,
        lm_usage: LmUsage,
    },

    #[error("failed to convert parsed value to output type")]
    Conversion {
        #[source] source: ConversionError,
        parsed: BamlValue,
    },
}
```

#### `ParseError`

Match the spec variants; you’ll be creating these in `ChatAdapter::parse_typed`.

* `MissingField { field, raw_response }`
* `ExtractionFailed { field, raw_response, reason }`
* `CoercionFailed { field, expected_type, raw_text, source }`
* `AssertFailed { field, label, expression, value }`
* `Multiple { errors, partial }`

**Note about `source` for `CoercionFailed`:**

* jsonish currently returns `anyhow::Error`.
* Wrap it in a `JsonishError(anyhow::Error)` newtype so your API doesn’t leak `anyhow` forever.

#### `ConversionError`

Map cleanly from baml-bridge’s `BamlConvertError`:

* type mismatch
* missing field
* unknown variant

You already have rich information in `BamlConvertError { path, expected, got, message }`.

---

### 1.4 Add `CallResult<O>` and per-field metadata

**File:** create `crates/dspy-rs/src/core/call_result.rs`

Per spec:

```rust
pub struct CallResult<O> {
    pub output: O,
    pub raw_response: String,
    pub lm_usage: LmUsage,
    pub tool_calls: Vec<rig::message::ToolCall>,
    pub tool_executions: Vec<String>,
    pub node_id: Option<usize>,
    fields: indexmap::IndexMap<String, FieldMeta>, // keyed by rust field name
}

pub struct FieldMeta {
    pub raw_text: String,
    pub flags: Vec<Flag>,
    pub checks: Vec<ConstraintResult>,
}

pub struct ConstraintResult {
    pub label: String,
    pub expression: String,
    pub passed: bool,
}
```

And implement:

* `field_flags(&self, field: &str) -> &[Flag]`
* `field_checks(&self, field: &str) -> &[ConstraintResult]`
* `field_raw(&self, field: &str) -> Option<&str>`

**Gotcha:** You already have `CallResult` in `utils/cache.rs`. Rename the cache one (Phase 6).

---

## Phase 2: Add the missing “typed value to prompt text” capability

This is the one spec-implied piece you don’t currently have: you need to format typed inputs and typed demos without relying on `serde_json::Value` in user code.

### 2.1 Add `ToBamlValue` to baml-bridge (bridge crate)

**File:** `crates/baml-bridge/src/lib.rs`

Add a new trait:

```rust
pub trait ToBamlValue {
    fn to_baml_value(&self) -> baml_types::BamlValue;
}
```

Implement it for:

* `String`, `bool`, numeric primitives already supported by baml-bridge
* `Option<T: ToBamlValue>`
* `Vec<T: ToBamlValue>`
* `HashMap<String, T: ToBamlValue>`, `BTreeMap<String, T: ToBamlValue>`

**Why here:** This keeps “BAML interop” in the BAML bridge crate, and it’s accessible from `dspy-rs`.

### 2.2 Extend `#[derive(BamlType)]` to also implement `ToBamlValue`

**File:** `crates/baml-bridge-derive/src/lib.rs`

You will:

* For structs: implement `ToBamlValue` by converting each field into a `BamlValue` and returning a `BamlValue::Class(name, map)`
* For enums: mirror how baml-bridge expects enums to be represented in `BamlValue` (likely `BamlValue::Enum` for unit enums, and for tagged unions either `Class` with tag field or whatever its conversion expects)

This is the most sensitive part of baml-bridge changes because you must align with how `BamlValueConvert` expects values to look.

**Implementation strategy to reduce risk:**

* Look at how baml-bridge derive currently generates conversion *from* `BamlValue`.
* Implement the exact inverse representation.
* Keep unit enums as `BamlValue::Enum(enum_name, variant_string)` if that’s how parsing returns them.

**Impact:** Once `ToBamlValue` exists, typed dspy-rs can:

* Render inputs into prompt markers.
* Render demos reliably (including nested custom types) as JSONish/JSON via `serde_json::to_string(&baml_value)`.

---

## Phase 3: Implement `#[derive(Signature)]` in dsrs-macros

This is the heart of the integration.

### 3.1 Replace the existing `#[Signature]` attribute macro

**File:** `crates/dsrs-macros/src/lib.rs`

Currently you have:

```rust
#[proc_macro_attribute]
pub fn Signature(attr: TokenStream, item: TokenStream) -> TokenStream { ... }
```

Spec requires:

* `#[derive(Signature)]` instead.

You cannot export two procedural macros with the same name cleanly. So you should:

* Rename the old attribute macro to `SignatureLegacy` (optional) and gate with a feature.
* Add:

```rust
#[proc_macro_derive(Signature, attributes(input, output, check, assert, alias))]
pub fn derive_signature(input: TokenStream) -> TokenStream { ... }
```

Also update the `sign!` macro in `dspy-rs` (Phase 7) to use `#[derive(Signature)]`.

---

### 3.2 What the derive macro must generate (concretely)

Given:

```rust
#[derive(Signature)]
/// Answer questions accurately
pub struct QA {
    #[input]
    pub question: String,

    #[input]
    pub context: Option<String>,

    #[output]
    pub answer: String,

    #[output]
    #[check("this >= 0.0 && this <= 1.0", label = "range")]
    pub confidence: f32,
}
```

Generate:

#### A) `QAInput` (public)

Only input fields.

```rust
#[derive(Debug, Clone, ::dspy_rs::BamlType)]
pub struct QAInput {
    pub question: String,
    pub context: Option<String>,
}
```

Also implement `ToBamlValue` if you want custom formatting, or rely on the field-level ToBamlValue. (If you add ToBamlValue to BamlType derive, this happens automatically.)

#### B) `__QAOutput` (hidden)

Only output fields, **including alias/constraints/docstrings** copied from original.

```rust
#[doc(hidden)]
#[derive(Debug, Clone, ::dspy_rs::BamlType)]
struct __QAOutput {
    pub answer: String,
    #[check("this >= 0.0 && this <= 1.0", label = "range")]
    pub confidence: f32,
}
```

This is what you use for:

* `output_format_content()` (schema rendering)
* final conversion from `BamlValue` to typed output fields

#### C) `__QAAll` (hidden)

All fields (inputs + outputs), used to give the original `QA` a BamlType implementation “by delegation” (since derive macros cannot add derives to the original item).

```rust
#[doc(hidden)]
#[derive(Debug, Clone, ::dspy_rs::BamlType)]
struct __QAAll {
    pub question: String,
    pub context: Option<String>,
    pub answer: String,
    pub confidence: f32,
}
```

#### D) Implement BAML traits for `QA` by delegating to `__QAAll`

In generated code:

* `impl ::dspy_rs::baml_bridge::BamlTypeInternal for QA { ... }`
* `impl ::dspy_rs::baml_bridge::BamlValueConvert for QA { ... }`
* `impl ::dspy_rs::baml_bridge::BamlType for QA { ... }`
* `impl ::dspy_rs::baml_bridge::ToBamlValue for QA { ... }` (optional but useful)

Each impl can be “convert to/from __QAAll”.

This satisfies spec’s “Original struct unchanged, but with BamlType impl”.

#### E) Implement `dspy_rs::core::Signature` for `QA`

```rust
impl ::dspy_rs::core::Signature for QA {
    type Input = QAInput;
    type Output = __QAOutput;

    fn instruction() -> &'static str { "Answer questions accurately" }

    fn input_fields() -> &'static [::dspy_rs::core::FieldSpec] { __QA_INPUT_FIELDS }
    fn output_fields() -> &'static [::dspy_rs::core::FieldSpec] { __QA_OUTPUT_FIELDS }

    fn output_format_content() -> &'static ::dspy_rs::OutputFormatContent {
        <__QAOutput as ::dspy_rs::BamlType>::baml_output_format()
    }

    fn from_parts(input: QAInput, output: __QAOutput) -> Self {
        Self { question: input.question, context: input.context, answer: output.answer, confidence: output.confidence }
    }

    fn into_parts(self) -> (QAInput, __QAOutput) { ... }
}
```

#### F) Generate `FieldSpec` slices and `TypeIR` constructors

Each output field needs:

* `name` (alias or rust field name)
* `rust_name`
* `description`
* `type_ir` fn pointer that returns TypeIR with constraints attached
* `constraints` static slice

Example for confidence:

```rust
fn __qa_confidence_type_ir() -> ::dspy_rs::TypeIR {
    let ty = <f32 as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir();
    ::dspy_rs::baml_bridge::with_constraints(ty, vec![
        ::dspy_rs::Constraint::new_check("range", "this >= 0.0 && this <= 1.0"),
    ])
}
```

Then:

```rust
const __QA_CONFIDENCE_CONSTRAINTS: &[::dspy_rs::core::ConstraintSpec] = &[
    ::dspy_rs::core::ConstraintSpec {
        kind: ::dspy_rs::core::ConstraintKind::Check,
        label: "range",
        expression: "this >= 0.0 && this <= 1.0",
    }
];

static __QA_OUTPUT_FIELDS: &[::dspy_rs::core::FieldSpec] = &[
    ::dspy_rs::core::FieldSpec {
        name: "confidence",
        rust_name: "confidence",
        description: "…doc comment…",
        type_ir: __qa_confidence_type_ir,
        constraints: __QA_CONFIDENCE_CONSTRAINTS,
    },
    // ...
];
```

**Why the `type_ir: fn() -> TypeIR`:**

* You can’t make `TypeIR` `const`, so a function pointer is the simplest static-friendly way.

---

### 3.3 Compile-time validation rules (match spec error messages)

Inside the derive macro, validate:

* Struct only, named fields only.
* Each field has exactly one of `#[input]` or `#[output]`.
* Must have ≥1 input and ≥1 output.
* `#[check(...)]` must include `label = "..."`
* `#[assert(...)]` label optional.
* Reject unsupported types? Prefer letting BamlType derive fail later, but signature macro should catch obvious forbidden patterns if needed.

Generate `syn::Error::new_spanned(...).to_compile_error()` messages matching spec examples.

---

## Phase 4: Implement typed prompting and parsing in ChatAdapter

You will keep the marker protocol, but swap schema rendering + jsonish parsing.

### 4.1 Add typed formatting function

**File:** `crates/dspy-rs/src/adapter/chat.rs`

Add new generic helpers *without* requiring `MetaSignature`.

#### A) Add helper: render field descriptions

New method:

```rust
fn format_field_descriptions<S: Signature>(&self) -> String
```

Implementation:

* Iterate `S::input_fields()` and `S::output_fields()`.
* Include index, `name`, and the type name:

  * Prefer `((field.type_ir)()).diagnostic_repr().to_string()` for stable user display.
* Append doc comment description if present.

#### B) Add helper: render “interaction structure” block

New method:

```rust
fn format_field_structure<S: Signature>(&self) -> String
```

This produces the same “All interactions will be structured…” section, but it should no longer include schemars JSON schema hints. The schema will be embedded separately as BAML schema.

You can keep the same marker skeleton:

```
[[ ## question ## ]]
question

[[ ## answer ## ]]
answer

[[ ## completed ## ]]
```

Use `FieldSpec.name` for the marker names (alias-aware).

#### C) Add helper: render system message (per spec 9.3)

New:

```rust
fn format_system_message_typed<S: Signature>(&self) -> Result<String, minijinja::Error>
```

Steps:

1. `let schema = S::output_format_content().render(RenderOptions::default())?;`
2. Combine:

   * field descriptions
   * field structure
   * “Answer in this schema:\n{schema}\n\n{instruction}”

#### D) Add helper: render user message from typed input

This requires the `ToBamlValue` work from Phase 2.

New:

```rust
fn format_user_message_typed<S: Signature>(&self, input: &S::Input) -> String
```

Approach:

* For each input field spec, you need the value. Rust has no reflection, so you have two choices:

**Choice 1 (recommended): macro-generated prompt parts**

* Have `#[derive(Signature)]` also generate for `QAInput`:

```rust
impl QAInput {
    pub fn __prompt_parts(&self) -> Vec<(&'static str, String)> { ... }
}
```

Where each entry is `(llm_field_name, rendered_value_string)`.

Then `ChatAdapter` can call `input.__prompt_parts()` via a trait bound:

Add a trait in dspy-rs:

```rust
pub trait PromptParts {
    fn prompt_parts(&self) -> Vec<(&'static str, String)>;
}
```

And have the macro implement it for `QAInput` and `__QAOutput`.

**Choice 2 (serde-based reflection):**

* Require `Serialize` on inputs, convert to `serde_json::Value`, index by field names.
* This violates “no HashMap/Value in user-facing code” spirit and adds hidden trait bounds, so I recommend Choice 1.

Rendered value rules:

* If value is a string: emit raw string
* Else: emit JSON via `serde_json::to_string(&value.to_baml_value()).unwrap()`

This yields consistent, schema-friendly examples without requiring `serde::Serialize` on user types.

#### E) Demo formatting

Similarly, for demos you need to create:

* user message from demo input
* assistant message from demo output fields

Use `demo.into_parts()` from `Signature` trait and then call `PromptParts` for each.

---

### 4.2 Add typed parsing function (marker extraction + jsonish)

**File:** `crates/dspy-rs/src/adapter/chat.rs`

Add:

```rust
fn extract_field(content: &str, llm_field_name: &str) -> Result<String, String>
```

Use the exact protocol from the spec (start marker `[[ ## name ## ]]`, end marker `[[ ## `).

Then:

```rust
pub fn parse_response_typed<S: Signature>(
    &self,
    response: &Message,
) -> Result<(S::Output, indexmap::IndexMap<String, FieldMeta>), ParseError>
```

Algorithm:

1. `let content = response.content();`

2. Initialize:

   * `let mut metas = IndexMap::new();`
   * `let mut errors = Vec::new();`
   * `let mut output_fields_map = indexmap::IndexMap::<String, BamlValue>::new();` (or HashMap)

3. For each output `FieldSpec` in `S::output_fields()`:

   * Determine:

     * `llm_name = spec.name`
     * `rust_name = spec.rust_name`
     * `let ty = (spec.type_ir)();`

   * Extract raw:

     * If start marker missing: push `ParseError::MissingField { field: rust_name.to_string(), raw_response: content.to_string() }` and continue
     * Else extract with trimming.
     * Save `raw_text`.

   * Parse via jsonish:

```rust
let parsed: baml_bridge::jsonish::BamlValueWithFlags =
    baml_bridge::jsonish::from_str(S::output_format_content(), &ty, &raw_text, true)
        .map_err(|e| ParseError::CoercionFailed { ... })?;
```

* Collect flags:

  * Copy the recursion logic from `crates/baml-bridge/src/lib.rs` (`collect_flags_recursive`) but operate on this one field node.
  * Store into `Vec<Flag>`.

* Convert parsed value into `BamlValue`:

  * In baml-bridge they do `let baml_value_with_meta: BamlValueWithMeta<TypeIR> = parsed.clone().into(); let baml_value: BamlValue = baml_value_with_meta.into();`
  * Use same conversion.

* Run constraints:

  * Call `run_user_checks(&baml_value, &ty)` (import it from baml-bridge’s `jsonish::deserializer::coercer::run_user_checks` as bridge does).
  * For each result:

    * If `Check`: push `ConstraintResult` into `FieldMeta.checks`
    * If `Assert` and failed: push `ParseError::AssertFailed { field: rust_name, label, expression, value: baml_value.clone() }`

* Save the parsed `BamlValue` into `output_fields_map` keyed by **llm name** or **rust name**:

  * If you later convert using `__QAOutput::try_from_baml_value`, it expects the map keys matching its field names or aliases. Since `__QAOutput` is generated with aliases, you can store with **llm name** to match what the LLM sees.
  * Metadata `metas` should be keyed by **rust name** to match `CallResult::field_raw("confidence")`.

* Set `metas.insert(rust_name.to_string(), FieldMeta { raw_text, flags, checks });`

4. If `errors` is non-empty:

   * Build `partial`:

     * `partial = Some(BamlValue::Class("__QAOutput".to_string(), output_fields_map_as_hashmap))` (exact constructor depends on BamlValue type)
   * Return `Err(ParseError::Multiple { errors, partial })`

5. If no errors:

   * Construct a full `BamlValue` for the output struct:

     * `BamlValue::Class("__QAOutput", map)`
   * Convert to `S::Output` using baml-bridge conversion:

     * `<S::Output as baml_bridge::BamlValueConvert>::try_from_baml_value(...)`
   * Return `(typed_output, metas)`

**Why parse per-field:** matches spec and preserves partial success.

---

## Phase 5: Implement `Predict::<S>` typed predictor and wire LM calls + metadata

### 5.1 Create a new generic predictor type

**File:** `crates/dspy-rs/src/predictors/predict.rs` (currently defines non-generic `Predict`)

You’ll likely rename the existing one to `PredictLegacy` or delete it.

Add:

```rust
pub struct Predict<S: Signature> {
    tools: Vec<Arc<dyn ToolDyn>>,
    demos: Vec<S>,
    instruction_override: Option<String>,
    // optional: lm override config
}
```

#### Construction API per spec

* `Predict::<S>::new()`
* `Predict::<S>::builder() ...`
* `.demo(...)`, `.with_demos(...)`
* `.add_tool(...)`, `.with_tools(...)`

### 5.2 Implement `call` and `call_with_meta`

**File:** same

Add:

```rust
impl<S: Signature> Predict<S> {
    pub async fn call(&self, input: S::Input) -> Result<S, PredictError> {
        Ok(self.call_with_meta(input).await?.output)
    }

    pub async fn call_with_meta(&self, input: S::Input) -> Result<CallResult<S>, PredictError> {
        // 1) build Chat via ChatAdapter typed formatting
        // 2) call LM
        // 3) parse response via jsonish
        // 4) build final S output (input preserved)
        // 5) return CallResult
    }
}
```

Implementation walkthrough inside `call_with_meta`:

1. Determine adapter + LM:

   * Currently you do:

```rust
let (adapter, lm) = {
  let guard = GLOBAL_SETTINGS.read().unwrap();
  let settings = guard.as_ref().unwrap();
  (settings.adapter.clone(), Arc::clone(&settings.lm))
};
```

For typed, you can keep this.
But `adapter` type is `Arc<dyn Adapter>` currently (legacy). You can:

* call `ChatAdapter` directly (since typed logic lives there), ignoring `Adapter` trait for typed calls, or
* introduce a new typed adapter trait. Keep it simple: call `ChatAdapter`.

2. Build system + user messages:

   * `let system = ChatAdapter.format_system_message_typed::<S>()?`
   * `let user = ChatAdapter.format_user_message_typed::<S>(&input)`
   * Add demos similarly.

3. Call `lm.call(chat, tools).await`:

   * On error: map into `PredictError::Lm { source: ... }`
   * You may initially wrap errors as `LmError::Provider { ... }`

4. Extract:

   * `raw_response = response.output.content()`
   * `lm_usage = response.usage`
   * `tool_calls`, `tool_executions`

5. Parse:

   * `let (typed_output_only, metas) = ChatAdapter.parse_response_typed::<S>(&response.output)?;`
   * On parse error: return `PredictError::Parse { source, raw_response, lm_usage }`

6. Build final return:

   * `let output = S::from_parts(input, typed_output_only);`

7. Return `CallResult<S>`:

   * output
   * raw_response
   * usage
   * tool calls and executions
   * node_id if tracing is on (see next subsection)
   * fields metadata = metas

---

### 5.3 Integrate tracing (so you don’t break existing trace module)

Right now `Predict::forward` records:

* Root node
* Predict node with `NodeType::Predict { signature_name, signature: Arc<dyn MetaSignature> }`
* Records Prediction output

With typed Predict, you can keep the node recording but you must **remove dependency on `MetaSignature`** in the trace graph (unless you keep a compatibility wrapper).

**File(s):**

* `crates/dspy-rs/src/trace/dag.rs`
* `crates/dspy-rs/src/trace/context.rs`
* `crates/dspy-rs/src/predictors/predict.rs`

Plan:

1. Update `NodeType::Predict` to store:

   * `signature_name: String`
   * `schema_fingerprint: Option<String>` (optional)
   * maybe `instruction: String` (optional)

2. In `Predict::<S>::call_with_meta`, when tracing:

   * record Predict node with signature name `std::any::type_name::<S>()` or `S::...` static name if you add one.
   * store node_id into `CallResult`.

This avoids threading trait objects through the graph.

---

## Phase 6: Fix naming collisions and caching type conflicts

### 6.1 Rename the cache `CallResult`

**File:** `crates/dspy-rs/src/utils/cache.rs`

Currently:

```rust
pub struct CallResult {
  pub prompt: String,
  pub prediction: Prediction,
}
```

Rename it to something like `CachedCall` or `CacheEntry`.

Then update all usages:

* `crates/dspy-rs/src/core/lm/mod.rs` imports `CallResult`
* `crates/dspy-rs/src/adapter/chat.rs` caching code sends `CallResult`

**Why:** you now have `core::CallResult<O>` as a public API.

### 6.2 Decide caching strategy for typed Predict

Your current caching keys are `Example` objects. Typed Predict doesn’t have Example.

Two good options:

1. **Cache at the LM layer by prompt hash**

   * compute a cache key as `(schema_fingerprint, instruction, rendered_user_message, demos?)`
   * store raw assistant response + usage
   * then parsing happens every time (cheap compared to LM)

2. **Cache at the typed layer by input struct + instruction**

   * requires stable serialization of input
   * easiest if you convert input to BamlValue via new ToBamlValue support and then `serde_json::to_string`

Given scope, I’d implement caching later. For the integration plan, mark it as “post-MVP” and keep caching only for legacy predictor until typed is stable.

---

## Phase 7: Update dspy-rs macros and docs to the new API

### 7.1 Update `sign!` macro to generate a derived signature

**File:** `crates/dspy-rs/src/lib.rs`

Today `sign!` macro generates:

```rust
#[Signature]
struct InlineSignature { ... }
InlineSignature::new()
```

Under new design:

* You want `#[derive(Signature)]`
* And you probably want the macro to return a type, not a runtime object, because typed Predict is generic over the signature type.

Suggested new behavior:

* `sign!` returns a `Predict::<InlineSignature>` or returns the signature type as a marker.

Example:

```rust
let predict = sign!{ (question: String) -> answer: String };
let out: InlineSignature = predict.call(InlineSignatureInput { question: ... }).await?;
```

If you want the macro to keep returning “something callable”, make it return `Predict::<InlineSignature>::new()`.

### 7.2 Deprecate `example!` and `prediction!` for user-facing API

Keep them for internal tooling if you want, but spec wants typed input/output structs.

You can:

* Leave them in place but move them under `legacy` module.
* Or keep but mark `#[deprecated(note = "...use QAInput instead...")]`.

---

## Phase 8: Update optimizer and evaluation internals (two paths)

The spec’s long-term direction is: optimizers operate on `BamlValue` interchange, not typed structs or `HashMap<String, Value>`.

You can do this in a second wave, but here’s the plan.

### Path A (minimum breakage now): keep optimizers on legacy `Predict`

* Keep existing `Predict` + `MetaSignature` pipeline for optimizers only.
* Typed Predict is user-facing and separate.
* You’ll have two ecosystems temporarily.

### Path B (spec-consistent): refactor Module/Optimizable to use `BamlValue` interchange

If you implement Path B, you must change:

**File:** `crates/dspy-rs/src/core/module.rs`

Replace current `Module` trait:

```rust
pub trait Module {
  async fn forward(&self, inputs: Example) -> Result<Prediction>;
}
```

With spec-like untyped:

```rust
pub trait Module: Send + Sync {
    fn forward_untyped(
        &self,
        input: BamlValue,
    ) -> impl Future<Output = Result<BamlValue, PredictError>> + Send;

    fn signature_spec(&self) -> &'static [FieldSpec]; // or a SignatureSpec trait object
}
```

Then implement `Module` for `Predict<S>`:

* Convert `BamlValue` -> `S::Input` via `BamlValueConvert`
* Call typed `call`
* Convert result `S` -> `BamlValue` via `ToBamlValue`

This requires ToBamlValue for signature struct `S` (derive(Signature) can generate it via delegation to `__SAll`).

Then update:

* `optimizer/*` to call `forward_untyped`
* evaluation logic to use BamlValue instead of Prediction

This is a larger change, but it aligns with spec section 10.

---

## “Glue decisions” you should lock in early

These are small but crucial to avoid later rewrites.

### 1) Metadata keying

* **Expose** metadata by **Rust field name** (`confidence`)
* **Parse markers** by **LLM-facing name** (alias or name)
* Store both in `FieldSpec` and always map error messages to rust_name.

### 2) Output schema source

Use the macro-generated `__SigOutput` for:

* `Signature::output_format_content()`
* `Signature::output_fields()` TypeIR functions

This guarantees the schema used for prompting matches the schema used for parsing.

### 3) Constraint evaluation

* Always run `run_user_checks(value, type_ir)` after parsing to get authoritative check/assert results.
* Store checks in metadata.
* Convert failed asserts to `ParseError::AssertFailed` (and then `PredictError::Parse`).

### 4) Partial parsing behavior

When multiple output fields fail, return:

* `ParseError::Multiple { errors, partial: Some(BamlValue) }`
  Where `partial` includes any successfully parsed output fields (and may omit failed ones).

---

## A concrete “implementation checklist” by file

Here’s the tightest “touch list” to keep you oriented.

### dspy-rs

* `crates/dspy-rs/Cargo.toml`

  * add `baml-bridge` dependency

* `crates/dspy-rs/src/lib.rs`

  * re-export BAML types
  * re-export `Signature`, `Predict`, `CallResult`, errors
  * update `sign!` macro

* `crates/dspy-rs/src/core/signature.rs`

  * implement new typed `Signature`, `FieldSpec`, `ConstraintSpec`, `ConstraintKind`

* `crates/dspy-rs/src/core/errors.rs` (new)

  * implement spec error hierarchy

* `crates/dspy-rs/src/core/call_result.rs` (new)

  * implement `CallResult<O>` + metadata accessors

* `crates/dspy-rs/src/adapter/chat.rs`

  * add typed formatting functions
  * embed BAML schema render
  * add per-field marker extraction
  * use `jsonish::from_str` + `run_user_checks`
  * return `ParseError` variants

* `crates/dspy-rs/src/predictors/predict.rs`

  * implement `Predict<S: Signature>` with `call` and `call_with_meta`
  * integrate tools and trace
  * map errors into `PredictError`

* `crates/dspy-rs/src/utils/cache.rs`

  * rename cache `CallResult` to `CacheEntry` (or similar)
  * update imports/usages

* `crates/dspy-rs/src/trace/*`

  * remove dependence on `Arc<dyn MetaSignature>` in NodeType, or add typed wrapper

### dsrs-macros

* `crates/dsrs-macros/src/lib.rs`

  * add `proc_macro_derive(Signature, attributes(...))`
  * rename/gate/remove old `#[Signature]` attribute macro
  * generate `QAInput`, `__QAOutput`, `__QAAll`, `FieldSpec` arrays, and impls described above

### baml-bridge

* `crates/baml-bridge/src/lib.rs`

  * add `ToBamlValue` trait and primitive/container impls

* `crates/baml-bridge-derive/src/lib.rs`

  * extend derive(BamlType) to implement `ToBamlValue` for structs/enums

---

## Final note: sequencing that avoids “refactor spirals”

If you implement in this order, you avoid needing to backtrack:

1. Add re-exports + new core types (`Signature`, errors, `CallResult`)
2. Add `ToBamlValue` + derive support (so prompting/demos are solvable)
3. Implement `#[derive(Signature)]` macro generation (OutputFormatContent + FieldSpecs)
4. Implement typed ChatAdapter formatting + parsing
5. Implement typed Predict + error mapping
6. Only then touch tracing/optimizers

That gets you to the spec’s acceptance criteria fastest, without getting stuck on the “how do we print demos/inputs” problem halfway through.

---

If you want, I can take one signature example (like the `QA` in `CURRENT_SPEC.md`) and write out exactly what the macro expansion should look like (generated structs + impl blocks + static arrays) so you can compare it against what your proc-macro emits while debugging.
