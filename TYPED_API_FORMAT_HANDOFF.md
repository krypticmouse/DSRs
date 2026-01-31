# Typed API Output Format Bug - Handoff

## Summary

The typed `Predict<S>` API path doesn't include output format instructions in the system prompt, causing LLM responses to be unparseable. Models output JSON, YAML, or freeform text instead of the expected `[[ ## field ## ]]` format.

## Symptoms

Running `cargo run --example rlm_trajectory --features rlm` fails with:
```
Error: predictor failed during action
Caused by:
    0: failed to parse LLM response
    1: 2 field(s) failed to parse
```

The parser expects headers like `[[ ## reasoning ## ]]` but models return various formats:
- JSON: `{"reasoning":"...", "code":"..."}`
- YAML: `reasoning: |`
- Jinja-style: `{{reasoning}}`

## Root Cause

**Location:** `crates/dspy-rs/src/core/signature/compiled.rs`

The `DEFAULT_SYSTEM_TEMPLATE` (lines 19-32) only describes field names/types:

```rust
pub const DEFAULT_SYSTEM_TEMPLATE: &str = r#"
Your input fields are:
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% endfor %}

Your output fields are:
{% for f in sig.outputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
...
"#;
```

**Missing:** Format structure instructions telling the model to use `[[ ## field ## ]]` markers.

**Contrast with untyped API:** `ChatAdapter::format_field_structure()` in `chat.rs` includes:
```rust
format!(
    "All interactions will be structured in the following way, with the appropriate values filled in.\n\n{input_field_structure}{output_field_structure}[[ ## completed ## ]]\n"
)
```

And `ChatAdapter::get_field_structure()` generates the actual `[[ ## field ## ]]` examples.

## Code Paths

### Untyped path (works)
```
LegacyPredict::forward()
  -> ChatAdapter::format_field_description()  // lists fields
  -> ChatAdapter::format_field_structure()    // [[ ## field ## ]] format ✓
  -> ChatAdapter::format_task_description()   // instruction
  -> ChatAdapter::format_user_message_untyped()
```

### Typed path (broken)
```
Predict<S>::call()
  -> ChatAdapter::format_system_message_with_instruction()
     -> CompiledSignature::render_system_message_with_ctx()
        -> DEFAULT_SYSTEM_TEMPLATE  // NO format structure ✗
  -> ChatAdapter::format_user_message()
```

## Parser Location

`crates/dspy-rs/src/adapter/chat.rs`:
- `FIELD_HEADER_PATTERN` (line 28-29): `r"^\[\[ ## (\w+) ## \]\]"`
- `parse_sections()` (line 460+): Extracts content between `[[ ## field ## ]]` markers

## Fix Options

### Option 1: Add format to DEFAULT_SYSTEM_TEMPLATE

Update `compiled.rs` to include format structure in the system template:

```rust
pub const DEFAULT_SYSTEM_TEMPLATE: &str = r#"
Your input fields are:
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% endfor %}

Your output fields are:
{% for f in sig.outputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% endfor %}

All interactions will be structured in the following way, with the appropriate values filled in.

{% for f in sig.inputs %}
[[ ## {{ f.llm_name }} ## ]]
{a]{{ f.llm_name }} value here}

{% endfor %}
{% for f in sig.outputs %}
[[ ## {{ f.llm_name }} ## ]]
{your {{ f.llm_name }} here}

{% endfor %}
[[ ## completed ## ]]
"#;
```

### Option 2: Add format in ChatAdapter

Modify `format_system_message_with_instruction()` to append format structure after rendering the template.

### Option 3: Add format in Predict::call()

Append format structure to the system message in the predictor itself.

## Related Changes Made This Session

1. **Made `temperature` optional** in `LM` struct (`Option<f32>` instead of `f32`)
   - Reasoning models (o1, o3, gpt-5.2) reject requests with temperature
   - Updated: `crates/dspy-rs/src/core/lm/mod.rs`
   - Updated all optimizer usages to use `Some(temperature)`

2. **Updated rlm_trajectory example** to use GPT 5.2:
   ```rust
   let lm = LM::builder()
       .model("openai:gpt-5.2".to_string())
       .additional_params(serde_json::json!({"reasoning_effort": "low"}))
       .build()
       .await?;
   ```

## Files Modified

- `crates/dspy-rs/src/core/lm/mod.rs` - temperature now `Option<f32>`
- `crates/dspy-rs/src/optimizer/mipro.rs` - `Some(temperature)` assignments
- `crates/dspy-rs/src/optimizer/gepa.rs` - `Some(temperature)` assignments  
- `crates/dspy-rs/src/optimizer/copro.rs` - `Some(temperature)` assignments
- `crates/dspy-rs/examples/rlm_trajectory.rs` - GPT 5.2 config

## Not Related to Storage Commit

This bug predates the storage commit (`nsqmkpro`). The typed API has never included format structure in its system prompt. It likely worked before because:
1. Different code paths were used
2. Models happened to guess the format
3. The RLM example wasn't being run regularly

## Test Command

```bash
cargo run --example rlm_trajectory --features rlm
```

Expected: Should complete RLM loop and print patterns
Actual: Fails with "2 field(s) failed to parse"
