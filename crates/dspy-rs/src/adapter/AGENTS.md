# Adapter Module

## Boundary

This directory: Prompt formatting and response parsing adapters that bridge signatures and LM calls.

Depends on: `crate::core` (Chat, Message, Example), `crate::baml_bridge` (type coercion, output formatting), `rig::tool::ToolDyn`

Depended on by: Modules (Predict, ChainOfThought), high-level orchestration code

NEVER: Put LM provider logic here. Adapters transform data; they do not make network calls directly.

## How to work here

The `Adapter` trait defines three methods:
- `format()` - Convert signature + inputs into a `Chat` (system/user messages)
- `parse_response()` - Extract structured output from LM response using `[[ ## field ## ]]` markers
- `call()` - Full pipeline: format, call LM, parse, handle caching

Golden pattern: `ChatAdapter` in `chat.rs`. Study these key methods:
- `format_system_message_untyped()` - Builds field descriptions + structure + task description
- `format_user_message_untyped()` - Formats input values with field markers
- `parse_sections()` - Regex-based extraction of `[[ ## field ## ]]` blocks

When adding a new adapter:
1. Implement the `Adapter` trait
2. Handle both typed (`Signature` trait) and untyped (`MetaSignature` trait) paths
3. Ensure `[[ ## completed ## ]]` marker is always included in format output
4. Test with `DummyLM` to verify format/parse roundtrip

## Verification

```bash
cargo test -p dspy-rs test_chat_adapter
cargo test -p dspy-rs test_adapters
```

Tests verify: prompt structure, field extraction, demo formatting, cache behavior.

## Gotchas

- **Tool handling**: When `tools` are provided to `call()`, tool calls and executions are injected into output under `tool_calls` and `tool_executions` keys. Check for these when parsing.

- **Message format**: The `[[ ## field ## ]]` pattern is load-bearing. Response parsing relies on `FIELD_HEADER_PATTERN` regex. Malformed markers cause silent field omission.

- **Typed vs untyped**: `ChatAdapter` has parallel code paths - `*_typed` methods use `Signature` trait bounds, untyped methods use `MetaSignature` (runtime reflection). Keep both in sync.

- **Schema rendering**: Complex types get JSON schema in prompts via `render_field_type_schema()`. Schema parsing failures surface as `RenderError`.
