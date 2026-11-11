# Typed history and tools

`History`, `HistoryMessage`, `Tool`, `ToolCalls` and `ToolInvocation` use Facet,
Serde and the current `Schema` blanket implementation. They replace the old
schemars placeholders; the earlier `BamlType` path has already been replaced by
the in-house type system. There is no `NoTool` placeholder: an empty `ToolCalls`
value represents no requested calls.

```rust,no_run
use dspy_rs::{Signature, Tool, ToolCalls};

#[derive(Signature, Clone, Debug)]
/// Choose tools for the request.
struct ChooseTools {
    #[input] request: String,
    #[input] tools: Vec<Tool>,
    #[output] calls: ToolCalls,
}
```

`Tool` is a serializable declaration, not an executable function. Construct one
from a rig `ToolDefinition` with `Tool::from_definition`. The `parameters` field
contains JSON Schema text; each declaration must have a nonempty unique name and
an object argument schema. `ToolCalls` contains a `calls` vector of typed
`ToolInvocation` values (`id`, `name`, JSON argument text).

The adapter recognizes these fields through `FieldSchema::special_kind()` and
`FieldDef::special_kind()`. Predictions with tool declaration inputs return
caller-managed requests through a sole `ToolCalls` output. Execute those requests
in host code, then supply their results on a subsequent turn. For automatic
execution, attach executable tools using `Predict::builder().add_tool(...)` or
an IR agent. Signature metadata never grants execution capabilities by itself.

Native function calling is the default. For a model or endpoint without it, set
`LM::builder().native_tools(false)` or `lm.with_native_tools(false)`. The LM sends
declarations and tool-choice instructions as text, decodes the documented JSON
tool envelope, and uses the same dispatch/round limits/error handling as native
tools. Follow-up requests encode tool calls/results as ordinary text messages;
no native tool fields reach the endpoint. This capability is explicit: provider
names alone cannot establish whether an arbitrary model supports tools.

Input `History { messages: Vec<HistoryMessage> }` is inserted as actual chat
messages before the current user input. Supported roles are `user` and
`assistant`; system instructions remain host-owned. It is distinct from execution
history inspection and persistent interaction storage. `ToolCalls` fields still
pass the normal coercion/constraint pipeline. Native tool requests are adapted to
the same typed representation returned by the prompt fallback.

The schema bridge for provider-native structured output still uses
`schemars::Schema`, the type required by rig's completion request. Special fields
do not derive or depend on schemars, and their migration does not remove that
separate, active bridge.
