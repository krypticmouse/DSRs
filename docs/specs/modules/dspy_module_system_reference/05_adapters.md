# Adapters: How Modules Talk to LMs

## What Adapters Do

An adapter sits between `Predict` and the LM. It has three jobs:
1. **Format**: Convert (signature, demos, inputs) into a list of chat messages
2. **Call**: Send messages to the LM
3. **Parse**: Extract typed output field values from the LM's text response

The critical path: `Predict.forward()` -> `adapter(lm, lm_kwargs, signature, demos, inputs)` -> messages -> LM -> completions -> parsed dicts -> Prediction.

---

## 1. Adapter Base Class

**File**: `dspy/adapters/base.py`

### Constructor

```python
class Adapter:
    def __init__(self, callbacks=None, use_native_function_calling=False,
                 native_response_types=None):
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.native_response_types = native_response_types or [Citations, Reasoning]
```

- `use_native_function_calling`: When True, detects `dspy.Tool` input fields and `dspy.ToolCalls` output fields, converts them to litellm tool definitions
- `native_response_types`: Types handled by native LM features rather than text parsing (e.g., `Reasoning` for o1-style models)

### The `__call__` Pipeline

```python
def __call__(self, lm, lm_kwargs, signature, demos, inputs):
    # Step 1: Preprocess - handle native tools and response types
    processed_signature, original_signature, lm_kwargs = self._call_preprocess(
        lm, lm_kwargs, signature, inputs
    )

    # Step 2: Format and call
    messages = self.format(processed_signature, demos, inputs)
    outputs = lm(messages=messages, **lm_kwargs)  # list[str | dict]

    # Step 3: Postprocess - parse each completion
    return self._call_postprocess(
        processed_signature, original_signature, outputs, lm, lm_kwargs
    )
```

### Step 1: `_call_preprocess()`

Handles two categories of "native" features:

**Native function calling** (when `use_native_function_calling=True`):
- Finds `dspy.Tool` / `list[dspy.Tool]` input fields
- Finds `dspy.ToolCalls` output fields
- Converts tools to litellm format via `tool.format_as_litellm_function_call()`
- Adds to `lm_kwargs["tools"]`
- **Removes** both tool input and ToolCalls output fields from the signature
- The LM handles tool calling natively instead of through text

**Native response types** (Reasoning, Citations):
- For each output field with a native response type annotation:
  - Calls `field.annotation.adapt_to_native_lm_feature(signature, name, lm, lm_kwargs)`
  - For `Reasoning`: checks if LM supports native reasoning (via `litellm.supports_reasoning()`). If yes, sets `reasoning_effort` in lm_kwargs and **deletes** the reasoning field from the signature. The model uses its built-in chain-of-thought.
  - Returns the modified signature (with native-handled fields removed)

### Step 3: `_call_postprocess()`

For each LM output:
1. If the output has text: call `self.parse(processed_signature, text)` -> dict of field values
2. Set missing fields (ones in original but not processed signature) to `None`
3. If tool_calls present: parse into `ToolCalls.from_dict_list()`
4. For native response types: call `field.annotation.parse_lm_response(output)` (e.g., extract `reasoning_content` from the response dict)
5. Handle logprobs

### Abstract Methods (subclasses must implement)

```python
def format_field_description(self, signature) -> str
def format_field_structure(self, signature) -> str
def format_task_description(self, signature) -> str
def format_user_message_content(self, signature, inputs, ...) -> str
def format_assistant_message_content(self, signature, outputs, ...) -> str
def parse(self, signature, completion) -> dict
```

### Concrete Methods in Base

**`format(signature, demos, inputs)`** -- The main formatting pipeline:

```python
def format(self, signature, demos, inputs):
    messages = []

    # 1. Check for History field; if present, extract conversation history
    history_field_name = ...  # find field with dspy.History type
    if history_field_name:
        signature = signature.delete(history_field_name)

    # 2. System message
    messages.append({
        "role": "system",
        "content": self.format_system_message(signature)
    })

    # 3. Demo messages (few-shot examples)
    messages.extend(self.format_demos(signature, demos))

    # 4. Conversation history (if any)
    if history_field_name:
        messages.extend(self.format_conversation_history(
            signature, history_field_name, inputs
        ))

    # 5. Current user input
    messages.append({
        "role": "user",
        "content": self.format_user_message_content(
            signature, inputs, main_request=True
        )
    })

    # 6. Handle custom types (Image, Audio, File)
    messages = split_message_content_for_custom_types(messages)

    return messages
```

**`format_system_message(signature)`**:
```python
def format_system_message(self, signature):
    return (
        self.format_field_description(signature) + "\n\n" +
        self.format_field_structure(signature) + "\n\n" +
        self.format_task_description(signature)
    )
```

**`format_demos(signature, demos)`** -- Sorts demos into complete and incomplete:

```python
def format_demos(self, signature, demos):
    messages = []

    # Separate complete (all fields) from incomplete (some missing)
    complete_demos = [d for d in demos if all fields present]
    incomplete_demos = [d for d in demos if has_input AND has_output but not all]

    # Incomplete demos come FIRST with a disclaimer
    for demo in incomplete_demos:
        # User message with "This is an example of the task, though some input
        # or output fields are not supplied."
        # Missing fields show: "Not supplied for this particular example."

    # Complete demos after
    for demo in complete_demos:
        # User/assistant message pair with all fields filled
```

---

## 2. ChatAdapter

**File**: `dspy/adapters/chat_adapter.py`

The default adapter. Uses `[[ ## field_name ## ]]` delimiters to separate fields.

### Fallback to JSONAdapter

```python
def __call__(self, lm, lm_kwargs, signature, demos, inputs):
    try:
        return super().__call__(...)
    except Exception as e:
        if isinstance(e, ContextWindowExceededError):
            raise  # Don't retry context window errors
        if isinstance(self, JSONAdapter):
            raise  # Already in JSON mode
        if not self.use_json_adapter_fallback:
            raise
        # Fallback: retry with JSONAdapter
        return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)
```

### `format_field_description(signature)`

```
Your input fields are:
1. `question` (str): The question to answer
2. `context` (list[str]): Relevant passages

Your output fields are:
1. `answer` (str): The answer, often between 1 and 5 words
```

### `format_field_structure(signature)`

Shows the expected format using `[[ ## field_name ## ]]` markers:

```
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## context ## ]]
{context}

[[ ## answer ## ]]
{answer}    # note: the value you produce must be a single str value

[[ ## completed ## ]]
```

The type hints come from `translate_field_type()`:

| Python Type | Prompt Hint |
|-------------|------------|
| `str` | (no hint) |
| `bool` | `"must be True or False"` |
| `int` / `float` | `"must be a single int/float value"` |
| `Enum` | `"must be one of: val1; val2; val3"` |
| `Literal["a", "b"]` | `"must exactly match (no extra characters) one of: a; b"` |
| Complex types | `"must adhere to the JSON schema: {...}"` (Pydantic JSON schema) |

### `format_task_description(signature)`

```
In adhering to this structure, your objective is:
    Answer questions with short factoid answers.
```

### `format_user_message_content(signature, inputs, main_request=True)`

```
[[ ## question ## ]]
What is the capital of France?

[[ ## context ## ]]
[1] <<France is a country in Western Europe. Its capital is Paris.>>

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`,
and then ending with the marker for `[[ ## completed ## ]]`.
```

The last line (output requirements) is only added when `main_request=True` (not for demos).

### `format_assistant_message_content(signature, outputs)`

```
[[ ## answer ## ]]
Paris

[[ ## completed ## ]]
```

### `format_field_value()` (from `utils.py`)

How values are formatted in messages:
- Lists of strings: numbered format `[1] <<text>>`, `[2] <<text>>`
- Dicts/lists of non-strings: `json.dumps(jsonable_value)`
- Primitives: `str(value)`
- Single items with delimiters: `<<value>>` or `<<<multi\nline>>>` for long values

### `parse(signature, completion)`

```python
def parse(self, signature, completion):
    # 1. Split on [[ ## field_name ## ]] headers
    sections = re.split(r"\[\[ ## (\w+) ## \]\]", completion)

    # 2. Group content under each header
    fields = {}
    for header, content in paired_sections:
        if header in signature.output_fields:
            fields[header] = content.strip()

    # 3. Parse each field value to its annotated type
    for name, raw_value in fields.items():
        annotation = signature.output_fields[name].annotation
        fields[name] = parse_value(raw_value, annotation)

    # 4. Validate all output fields are present
    if not all(name in fields for name in signature.output_fields):
        raise AdapterParseError(...)

    return fields
```

**`parse_value(value_string, annotation)`** (from `utils.py`):
1. `str` -> return as-is
2. `Enum` -> find matching member by value or name
3. `Literal` -> validate against allowed values, strip wrapper syntax
4. `bool/int/float` -> type cast
5. Complex types -> `json_repair.loads()` then `pydantic.TypeAdapter(annotation).validate_python()`
6. DSPy Type subclasses -> try custom parsing

---

## 3. JSONAdapter

**File**: `dspy/adapters/json_adapter.py`

Extends ChatAdapter. Key differences: outputs are JSON instead of delimited text.

### Structured Outputs Support

```python
def __call__(self, lm, lm_kwargs, signature, demos, inputs):
    # Try 1: json_object mode
    result = self._json_adapter_call_common(...)
    if result: return result

    try:
        # Try 2: OpenAI Structured Outputs (full schema)
        structured_output_model = _get_structured_outputs_response_format(signature)
        lm_kwargs["response_format"] = structured_output_model
        return super().__call__(...)
    except:
        # Try 3: json_object mode (simpler)
        lm_kwargs["response_format"] = {"type": "json_object"}
        return super().__call__(...)
```

### Output Format Differences

**ChatAdapter output**:
```
[[ ## answer ## ]]
Paris

[[ ## completed ## ]]
```

**JSONAdapter output**:
```json
{
  "answer": "Paris"
}
```

### `format_field_structure(signature)` -- Different from ChatAdapter

User inputs still use `[[ ## field_name ## ]]` markers, but outputs are described as JSON:

```
Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.
{
  "answer": "{answer}"    // note: must adhere to JSON schema: ...
}
```

### `parse(signature, completion)` -- JSON parsing

```python
def parse(self, signature, completion):
    # 1. Parse with json_repair (handles malformed JSON)
    result = json_repair.loads(completion)

    # 2. If not a dict, try regex extraction of JSON object
    if not isinstance(result, dict):
        match = regex.search(r"\{(?:[^{}]|(?R))*\}", completion)
        result = json_repair.loads(match.group())

    # 3. Filter to known output fields
    result = {k: v for k, v in result.items() if k in signature.output_fields}

    # 4. Parse each value to its annotated type
    for name, value in result.items():
        result[name] = parse_value(value, signature.output_fields[name].annotation)

    # 5. Validate all fields present
    if not all(name in result for name in signature.output_fields):
        raise AdapterParseError(...)

    return result
```

### Structured Outputs Model Generation

`_get_structured_outputs_response_format(signature)` builds a Pydantic model from output fields with OpenAI's requirements:
- `extra="forbid"` (no additional properties)
- Recursive `enforce_required()` ensures all nested objects have `required` and `additionalProperties: false`

---

## 4. Other Adapters

### XMLAdapter

**File**: `dspy/adapters/xml_adapter.py`

Uses `<field_name>...</field_name>` XML tags instead of `[[ ## ]]` delimiters. Otherwise similar to ChatAdapter.

### TwoStepAdapter

**File**: `dspy/adapters/two_step_adapter.py`

Uses two LM calls:
1. First call: natural language prompt, get a free-form response
2. Second call: use ChatAdapter to extract structured fields from the free-form response

Useful for models that struggle with strict formatting.

---

## 5. Complete Message Assembly Example

For a `ChainOfThought("question -> answer")` with 2 demos and the input "What is 2+2?":

### System Message

```
Your input fields are:
1. `question` (str)

Your output fields are:
1. `reasoning` (str): ${reasoning}
2. `answer` (str)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## reasoning ## ]]
{reasoning}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]

In adhering to this structure, your objective is:
    Given the fields `question`, produce the fields `reasoning`, `answer`.
```

### Demo 1 (User)

```
[[ ## question ## ]]
What is the capital of France?
```

### Demo 1 (Assistant)

```
[[ ## reasoning ## ]]
The question asks about the capital of France. France is a country in Europe, and its capital city is Paris.

[[ ## answer ## ]]
Paris

[[ ## completed ## ]]
```

### Demo 2 (User + Assistant)

(Same pattern)

### Current Input (User)

```
[[ ## question ## ]]
What is 2+2?

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`,
and then ending with the marker for `[[ ## completed ## ]]`.
```

### LM Response (Assistant)

```
[[ ## reasoning ## ]]
The question asks for the sum of 2 and 2. Basic arithmetic: 2 + 2 = 4.

[[ ## answer ## ]]
4

[[ ## completed ## ]]
```

### Parsed Result

```python
{"reasoning": "The question asks for the sum of 2 and 2. Basic arithmetic: 2 + 2 = 4.",
 "answer": "4"}
```

---

## 6. Settings and Adapter Configuration

### Global Configuration

```python
dspy.configure(
    lm=dspy.LM("openai/gpt-4"),
    adapter=dspy.ChatAdapter(),  # Default if not set
)
```

### Per-Call Override

```python
with dspy.context(adapter=dspy.JSONAdapter()):
    result = predict(question="...")
```

### LM Resolution in Predict

```python
# In _forward_preprocess:
adapter = settings.adapter or ChatAdapter()  # Global or default
lm = kwargs.pop("lm", self.lm) or settings.lm  # Per-call > per-predict > global
```

---

## 7. Custom Types and Special Handling

### Image (`dspy/adapters/types/image.py`)

- Subclass of `dspy.Type`
- `format()` returns `[{"type": "image_url", "image_url": {"url": data_uri}}]`
- Serialized with custom markers: `<<CUSTOM-TYPE-START-IDENTIFIER>>json<<CUSTOM-TYPE-END-IDENTIFIER>>`
- `split_message_content_for_custom_types()` finds these markers and splits the user message into multimodal content blocks (text + image_url parts), matching OpenAI's multimodal message format

### Reasoning (`dspy/adapters/types/reasoning.py`)

- String-like custom type
- `adapt_to_native_lm_feature()`: If LM supports native reasoning, sets `reasoning_effort` in lm_kwargs and removes the reasoning field from signature
- `parse_lm_response()`: Extracts `reasoning_content` from the response dict
- Falls back to text-based reasoning for non-reasoning models

### Tool / ToolCalls (`dspy/adapters/types/tool.py`)

- Handled in `_call_preprocess`: tools converted to litellm function calling format
- Tool and ToolCalls fields removed from signature before formatting
- In `_call_postprocess`: tool calls from LM response parsed back into `ToolCalls` objects

---

## 8. Adapter Summary Table

| Adapter | Input Format | Output Format | Fallback | Native Structured |
|---------|-------------|---------------|----------|-------------------|
| **ChatAdapter** | `[[ ## field ## ]]` markers | `[[ ## field ## ]]` markers | Falls back to JSONAdapter on parse error | No |
| **JSONAdapter** | `[[ ## field ## ]]` markers | JSON object | Falls back to `json_object` mode | Yes (OpenAI Structured Outputs) |
| **XMLAdapter** | `<field>...</field>` tags | `<field>...</field>` tags | Inherits ChatAdapter fallback | No |
| **TwoStepAdapter** | Natural language | Second LM call to extract | ChatAdapter for extraction | No |

---

## 9. Key Files

| File | Role |
|------|------|
| `dspy/adapters/base.py` | Abstract base, pipeline orchestration, demo formatting |
| `dspy/adapters/chat_adapter.py` | Default adapter with `[[ ## ]]` delimiters |
| `dspy/adapters/json_adapter.py` | JSON/structured output adapter |
| `dspy/adapters/xml_adapter.py` | XML tag-based adapter |
| `dspy/adapters/two_step_adapter.py` | Two-LM extraction adapter |
| `dspy/adapters/utils.py` | `format_field_value`, `parse_value`, `translate_field_type`, `serialize_for_json` |
| `dspy/adapters/types/base_type.py` | `Type` base class, multimodal content splitting |
| `dspy/adapters/types/image.py` | Image type with base64 encoding |
| `dspy/adapters/types/reasoning.py` | Native reasoning support |
| `dspy/adapters/types/tool.py` | Native tool calling support |
