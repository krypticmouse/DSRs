# Signatures: DSPy's Type System

## What a Signature Is

A Signature is a **typed contract** between a module and an LM: named input fields -> named output fields, with instructions. It's the thing that makes DSPy declarative -- you say "question -> answer" and the framework handles prompt construction, output parsing, and type validation.

**Critical implementation detail**: A Signature is a **class**, not an instance. When you write `dspy.Signature("question -> answer")`, you get back a new *type* (a dynamically-created Pydantic BaseModel subclass), not an object. Operations like `prepend`, `with_instructions`, `delete` all return *new classes*. This is metaclass-heavy Python.

---

## 1. File Layout

```
dspy/signatures/
  signature.py   -- Signature class, SignatureMeta metaclass, make_signature(), parsing
  field.py       -- InputField(), OutputField() factory functions
  utils.py       -- get_dspy_field_type() helper
```

---

## 2. InputField and OutputField

These are **factory functions** (not classes) that return `pydantic.Field()` instances with DSPy metadata stuffed into `json_schema_extra`:

```python
# dspy/signatures/field.py

def InputField(**kwargs):
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))

def OutputField(**kwargs):
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))
```

`move_kwargs` separates DSPy-specific arguments from Pydantic-native arguments:

**DSPy-specific** (stored in `json_schema_extra`):
| Argument | Type | Purpose |
|----------|------|---------|
| `__dspy_field_type` | `"input"` or `"output"` | The discriminator -- how the system tells inputs from outputs |
| `desc` | `str` | Field description shown to the LM in the prompt |
| `prefix` | `str` | Prompt prefix for this field (e.g., `"Question:"`) |
| `format` | `callable` | Optional formatting function |
| `parser` | `callable` | Optional parsing function |
| `constraints` | `str` | Human-readable constraint strings |

**Pydantic-native** (passed through to `pydantic.Field`):
| Argument | Purpose |
|----------|---------|
| `gt`, `ge`, `lt`, `le` | Numeric constraints |
| `min_length`, `max_length` | String/collection length |
| `default` | Default value |

**Constraint translation**: Pydantic constraints are automatically converted to human-readable strings. `OutputField(ge=5, le=10)` generates `constraints="greater than or equal to: 5, less than or equal to: 10"` which gets included in the prompt so the LM knows the bounds.

---

## 3. SignatureMeta: The Metaclass

`SignatureMeta` extends `type(BaseModel)` (Pydantic's metaclass). It does three key things:

### 3.1 `__call__` -- String Shorthand Interception

```python
class SignatureMeta(type(BaseModel)):
    def __call__(cls, *args, **kwargs):
        # If called with a string like Signature("question -> answer"),
        # route to make_signature() to create a new class (not instance)
        if cls is Signature:
            if len(args) == 1 and isinstance(args[0], (str, dict)):
                return make_signature(args[0], kwargs.pop("instructions", None))
        # Otherwise, create an actual instance (rare in normal DSPy usage)
        return super().__call__(*args, **kwargs)
```

This means `dspy.Signature("question -> answer")` returns a **new class**, not an instance.

### 3.2 `__new__` -- Class Creation

When a Signature class is being *defined* (either via `class QA(dspy.Signature)` or via `make_signature()`):

```python
def __new__(mcs, signature_name, bases, namespace):
    # 1. Set str as default type for fields without annotations
    for name in namespace:
        if name not in annotations:
            annotations[name] = str

    # 2. Preserve field ordering: inputs before outputs
    # (reorder annotations dict to match declaration order)

    # 3. Let Pydantic create the class
    cls = super().__new__(mcs, signature_name, bases, namespace)

    # 4. Set default instructions if none given
    if not cls.__doc__:
        inputs = ", ".join(f"`{k}`" for k in cls.input_fields)
        outputs = ", ".join(f"`{k}`" for k in cls.output_fields)
        cls.__doc__ = f"Given the fields {inputs}, produce the fields {outputs}."

    # 5. Validate: every field must have InputField or OutputField
    for name, field in cls.model_fields.items():
        if "__dspy_field_type" not in (field.json_schema_extra or {}):
            raise TypeError(f"Field '{name}' must use InputField or OutputField")

    # 6. Auto-generate prefix and desc for fields that don't have them
    for name, field in cls.model_fields.items():
        extra = field.json_schema_extra
        if "prefix" not in extra:
            extra["prefix"] = infer_prefix(name)  # snake_case -> "Title Case:"
        if "desc" not in extra:
            extra["desc"] = f"${{{name}}}"  # template placeholder
```

### 3.3 `infer_prefix()` -- Name to Prompt Prefix

Converts field names to human-readable prefixes:
- `"question"` -> `"Question:"`
- `"some_attribute_name"` -> `"Some Attribute Name:"`
- `"HTMLParser"` -> `"HTML Parser:"`

Uses regex to split on underscores and camelCase boundaries, then title-cases and joins.

---

## 4. Two Ways to Define Signatures

### Class-Based (Full Control)

```python
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

Here `QA` is a class. `QA.__doc__` becomes the instructions. Fields are declared as class attributes with type annotations and InputField/OutputField defaults.

### String Shorthand (Quick)

```python
sig = dspy.Signature("question -> answer")
sig = dspy.Signature("question: str, context: list[str] -> answer: str")
sig = dspy.Signature("question -> answer", "Answer the question.")
```

When `SignatureMeta.__call__` sees a string, it routes to `make_signature()`.

### The String Parser

The parser is clever -- it uses Python's **AST module**:

```python
def _parse_field_string(field_string: str, names=None):
    # Wraps the field string as function parameters and parses with ast
    args = ast.parse(f"def f({field_string}): pass").body[0].args.args
```

This means field strings follow Python function parameter syntax: `question: str, context: list[int]` is valid because it would be valid as `def f(question: str, context: list[int]): pass`.

**Type resolution** happens in `_parse_type_node()`, which recursively walks the AST:
- Simple: `int`, `str`, `float`, `bool`
- Generic: `list[int]`, `dict[str, float]`, `tuple[str, int]`
- Union: `Union[int, str]`, `Optional[str]`, PEP 604 `int | str`
- Nested: `dict[str, list[Optional[Tuple[int, str]]]]`
- Custom: looked up via a `names` dict or by walking the Python call stack

**Custom type auto-detection** (`_detect_custom_types_from_caller`): When you write `Signature("input: MyType -> output")`, the metaclass walks up the call stack (up to 100 frames) looking in `f_locals` and `f_globals` for `MyType`. This is fragile but convenient. The reliable alternative is passing `custom_types={"MyType": MyType}`.

### `make_signature()` -- The Factory

```python
def make_signature(signature, instructions=None, signature_name="StringSignature"):
    """
    Accepts either:
    - A string: "question -> answer" (parsed into fields)
    - A dict: {"question": InputField(), "answer": OutputField()} (used directly)

    Creates a new Signature class via pydantic.create_model().
    """
    if isinstance(signature, str):
        fields = _parse_signature(signature)
    else:
        fields = signature  # dict of {name: (type, FieldInfo)}

    # pydantic.create_model creates a new BaseModel subclass dynamically
    model = pydantic.create_model(
        signature_name,
        __base__=Signature,
        __doc__=instructions,
        **fields,
    )
    return model
```

---

## 5. Signature Properties (Class-Level)

These are properties on the *metaclass*, meaning they're accessed on the class itself (not instances):

```python
@property
def instructions(cls) -> str:
    """The cleaned docstring. This is the task description shown to the LM."""
    return cls.__doc__

@property
def input_fields(cls) -> dict[str, FieldInfo]:
    """Fields where __dspy_field_type == "input", in declaration order"""
    return {k: v for k, v in cls.model_fields.items()
            if v.json_schema_extra["__dspy_field_type"] == "input"}

@property
def output_fields(cls) -> dict[str, FieldInfo]:
    """Fields where __dspy_field_type == "output", in declaration order"""
    return {k: v for k, v in cls.model_fields.items()
            if v.json_schema_extra["__dspy_field_type"] == "output"}

@property
def fields(cls) -> dict[str, FieldInfo]:
    """All fields: {**input_fields, **output_fields}"""
    return {**cls.input_fields, **cls.output_fields}

@property
def signature(cls) -> str:
    """String representation: "input1, input2 -> output1, output2" """
    inputs = ", ".join(cls.input_fields.keys())
    outputs = ", ".join(cls.output_fields.keys())
    return f"{inputs} -> {outputs}"
```

---

## 6. Signature Manipulation

**All manipulation methods return new Signature classes.** The original is never mutated. This is the immutable pattern.

### `with_instructions(instructions: str) -> type[Signature]`

```python
def with_instructions(cls, instructions: str):
    """New Signature with different instructions, same fields."""
    return Signature(cls.fields, instructions)
```

### `with_updated_fields(name, type_=None, **kwargs) -> type[Signature]`

```python
def with_updated_fields(cls, name, type_=None, **kwargs):
    """Deep-copies fields, updates json_schema_extra for the named field, creates new Signature."""
    fields_copy = deepcopy(cls.fields)
    fields_copy[name].json_schema_extra = {**fields_copy[name].json_schema_extra, **kwargs}
    if type_ is not None:
        fields_copy[name].annotation = type_
    return Signature(fields_copy, cls.instructions)
```

Used by COPRO to change field prefixes: `sig.with_updated_fields("answer", prefix="Final Answer:")`.

### `prepend(name, field, type_=None)` / `append(name, field, type_=None)`

Both delegate to `insert()`:

```python
def prepend(cls, name, field, type_=None):
    return cls.insert(0, name, field, type_)

def append(cls, name, field, type_=None):
    return cls.insert(-1, name, field, type_)
```

### `insert(index, name, field, type_=None)`

```python
def insert(cls, index, name, field, type_=None):
    """
    Splits fields into input_fields and output_fields lists.
    Determines which list based on __dspy_field_type.
    Inserts at the given index.
    Recombines and creates a new Signature.
    """
    input_fields = list(cls.input_fields.items())
    output_fields = list(cls.output_fields.items())

    lst = input_fields if field.json_schema_extra["__dspy_field_type"] == "input" else output_fields
    lst.insert(index, (name, (type_ or str, field)))

    new_fields = dict(input_fields + output_fields)
    return Signature(new_fields, cls.instructions)
```

### `delete(name)`

```python
def delete(cls, name):
    """Removes the named field. Returns new Signature."""
    fields_copy = dict(cls.fields)
    fields_copy.pop(name, None)
    return Signature(fields_copy, cls.instructions)
```

---

## 7. How Modules Modify Signatures

This is the core of the "augmentation pattern." Each module type manipulates the signature differently:

### ChainOfThought -- Prepend Reasoning

```python
extended_signature = signature.prepend(
    name="reasoning",
    field=dspy.OutputField(
        prefix="Reasoning: Let's think step by step in order to",
        desc="${reasoning}"
    ),
    type_=str
)
```

`"question -> answer"` becomes `"question -> reasoning, answer"`. The LM is forced to produce reasoning before the answer.

### ReAct -- Build From Scratch

```python
react_signature = (
    dspy.Signature({**signature.input_fields}, "\n".join(instr))
    .append("trajectory", dspy.InputField(), type_=str)
    .append("next_thought", dspy.OutputField(), type_=str)
    .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
    .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
)
```

Note `Literal[tuple(tools.keys())]` -- the type system constrains what the LM can output for tool selection.

### MultiChainComparison -- Append Input Fields + Prepend Output

```python
for idx in range(M):
    signature = signature.append(
        f"reasoning_attempt_{idx+1}",
        InputField(prefix=f"Student Attempt #{idx+1}:")
    )
signature = signature.prepend("rationale", OutputField(prefix="Accurate Reasoning: ..."))
```

### Refine -- Dynamic Injection at Call Time

```python
signature = signature.append("hint_", InputField(desc="A hint from an earlier run"))
```

Done *inside the adapter wrapper* at call time, not at construction time. This is unique -- most modules modify signatures at `__init__`.

---

## 8. Signature Serialization

### `dump_state()` / `load_state(state)`

```python
def dump_state(cls):
    """Dumps instructions + per-field prefix and description."""
    return {
        "instructions": cls.instructions,
        "fields": {
            name: {
                "prefix": field.json_schema_extra.get("prefix"),
                "desc": field.json_schema_extra.get("desc"),
            }
            for name, field in cls.fields.items()
        }
    }

def load_state(cls, state):
    """Creates a new Signature from stored state.
    Updates instructions and field prefix/desc from the saved state."""
    new_sig = cls.with_instructions(state["instructions"])
    for name, field_state in state.get("fields", {}).items():
        if name in new_sig.fields:
            new_sig = new_sig.with_updated_fields(name, **field_state)
    return new_sig
```

This is what `Predict.dump_state()` calls under `state["signature"]`. It preserves the optimized instructions and field metadata while the field types and structure come from the code.

---

## 9. Pydantic Integration

### How Types Map to Prompts

The adapter uses `translate_field_type()` to generate type hints for the LM:

| Python Type | Prompt Hint |
|-------------|------------|
| `str` | (no hint) |
| `bool` | `"must be True or False"` |
| `int` / `float` | `"must be a single int/float value"` |
| `Enum` | `"must be one of: val1; val2; val3"` |
| `Literal["a", "b"]` | `"must exactly match one of: a; b"` |
| Complex types | `"must adhere to the JSON schema: {...}"` (Pydantic JSON schema) |

### How Parsing Works

Parsing happens in `parse_value()` (`dspy/adapters/utils.py`):

1. `str` annotation -> return raw string
2. `Enum` -> find matching member by value or name
3. `Literal` -> validate against allowed values
4. `bool/int/float` -> type cast
5. Complex types -> `json_repair.loads()` then `pydantic.TypeAdapter(annotation).validate_python()`
6. DSPy Type subclasses -> custom parsing

---

## 10. The Signature as Contract

A Signature encodes:

| Aspect | How |
|--------|-----|
| **What inputs are needed** | `input_fields` dict |
| **What outputs are produced** | `output_fields` dict |
| **How to describe the task** | `instructions` (docstring) |
| **How to present each field** | `prefix` and `desc` per field |
| **What types are expected** | Python type annotations per field |
| **What constraints apply** | Pydantic constraints -> `constraints` string |
| **Field ordering** | Dict insertion order (inputs first, then outputs) |

The signature flows through the entire system:
- **Module** holds it on `self.signature`
- **Adapter.format()** reads it to build the prompt
- **Adapter.parse()** reads it to know what to extract
- **Optimizers** modify `instructions` and field `prefix`/`desc`
- **save()/load()** serializes/deserializes it
