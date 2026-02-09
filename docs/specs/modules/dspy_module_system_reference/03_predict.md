# Predict: The Foundation Primitive

## What Predict Is

`Predict` is the **only** leaf node in the DSPy module tree. It is the only class that inherits from both `Module` (callable, composable) and `Parameter` (discoverable by optimizers). Every higher-level module (ChainOfThought, ReAct, etc.) ultimately delegates to one or more Predict instances.

A Predict takes a Signature, formats it into a prompt via an adapter, calls an LM, parses the response back into typed outputs, and returns a Prediction.

---

## 1. Construction

```python
class Predict(Module, Parameter):
    def __init__(self, signature: str | type[Signature], callbacks=None, **config):
        super().__init__(callbacks=callbacks)
        self.stage = random.randbytes(8).hex()  # Unique ID for tracing
        self.signature = ensure_signature(signature)  # Parse string -> Signature class
        self.config = config  # Default LM kwargs (temperature, n, etc.)
        self.reset()

    def reset(self):
        """Clears all learned/optimizable state."""
        self.lm = None      # Per-predictor LM override (None = use settings.lm)
        self.traces = []     # Execution traces (for optimization)
        self.train = []      # Training examples
        self.demos = []      # Few-shot examples (THE primary optimizable state)
```

### Key Attributes

| Attribute | Type | Purpose | Optimizable? |
|-----------|------|---------|-------------|
| `signature` | `type[Signature]` | The typed I/O contract | Yes (instructions, field prefixes) |
| `demos` | `list[Example]` | Few-shot examples prepended to prompt | Yes (primary optimization lever) |
| `lm` | `LM \| None` | Per-predictor LM override | Yes (BootstrapFinetune replaces this) |
| `config` | `dict` | Default LM kwargs (temp, n, etc.) | No (set at construction) |
| `stage` | `str` | Random hex ID for tracing | No |
| `traces` | `list` | Execution traces for optimization | Bookkeeping |
| `train` | `list` | Training examples | Bookkeeping |

### `ensure_signature()`

Converts various inputs to a Signature class:
- String `"question -> answer"` -> parse into a Signature class
- Existing Signature class -> return as-is
- Dict of fields -> create a Signature class

---

## 2. The Forward Pipeline

`Predict.__call__(**kwargs)` -> `Module.__call__` (callbacks, tracking) -> `Predict.forward(**kwargs)`.

Note: `Predict.__call__` first validates that no positional args are passed (must use keyword args matching signature fields):

```python
def __call__(self, *args, **kwargs):
    if args:
        raise ValueError(self._get_positional_args_error_message())
    return super().__call__(**kwargs)
```

### 2.1 `forward()` -- Three Steps

```python
def forward(self, **kwargs):
    # Step 1: Resolve LM, merge config, extract demos
    lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

    # Step 2: Get adapter and run the full pipeline
    adapter = settings.adapter or ChatAdapter()

    if self._should_stream():
        with settings.context(caller_predict=self):
            completions = adapter(lm, lm_kwargs=config, signature=signature,
                                  demos=demos, inputs=kwargs)
    else:
        with settings.context(send_stream=None):
            completions = adapter(lm, lm_kwargs=config, signature=signature,
                                  demos=demos, inputs=kwargs)

    # Step 3: Build Prediction, record trace
    return self._forward_postprocess(completions, signature, **kwargs)
```

### 2.2 `_forward_preprocess()` -- The Critical Setup

This method extracts "privileged" kwargs that override Predict's defaults, resolves the LM, and prepares everything for the adapter call.

```python
def _forward_preprocess(self, **kwargs):
    # 1. Extract privileged kwargs (these are NOT passed to the LM as inputs)
    signature = kwargs.pop("signature", self.signature)
    signature = ensure_signature(signature)

    demos = kwargs.pop("demos", self.demos)

    config = {**self.config, **kwargs.pop("config", {})}

    lm = kwargs.pop("lm", self.lm) or settings.lm

    # 2. Validate LM exists and is the right type
    if lm is None or not isinstance(lm, BaseLM):
        raise ValueError("No LM is loaded / invalid LM type")

    # 3. Auto-adjust temperature for multi-generation
    if config.get("n", 1) > 1 and config.get("temperature", 0) <= 0.15:
        config["temperature"] = 0.7  # Prevent deterministic multi-gen

    # 4. Handle OpenAI predicted outputs
    if "prediction" in kwargs:
        config["prediction"] = kwargs.pop("prediction")

    # 5. Fill missing input fields with Pydantic defaults
    for field_name, field_info in signature.input_fields.items():
        if field_name not in kwargs:
            if field_info.default is not PydanticUndefined:
                kwargs[field_name] = field_info.default

    # 6. Warn about missing required inputs
    for field_name in signature.input_fields:
        if field_name not in kwargs:
            logger.warning(f"Missing input: {field_name}")

    return lm, config, signature, demos, kwargs
```

**LM resolution order**: `kwargs["lm"]` > `self.lm` > `settings.lm`

**Config merge**: `{**self.config, **kwargs["config"]}` -- per-call config overrides construction-time config.

### 2.3 `_forward_postprocess()` -- Tracing

```python
def _forward_postprocess(self, completions, signature, **kwargs):
    # 1. Build Prediction from completions
    pred = Prediction.from_completions(completions, signature=signature)

    # 2. Append to trace if tracing is enabled
    if kwargs.pop("_trace", True) and settings.trace is not None:
        trace = settings.trace
        if len(trace) >= settings.max_trace_size:
            trace.pop(0)  # LRU eviction
        trace.append((self, {**kwargs}, pred))
        # Tuple: (predictor_instance, input_kwargs_dict, prediction_output)

    return pred
```

**The trace tuple** `(self, inputs, prediction)` is how optimizers connect outputs back to specific Predict instances. BootstrapFewShot reads these traces to create demos.

---

## 3. Predict State Management

### `dump_state()` -- Serialization

```python
def dump_state(self, json_mode=True):
    state_keys = ["traces", "train"]
    state = {k: getattr(self, k) for k in state_keys}

    # Serialize demos (the main optimizable state)
    state["demos"] = []
    for demo in self.demos:
        demo = demo.copy()
        for field in demo:
            demo[field] = serialize_object(demo[field])  # Pydantic models -> dicts
        if isinstance(demo, dict) or not json_mode:
            state["demos"].append(demo)
        else:
            state["demos"].append(demo.toDict())

    # Signature state (instructions + field prefixes/descriptions)
    state["signature"] = self.signature.dump_state()

    # LM state (model config) or None
    state["lm"] = self.lm.dump_state() if self.lm else None

    return state
```

### `load_state()` -- Deserialization

```python
def load_state(self, state):
    excluded_keys = ["signature", "extended_signature", "lm"]
    for name, value in state.items():
        if name not in excluded_keys:
            setattr(self, name, value)  # demos, traces, train

    # Reconstruct signature from saved instructions/field metadata
    self.signature = self.signature.load_state(state["signature"])

    # Reconstruct LM from saved config
    self.lm = LM(**state["lm"]) if state["lm"] else None
```

### What Gets Serialized

| Field | Serialized? | Format |
|-------|------------|--------|
| `demos` | Yes | List of dicts (Example.toDict()) |
| `traces` | Yes | Raw list |
| `train` | Yes | Raw list |
| `signature` | Yes | `{instructions, fields: {name: {prefix, desc}}}` |
| `lm` | Yes (if set) | LM config dict (model name, kwargs) |
| `config` | No | Comes from code |
| `stage` | No | Random, regenerated |
| `callbacks` | No | Transient |

---

## 4. The Adapter Call

Inside `forward()`, the adapter call is the heart of the computation:

```python
adapter = settings.adapter or ChatAdapter()
completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)
```

The adapter does:
1. **`_call_preprocess()`**: Handle native tool calls, reasoning types. May remove fields from signature.
2. **`format(signature, demos, inputs)`**: Build message list (system + demos + user).
3. **`lm(messages=messages, **kwargs)`**: Actually call the LM.
4. **`_call_postprocess()`**: Parse each completion via `parse(signature, text)`.

The result is a list of dicts, one per completion, each containing the output field values.

Then `Prediction.from_completions()` wraps this into a Prediction object.

---

## 5. Prediction and Example

### Example (`dspy/primitives/example.py`)

Dict-like container with input/label separation:

```python
class Example:
    def __init__(self, **kwargs):
        self._store = kwargs          # The actual data
        self._input_keys = set()      # Which keys are inputs
        self._demos = []              # Attached demos (rarely used)

    def with_inputs(self, *keys):
        """Mark which fields are inputs. Returns self (mutates)."""
        self._input_keys = set(keys)
        return self

    def inputs(self):
        """Returns Example with only input keys."""
        return {k: v for k, v in self._store.items() if k in self._input_keys}

    def labels(self):
        """Returns Example with only non-input keys."""
        return {k: v for k, v in self._store.items() if k not in self._input_keys}
```

Training data and demos are both Examples. The `.with_inputs()` call marks the boundary between what gets passed as input and what's a label.

### Prediction (`dspy/primitives/prediction.py`)

Subclass of Example, returned by all modules:

```python
class Prediction(Example):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._completions = None   # All completions (not just the first)
        self._lm_usage = None      # Token usage tracking

    @classmethod
    def from_completions(cls, list_or_dict, signature=None):
        """
        Wraps completions into a Prediction.
        - Stores all completions as a Completions object
        - pred._store = {k: v[0] for k, v in completions.items()}
          (first completion is the default)
        """
        obj = cls()
        obj._completions = Completions(list_or_dict, signature=signature)
        # Set primary values to first completion
        obj._store = {k: v[0] for k, v in obj._completions.items()}
        return obj
```

Attribute access (`pred.answer`) returns the first completion's value. `pred.completions.answer` returns all completions for that field.

---

## 6. The Complete Flow

Putting it all together for a single `predict(question="What is 2+2?")` call:

```
1. Predict.__call__(question="What is 2+2?")
   -> Validates no positional args
   -> Module.__call__(**kwargs)
      -> @with_callbacks: on_module_start
      -> Push self to caller_modules stack
      -> Predict.forward(question="What is 2+2?")

2. _forward_preprocess(question="What is 2+2?")
   -> signature = self.signature (e.g., "question -> answer")
   -> demos = self.demos (e.g., 3 few-shot examples)
   -> config = {**self.config} (e.g., {temperature: 0})
   -> lm = self.lm or settings.lm
   -> kwargs = {question: "What is 2+2?"}
   -> return (lm, config, signature, demos, kwargs)

3. adapter = settings.adapter or ChatAdapter()

4. completions = adapter(lm, lm_kwargs=config, signature=signature,
                         demos=demos, inputs=kwargs)

   Inside adapter.__call__:
   a. _call_preprocess: check for tools/native types, may modify signature
   b. format(signature, demos, inputs):
      - System message: field descriptions + format structure + instructions
      - Demo messages: few-shot examples as user/assistant pairs
      - User message: current inputs + output format reminder
   c. lm(messages=messages, **lm_kwargs):
      - litellm call to the actual LM
      - Returns list of completion strings
   d. _call_postprocess: for each completion:
      - parse(signature, text): extract output field values
      - Returns list of dicts: [{answer: "4"}, ...]

5. _forward_postprocess(completions, signature, question="What is 2+2?")
   -> Prediction.from_completions([{answer: "4"}])
   -> Append (self, {question: "What is 2+2?"}, prediction) to settings.trace
   -> Return prediction

6. Module.__call__ returns
   -> @with_callbacks: on_module_end
   -> Return Prediction(answer="4")
```
