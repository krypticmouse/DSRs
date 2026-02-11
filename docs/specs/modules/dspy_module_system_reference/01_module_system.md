# The Module System: BaseModule, Module, Parameter

## Current Scope Addendum (2026-02-12)

This document is historical DSPy/Python reference material, preserved for context.

It is not the active Rust runtime contract for `dspy-rs`. In current V1–V5 typed scope:
- Public module calls are typed and return `Result<Predicted<O>, PredictError>`.
- `_compiled`, `BaseModule`, and public `named_parameters()` are not part of the active Rust API surface.
- Optimizer discovery is internal via Facet-based predictor walking.

Refer to the active contracts in:
- `docs/specs/modules/design_reference.md`
- `docs/specs/modules/breadboard.md`

## Three Layers

The module system has three layers, each adding capabilities:

1. **`Parameter`** (`dspy/predict/parameter.py`) -- Empty marker class. Makes things discoverable by optimizers.
2. **`BaseModule`** (`dspy/primitives/base_module.py`) -- Tree traversal, serialization, copy mechanics.
3. **`Module`** (`dspy/primitives/module.py`) -- The `__call__` -> `forward()` protocol, callbacks, metaclass magic.

`Predict` inherits from both `Module` and `Parameter`, making it both callable and optimizable.

---

## 1. Parameter: The Marker

```python
# dspy/predict/parameter.py
class Parameter:
    pass
```

That's the entire class. No methods, no state. It exists so `isinstance(obj, Parameter)` can distinguish "things optimizers can tune" from "things that are just structural." In the current codebase, `Predict` is the *only* class that inherits from `Parameter`.

**Why this matters**: When `BaseModule.named_parameters()` walks the object graph, it collects everything that passes `isinstance(value, Parameter)`. Since only `Predict` does, optimizers only ever see `Predict` instances. Higher-level modules (ChainOfThought, ReAct) are invisible to optimizers -- they're just containers that *hold* Predict instances.

---

## 2. BaseModule: The Tree

`BaseModule` provides the infrastructure for treating a module hierarchy as a traversable tree.

### 2.1 `named_parameters()` -- DFS Parameter Discovery

This is the most important method in the entire module system. Every optimizer calls it.

```python
def named_parameters(self):
    """
    DFS walk of self.__dict__. Finds all Parameter instances (i.e., Predict objects).
    Returns list of (dotted_path_string, Parameter_instance) tuples.

    Rules:
    - If self is a Parameter, includes ("self", self)
    - Parameter instances in __dict__ -> added directly
    - Module instances in __dict__ -> recurse (unless _compiled=True)
    - Lists/tuples -> iterate with indexed names: "name[0]", "name[1]"
    - Dicts -> iterate with keyed names: "name['key']"
    - Tracks visited set by id() to handle diamond DAGs (same object reachable via multiple paths)
    """
    import dspy
    from dspy.predict.parameter import Parameter

    visited = set()
    named_parameters = []

    def add_parameter(param_name, param_value):
        if isinstance(param_value, Parameter):
            if id(param_value) not in visited:
                visited.add(id(param_value))
                named_parameters.append((param_name, param_value))
        elif isinstance(param_value, dspy.Module):
            # CRITICAL: _compiled modules are FROZEN -- we don't recurse into them.
            # This is how pre-optimized sub-modules keep their state.
            if not getattr(param_value, "_compiled", False):
                for sub_name, param in param_value.named_parameters():
                    add_parameter(f"{param_name}.{sub_name}", param)

    if isinstance(self, Parameter):
        add_parameter("self", self)

    for name, value in self.__dict__.items():
        if isinstance(value, Parameter):
            add_parameter(name, value)
        elif isinstance(value, dspy.Module):
            if not getattr(value, "_compiled", False):
                for sub_name, param in value.named_parameters():
                    add_parameter(f"{name}.{sub_name}", param)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                add_parameter(f"{name}[{idx}]", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                add_parameter(f"{name}['{key}']", item)

    return named_parameters
```

**Example**: Given a module `MyProgram` with:
```python
class MyProgram(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought("question -> answer")
        self.summarize = dspy.Predict("text -> summary")
```

`named_parameters()` returns:
```
[
    ("cot.predict", <Predict instance>),   # ChainOfThought holds self.predict
    ("summarize",   <Predict instance>),   # Predict IS a Parameter
]
```

The dotted path names are how optimizers map traces back to specific predictors and how `save()`/`load()` serialize state.

### 2.2 `named_sub_modules()` -- BFS Module Discovery

```python
def named_sub_modules(self, type_=None, skip_compiled=False):
    """
    BFS traversal of ALL BaseModule instances in the tree.
    Different from named_parameters:
    - BFS not DFS
    - Returns ALL modules, not just Parameters
    - Optional type filter and compiled-skip flag
    """
    if type_ is None:
        type_ = BaseModule

    queue = deque([("self", self)])
    seen = {id(self)}

    def add_to_queue(name, item):
        if id(item) not in seen:
            seen.add(id(item))
            queue.append((name, item))

    while queue:
        name, item = queue.popleft()
        if isinstance(item, type_):
            yield name, item
        if isinstance(item, BaseModule):
            if skip_compiled and getattr(item, "_compiled", False):
                continue
            for sub_name, sub_item in item.__dict__.items():
                add_to_queue(f"{name}.{sub_name}", sub_item)
        elif isinstance(item, (list, tuple)):
            for i, sub_item in enumerate(item):
                add_to_queue(f"{name}[{i}]", sub_item)
        elif isinstance(item, dict):
            for key, sub_item in item.items():
                add_to_queue(f"{name}[{key}]", sub_item)
```

### 2.3 `deepcopy()` -- Safe Deep Copying

```python
def deepcopy(self):
    """
    Strategy:
    1. Try copy.deepcopy(self) -- works if all attributes are picklable
    2. If that fails, manual fallback:
       - Create empty instance via __new__ (no __init__)
       - For each attr in __dict__:
         - BaseModule -> recursive deepcopy()
         - Other -> try deepcopy, fallback copy.copy, fallback reference
    """
    try:
        return copy.deepcopy(self)
    except Exception:
        pass

    new_instance = self.__class__.__new__(self.__class__)
    for attr, value in self.__dict__.items():
        if isinstance(value, BaseModule):
            setattr(new_instance, attr, value.deepcopy())
        else:
            try:
                setattr(new_instance, attr, copy.deepcopy(value))
            except Exception:
                try:
                    setattr(new_instance, attr, copy.copy(value))
                except Exception:
                    setattr(new_instance, attr, value)
    return new_instance
```

**Why the fallback matters**: Some modules hold references to non-picklable objects (LM connections, thread pools). The manual fallback ensures the module tree is still copyable even when `copy.deepcopy` chokes.

### 2.4 `reset_copy()` -- Fresh Copy for Optimization

```python
def reset_copy(self):
    """Deep copy, then reset() every parameter.
    Creates a fresh copy with architecture intact but all learned state cleared.
    Used by optimizers to create candidate programs."""
    new_instance = self.deepcopy()
    for param in new_instance.parameters():
        param.reset()
    return new_instance
```

`param.reset()` on a Predict clears `self.lm`, `self.traces`, `self.train`, and `self.demos`. The architecture (signature, config) is preserved; the learned state is wiped.

### 2.5 `dump_state()` / `load_state()` -- Serialization

```python
def dump_state(self, json_mode=True):
    """Serializes every parameter: {dotted_path: param.dump_state()}"""
    return {name: param.dump_state(json_mode=json_mode)
            for name, param in self.named_parameters()}

def load_state(self, state):
    """Deserializes: walks named_parameters(), calls each param.load_state()"""
    for name, param in self.named_parameters():
        param.load_state(state[name])
```

For a Predict, `dump_state()` serializes:
- `traces` (execution traces)
- `train` (training examples)
- `demos` (few-shot examples, serialized via `serialize_object` for JSON safety)
- `signature` state (instructions + field prefixes/descriptions)
- `lm` state (model config) or None

### 2.6 `save()` / `load()` -- File I/O

Two modes:

**State-only (default)**: Saves just the optimized state (demos, instructions, etc.) to `.json` or `.pkl`.
```python
def save(self, path, save_program=False):
    # state = self.dump_state() + metadata (python/dspy/cloudpickle versions)
    # Write to JSON or pickle based on file extension
```

**Full program** (`save_program=True`): Uses `cloudpickle` to serialize the entire module object (architecture + state) to a directory containing `program.pkl` + `metadata.json`.

`load()` reads state and calls `self.load_state(state)`. Note: this loads state *into* an existing module. For loading a whole program from pickle, there's a separate `dspy.load()` function.

---

## 3. Module: The Call Protocol

`Module` extends `BaseModule` with the call/forward protocol, a metaclass that ensures safe initialization, and convenience methods.

### 3.1 `ProgramMeta` -- The Metaclass

```python
class ProgramMeta(type):
    """Ensures _base_init runs BEFORE __init__, even if subclass forgets super().__init__().

    When you do MyModule(args):
    1. __new__ creates the instance (no __init__ yet)
    2. Module._base_init(obj) -- sets _compiled, callbacks, history
    3. cls.__init__(obj, args) -- the user's actual __init__
    4. Safety: ensures callbacks and history exist even if __init__ didn't set them
    """
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            Module._base_init(obj)
            cls.__init__(obj, *args, **kwargs)
            if not hasattr(obj, "callbacks"):
                obj.callbacks = []
            if not hasattr(obj, "history"):
                obj.history = []
        return obj
```

**Why this exists**: If a user writes `class MyModule(dspy.Module)` and forgets `super().__init__()`, the module would lack `_compiled`, `callbacks`, and `history`. The metaclass guarantees these always exist.

### 3.2 Module Attributes

```python
class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False    # Has this module been optimized?
        self.callbacks = []       # List of BaseCallback instances
        self.history = []         # LM call history

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False
        self.history = []
```

### 3.3 `__call__()` -- The Central Dispatch

```python
@with_callbacks  # Wraps with on_module_start / on_module_end callbacks
def __call__(self, *args, **kwargs):
    """
    1. Get caller_modules stack from settings (tracks nested module calls)
    2. Append self to the stack
    3. In a settings.context with updated caller_modules:
       a. If usage tracking enabled and no tracker yet, create one
       b. Call self.forward(*args, **kwargs)
       c. If tracking, attach token usage to the Prediction
    4. Return the Prediction
    """
    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and no_tracker_yet:
            with track_usage() as usage_tracker:
                output = self.forward(*args, **kwargs)
            tokens = usage_tracker.get_total_tokens()
            self._set_lm_usage(tokens, output)
            return output
        return self.forward(*args, **kwargs)
```

**`__call__` vs `forward()`**: `__call__` is the public entry point. It handles callbacks, usage tracking, and the module call stack. `forward()` is the actual logic that subclasses override. There is a `__getattribute__` override that **warns** if you call `.forward()` directly (it inspects the call stack):

```python
def __getattribute__(self, name):
    attr = super().__getattribute__(name)
    if name == "forward" and callable(attr):
        stack = inspect.stack()
        forward_called_directly = len(stack) <= 1 or stack[1].function != "__call__"
        if forward_called_directly:
            logger.warning("Calling module.forward() directly is discouraged. Use module() instead.")
    return attr
```

### 3.4 Pickle Support

```python
def __getstate__(self):
    """Excludes history and callbacks (transient state) from pickle"""
    state = self.__dict__.copy()
    state.pop("history", None)
    state.pop("callbacks", None)
    return state

def __setstate__(self, state):
    """Restores history and callbacks as empty on unpickle"""
    self.__dict__.update(state)
    if not hasattr(self, "history"):
        self.history = []
    if not hasattr(self, "callbacks"):
        self.callbacks = []
```

### 3.5 Convenience Methods

```python
def named_predictors(self):
    """Filters named_parameters() to only Predict instances"""
    from dspy.predict.predict import Predict
    return [(name, param) for name, param in self.named_parameters()
            if isinstance(param, Predict)]

def predictors(self):
    """Just the Predict objects, no names"""
    return [param for _, param in self.named_predictors()]

def set_lm(self, lm):
    """Sets the LM on ALL predictors in the tree"""
    for _, param in self.named_predictors():
        param.lm = lm

def get_lm(self):
    """Returns the LM if all predictors share one, raises if they differ"""

def map_named_predictors(self, func):
    """Applies func to each predictor and replaces it in the tree.
    Uses magicattr.set for nested path assignment (handles dotted paths)."""
    for name, predictor in self.named_predictors():
        set_attribute_by_name(self, name, func(predictor))
    return self
```

---

## 4. The `_compiled` Flag

`_compiled` is a boolean that controls optimizer traversal:

1. Initialized to `False` on every new Module (via `_base_init`)
2. Set to `True` by optimizers after compilation (e.g., `student._compiled = True`)
3. When `True`, `named_parameters()` **stops recursing** into this module -- its Predict instances are invisible to further optimization
4. This is how you compose pre-optimized modules: a compiled sub-module's demos and signature instructions won't be overwritten by a parent optimizer

**Example**:
```python
# Pre-optimize a sub-module
optimized_qa = bootstrap.compile(qa_module, trainset=data)
# optimized_qa._compiled is now True

# Use it in a larger program
class Pipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Predict("query -> passages")
        self.qa = optimized_qa  # _compiled=True, frozen

# When a parent optimizer runs on Pipeline:
# named_parameters() finds: [("retrieve", <Predict>)]
# It does NOT find optimized_qa's internal Predict -- it's frozen.
```

---

## 5. The Full Hierarchy

```
BaseModule
  |-- named_parameters()        # DFS, finds Parameters (Predict instances)
  |-- named_sub_modules()       # BFS, finds all Modules
  |-- deepcopy() / reset_copy() # Safe copying
  |-- dump_state() / load_state() / save() / load()  # Serialization
  |
  +-- Module (metaclass=ProgramMeta)
        |-- __call__() -> forward()   # The call protocol
        |-- callbacks, history        # Transient state
        |-- _compiled                 # Freeze flag
        |-- named_predictors()        # Convenience filter
        |-- set_lm() / get_lm()      # LM management
        |
        +-- Predict (also inherits Parameter)
              |-- signature, demos, lm, config  # Optimizable state
              |-- forward() -> adapter -> LM -> parse -> Prediction
              |-- traces, train                 # Optimization bookkeeping
              |-- reset()                       # Clear learned state
```

**The dual inheritance of Predict is the key design decision**: It is both a `Module` (callable, composable, has forward()) and a `Parameter` (discoverable by optimizers). Everything else in the system follows from this.
