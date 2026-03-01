# Investigation: Why PyO3 Doesn't Generate Real Python Dataclasses

## Summary

PyO3's `#[pyclass]` generates **opaque native extension types**, not Python dataclasses. This is not a bug or oversight — it's a fundamental consequence of Rust's ownership model being incompatible with the Python dataclass protocol. The gap is architectural, not accidental, and PyO3 has no plans to close it fully. Your project (DSRs) already works around this extensively via the `rlm-derive` crate.

## The Lie

When you write this in Rust:

```rust
#[pyclass(get_all, set_all, frozen)]
struct Answer {
    question: String,
    answer: String,
    confidence: f64,
}
```

Python sees **this**:

```python
>>> type(answer)
<class 'my_module.Answer'>

>>> dataclasses.is_dataclass(answer)
False

>>> dataclasses.asdict(answer)
TypeError: asdict() should be called on dataclass instances

>>> answer.__dict__
AttributeError: 'Answer' object has no attribute '__dict__'

>>> answer.__dataclass_fields__
AttributeError: ...

>>> vars(answer)
TypeError: vars() argument must have __dict__ attribute
```

It *looks* like a class with attributes. It *is not* a Python dataclass, a namedtuple, an attrs class, or anything from the Python data-object ecosystem. It's a CPython extension type backed by a Rust struct, with property descriptors bolted on.

## Why It Can't Be a Real Dataclass

### 1. The Dataclass Protocol Is Pure Python Decoration

Python's `@dataclass` works by:
1. Reading `__annotations__` from the class
2. Using `exec()` to generate `__init__`, `__repr__`, `__eq__`, `__hash__` as string-based function definitions
3. Setting `__dataclass_fields__` (an `OrderedDict[str, Field]`) on the class
4. Setting `__dataclass_params__` with decorator configuration

`dataclasses.is_dataclass()` is literally:
```python
def is_dataclass(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, '__dataclass_fields__')
```

PyO3's `#[pyclass]` never sets `__dataclass_fields__` because it has no concept of Python `Field` objects at the Rust level. The Rust struct's fields exist in Rust memory; they're exposed to Python through `#[getter]` property descriptors, not through the dataclass field protocol.

### 2. Memory Layout Is Fundamentally Different

A Python dataclass stores its fields as Python objects in `__dict__` (or `__slots__`). A PyO3 `#[pyclass]` stores its data as a Rust struct in a `PyCell<T>` wrapper that does **runtime borrow checking** (like `RefCell<T>`). There is no `__dict__`. The "fields" are property descriptors that call into Rust getters.

This means:
- `vars(obj)` → fails (no `__dict__`)
- `obj.__dict__` → fails
- `dataclasses.asdict(obj)` → fails (no `__dataclass_fields__`)
- `json.dumps(obj)` → fails
- Pydantic `model_validate(obj)` → fails (can't introspect)

### 3. Ownership & GIL Semantics Block Transparent Bridging

Rust structs have ownership semantics. Once a struct is handed to the Python runtime:
- The Rust borrow checker can no longer reason about `&mut` references
- PyO3 must do runtime borrow checking (the `RefCell`-like pattern)
- Lifetimes and generics cannot be expressed in Python's type system
- The object must be `Send + Sync` (Python threads may access it)

This means you can't just "wrap" a Rust struct as a Python dict — the Rust struct owns its memory, and Python's dataclass machinery expects to own `__dict__` entries.

### 4. `get_all`/`set_all` Is Lipstick on a Pig

PyO3's `#[pyclass(get_all, set_all)]` (added in 0.18) generates property descriptors for all fields. This makes `obj.field_name` work. But:

- These are `@property`-style descriptors, not dataclass fields
- Each access calls a Rust getter function that clones the value into a Python object
- There is no field metadata, no `Field()` objects, no defaults mechanism
- Python tooling (IDE autocomplete, mypy, pydantic, `dataclasses.asdict`) doesn't know they exist

### 5. The Stub Generation Problem Compounds This

PyO3 is actively working on `.pyi` stub file generation (issue #5137, ~28/40 tasks done), but stubs declare the class as:

```python
class Answer:
    question: str
    answer: str
    confidence: float
```

This *looks* like it could be a dataclass to a type checker, but at runtime it's not. The mismatch between static type stubs and runtime behavior is actively confusing.

## What PyO3 0.28.2 Has Actually Shipped (as of 2026-02-18)

You can now combine these to get ~80% of a dataclass:

```rust
#[pyclass(frozen, get_all, eq, hash, str, new = "from_fields")]
#[derive(Clone, PartialEq, Hash)]
struct Answer {
    question: String,
    answer: String,
    confidence: f64,
}

impl std::fmt::Display for Answer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Answer(question={:?}, answer={:?}, confidence={})",
            self.question, self.answer, self.confidence)
    }
}
```

This gives you:
- `Answer(question="what?", answer="42", confidence=0.9)` — construction with kwargs ✅
- `a.question`, `a.answer`, `a.confidence` — attribute access ✅
- `a == b` — equality ✅
- `hash(a)` — hashable ✅
- `str(a)` — string representation ✅
- Frozen/immutable ✅

| Feature | Status | What It Does |
|---------|--------|--------------|
| `#[pyclass(get_all)]` | ✅ 0.18+ | Generates `@property` getters for all fields |
| `#[pyclass(set_all)]` | ✅ 0.18+ | Generates `@property` setters for all fields |
| `#[pyclass(frozen)]` | ✅ | Removes runtime borrow overhead for immutable types |
| `#[pyclass(eq)]` | ✅ 0.22+ | `__eq__` from `PartialEq` |
| `#[pyclass(ord)]` | ✅ 0.22+ | `__lt__`/`__gt__` etc. from `PartialOrd` |
| `#[pyclass(hash)]` | ✅ 0.22+ | `__hash__` from `Hash` (requires `eq` + `frozen`) |
| `#[pyclass(str)]` | ✅ 0.22+ | `__str__` from `Display` |
| `#[pyclass(new = "from_fields")]` | ✅ **0.28+** | Generates `__new__` with all fields as kwargs |
| `#[init]` in `#[pymethods]` | ✅ **0.28+** | Custom `__init__` for subclass init flow |
| `#[pyclass(repr)]` | 🚧 | `__repr__` from `Debug` (stuck on type info problem) |
| `#[pyclass(dataclass)]` | ❌ | Proposed but no implementation |
| `__dataclass_fields__` | ❌ | Not on the roadmap |
| `dataclasses.is_dataclass()` | ❌ | Will always be `False` |
| `dataclasses.asdict()` compat | ❌ | Not possible without the protocol |
| `vars()` / `__dict__` | ❌* | Only with `#[pyclass(dict)]`, but empty — fields aren't in it |
| `.pyi` stub generation | 🚧 ~70% | `experimental-inspect` feature, `pyo3-introspection` crate |

## What Your Project Does About It

### The `rlm-derive` Crate Workarounds

Your `#[rlm_type]` attribute macro (in `crates/rlm-derive/`) is essentially a PyO3 dataclass polyfill:

1. **Auto-applies `#[pyclass]`** — so users don't write it manually
2. **Generates `#[getter]` for each field** — with smart strategy (returns `&str` for `String`, copies for primitives, clones otherwise)
3. **Generates `__repr__`** — custom BamlValue-based repr with truncation
4. **Generates `__baml__`** — a `dict`-returning method that does what `dataclasses.asdict()` would
5. **Generates `__len__`/`__iter__`/`__getitem__`** — for list-like collection fields
6. **No `__init__` or `__eq__`** — Python code can't construct or compare these objects

### The `py_bridge.rs` Workarounds

In `py_bridge.rs`, the code that converts Python objects to Rust values has a **priority chain** of dict-extraction strategies:

```rust
// Try model_dump() (Pydantic v2)
// Try dict() (Pydantic v1 / generic)
// Try _asdict() (namedtuple)
// Try dataclasses.asdict() (real dataclasses)
// Try attr.asdict() (attrs library)
// Try __dict__ (plain classes)
// Give up, return the object
```

This is *because* your own types won't match any of these protocols — they need the `__baml__()` method you generate. But when interacting with user-provided Python types, you need to handle all these flavors.

## Alternatives and Workarounds

### 1. Fake the Protocol (Monkey-Patch `__dataclass_fields__`)

You *could* set `__dataclass_fields__` on your pyclass from Rust:

```rust
#[pymethods]
impl Answer {
    #[classattr]
    fn __dataclass_fields__() -> PyResult<PyObject> {
        // Build a dict of field name -> dataclasses.Field(...)
    }
}
```

**Problem**: This makes `dataclasses.is_dataclass()` return `True`, but `dataclasses.asdict()` will still fail because it tries to call the `default`/`default_factory` protocol, access `__init__`, etc. You'd have a half-working lie.

### 2. Generate a Pure Python Wrapper

Generate a Python `@dataclass` that wraps the Rust type:

```python
@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    confidence: float

    @staticmethod
    def _from_rust(obj) -> "Answer":
        return Answer(question=obj.question, ...)
```

**Problem**: Double memory, copy overhead, and the wrapper falls out of sync.

### 3. Define Types in Python, Convert in Rust

Define your data types as pure Python dataclasses and handle conversion in Rust:

```python
@dataclass
class Answer:
    question: str
    answer: str
    confidence: float
```

Then in Rust, extract fields from the Python object. This is what BAML/DSPy-style frameworks often do — the schema lives in Python, the execution in Rust.

### 4. Use `__baml__()` Convention (What You Do)

Your current approach: generate a `__baml__()` method that returns a dict. This is honest — it doesn't pretend to be a dataclass, it just gives you a serializable representation when you need one. The downside is it's non-standard — nobody knows to call `.__baml__()`.

### 5. Wait for PyO3 `#[pyclass(dataclass)]`

This has been discussed but has no implementation timeline. The PyO3 team acknowledges it's wanted but the design is complex.

## Root Cause

**PyO3 was designed to expose Rust code to Python, not to make Rust structs behave like Python data objects.** The `#[pyclass]` model is fundamentally an FFI bridge, not a data protocol implementation. Every "field" is a property descriptor backed by a Rust function call, not a Python-native data slot. The dataclass protocol requires deep Python-level metadata (`__dataclass_fields__`, `__dataclass_params__`, `__annotations__` + `__dict__` storage) that doesn't exist in the PyO3 object model.

This isn't PyO3 lying — it's two type systems with fundamentally different assumptions about what a "field" is.

## Recommendations (Updated for 0.28.2)

### The 0.28 combo gets you most of the way

With PyO3 0.28, you can now write:
```rust
#[pyclass(frozen, get_all, eq, hash, str, new = "from_fields")]
```

This gives you construction, attribute access, equality, hashing, and string representation — the core behavior of `@dataclass(frozen=True)`. The remaining gaps are:

1. **`__repr__`** — `str` gives you `__str__`, but Python convention is `__repr__` for the unambiguous representation. You still need `#[pymethods]` for this (or wait for `#[pyclass(repr)]`).

2. **`dataclasses.asdict()`** — Will never work. Keep your `__baml__()` approach, but also consider adding a standard `_asdict()` method (namedtuple convention) that returns a dict.

3. **`dataclasses.is_dataclass()`** — Will always be `False`. If you need this, you could fake it with `#[classattr] fn __dataclass_fields__()` but it's a rabbit hole.

4. **`vars()`/`__dict__`** — Fields are descriptors, not dict entries. `#[pyclass(dict)]` gives an empty dict.

5. **Type stubs** — Use `experimental-inspect` + `pyo3-introspection` to generate `.pyi` files so mypy/pyright see the fields.

### For your `rlm-derive` crate specifically

Your `#[rlm_type]` currently generates:
- Per-field `#[getter]` (smart return types)
- `__repr__` (BamlValue-based)
- `__baml__()` → dict
- `__len__`/`__iter__`/`__getitem__` for collections

**Consider upgrading to use the 0.28 builtins:**
- Replace per-field getter generation with `#[pyclass(get_all)]` — less code to maintain
- Add `new = "from_fields"` so Python code can construct these objects
- Add `eq` + `hash` + `frozen` for free equality/hashing
- Keep your custom `__repr__` (it's better than what `repr` would give)
- Keep `__baml__()` for dict serialization
- Consider adding `_asdict()` as an alias for `__baml__()` for broader compat

### If you truly need full dataclass compat

You could generate `__dataclass_fields__`, `__dataclass_params__`, and `__match_args__` from Rust via `#[classattr]`. Combined with `new = "from_fields"` (which gives you a working constructor), `dataclasses.asdict()` *might* work if you carefully construct `Field` objects. But this is fragile — CPython's `dataclasses.asdict()` walks `__dataclass_fields__` and calls `copy.deepcopy` on values, which may not work with PyO3's property descriptors. Test thoroughly before committing.
