# dsrs-dspy (dspy-py)

A drop-in [DSPy](https://github.com/stanfordnlp/dspy) adapter backed by DSRs:
BAML-style schema rendering and JSONish response parsing, exposed to Python
through a PyO3 extension.

Compared to DSPy's stock `ChatAdapter`, this adapter:

- renders output-field schemas in BAML's compact, LLM-friendly format
  (including nested pydantic models, enums, unions, `$defs`/`$ref`), and
- parses completions with JSONish, which tolerates the sins real models
  commit: trailing commas, Python literals (`True`/`False`/`None`),
  single quotes, and whitespace-mangled `[[ ## field ## ]]` markers.

There is no silent fallback to a second model call: parse failures surface as
`AdapterParseError`.

## Install

Not published to PyPI. Install from git (requires a Rust toolchain; the wheel
builds from source via maturin):

```bash
uv add "dsrs-dspy @ git+https://github.com/krypticmouse/DSRs@main#subdirectory=crates/dspy-py"
# or
pip install "dsrs-dspy @ git+https://github.com/krypticmouse/DSRs@main#subdirectory=crates/dspy-py"
```

## Use

```python
import dspy
from dsrs_dspy import DSRSBAMLAdapter

dspy.configure(lm=dspy.LM("openai/gpt-5.2"), adapter=DSRSBAMLAdapter())
```

Everything else is ordinary DSPy — signatures, modules, optimizers.

## Python API surface

The compiled module is `dsrs_dspy._dsrs_dspy`:

- `render_field_structure(spec_json: str) -> str`
  — compiles DSPy/pydantic field schemas into BAML `TypeIR` and returns the
  BAML-style field structure text.
- `parse_response(spec_json: str, completion: str, is_done: bool = True) -> str`
  — parses `[[ ## field ## ]]` sections with JSONish and returns a JSON string
  of parsed output fields.

`DSRSBAMLAdapter` (pure Python, in `dsrs_dspy/adapter.py`) subclasses DSPy's
`ChatAdapter` and overrides `format_field_structure()` and `parse()` with the
functions above, keeping DSPy's own field coercion via
`dspy.adapters.utils.parse_value`.

### Spec shape (`spec_json`)

```json
{
  "input_fields": [
    {"name": "question", "description": "", "format": null, "schema": {"type": "string"}}
  ],
  "output_fields": [
    {"name": "answer", "description": "", "format": null, "schema": {"type": "string"}}
  ],
  "instruction": "Given ..."
}
```

`schema` is pydantic/JSON-Schema-like and may include `$defs` + local `$ref`.

Supported schema features: primitives, objects (`properties`/`required`),
maps (`additionalProperties`), arrays (`items`/`prefixItems`), unions
(`anyOf`/`oneOf`/`allOf`/`type: [..]`), enums, `const`, and local
`#/$defs/...` / `#/definitions/...` references.

## Tests

Rust unit + property tests (schema fuzzing, JSONish noise, description
propagation, and a matrix driven by real `pydantic.TypeAdapter(...).json_schema()`
output — that last one needs a Python with pydantic importable and is skipped
otherwise):

```bash
cargo test -p dspy-py
```

Python-side adapter tests against a real DSPy install:

```bash
uv venv .venv && VIRTUAL_ENV=$PWD/.venv uv pip install . dspy pytest
.venv/bin/pytest tests/python/
```
