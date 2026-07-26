"""DSPy adapter backed by DSRS BAML rendering + JSONish parsing via PyO3."""

from __future__ import annotations

import json
import types
from functools import lru_cache
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import format_field_value as original_format_field_value
from dspy.adapters.utils import parse_value
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

_IMPORT_ERROR: Exception | None = None
try:
    from ._dsrs_dspy import parse_response as _rust_parse_response
    from ._dsrs_dspy import render_field_structure as _rust_render_field_structure
except Exception as exc:  # pragma: no cover - import availability is environment-specific.
    _IMPORT_ERROR = exc
    _rust_parse_response = None
    _rust_render_field_structure = None


def _field_description(field_info: Any) -> str:
    extra = field_info.json_schema_extra or {}
    desc = extra.get("desc")
    if isinstance(desc, str):
        # DSPy auto-infers placeholder desc like "${field_name}".
        if desc.startswith("${") and desc.endswith("}"):
            return ""
        return desc
    return field_info.description or ""


def _field_spec(name: str, field_info: Any) -> dict[str, Any]:
    schema = TypeAdapter(field_info.annotation).json_schema()
    return {
        "name": name,
        "description": _field_description(field_info),
        "format": (field_info.json_schema_extra or {}).get("format"),
        "schema": schema,
    }


def _allows_none(annotation: Any) -> bool:
    return annotation is type(None) or type(None) in get_args(annotation)


def _restore_pydantic_defaults(value: Any, annotation: Any) -> Any:
    """Map BAML's null-for-defaultable-field representation to Pydantic defaults.

    BAML renders a non-required model property as ``T or null``. Pydantic may
    instead define it as a non-nullable ``T`` with a default/default_factory.
    A null at that boundary means "use the declared default", not a literal
    value to validate against ``T``.
    """
    if value is None:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list and isinstance(value, list) and args:
        return [_restore_pydantic_defaults(item, args[0]) for item in value]
    if origin is dict and isinstance(value, dict) and len(args) == 2:
        return {
            key: _restore_pydantic_defaults(item, args[1]) for key, item in value.items()
        }
    if origin in (types.UnionType, Union):
        non_null = [choice for choice in args if choice is not type(None)]
        if len(non_null) == 1:
            return _restore_pydantic_defaults(value, non_null[0])

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if not isinstance(value, dict):
            return value
        restored = dict(value)
        for field_name, field in annotation.model_fields.items():
            if field_name not in restored:
                continue
            field_value = restored[field_name]
            if field_value is None and not field.is_required() and not _allows_none(
                field.annotation
            ):
                restored[field_name] = field.get_default(call_default_factory=True)
            else:
                restored[field_name] = _restore_pydantic_defaults(
                    field_value, field.annotation
                )
        return restored
    return value


@lru_cache(maxsize=256)
def _signature_spec_json(signature: type[Signature]) -> str:
    spec = {
        "input_fields": [_field_spec(name, info) for name, info in signature.input_fields.items()],
        "output_fields": [_field_spec(name, info) for name, info in signature.output_fields.items()],
        "instruction": signature.instructions,
    }
    return json.dumps(spec, ensure_ascii=False, separators=(",", ":"))


class DSRSBAMLAdapter(ChatAdapter):
    """
    DSPy adapter that swaps in DSRS BAML-style rendering and JSONish parsing.

    This adapter is a drop-in `ChatAdapter` replacement for DSPy. It keeps DSPy's
    runtime/callback behavior, but disables ChatAdapter's silent second-model-call
    fallback and delegates output-format rendering and
    response parsing to the Rust extension module (`_dsrs_dspy`).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("use_json_adapter_fallback", False)
        super().__init__(*args, **kwargs)

    def _require_rust_extension(self) -> None:
        if _rust_parse_response is None or _rust_render_field_structure is None:
            detail = repr(_IMPORT_ERROR) if _IMPORT_ERROR else "unknown import error"
            raise RuntimeError(
                "_dsrs_dspy extension is unavailable. Build/install the PyO3 module first. "
                f"Import error: {detail}"
            )

    def format_field_structure(self, signature: type[Signature]) -> str:
        self._require_rust_extension()
        spec_json = _signature_spec_json(signature)

        try:
            return _rust_render_field_structure(spec_json)
        except Exception as exc:  # pragma: no cover - depends on LM/schema shape.
            raise AdapterParseError(
                adapter_name="DSRSBAMLAdapter",
                signature=signature,
                lm_response="",
                message=f"Failed to render BAML field structure: {exc}",
            ) from exc

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]
        for key, field_info in signature.input_fields.items():
            if key not in inputs:
                continue

            value = inputs.get(key)
            if isinstance(value, BaseModel):
                formatted_value = value.model_dump_json(indent=2, by_alias=True)
            else:
                formatted_value = original_format_field_value(field_info=field_info, value=value)

            messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(message for message in messages if message).strip()

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        self._require_rust_extension()
        spec_json = _signature_spec_json(signature)

        try:
            parsed_json = _rust_parse_response(spec_json, completion, True)
            parsed_fields = json.loads(parsed_json)
        except Exception as exc:
            raise AdapterParseError(
                adapter_name="DSRSBAMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Rust JSONish parse failed: {exc}",
            ) from exc

        coerced: dict[str, Any] = {}
        for field_name, field in signature.output_fields.items():
            if field_name not in parsed_fields:
                raise AdapterParseError(
                    adapter_name="DSRSBAMLAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed_fields,
                    message=f"Missing parsed output field `{field_name}`",
                )

            try:
                value = _restore_pydantic_defaults(
                    parsed_fields[field_name], field.annotation
                )
                coerced[field_name] = parse_value(value, field.annotation)
            except Exception as exc:
                raise AdapterParseError(
                    adapter_name="DSRSBAMLAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed_fields,
                    message=(
                        f"Failed to coerce parsed output field `{field_name}` "
                        f"to annotation `{field.annotation}`: {exc}"
                    ),
                ) from exc

        if coerced.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="DSRSBAMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=coerced,
            )

        return coerced
