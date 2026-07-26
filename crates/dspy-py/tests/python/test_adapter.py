"""End-to-end tests: DSRSBAMLAdapter render + parse against a real DSPy install.

No network and no LM calls — only formatting and parsing.
"""

import dspy
import pytest
from pydantic import BaseModel, Field

from dsrs_dspy import DSRSBAMLAdapter


class Address(BaseModel):
    street: str
    city: str = "Unknown"


class Patient(BaseModel):
    name: str
    age: int | None
    address: Address
    tags: list[str] = Field(default_factory=list)


class Extract(dspy.Signature):
    """Extract patient information from a clinical note."""

    note: str = dspy.InputField(desc="Clinical note")
    patient: Patient = dspy.OutputField(desc="Extracted patient object")
    confidence: float = dspy.OutputField()


@pytest.fixture()
def adapter() -> DSRSBAMLAdapter:
    return DSRSBAMLAdapter()


def test_render_field_structure(adapter: DSRSBAMLAdapter) -> None:
    rendered = adapter.format_field_structure(Extract)

    assert "[[ ## note ## ]]" in rendered
    assert "[[ ## patient ## ]]" in rendered
    assert "[[ ## confidence ## ]]" in rendered
    # Nested model fields appear in the BAML schema block.
    assert "street" in rendered
    assert "city" in rendered
    assert rendered.rstrip().endswith("[[ ## completed ## ]]")


def test_parse_tolerates_llm_noise(adapter: DSRSBAMLAdapter) -> None:
    # Python literals, trailing comma, and a missing space in the marker.
    completion = (
        "[[ ## patient ##]]\n"
        "{'name': 'Ada', 'age': None,"
        " 'address': {'street': 'Main St', 'city': 'Springfield'}, 'tags': ['a', 'b'],}\n\n"
        "[[ ## confidence ## ]]\n0.95\n\n[[ ## completed ## ]]\n"
    )

    parsed = adapter.parse(Extract, completion)

    patient = parsed["patient"]
    assert isinstance(patient, Patient)
    assert patient.name == "Ada"
    assert patient.age is None
    assert patient.address.street == "Main St"
    assert patient.address.city == "Springfield"
    assert patient.tags == ["a", "b"]
    assert parsed["confidence"] == 0.95


def test_parse_restores_pydantic_defaults(adapter: DSRSBAMLAdapter) -> None:
    # BAML renders defaultable fields as nullable; a null there means
    # "use the pydantic default", not a literal None.
    completion = (
        "[[ ## patient ## ]]\n"
        '{"name": "Ada", "age": 3, "address": {"street": "Main", "city": null}, "tags": null}\n\n'
        "[[ ## confidence ## ]]\n1.0\n\n[[ ## completed ## ]]\n"
    )

    parsed = adapter.parse(Extract, completion)

    assert parsed["patient"].address.city == "Unknown"
    assert parsed["patient"].tags == []


def test_parse_missing_field_raises(adapter: DSRSBAMLAdapter) -> None:
    from dspy.utils.exceptions import AdapterParseError

    with pytest.raises(AdapterParseError):
        adapter.parse(Extract, "[[ ## completed ## ]]")


def test_format_produces_chat_messages(adapter: DSRSBAMLAdapter) -> None:
    messages = adapter.format(Extract, demos=[], inputs={"note": "Ada, Main St"})

    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assert any("[[ ## note ## ]]" in content for content in user_contents)
