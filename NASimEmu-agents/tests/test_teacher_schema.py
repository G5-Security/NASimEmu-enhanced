"""Master plan Sec 15.7/15.13. Guards the teacher output contract's shape and
its consistency with goal_ontology -- output_schema.py's JSON_SCHEMA is passed
straight to Ollama as constrained-decoding `format`, so a drift here would
silently change what the teacher is even allowed to say."""
from llm_teacher.goal_ontology import GOAL_NAMES
from llm_teacher.output_schema import (
    JSON_SCHEMA, REQUIRED_FIELDS, OPTIONAL_FIELDS, MIN_HORIZON, MAX_HORIZON, OUTPUT_SCHEMA_VERSION,
)


def test_required_fields_present_in_json_schema():
    assert set(REQUIRED_FIELDS) <= set(JSON_SCHEMA["properties"].keys())
    assert JSON_SCHEMA["required"] == REQUIRED_FIELDS


def test_optional_fields_are_declared_but_not_required():
    for field in OPTIONAL_FIELDS:
        assert field in JSON_SCHEMA["properties"]
        assert field not in JSON_SCHEMA["required"]


def test_goal_enum_matches_ontology_exactly():
    assert JSON_SCHEMA["properties"]["goal"]["enum"] == list(GOAL_NAMES)


def test_horizon_bounds_are_sane():
    assert isinstance(MIN_HORIZON, int) and isinstance(MAX_HORIZON, int)
    assert 1 <= MIN_HORIZON < MAX_HORIZON


def test_confidence_and_rationale_fields_typed_correctly():
    props = JSON_SCHEMA["properties"]
    assert props["confidence"]["type"] == "number"
    assert props["rationale"]["type"] == "string"
    assert props["reason_code"]["type"] == "string"


def test_target_fields_are_nullable():
    props = JSON_SCHEMA["properties"]
    for field in ("target_subnet", "target_host"):
        assert "null" in props[field]["type"]


def test_schema_version_is_a_positive_int():
    assert isinstance(OUTPUT_SCHEMA_VERSION, int)
    assert OUTPUT_SCHEMA_VERSION >= 1
