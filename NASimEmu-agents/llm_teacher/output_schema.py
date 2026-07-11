"""Teacher output contract -- master plan Sec 15.7. Versioned so a stored
dataset record can always be traced back to the exact schema it was
validated against."""
from .goal_ontology import GOAL_NAMES

OUTPUT_SCHEMA_VERSION = 1

REQUIRED_FIELDS = ["goal", "horizon", "confidence", "reason_code", "rationale"]
OPTIONAL_FIELDS = ["target_subnet", "target_host"]

MIN_HORIZON, MAX_HORIZON = 1, 8

# Passed as Ollama's `format` (JSON-schema-constrained decoding) -- this only
# guarantees shape (types, enum membership, required keys). It does NOT
# guarantee the semantic checks in validator.py (no hallucinated target, no
# structurally-incompatible goal/target pairing) -- those need the actual
# summary and are checked after generation.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "enum": list(GOAL_NAMES)},
        "target_subnet": {"type": ["string", "null"]},
        "target_host": {"type": ["string", "null"]},
        "horizon": {"type": "integer"},
        "confidence": {"type": "number"},
        "reason_code": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": REQUIRED_FIELDS,
}

# cascade.py's veto/critique escalation call (docs/llm_teacher_workstation_upgrade_walkthrough.md
# Sec 5): the escalation model either endorses the primary's answer
# (defensible=true, the rest of the fields ignored) or provides a full
# replacement (defensible=false, validated exactly like a normal teacher
# answer). Same required fields as JSON_SCHEMA plus "defensible".
CRITIQUE_REQUIRED_FIELDS = ["defensible"] + REQUIRED_FIELDS
CRITIQUE_JSON_SCHEMA = {
    "type": "object",
    "properties": dict(JSON_SCHEMA["properties"], defensible={"type": "boolean"}),
    "required": CRITIQUE_REQUIRED_FIELDS,
}
