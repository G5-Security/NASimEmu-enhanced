"""
Reject/repair the teacher's output -- master plan Sec 15.7 + 15.14.

Rejected records are returned with `valid=False` and a `reject_reason`, never
silently dropped (Sec 15.7: "Rejected outputs are retained, not discarded,
for later error analysis") -- dataset_writer.py stores them as-is so
audit_dataset-style analysis stays possible later.
"""
import json

from .goal_ontology import GOAL_NAMES
from .output_schema import REQUIRED_FIELDS, MIN_HORIZON, MAX_HORIZON
from .state_summarizer import known_addrs

# Sec 15.14: "must never output or be permitted to execute shell commands ...
# or any executable code". This is a coarse denylist over the free-text
# fields only (rationale/reason_code) -- the goal itself is already
# constrained to the fixed enum by the JSON schema, so it cannot smuggle
# executable text through that field.
_SHELL_DENYLIST = (
    ";", "|", "`", "$(", "&&", "/bin/", "rm -rf", "exec(", "eval(",
    "os.system", "subprocess", "import os", "<script", "curl ", "wget ",
)

# Sec 15.7: "the target is structurally incompatible with the chosen goal".
# Only host-targeted goals have an unambiguous structural precondition given
# what's in the summary; the rest are left unconstrained (target optional).
_ACCESS_PRECONDITION = {
    "GAIN_INITIAL_ACCESS": lambda host: host["access"] == "none" and host["reachable"],
    "ESCALATE_PRIVILEGE": lambda host: host["access"] == "user",
}


def _find_host(summary, addr):
    for h in summary["hosts"]:
        if h["addr"] == addr:
            return h
    return None


def validate_teacher_output(parsed, summary):
    """
    parsed: dict already produced by json.loads (or None if parsing failed
        upstream -- teacher_client.py handles the parse step itself).
    summary: the exact dict passed into the prompt for this record (Sec 15.7:
        "the named target host or subnet was not present in the input
        summary").
    Returns (is_valid: bool, reject_reason: str or None).
    """
    if parsed is None or not isinstance(parsed, dict):
        return False, "not_valid_json"

    for field in REQUIRED_FIELDS:
        if field not in parsed:
            return False, f"missing_field:{field}"

    goal = parsed.get("goal")
    if goal not in GOAL_NAMES:
        return False, "goal_not_in_ontology"

    horizon = parsed.get("horizon")
    if not isinstance(horizon, (int, float)) or not (MIN_HORIZON <= horizon <= MAX_HORIZON):
        return False, "horizon_out_of_range"

    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        return False, "confidence_out_of_range"

    for text_field in ("reason_code", "rationale"):
        text = str(parsed.get(text_field, ""))
        low = text.lower()
        if any(tok in low for tok in _SHELL_DENYLIST):
            return False, f"executable_text_in_{text_field}"

    host_addrs, subnet_ids = known_addrs(summary)

    target_host = parsed.get("target_host")
    if target_host is not None:
        if target_host not in host_addrs:
            return False, "hallucinated_target_host"
        host = _find_host(summary, target_host)
        precondition = _ACCESS_PRECONDITION.get(goal)
        if precondition is not None and not precondition(host):
            return False, "target_incompatible_with_goal"

    target_subnet = parsed.get("target_subnet")
    if target_subnet is not None and target_subnet not in subnet_ids:
        return False, "hallucinated_target_subnet"

    return True, None


def parse_and_validate(raw_text, summary):
    """Convenience wrapper: raw model text -> (parsed_or_None, is_valid, reject_reason)."""
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None, False, "not_valid_json"

    is_valid, reject_reason = validate_teacher_output(parsed, summary)
    return parsed, is_valid, reject_reason
