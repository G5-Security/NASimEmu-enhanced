"""docs/llm_teacher_workstation_upgrade_walkthrough.md Sec 5. should_escalate()
is pure logic (no network) and gets full coverage here; label_one_state_cascade's
plumbing gets one real end-to-end smoke test against whatever teacher model is
actually available locally (same model in both cascade slots -- this project's
laptop only has qwen2.5:3b-instruct, not the workstation-target
mistral-small3.2:24b/qwen3.5:9b; that comparison is explicitly a workstation-time
question, not something to fake here)."""
import socket

import pytest

from llm_teacher.cascade import should_escalate, label_one_state_cascade
from llm_teacher.teacher_client import DEFAULT_MODEL


def _ollama_reachable():
    try:
        with socket.create_connection(("localhost", 11434), timeout=1):
            return True
    except OSError:
        return False


SUMMARY = {
    "ontology_version": 1, "step": 3, "remaining_frac": 0.9,
    "subnets": [{"id": "1", "discovered_hosts": 1, "compromised_hosts": 0}],
    "hosts": [{"addr": "1.0", "reachable": True, "compromised": False, "access": "none",
               "os": ["linux"], "services": ["ssh"], "processes": [], "ids_level": 0.0, "ids_threshold": 0.0}],
    "recent_actions": [], "active_goal": None, "active_goal_remaining_horizon": None,
    "available_goals": ["DISCOVER_SUBNET", "ENUMERATE_HOST", "GAIN_INITIAL_ACCESS",
                         "ESCALATE_PRIVILEGE", "PIVOT", "CAPTURE_SENSITIVE_HOST",
                         "REDUCE_DETECTION", "RECOVER_OR_REPLAN"],
}


def _record(goal="ENUMERATE_HOST", target_host=None, confidence=0.9, valid=True):
    return {
        "valid": valid,
        "parsed_output": {
            "goal": goal, "target_subnet": None, "target_host": target_host,
            "horizon": 4, "confidence": confidence, "reason_code": "x", "rationale": "x",
        } if valid else None,
    }


def test_invalid_primary_record_always_escalates():
    should, reason = should_escalate(_record(valid=False), SUMMARY)
    assert should
    assert reason == "invalid_or_missing_output"


def test_low_confidence_escalates():
    should, reason = should_escalate(_record(confidence=0.3), SUMMARY)
    assert should
    assert reason == "low_confidence"


def test_high_confidence_does_not_escalate():
    should, reason = should_escalate(_record(goal="ENUMERATE_HOST", confidence=0.95), SUMMARY)
    assert not should
    assert reason is None


def test_gain_initial_access_already_unsatisfiable_escalates():
    summary = dict(SUMMARY, hosts=[dict(SUMMARY["hosts"][0], access="root")])  # no unclaimed host left
    should, reason = should_escalate(_record(goal="GAIN_INITIAL_ACCESS", target_host=None), summary)
    assert should
    assert reason == "goal_already_unsatisfiable"


def test_gain_initial_access_with_eligible_host_does_not_escalate_on_that_trigger():
    should, reason = should_escalate(_record(goal="GAIN_INITIAL_ACCESS", target_host=None, confidence=0.95), SUMMARY)
    assert not should  # SUMMARY's one host has access=="none" and reachable -- still eligible


def test_repeated_failure_not_recognized_escalates():
    summary = dict(SUMMARY, recent_actions=[
        {"action": "Exploit", "target": "1.0", "success": False},
        {"action": "Exploit", "target": "1.0", "success": False},
        {"action": "Exploit", "target": "1.0", "success": False},
    ])
    should, reason = should_escalate(_record(goal="ENUMERATE_HOST", confidence=0.95), summary)
    assert should
    assert reason == "repeated_failure_same_target_not_recognized"


def test_repeated_failure_recognized_as_recover_does_not_escalate_on_that_trigger():
    summary = dict(SUMMARY, recent_actions=[
        {"action": "Exploit", "target": "1.0", "success": False},
        {"action": "Exploit", "target": "1.0", "success": False},
        {"action": "Exploit", "target": "1.0", "success": False},
    ])
    should, reason = should_escalate(_record(goal="RECOVER_OR_REPLAN", confidence=0.95), summary)
    assert not should


def test_three_failures_on_different_targets_is_not_a_trigger():
    summary = dict(SUMMARY, recent_actions=[
        {"action": "Exploit", "target": "1.0", "success": False},
        {"action": "Exploit", "target": "1.1", "success": False},
        {"action": "Exploit", "target": "1.2", "success": False},
    ])
    should, reason = should_escalate(_record(confidence=0.95), summary)
    assert not should


@pytest.mark.skipif(not _ollama_reachable(), reason="no local Ollama server reachable")
def test_cascade_end_to_end_smoke():
    """Not testing model quality (that's a workstation-time question) --
    testing that the primary->should_escalate->critique->fallback plumbing
    runs without exceptions and returns a well-shaped record either way."""
    record = label_one_state_cascade(
        SUMMARY, primary_model=DEFAULT_MODEL, escalation_model=DEFAULT_MODEL,
        low_conf_threshold=0.99,  # force escalation on every call regardless of primary's confidence
    )
    assert record["decided_by"] in ("primary", "escalation")
    assert record["escalate_reason"] == "low_confidence"
    assert "critique_model" in record  # confirms the escalation path actually ran
    assert record["parsed_output"] is not None or not record["valid"]
