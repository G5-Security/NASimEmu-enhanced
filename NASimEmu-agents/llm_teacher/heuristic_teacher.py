"""
Rule-based heuristic teacher -- master plan Ch.16 condition 6: "a rule-based
heuristic teacher in place of the LLM -- if simple hand-written heuristics
match the LLM's effect, the LLM is not doing anything a much cheaper
mechanism could not."

Implements the exact same call signature and return shape as
teacher_client.label_one_state, so label_states.py / evaluate_llm_selector.py
can swap teacher backends (--teacher_backend llm|heuristic) without a
separate code path. Deterministic, no LLM, no network call: a fixed-priority
decision tree over the same sanitized summary dict the LLM teacher sees, so
it is bound by exactly the same hidden-state constraints (Sec 15.6/15.14) --
it has no access to anything this module doesn't also read out of `summary`.

Every decision is still routed through the real validator.py rather than
assumed correct, so a bug in this decision tree fails the same way a bad LLM
answer would (rejected, not silently accepted into the dataset).
"""
import hashlib
import json
import time

from .validator import validate_teacher_output

MODEL_NAME = "heuristic-v1"
DEFAULT_HORIZON = 4


def _summary_hash(summary):
    blob = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _decide(summary):
    """Returns (goal, target_host_addr_or_None, reason_code). Priority order
    mirrors a plausible greedy pentesting strategy, cheapest signal first."""
    hosts = summary["hosts"]

    for h in hosts:
        if h["access"] == "user":
            return "ESCALATE_PRIVILEGE", h["addr"], "user_access_not_yet_root"

    for h in hosts:
        if h["access"] == "none" and h["reachable"] and (h["os"] or h["services"] or h["processes"]):
            return "GAIN_INITIAL_ACCESS", h["addr"], "enumerated_host_not_yet_accessed"

    for h in hosts:
        if h["reachable"] and not (h["os"] or h["services"] or h["processes"]):
            return "ENUMERATE_HOST", h["addr"], "reachable_host_unscanned"

    for h in hosts:
        if h["access"] == "root":
            return "CAPTURE_SENSITIVE_HOST", h["addr"], "root_access_host_present"

    for h in hosts:
        if h["ids_threshold"] > 0 and h["access"] != "none" and h["ids_level"] >= 0.8 * h["ids_threshold"]:
            return "REDUCE_DETECTION", None, "ids_level_near_threshold"

    recent = summary.get("recent_actions") or []
    last_three = recent[-3:]
    same_target = len(last_three) == 3 and len({a["target"] for a in last_three}) == 1
    all_failed = len(last_three) == 3 and all(not a["success"] for a in last_three)
    if same_target and all_failed:
        return "RECOVER_OR_REPLAN", None, "three_consecutive_failures_same_target"

    return "DISCOVER_SUBNET", None, "default_exploration"


def label_one_state(summary, model=MODEL_NAME, host=None, timeout_s=None):
    """Drop-in replacement for teacher_client.label_one_state. `host`/
    `timeout_s` accepted and ignored, for call-signature compatibility with
    the real (network-backed) teacher."""
    t0 = time.time()
    goal, target_addr, reason_code = _decide(summary)

    parsed = {
        "goal": goal,
        "target_subnet": None,
        "target_host": target_addr,
        "horizon": DEFAULT_HORIZON,
        "confidence": 1.0,
        "reason_code": reason_code,
        "rationale": f"heuristic rule fired: {reason_code}",
    }

    is_valid, reject_reason = validate_teacher_output(parsed, summary)

    return {
        "model": MODEL_NAME,
        "prompt_version": "heuristic-v1",
        "output_schema_version": "heuristic-v1",
        "input_hash": _summary_hash(summary),
        "state_summary": summary,
        "raw_output": json.dumps(parsed),
        "parsed_output": parsed,  # kept even if invalid -- Sec 15.7, matches teacher_client.py
        "valid": bool(is_valid),
        "reject_reason": reject_reason,
        "latency_s": time.time() - t0,
        "generated_at": time.time(),
    }
