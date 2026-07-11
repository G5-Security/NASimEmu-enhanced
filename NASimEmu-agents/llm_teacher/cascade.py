"""
Primary/escalation teacher cascade -- docs/llm_teacher_workstation_upgrade_walkthrough.md
Sec 5. Workstation-scale addition: query a primary model on every state, and
only escalate to a second model on a small set of concrete, checkable
triggers -- not a fixed fraction of calls, and not raw self-reported
confidence alone (small instruct models are poorly calibrated; on this
project's own laptop smoke dataset, qwen2.5:3b's confidence was >=0.9 on
every one of 8 states, so a naive threshold would never fire).

should_escalate()'s design deliberately does NOT use
compute_all_potentials()/argmax-over-potentials as a disagreement signal --
an earlier version of this idea was tried and retracted (see the walkthrough
doc Sec 5): those functions are potential-based-reward-shaping progress
values ("how much progress toward goal g"), not "which goal should be
selected next" scores, and using them that way silently conflates two
different semantics (e.g. RECOVER_OR_REPLAN is hardcoded to phi=0 and could
never win an argmax, exactly the situation it exists to catch).

The escalation call uses the veto/critique design (recommended default over
independent re-decide): the escalation model sees the primary's answer and
either endorses it or replaces it, which is a narrower, cheaper judgment than
a fresh 8-way classification and easier to validate.
"""
import time

from .output_schema import CRITIQUE_JSON_SCHEMA
from .prompt_templates import SYSTEM_PROMPT, build_critique_prompt
from .teacher_client import (
    DEFAULT_HOST, MAX_RETRIES, _call_ollama_chat, label_one_state, summary_hash,
)
from .validator import validate_teacher_output

import json
import urllib.error


def should_escalate(teacher_record, summary, low_conf_threshold=0.6):
    """Returns (should_escalate: bool, reason: str or None). Any one trigger
    fires escalation; each is a cheap, teacher-free signal computable from
    the record/summary already in hand -- no extra LLM call needed just to
    decide whether to make another LLM call."""
    parsed = teacher_record.get("parsed_output")
    if not teacher_record.get("valid") or parsed is None:
        return True, "invalid_or_missing_output"

    goal = parsed["goal"]
    target_host = parsed.get("target_host")

    # NOTE: deliberately not re-checking the GAIN_INITIAL_ACCESS/ESCALATE_PRIVILEGE
    # access precondition against an explicit target_host here --
    # teacher_record["valid"] is already guaranteed True at this point (the
    # branch above returns early otherwise), and validator.py's
    # validate_teacher_output() already ran exactly that check before the
    # record could be marked valid. Re-running it here would be dead code.

    # This path (no explicit target given) is NOT covered by validator.py,
    # unlike the case above -- the primary said "go get initial access"
    # without naming a host, but there may be no eligible host left at all.
    if goal == "GAIN_INITIAL_ACCESS" and target_host is None:
        any_unclaimed = any(h["access"] == "none" and h["reachable"] for h in summary["hosts"])
        if not any_unclaimed:
            return True, "goal_already_unsatisfiable"

    # Sec 15.8's "three consecutive failures" trigger, repurposed here as a
    # disagreement check: narrowed to three failures against the SAME target
    # (ordinary exploration noise otherwise) that the primary did NOT
    # recognize as a reason to replan.
    recent_actions = summary.get("recent_actions") or []
    last_three = recent_actions[-3:]
    same_target = len(last_three) == 3 and len({a["target"] for a in last_three}) == 1
    all_failed = len(last_three) == 3 and all(not a["success"] for a in last_three)
    if same_target and all_failed and goal != "RECOVER_OR_REPLAN":
        return True, "repeated_failure_same_target_not_recognized"

    if parsed.get("confidence", 1.0) < low_conf_threshold:
        return True, "low_confidence"  # secondary/OR condition only, per the calibration note above

    return False, None


def _critique_call(summary, primary_answer, model, host, timeout_s):
    """Thin retry loop specific to the critique response shape (adds
    'defensible' on top of the standard schema) -- can't reuse
    teacher_client._label_with_prompt's retry loop as-is, since that
    validates against the standard REQUIRED_FIELDS only."""
    prompt = build_critique_prompt(summary, primary_answer)
    last_raw, last_parsed, last_thinking = None, None, None
    for _attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_text, thinking = _call_ollama_chat(
                prompt, model, host, timeout_s, system_prompt=SYSTEM_PROMPT, schema=CRITIQUE_JSON_SCHEMA
            )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return None, None, f"api_error:{e}", None
        last_raw, last_thinking = raw_text, thinking
        try:
            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict) or "defensible" not in parsed:
            continue
        last_parsed = parsed
        break
    else:
        return last_raw, None, "not_valid_json_or_missing_defensible", last_thinking
    return last_raw, last_parsed, None, last_thinking


def label_one_state_cascade(summary, primary_model, escalation_model,
                             host=DEFAULT_HOST, timeout_s=120, low_conf_threshold=0.6):
    """
    Primary/escalation cascade entry point -- same return shape as
    teacher_client.label_one_state, plus "decided_by": "primary"|"escalation"
    and "escalate_reason" (None if never escalated) for later analysis of
    what fraction of labels needed escalation.

    Fail-safe by construction: any failure in the escalation path (API
    error, unparsable critique, an escalation replacement that itself fails
    validate_teacher_output) falls back to the primary's own answer rather
    than raising or silently producing a worse label than not escalating at
    all -- a smaller/differently-tuned escalation model being wrong is a
    real, expected outcome (Sec 5's "9B-vs-24B caution": don't let a second
    opinion override the first just because it was consulted second).
    """
    primary_record = label_one_state(summary, model=primary_model, host=host, timeout_s=timeout_s)
    escalate, escalate_reason = should_escalate(primary_record, summary, low_conf_threshold=low_conf_threshold)

    record = dict(primary_record)
    record["decided_by"] = "primary"
    record["escalate_reason"] = escalate_reason if escalate else None

    if not escalate:
        return record

    primary_answer = primary_record.get("parsed_output") or {}
    t0 = time.time()
    raw, critique_parsed, critique_error, critique_thinking = _critique_call(
        summary, primary_answer, escalation_model, host, timeout_s
    )
    critique_latency_s = time.time() - t0

    record["critique_model"] = escalation_model
    record["critique_raw_output"] = raw
    record["critique_thinking"] = critique_thinking
    record["critique_latency_s"] = critique_latency_s

    if critique_error is not None or critique_parsed is None:
        record["critique_error"] = critique_error
        return record  # fall back to primary, decided_by stays "primary"

    if critique_parsed.get("defensible") is True:
        return record  # endorsed -- primary's own already-validated answer stands

    # defensible == False: escalation model provides a full replacement,
    # validated exactly like any other teacher answer -- not trusted just
    # because it came from the escalation slot.
    is_valid, reject_reason = validate_teacher_output(critique_parsed, summary)
    if not is_valid:
        record["critique_error"] = f"escalation_replacement_invalid:{reject_reason}"
        return record  # fall back to primary

    record["decided_by"] = "escalation"
    record["parsed_output"] = critique_parsed
    record["valid"] = True
    record["reject_reason"] = None
    record["model"] = escalation_model
    record["input_hash"] = summary_hash(summary)
    return record
