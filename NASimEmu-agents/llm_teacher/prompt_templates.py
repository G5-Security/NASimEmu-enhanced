"""Teacher prompt -- master plan Sec 15.6/15.14. Versioned so every dataset
record can record exactly which prompt text produced it."""
import json

from .goal_ontology import GOAL_NAMES

PROMPT_VERSION = 1

SYSTEM_PROMPT = (
    "You are a planning assistant for an authorized, simulated network "
    "penetration-testing research environment (NASimEmu). You never see raw "
    "network traffic, banners, or file contents -- only a structured summary "
    "of what the acting agent has already discovered. You must choose exactly "
    "one abstract next goal from a fixed, closed list. You never propose, "
    "output, or reference shell commands, exploit code, Metasploit console "
    "commands, or any executable payload -- only one of the eight named goal "
    "types below, plus an optional already-discovered target. Never invent a "
    "host or subnet id that is not present in the state summary you are given."
)

GOAL_DESCRIPTIONS = {
    "DISCOVER_SUBNET": "Reveal new reachable subnet structure",
    "ENUMERATE_HOST": "Learn OS/services/processes of a discovered host",
    "GAIN_INITIAL_ACCESS": "Obtain user or root access on a discovered, reachable host remotely",
    "ESCALATE_PRIVILEGE": "Upgrade an already-held user access to root on the same host",
    "PIVOT": "Move toward/through a newly reachable, deeper subnet",
    "CAPTURE_SENSITIVE_HOST": "Prioritize progress on a host that looks high-value",
    "REDUCE_DETECTION": "Prefer a lower-detection-risk action or target over a riskier one",
    "RECOVER_OR_REPLAN": "Abandon a goal that has repeatedly failed and pick a feasible alternative",
}


def build_user_prompt(summary: dict) -> str:
    goal_list = "\n".join(f"- {g}: {GOAL_DESCRIPTIONS[g]}" for g in GOAL_NAMES)
    return (
        f"Fixed goal ontology (choose exactly one 'goal' value from this list):\n{goal_list}\n\n"
        f"STATE SUMMARY (JSON, only what the agent has already observed):\n"
        f"{json.dumps(summary, separators=(',', ':'))}\n\n"
        "Respond with strict JSON only, matching the required schema: "
        "goal (one of the names above), optional target_subnet/target_host "
        "(only an id that literally appears in the summary above, else null), "
        "horizon (integer steps, 1-8), confidence (0-1), reason_code (short "
        "snake_case string), rationale (one sentence, plain English, metadata "
        "only -- never executable text)."
    )


def build_critique_prompt(summary: dict, primary_answer: dict) -> str:
    """cascade.py's veto/critique escalation prompt (docs/llm_teacher_workstation_upgrade_walkthrough.md
    Sec 5, recommended default over independent re-decide): shown the same
    state summary plus the primary model's already-produced answer, and
    asked to endorse or replace it -- a narrower, cheaper judgment than a
    fresh 8-way classification from scratch."""
    goal_list = "\n".join(f"- {g}: {GOAL_DESCRIPTIONS[g]}" for g in GOAL_NAMES)
    return (
        f"Fixed goal ontology:\n{goal_list}\n\n"
        f"STATE SUMMARY (JSON, only what the agent has already observed):\n"
        f"{json.dumps(summary, separators=(',', ':'))}\n\n"
        f"A first-pass planner proposed this next goal:\n"
        f"{json.dumps(primary_answer, separators=(',', ':'))}\n\n"
        "Is this goal choice defensible given the state summary above -- a "
        "reasonable next step, even if not the one you would have picked "
        "first? Respond with strict JSON only: defensible (boolean -- true "
        "if the proposed goal is reasonable, false only if it is clearly "
        "wrong given the summary, e.g. targets a host in a state that "
        "contradicts the summary, or ignores an obviously more urgent "
        "signal). If defensible is false, also fill in your own replacement "
        "answer using the same fields as the original schema: goal, optional "
        "target_subnet/target_host (only an id that literally appears in the "
        "summary above, else null), horizon (1-8), confidence (0-1), "
        "reason_code (short snake_case string), rationale (one sentence, "
        "plain English, metadata only -- never executable text). If "
        "defensible is true, fill those fields with the same values as the "
        "proposed goal above."
    )
