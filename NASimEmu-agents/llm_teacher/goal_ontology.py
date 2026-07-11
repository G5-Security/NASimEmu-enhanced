"""
Fixed semantic goal ontology (version 1) binding Pen-DHRL's 8 anonymous
goal-bank slots (nasim_net_base_hrl.py: self.goal_bank, num_subgoals=8) to
named types, per docs/llm_integration_plan.pdf Sec 5.2 (Phase 1).

The learned embedding per slot remains exactly as trainable as before -- only
the *index -> name* binding is fixed, so an LLM-authored potential function
can be written against a stable name instead of an opaque integer.
"""

GOAL_ONTOLOGY_VERSION = 1

GOAL_NAMES = [
    "DISCOVER_SUBNET",
    "ENUMERATE_HOST",
    "GAIN_INITIAL_ACCESS",
    "ESCALATE_PRIVILEGE",
    "PIVOT",
    "CAPTURE_SENSITIVE_HOST",
    "REDUCE_DETECTION",
    "RECOVER_OR_REPLAN",
]

GOAL_INDEX = {name: i for i, name in enumerate(GOAL_NAMES)}

assert len(GOAL_NAMES) == 8, "goal ontology must match NASimNetDHRL.num_subgoals (default 8)"
