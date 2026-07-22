"""Master plan Ch.16 condition 6. The heuristic teacher must always produce
validator-passing output against real state summaries -- it's meant to be a
fair alternative backend, not a strawman that fails validation more often
than the LLM."""
import random

import numpy as np
import pytest

from nasimemu.env import NASimEmuEnv
from llm_teacher.state_summarizer import summarize_state
from llm_teacher.heuristic_teacher import label_one_state, _decide
from llm_teacher.goal_ontology import GOAL_NAMES
import os

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


def test_escalate_privilege_chosen_for_user_access_host():
    summary = {"hosts": [{"addr": "1.0", "access": "user", "reachable": True, "os": [], "services": [], "processes": [],
                          "ids_level": 0.0, "ids_threshold": 0.0, "compromised": True}],
               "recent_actions": []}
    goal, target, _ = _decide(summary)
    assert goal == "ESCALATE_PRIVILEGE"
    assert target == "1.0"


def test_gain_initial_access_chosen_for_enumerated_unowned_host():
    summary = {"hosts": [{"addr": "1.0", "access": "none", "reachable": True, "os": ["linux"], "services": [], "processes": [],
                          "ids_level": 0.0, "ids_threshold": 0.0, "compromised": False}],
               "recent_actions": []}
    goal, target, _ = _decide(summary)
    assert goal == "GAIN_INITIAL_ACCESS"
    assert target == "1.0"


def test_enumerate_host_chosen_for_unscanned_reachable_host():
    summary = {"hosts": [{"addr": "1.0", "access": "none", "reachable": True, "os": [], "services": [], "processes": [],
                          "ids_level": 0.0, "ids_threshold": 0.0, "compromised": False}],
               "recent_actions": []}
    goal, target, _ = _decide(summary)
    assert goal == "ENUMERATE_HOST"
    assert target == "1.0"


def test_reduce_detection_fires_on_high_observable_ids_level_overriding_escalate():
    # A user-access host would normally trigger ESCALATE_PRIVILEGE, but once its
    # observable detection level is high the stealth emergency takes priority.
    # ids_threshold stays 0 (hidden from observation) on purpose -- the rule must
    # fire on ids_level alone.
    summary = {"hosts": [{"addr": "1.0", "access": "user", "reachable": True, "os": ["linux"], "services": [],
                          "processes": [], "ids_level": 0.75, "ids_threshold": 0.0, "compromised": True}],
               "recent_actions": []}
    goal, target, _ = _decide(summary)
    assert goal == "REDUCE_DETECTION"
    assert target is None


def test_reduce_detection_overrides_capture_for_root_host_under_detection():
    summary = {"hosts": [{"addr": "2.3", "access": "root", "reachable": True, "os": ["linux"], "services": [],
                          "processes": [], "ids_level": 0.6, "ids_threshold": 0.0, "compromised": True}],
               "recent_actions": []}
    goal, _target, _ = _decide(summary)
    assert goal == "REDUCE_DETECTION"


def test_reduce_detection_does_not_fire_when_detection_low():
    # Low observable detection -> normal priority resumes (escalate the user host).
    summary = {"hosts": [{"addr": "1.0", "access": "user", "reachable": True, "os": ["linux"], "services": [],
                          "processes": [], "ids_level": 0.2, "ids_threshold": 0.0, "compromised": True}],
               "recent_actions": []}
    goal, target, _ = _decide(summary)
    assert goal == "ESCALATE_PRIVILEGE"
    assert target == "1.0"


def test_default_discover_subnet_on_empty_state():
    summary = {"hosts": [], "subnets": [], "recent_actions": []}
    goal, target, _ = _decide(summary)
    assert goal == "DISCOVER_SUBNET"
    assert target is None


def test_recover_or_replan_on_repeated_same_target_failure():
    summary = {"hosts": [], "recent_actions": [
        {"action": "Exploit", "target": "1.0", "success": False},
        {"action": "Exploit", "target": "1.0", "success": False},
        {"action": "Exploit", "target": "1.0", "success": False},
    ]}
    goal, _target, _ = _decide(summary)
    assert goal == "RECOVER_OR_REPLAN"


def test_output_always_passes_validation_against_real_rollout_states():
    random.seed(3)
    np.random.seed(3)
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=40, random_init=True,
                       observation_format="graph_v2", seed=3)
    env.reset()
    recent_actions = []
    for t in range(15):
        addr = tuple(random.choice(list(env.host_index)))
        action_id = random.randrange(len(env.action_list))
        _s, r, d, _info = env.step((addr, action_id))
        recent_actions.append({"action": "x", "target": f"{addr[0]}.{addr[1]}", "success": bool(r > 0)})
        summary = summarize_state(env.s_raw, t, 40, recent_actions=recent_actions)

        record = label_one_state(summary)
        assert record["valid"], f"heuristic teacher produced an invalid record: {record['reject_reason']} on {summary}"
        assert record["parsed_output"]["goal"] in GOAL_NAMES
        if d:
            break


def test_same_call_signature_as_teacher_client():
    """label_states.py calls label_one_state(summary, model=...) positionally
    the same way regardless of backend -- confirm the heuristic accepts the
    same kwargs without error."""
    summary = {"hosts": [], "subnets": [], "recent_actions": []}
    record = label_one_state(summary, model="ignored", host="ignored", timeout_s=999)
    assert record["valid"]
