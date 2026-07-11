"""Master plan Sec 15.6/15.14: the teacher must only ever see what the agent
has actually observed. This is the test the master plan's own repo layout
names explicitly ("asserts no hidden-state leakage") -- it is the single
highest-stakes test in llm_teacher/, since a failure here means privileged
simulator-only information (true sensitive-host value, undiscovered hosts)
could leak into training via the teacher's labels.

Uses a real NASimEmuEnv rollout (not hand-built byte vectors) so the test
exercises the actual HostVector layout rather than a maintainer's guess at it.
"""
import json
import os
import random

import numpy as np
import pytest

from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector
from llm_teacher.state_summarizer import summarize_state, known_addrs

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)

ALLOWED_HOST_KEYS = {"addr", "reachable", "compromised", "access", "os", "services", "processes",
                     "ids_level", "ids_threshold"}
FORBIDDEN_KEY_SUBSTRINGS = ("value", "firewall", "true_", "hidden")


@pytest.fixture
def partial_rollout():
    """A real env, stepped a few times with random actions so some but not
    all hosts end up discovered -- the actual condition this module must
    handle correctly."""
    random.seed(7)
    np.random.seed(7)
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=30, random_init=True,
                       observation_format="graph_v2", seed=7)
    env.reset()
    for _ in range(10):
        addr = tuple(random.choice(list(env.host_index)))
        action_id = random.randrange(len(env.action_list))
        _s, _r, d, _info = env.step((addr, action_id))
        if d:
            break
    return env


def test_only_discovered_hosts_appear(partial_rollout):
    env = partial_rollout
    summary = summarize_state(env.s_raw, ep_t=10, ep_limit=30)

    discovered_addrs = set()
    for row in env.s_raw[:-1]:
        hv = HostVector(np.asarray(row, dtype=np.float32))
        if hv.address != (0, 0) and bool(hv.discovered):
            discovered_addrs.add(f"{int(hv.address[0])}.{int(hv.address[1])}")

    summary_addrs = {h["addr"] for h in summary["hosts"]}
    assert summary_addrs == discovered_addrs
    # sanity: this scenario should actually produce a partial-discovery state,
    # otherwise the test isn't exercising the exclusion logic at all
    assert 0 < len(discovered_addrs) < len(env.host_index)


def test_known_addrs_matches_summary_exactly(partial_rollout):
    summary = summarize_state(partial_rollout.s_raw, ep_t=10, ep_limit=30)
    host_addrs, subnet_ids = known_addrs(summary)
    assert host_addrs == {h["addr"] for h in summary["hosts"]}
    assert subnet_ids == {s["id"] for s in summary["subnets"]}


def test_host_records_only_expose_allowed_fields(partial_rollout):
    summary = summarize_state(partial_rollout.s_raw, ep_t=10, ep_limit=30)
    for host in summary["hosts"]:
        assert set(host.keys()) == ALLOWED_HOST_KEYS


def test_no_privileged_field_names_anywhere_in_output(partial_rollout):
    summary = summarize_state(partial_rollout.s_raw, ep_t=10, ep_limit=30)
    blob = json.dumps(summary).lower()
    for forbidden in FORBIDDEN_KEY_SUBSTRINGS:
        assert forbidden not in blob, f"forbidden substring '{forbidden}' leaked into state summary"


def test_empty_rollout_produces_empty_hosts():
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=30, random_init=True,
                       observation_format="graph_v2", seed=11)
    env.reset()
    summary = summarize_state(env.s_raw, ep_t=0, ep_limit=30)
    # at t=0 only the agent's own starting foothold (if any) is discovered --
    # never the full network
    assert len(summary["hosts"]) < len(env.host_index)
