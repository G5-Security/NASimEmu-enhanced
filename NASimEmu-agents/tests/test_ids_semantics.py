"""Characterization tests for the IDS semantics documented in
docs/eval_followup_2026-09-24/C_ids_semantics (reviewer question: "what happens
when the sampled threshold is exceeded? how do alerts accumulate?").

Each test pins one behaviour of HostVector.update_detection() /
_handle_detection() as it is, so the write-up cannot drift from the code.
"""
import os

import pytest

from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


class _Action:
    """The Action interface update_detection() branches on."""
    def __init__(self, kind):
        self.kind = kind

    def is_scan(self): return self.kind in ("subnet_scan", "service_scan", "os_scan", "process_scan")
    def is_subnet_scan(self): return self.kind == "subnet_scan"
    def is_service_scan(self): return self.kind == "service_scan"
    def is_os_scan(self): return self.kind == "os_scan"
    def is_process_scan(self): return self.kind == "process_scan"
    def is_exploit(self): return self.kind == "exploit"
    def is_privilege_escalation(self): return self.kind == "privesc"


def _host(seed=1):
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=50, observation_format="graph_v2",
                      seed=seed, training_mode=False)
    env.reset()
    state = env.env.current_state
    hosts = [state.get_host(a) for a in state.host_num_map]
    return next(h for h in hosts if any(h.services.values()))  # a host with at least one running service


def _increase(host, kind, step, success=True):
    """The pre-decay increase an action added (recovered from the level change)."""
    before = host.detection_level
    host.update_detection(_Action(kind), success=success, current_step=step)
    return host.detection_level / HostVector.ids_config["detection_decay"] - before


def test_decay_applies_when_the_host_itself_is_updated_not_per_environment_step():
    h = _host()
    h.detection_level = 1.0
    h.detection_threshold = 99.0
    h.update_detection(_Action("noop"), success=True, current_step=1)
    assert h.detection_level == pytest.approx(1.0 * HostVector.ids_config["detection_decay"])
    other = _host(seed=2)
    other.detection_level = 1.0
    h.update_detection(_Action("noop"), success=True, current_step=2)  # activity on `h` must not decay `other`
    assert other.detection_level == 1.0


def test_two_scans_within_five_steps_double_the_second_increase():
    h = _host()
    h.detection_threshold = 99.0
    first = _increase(h, "service_scan", step=10)
    second = _increase(h, "service_scan", step=12)   # 2 steps later: "rapid scanning"
    later = _increase(h, "service_scan", step=30)    # 18 steps later: normal again
    assert second == pytest.approx(2 * first, rel=0.05)
    assert later == pytest.approx(first, rel=0.05)


def test_failed_exploit_increase_compounds_with_the_failure_count():
    h = _host()
    h.detection_threshold = 99.0
    incs = [_increase(h, "exploit", step=i, success=False) for i in range(4)]
    assert all(b > a for a, b in zip(incs, incs[1:]))
    base = HostVector.ids_config["detection_increase"]["exploit_failed"]
    mult = HostVector.ids_config["failed_exploit_multiplier"]
    assert incs[0] == pytest.approx(base * (1 + 0.3 * 1 * mult), rel=0.05)


def test_a_detection_does_not_reset_the_level_so_later_actions_keep_alerting():
    h = _host()
    h.detection_threshold = 0.05
    statuses = [h.update_detection(_Action("exploit"), success=False, current_step=i)[0] for i in range(6)]
    assert statuses == ["DETECTED"] * 6
    assert h.detection_level > h.detection_threshold


@pytest.mark.parametrize("response,penalty", [("quarantine", -50), ("patch", -20), ("monitor", -5)])
def test_each_response_type_has_its_penalty_and_effect(response, penalty):
    h = _host()
    old = dict(HostVector.ids_config)
    try:
        HostVector.ids_config = {**old, "response_types": {"quarantine": float(response == "quarantine"),
                                                            "patch": float(response == "patch"),
                                                            "monitor": float(response == "monitor")}}
        out = h._handle_detection()
    finally:
        HostVector.ids_config = old
    assert out["type"] == response and out["penalty"] == penalty
    if response == "patch":
        assert 1 <= len(out["patched_services"]) <= 2 and set(out["patched_services"]) <= h.patched_services
    if response == "monitor":
        assert h.detection_multiplier == 2.0
