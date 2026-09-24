"""Follow-up E: scan noise and the agent's observation.

Two behaviours are pinned here, deliberately together:

1. DEFAULT (flag off): scan false-positive/negative noise is computed and
   recorded, but never reaches what the agent observes. `HostVector.observe()`
   copies the true host bits, and nothing reads the noisy `ActionResult`
   fields. This is a characterization test of the simulator as it has always
   behaved, found while analysing the Experiment 4 no-scan-noise null (0 of 500
   episodes changed). It is not endorsed behaviour; a future fix that makes it
   fail should update this test on purpose.
2. FLAG ON (`HostVector.set_scan_noise_observed(True)`): the noisy scan result
   is what the agent sees, and with zero noise nothing changes at all.
"""
import os

import numpy as np
import pytest

from experiments.dynamics_recorder import DynamicsRecorder
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


@pytest.fixture(autouse=True)
def _reset_class_state():
    HostVector.set_scan_noise_observed(False)
    HostVector.set_event_recorder(None)
    yield
    HostVector.set_scan_noise_observed(False)
    HostVector.set_event_recorder(None)


def _run(fn, fp, observed, steps=30):
    """Scripted scans (action ids 0, 1, 2 = service/OS/process scan) of the first
    known host, under the given scan-noise rates; returns the agent-visible state
    and how many scan bits were reported wrongly."""
    HostVector.set_scan_noise_observed(observed)
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=steps + 5, observation_format="graph_v2",
                       seed=1_000_000, training_mode=False)
    env.reset()
    HostVector.set_scan_noise({t: {"false_negative_rate": fn, "false_positive_rate": fp}
                               for t in ("service_scan", "os_scan", "process_scan")})
    rec = DynamicsRecorder()
    HostVector.set_event_recorder(rec)
    for t in range(steps):
        env.step((np.array(HostVector(env.s_raw[0]).address), t % 3))
    flips = rec.count("scan_noise_trial", flipped="fn") + rec.count("scan_noise_trial", flipped="fp")
    return env.s_raw.copy(), flips


def test_flag_defaults_to_off():
    assert HostVector.scan_noise_observed is False


def test_default_scan_noise_is_computed_but_never_reaches_the_agent():
    quiet, flips_quiet = _run(0.0, 0.0, observed=False)
    loud, flips_loud = _run(1.0, 1.0, observed=False)
    assert flips_quiet == 0
    assert flips_loud > 0, "the noise did not fire, so this test proves nothing"
    assert np.array_equal(quiet, loud)


def test_flag_on_makes_scan_noise_reach_the_agent():
    quiet, _ = _run(0.0, 0.0, observed=True)
    loud, flips = _run(1.0, 1.0, observed=True)
    assert flips > 0
    assert not np.array_equal(quiet, loud)


def test_flag_on_with_zero_noise_is_bit_identical_to_flag_off():
    off, _ = _run(0.0, 0.0, observed=False)
    on, _ = _run(0.0, 0.0, observed=True)
    assert np.array_equal(off, on)
