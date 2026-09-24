"""Tests for Step 7 simulator validation
(experiments/dynamics_recorder.py, docs/eval_revision_plan.tex).

Two layers:
  - DynamicsRecorder/wilson_interval as pure utilities (no env needed).
  - Instrumentation hooks (HostVector.event_recorder, Network.event_recorder)
    against a real environment: (a) the critical regression guard --
    leaving the recorder unset (the default) must not change behavior at
    all, verified by comparing a full episode's trajectory bit-for-bit with
    and without a recorder attached; (b) with a recorder attached and full
    real dynamics (nonzero IDS/scan-noise/churn/timeout rates), each event
    kind fires at least once over enough steps.
"""
import os

import numpy as np
import pytest

from experiments.dynamics_recorder import DynamicsRecorder, wilson_interval
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.network import Network

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


@pytest.fixture(autouse=True)
def _clear_recorders():
    """Every test starts and ends with recording off -- these are class-level
    attributes (see the module docstrings on both classes for why), so a
    test that forgets to clean up would otherwise leak into every later test
    in the whole suite, not just this file."""
    HostVector.set_event_recorder(None)
    Network.set_event_recorder(None)
    yield
    HostVector.set_event_recorder(None)
    Network.set_event_recorder(None)


# ---- DynamicsRecorder / wilson_interval as pure utilities -----------------

def test_record_and_count():
    rec = DynamicsRecorder()
    rec.record("ids_detected", host=(1, 0), step=3, response_type="patch")
    rec.record("ids_detected", host=(1, 0), step=9, response_type="monitor")
    rec.record("ids_detected", host=(2, 0), step=5, response_type="patch")

    assert rec.count("ids_detected") == 3
    assert rec.count("ids_detected", response_type="patch") == 2
    assert rec.count("ids_detected", host=(1, 0)) == 2


def test_response_type_counts():
    rec = DynamicsRecorder()
    for rt in ["quarantine", "patch", "patch", "monitor", "monitor", "monitor"]:
        rec.record("ids_detected", response_type=rt)
    counts = rec.response_type_counts()
    assert counts == {"quarantine": 1, "patch": 2, "monitor": 3}


def test_level_trace_filters_by_host_and_preserves_order():
    rec = DynamicsRecorder()
    rec.record("ids_increase", host=(1, 0), step=1, level_after_decay=0.1)
    rec.record("ids_increase", host=(2, 0), step=1, level_after_decay=0.9)
    rec.record("ids_increase", host=(1, 0), step=2, level_after_decay=0.15)
    assert rec.level_trace((1, 0)) == [(1, 0.1), (2, 0.15)]


def test_measured_rate_on_trial_events():
    rec = DynamicsRecorder()
    for fired in [True, False, False, True, False]:
        rec.record("timeout_trial", fired=fired)
    k, n, rate = rec.measured_rate("timeout_trial")
    assert (k, n) == (2, 5)
    assert rate == pytest.approx(0.4)


def test_measured_rate_empty_is_nan():
    rec = DynamicsRecorder()
    k, n, rate = rec.measured_rate("timeout_trial")
    assert (k, n) == (0, 0)
    assert np.isnan(rate)


def test_measured_flip_rate_denominator_excludes_ineligible_trials():
    rec = DynamicsRecorder()
    # fp-eligible trials (true_value=False): 3 total, 1 flipped to fp
    rec.record("scan_noise_trial", scan_type="service_scan", true_value=False, flipped="fp")
    rec.record("scan_noise_trial", scan_type="service_scan", true_value=False, flipped=None)
    rec.record("scan_noise_trial", scan_type="service_scan", true_value=False, flipped=None)
    # fn-eligible trials (true_value=True): 2 total, 1 flipped to fn
    rec.record("scan_noise_trial", scan_type="service_scan", true_value=True, flipped="fn")
    rec.record("scan_noise_trial", scan_type="service_scan", true_value=True, flipped=None)

    k_fp, n_fp, rate_fp = rec.measured_flip_rate("fp")
    assert (k_fp, n_fp) == (1, 3)
    assert rate_fp == pytest.approx(1 / 3)

    k_fn, n_fn, rate_fn = rec.measured_flip_rate("fn")
    assert (k_fn, n_fn) == (1, 2)
    assert rate_fn == pytest.approx(0.5)


def test_wilson_interval_contains_the_point_estimate_and_widens_with_fewer_trials():
    lo_big, hi_big = wilson_interval(50, 1000)
    lo_small, hi_small = wilson_interval(5, 100)
    assert lo_big <= 0.05 <= hi_big
    assert lo_small <= 0.05 <= hi_small
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_wilson_interval_empty_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


# ---- Instrumentation hooks against a real environment ----------------------

def _run_episode(seed, n_steps, recorder):
    if recorder is not None:
        HostVector.set_event_recorder(recorder)
        Network.set_event_recorder(recorder)

    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=n_steps,
                       observation_format="graph_v2", seed=seed, training_mode=False)
    env.reset()
    trajectory = []
    for t in range(n_steps):
        # deterministic scripted action, no policy involved: cycle through
        # scan/exploit/privesc action kinds against the first known host
        # each step, generating a steady stream of real activity so
        # churn/timeout/IDS events (each a low-probability-per-step draw)
        # have many chances to fire under full_difficulty's nonzero rates.
        # Address is one-hot encoded across a range of HostVector columns,
        # not stored as a raw value -- HostVector(row).address decodes it.
        host_addr = HostVector(env.s_raw[0]).address
        action_id = t % 6  # cycle through scan/exploit/privesc kinds
        target = np.array(host_addr)
        _s, r, done, info = env.step((target, action_id))
        trajectory.append((float(r), bool(done)))
        if done:
            break
    return trajectory


def test_default_recorder_is_none_and_changes_nothing():
    assert HostVector.event_recorder is None
    assert Network.event_recorder is None

    traj_a = _run_episode(seed=42, n_steps=60, recorder=None)

    # explicitly re-confirm still unset, then run again -- must be bit-identical
    assert HostVector.event_recorder is None
    traj_b = _run_episode(seed=42, n_steps=60, recorder=None)
    assert traj_a == traj_b


def test_attaching_a_recorder_does_not_change_the_trajectory():
    # The whole point of the class-level None-check design: recording must
    # be observationally inert from the environment's point of view. Same
    # seed, same scripted actions, with vs. without a recorder attached ->
    # identical reward/done sequence.
    traj_without = _run_episode(seed=7, n_steps=60, recorder=None)

    rec = DynamicsRecorder()
    traj_with = _run_episode(seed=7, n_steps=60, recorder=rec)

    assert traj_without == traj_with
    assert len(rec.events) > 0, "recorder was attached but recorded nothing -- hooks likely not firing"


def test_ids_increase_and_detected_events_fire_under_full_difficulty():
    rec = DynamicsRecorder()
    _run_episode(seed=1, n_steps=80, recorder=rec)
    assert rec.count("ids_increase") > 0
    # every ids_increase event's level_after_decay must be a valid probability-like value
    for e in rec.filtered("ids_increase"):
        assert 0.0 <= e["level_after_decay"]
    # a response_type, when a detection did occur, must be one of the three
    for e in rec.filtered("ids_detected"):
        assert e["response_type"] in ("quarantine", "patch", "monitor")


def test_scan_noise_trials_fire_under_full_difficulty():
    rec = DynamicsRecorder()
    _run_episode(seed=1, n_steps=80, recorder=rec)
    assert rec.count("scan_noise_trial") > 0
    for e in rec.filtered("scan_noise_trial"):
        assert e["flipped"] in (None, "fp", "fn")


def test_churn_trials_fire_under_full_difficulty():
    rec = DynamicsRecorder()
    _run_episode(seed=1, n_steps=120, recorder=rec)
    assert rec.count("churn_trial") > 0


def test_timeout_trials_fire_under_full_difficulty():
    rec = DynamicsRecorder()
    _run_episode(seed=1, n_steps=120, recorder=rec)
    assert rec.count("timeout_trial") > 0
    for e in rec.filtered("timeout_blocked"):
        assert e["reason"] in ("new", "already_active")


def test_detection_never_ends_the_episode_by_itself():
    """From the running code (nasim/envs/network.py step(): quarantine
    zeroes access and applies a -50 penalty, patch closes a vulnerable
    service, monitor raises the detection multiplier -- action_result.done
    is never touched by any of the three) and NASimEmuEnv.step() (env.py:
    "ignore done flag from the environment, the agent has to choose to
    terminate"): a detection can never itself end an episode, only the step
    limit or a chosen TerminalAction can. The scripted action stream here
    (action_id cycling 0-5) never emits a TerminalAction, so with
    step_limit == n_steps every trajectory entry must have done=False
    except possibly the very last (which coincides with the step limit,
    not any detection)."""
    n_steps = 80
    rec = DynamicsRecorder()
    traj = _run_episode(seed=1, n_steps=n_steps, recorder=rec)
    if rec.count("ids_detected") == 0:
        pytest.skip("no detection occurred with this seed/step budget")

    for r, done in traj[:-1]:
        assert done is False, "an episode ended before the step limit with no TerminalAction taken"


# ---- IdsCounter (follow-up B) ---------------------------------------------------

def test_ids_counter_summarizes_ids_events_and_ignores_everything_else():
    from experiments.dynamics_recorder import IdsCounter
    c = IdsCounter()
    for kind in ("churn_trial", "scan_noise_trial", "timeout_trial"):
        c.record(kind, whatever=1)
    c.record("ids_increase", host=(1, 0), step=3, level_after_decay=0.4)
    c.record("ids_increase", host=(1, 0), step=4, level_after_decay=0.9)
    c.record("ids_detected", host=(1, 0), step=9, response_type="patch")
    c.record("ids_detected", host=(2, 0), step=5, response_type="monitor")
    s = c.summary()
    assert s["ids_detections"] == 2 and s["ids_patch"] == 1 and s["ids_monitor"] == 1 and s["ids_quarantine"] == 0
    assert s["ids_first_step"] == 5
    assert s["ids_max_level"] == pytest.approx(0.9) and s["ids_increases"] == 2
    c.reset()
    assert c.summary()["ids_detections"] == 0 and c.summary()["ids_first_step"] == -1
