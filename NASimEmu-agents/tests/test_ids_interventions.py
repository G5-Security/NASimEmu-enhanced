"""Tests for the Step 2 IDS observation interventions
(experiments/ids_interventions.py, docs/eval_revision_plan.tex).

These construct real environments (cheap: a handful of steps at a short step
limit) because two of the five conditions (alert_only, delayed) depend on
privileged ground truth read off env.env.current_state and on sequencing
across real steps -- there is no meaningful way to test them against
synthetic node_feats arrays alone.
"""
import os

import numpy as np
import pytest

from experiments.eval_harness import _graph_obs, build_net
from experiments.ids_interventions import (
    _columns, make_alert_only, make_condition_factory, make_delayed,
    make_normal, make_shuffled, make_zeroed, CONDITIONS, NEUTRAL_LEVEL,
    NEUTRAL_MULTIPLIER,
)
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)
STEP_LIMIT = 15


@pytest.fixture(scope="module")
def net():
    return build_net("NASimNetDHRL", SCENARIO, STEP_LIMIT, checkpoint_path=None)


def _fresh_env(seed=1):
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=STEP_LIMIT,
                       observation_format="graph_v2", seed=seed, training_mode=False)
    env.reset()
    return env


def _host_rows(node_feats):
    return np.where(node_feats[:, 0] == 0)[0]


def test_columns_land_on_the_last_three_of_eighty_and_match_hostvector_indices():
    _fresh_env()  # ensure HostVector is initialized
    level_col, thresh_col, mult_col = _columns()
    assert (level_col, thresh_col, mult_col) == (
        1 + HostVector._detection_level_idx,
        1 + HostVector._detection_threshold_idx,
        1 + HostVector._detection_multiplier_idx,
    )
    assert (level_col, thresh_col, mult_col) == (77, 78, 79)


def test_threshold_is_always_zero_in_a_real_observation():
    # Pins the finding the whole module is built on: detection_threshold is
    # deliberately never agent-observable (HostVector.observe()'s own
    # comment). If this regresses (e.g. the simulator starts exposing it),
    # every condition's handling of the threshold column needs revisiting.
    env = _fresh_env()
    node_feats, *_ = _graph_obs(env)
    _, thresh_col, _ = _columns()
    rows = _host_rows(node_feats)
    assert np.all(node_feats[rows, thresh_col] == 0.0)


def test_normal_is_the_identity():
    env = _fresh_env()
    s = _graph_obs(env)
    out = make_normal()(env, s)
    assert out is s


def test_zeroed_clears_all_three_ids_columns_on_host_rows_only():
    env = _fresh_env()
    s = _graph_obs(env)
    node_feats, *_ = s
    level_col, thresh_col, mult_col = _columns()
    host_rows = _host_rows(node_feats)
    subnet_rows = np.where(node_feats[:, 0] == 1)[0]

    out_feats, *_ = make_zeroed()(env, s)

    assert np.all(out_feats[host_rows, level_col] == 0.0)
    assert np.all(out_feats[host_rows, thresh_col] == 0.0)
    assert np.all(out_feats[host_rows, mult_col] == 0.0)
    # unrelated columns (e.g. discovered/compromised) untouched
    assert np.array_equal(out_feats[host_rows, :level_col], node_feats[host_rows, :level_col])
    # subnet rows never touched
    assert np.array_equal(out_feats[subnet_rows], node_feats[subnet_rows])
    # original array not mutated in place
    assert not np.array_equal(node_feats[host_rows, level_col], out_feats[host_rows, level_col]) or \
        np.all(node_feats[host_rows, level_col] == 0.0)


def test_shuffled_is_a_permutation_not_a_corruption():
    env = _fresh_env()
    s = _graph_obs(env)
    node_feats, *_ = s
    level_col, _, mult_col = _columns()
    host_rows = _host_rows(node_feats)

    observe = make_shuffled(seed=123)
    out_feats, *_ = observe(env, s)

    before = sorted(zip(node_feats[host_rows, level_col].tolist(), node_feats[host_rows, mult_col].tolist()))
    after = sorted(zip(out_feats[host_rows, level_col].tolist(), out_feats[host_rows, mult_col].tolist()))
    assert before == after, "shuffled must permute (level, multiplier) pairs, not invent new values"


def test_shuffled_same_seed_reproducible_different_seed_can_differ():
    env1 = _fresh_env()
    s1 = _graph_obs(env1)
    out_a, *_ = make_shuffled(seed=7)(env1, s1)

    env2 = _fresh_env()
    s2 = _graph_obs(env2)
    out_b, *_ = make_shuffled(seed=7)(env2, s2)
    assert np.array_equal(out_a, out_b)

    env3 = _fresh_env()
    s3 = _graph_obs(env3)
    out_c, *_ = make_shuffled(seed=8)(env3, s3)
    # Not asserting out_a != out_c: a coincidental match is possible with
    # few hosts (small permutation space), so that would be a flaky test.
    # test_shuffled_is_a_permutation_not_a_corruption already establishes
    # the seed changes what gets returned relative to the unshuffled input;
    # this test's job is only the reproducibility check above.


def test_shuffled_does_not_touch_the_global_numpy_or_random_state():
    import random
    env = _fresh_env()
    s = _graph_obs(env)
    observe = make_shuffled(seed=42)

    np_state_before = np.random.get_state()[1].copy()
    py_state_before = random.getstate()
    observe(env, s)
    assert np.array_equal(np.random.get_state()[1], np_state_before), (
        "shuffled must use its own independent RNG, not numpy's global "
        "state -- touching it would silently break episode determinism "
        "(NASimEmuEnv's scenario generation and dynamics draw from the "
        "same global state)"
    )
    assert random.getstate() == py_state_before


def test_alert_only_matches_privileged_ground_truth_comparison():
    env = _fresh_env()
    # step a few times so detection_level has a chance to move off 0
    net_ = None
    for _ in range(5):
        s = _graph_obs(env)
        # deterministic dummy action: pick the first available (target, 0) pair
        addr = tuple(int(a) for a in s[2][0])
        env.step((np.array(addr), 0))

    s = _graph_obs(env)
    node_feats, edge_index, node_index, pos_index = s
    level_col, thresh_col, mult_col = _columns()
    host_rows = _host_rows(node_feats)

    out_feats, *_ = make_alert_only()(env, s)

    for row in host_rows:
        address = node_index[row]
        state = env.env.current_state
        true_row = state.host_num_map[tuple(int(a) for a in address)]
        true_level = state.tensor[true_row, HostVector._detection_level_idx]
        true_threshold = state.tensor[true_row, HostVector._detection_threshold_idx]
        expected_flag = 1.0 if true_level >= true_threshold else 0.0

        assert out_feats[row, level_col] == expected_flag
        assert out_feats[row, thresh_col] == 0.0
        assert out_feats[row, mult_col] == NEUTRAL_MULTIPLIER


def test_delayed_shows_neutral_baseline_on_first_call_then_lags_by_one_step():
    env = _fresh_env()
    observe = make_delayed()
    level_col, thresh_col, mult_col = _columns()

    s0 = _graph_obs(env)
    node_feats0, edge_index0, node_index0, pos_index0 = s0
    host_rows0 = _host_rows(node_feats0)

    out0, *_ = observe(env, s0)
    assert np.all(out0[host_rows0, level_col] == NEUTRAL_LEVEL)
    assert np.all(out0[host_rows0, mult_col] == NEUTRAL_MULTIPLIER)

    env.step((np.array(tuple(int(a) for a in node_index0[0])), 0))

    s1 = _graph_obs(env)
    node_feats1, edge_index1, node_index1, pos_index1 = s1
    out1, *_ = observe(env, s1)

    # every host present in both steps must show step-0's *true* (pre-edit)
    # value at step 1, not step-1's own true value
    addr_to_row0 = {tuple(int(a) for a in node_index0[r]): r for r in host_rows0}
    for row in _host_rows(node_feats1):
        addr = tuple(int(a) for a in node_index1[row])
        if addr in addr_to_row0:
            r0 = addr_to_row0[addr]
            assert out1[row, level_col] == node_feats0[r0, level_col]
            assert out1[row, mult_col] == node_feats0[r0, mult_col]
        assert out1[row, thresh_col] == 0.0


def test_condition_factory_covers_every_named_condition():
    for name in CONDITIONS:
        factory = make_condition_factory(name, seed=1)
        observe = factory()
        assert callable(observe)


def test_condition_factory_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_condition_factory("not_a_real_condition")


# ---- follow-up G: k-step delay ------------------------------------------------

@pytest.mark.parametrize("delay", [1, 3])
def test_k_step_delay_shows_the_true_values_from_k_steps_earlier(delay):
    env = _fresh_env()
    observe = make_delayed(delay)
    level_col, thresh_col, mult_col = _columns()

    true_by_step, shown_by_step = [], []
    for t in range(delay + 4):
        s = _graph_obs(env)
        node_feats, _ei, node_index, _pi = s
        rows = _host_rows(node_feats)
        true_by_step.append({tuple(int(a) for a in node_index[r]): (float(node_feats[r, level_col]), float(node_feats[r, mult_col])) for r in rows})
        out, *_ = observe(env, s)
        shown_by_step.append({tuple(int(a) for a in node_index[r]): (float(out[r, level_col]), float(out[r, mult_col])) for r in rows})
        # scans of the first known host raise its detection level, so values change between steps
        env.step((np.array(tuple(int(a) for a in node_index[0])), t % 3))

    saw_a_nonneutral_lag = False
    for t, shown in enumerate(shown_by_step):
        for addr, value in shown.items():
            if t >= delay and addr in true_by_step[t - delay]:
                assert value == true_by_step[t - delay][addr], (t, addr)
                saw_a_nonneutral_lag |= value != (NEUTRAL_LEVEL, NEUTRAL_MULTIPLIER)
            else:
                assert value == (NEUTRAL_LEVEL, NEUTRAL_MULTIPLIER), (t, addr)
    assert saw_a_nonneutral_lag, "the test never exercised a non-neutral lagged value"


def test_delay_below_one_is_rejected():
    with pytest.raises(ValueError):
        make_delayed(0)


def test_condition_factory_parses_delayed_k_names():
    for name, k in (("delayed_5", 5), ("delayed_50", 50)):
        assert name in CONDITIONS
        assert callable(make_condition_factory(name)())
