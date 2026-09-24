"""Tests for experiments/analyze_goals.py (follow-up A trace analysis) on synthetic traces
where the truth is known: a goal that depends on the level has high MI, an independent one
does not, and the permutation null / cluster bootstrap behave."""
import csv
import os

import numpy as np
import pytest

from experiments import analyze_goals as ag


def _trace(dependent, n_episodes=40, steps=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(n_episodes):
        for t in range(steps):
            level = float(rng.choice([0.0, 0.1, 0.4, 0.6, 0.9]))
            goal = ag.level_bin(level) if dependent else int(rng.integers(0, 8))
            rows.append({"seed": float(ep), "step": float(t), "goal_idx": float(goal), "goal_entropy": 1.0,
                         "goal_maxprob": 0.5, "obs_max_level": level, "true_max_level": level,
                         "target_level": level, "detections_so_far": 0.0})
    return rows


def test_level_bins_cover_the_range_and_are_ordered():
    assert [ag.level_bin(x) for x in (0.0, 0.1, 0.25, 0.3, 0.5, 0.6, 0.75, 2.0)] == [0, 1, 1, 2, 2, 3, 3, 4]


def test_mutual_information_is_zero_for_constant_goal_and_ln_k_for_a_deterministic_map():
    bins = [0, 1, 2, 3] * 50
    assert ag.mutual_information([5] * len(bins), bins) == pytest.approx(0.0, abs=1e-12)
    assert ag.mutual_information(bins, bins) == pytest.approx(np.log(4))


def test_dependent_goals_have_mi_far_above_the_permutation_null():
    s = ag.summarize(_trace(dependent=True), n_perm=50, n_boot=50)
    assert s["mi"] > 1.0
    assert s["mi"] > 10 * s["mi_null_p95"]
    assert s["mi_ci"][0] > s["mi_null_p95"]


def test_independent_goals_have_mi_close_to_the_null():
    s = ag.summarize(_trace(dependent=False), n_perm=100, n_boot=50)
    assert s["mi"] < 3 * s["mi_null_p95"] + 0.01
    assert abs(s["mi_excess"]) < 0.01


def test_shares_sum_to_one_per_populated_bin():
    s = ag.summarize(_trace(dependent=True), n_perm=10, n_boot=10)
    for v in s["share_by_bin"].values():
        if v["n"]:
            assert sum(v["share"]) == pytest.approx(1.0)


def test_switch_rate_counts_goal_changes_within_episodes_only():
    rows = [{"seed": 1.0, "goal_idx": g} for g in (0, 0, 1, 1)] + [{"seed": 2.0, "goal_idx": g} for g in (3, 3)]
    assert ag.switch_rate(rows) == pytest.approx(1 / 4)  # 4 within-episode transitions, 1 change


def test_write_report_produces_tables_and_a_plot(tmp_path):
    paths = {}
    for name, dep in (("normal", True), ("zeroed", False)):
        p = tmp_path / f"{name}.csv"
        rows = _trace(dependent=dep, n_episodes=10)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        paths[name] = str(p)
    out = str(tmp_path / "out")
    ag.write_report(paths, out, n_perm=10, n_boot=10)
    md = open(os.path.join(out, "trace_analysis.md")).read()
    assert "| normal |" in md and "| zeroed |" in md and "only the excess over that means anything" in md
    assert os.path.getsize(os.path.join(out, "plots", "goal_mix_by_level.png")) > 0


# ---- association controlled for time within the episode -------------------------------

def _time_confounded_trace(n_episodes=60, steps=200, seed=0):
    """Goal depends only on the step; the IDS level also rises with the step. A raw goal-level MI
    is high, but conditional on time it must vanish."""
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(n_episodes):
        for t in range(steps):
            phase = t // 50
            level = [0.0, 0.1, 0.4, 0.9][phase] if rng.random() < 0.85 else float(rng.choice([0.0, 0.1, 0.4, 0.9]))
            rows.append({"seed": float(ep), "step": float(t), "goal_idx": float(phase), "goal_entropy": 1.0,
                         "goal_maxprob": 0.5, "obs_max_level": level, "true_max_level": level,
                         "target_level": level, "detections_so_far": 0.0})
    return rows


def test_a_time_confounded_association_vanishes_once_time_is_controlled():
    s = ag.summarize(_time_confounded_trace(), n_perm=20, n_boot=10)
    assert s["mi"] > 0.5                       # looks like the IDS level drives the goal...
    assert s["mi_goal_time"] > s["mi"] * 0.9   # ...but the goal is a function of time
    assert s["mi_given_time"] < 0.05           # and nothing is left once time is held fixed


def test_a_real_dependence_survives_controlling_for_time():
    s = ag.summarize(_trace(dependent=True), n_perm=20, n_boot=10)
    assert s["mi_given_time"] > 10 * s["mi_given_time_null_p95"]
