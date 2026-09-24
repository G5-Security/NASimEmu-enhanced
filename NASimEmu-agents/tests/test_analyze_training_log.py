"""Tests for experiments/analyze_training_log.py (follow-up H1) on a synthetic log
with known stage behaviour and one planted collapse."""
import json
import os

import pytest

from experiments import analyze_training_log as atl

GOALS = ["DISCOVER_SUBNET", "ENUMERATE_HOST", "GAIN_INITIAL_ACCESS", "ESCALATE_PRIVILEGE",
         "PIVOT", "CAPTURE_SENSITIVE_HOST", "REDUCE_DETECTION", "RECOVER_OR_REPLAN"]
STAGES = [("baseline", 0.0, 0.3), ("medium_difficulty", 0.3, 0.6), ("full_difficulty", 0.6, 1.0)]


def _row(i, n, collapsed=False):
    stage = 0 if i < 0.3 * n else (1 if i < 0.6 * n else 2)
    hist = {g: 1000 for g in GOALS}
    if stage == 1:
        hist["ESCALATE_PRIVILEGE"] = 9000  # a distinctive stage-1 signature
    return {"goal_hist": hist, "manager_entropy": 2.0 + 0.01 * stage,
            "eval_tst": {"reward_avg": 0.1 if stage == 0 else 0.5, "captured_avg": 0.2 if collapsed else (10.0 if stage == 0 else 30.0), "eplen_avg": 400.0},
            "eval_trn": {"reward_avg": 0.05, "captured_avg": 8.0, "eplen_avg": 400.0}}


@pytest.fixture
def log(tmp_path):
    n = 100
    path = tmp_path / "run.json"
    path.write_text("\n".join(json.dumps(_row(i, n, collapsed=(i == 75))) for i in range(n)) + "\n")
    return str(path)


def test_windows_split_at_the_curriculum_boundaries(log):
    rows = atl.load_log(log)
    assert atl.stage_windows(rows, STAGES) == [("baseline", 0, 30), ("medium_difficulty", 30, 60), ("full_difficulty", 60, 100)]


def test_window_summary_reports_goal_shares_that_sum_to_one_and_the_planted_stage_signature(log):
    rows = atl.load_log(log)
    s0 = atl.summarize_window(rows, 0, 30)
    s1 = atl.summarize_window(rows, 30, 60)
    assert sum(s0["goal_share"].values()) == pytest.approx(1.0)
    assert s0["goal_share"]["ESCALATE_PRIVILEGE"] == pytest.approx(1 / 8)
    assert s1["goal_share"]["ESCALATE_PRIVILEGE"] == pytest.approx(9 / 16)
    assert s1["eval_tst_reward_per_step"] == pytest.approx(0.5)


def test_collapse_detection_finds_only_the_planted_epoch(log):
    rows = atl.load_log(log)
    collapses, med = atl.collapsed_epochs(rows, start=40)
    assert [c["epoch"] for c in collapses] == [75]
    assert med == pytest.approx(30.0)


def test_plateau_spread_is_a_within_run_sd(log):
    rows = atl.load_log(log)
    spread = atl.plateau_spread(rows, 40)
    assert spread["eval_tst captured"]["min"] == pytest.approx(0.2)
    assert spread["eval_tst captured"]["sd"] > 0
    assert spread["eval_trn reward/step"]["sd"] == pytest.approx(0.0)


def test_build_writes_summary_plots_and_the_seed_variance_caveat(log, tmp_path):
    out = str(tmp_path / "out")
    atl.build(log, out, STAGES)
    for name in ("training_log_summary.md", "training_log_summary.json", "learning_curves.png", "goal_usage.png", "manager_entropy.png"):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    md = open(os.path.join(out, "training_log_summary.md")).read()
    assert "not** variability across training seeds" in md
    assert "| 75 |" in md
