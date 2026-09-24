"""Tests for Step 8's statistics engine (experiments/analyze.py,
docs/eval_revision_plan.tex).

Two layers, same discipline as Steps 1-7:
  - Pure-utility tests (bootstrap_ci, median_iqr, group_by, paired_difference,
    table rendering) against hand-built rows, no env needed.
  - An end-to-end test against a real CSV written by
    experiments/eval_harness.py running an *untrained* NASimNetDHRL -- proves
    the whole load -> summarize -> render pipeline works against the actual
    file format Step 1 produces, without needing a trained checkpoint.
"""
import os

import numpy as np
import pytest

from experiments import analyze
from experiments.eval_harness import build_net, run_eval

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


def _row(seed, model="NASimNetDHRL", scenario="s", condition="full_dynamics",
         episode_return=10.0, episode_len=50, captured=2,
         reward_at_400=10.0, captured_at_400=2,
         terminal_action=False, hit_step_limit=True):
    return {
        "model": model, "scenario": scenario, "condition": condition,
        "episode_idx": seed, "seed": seed,
        "episode_return": episode_return, "episode_len": episode_len, "captured": captured,
        "reward_at_100": episode_return, "captured_at_100": captured,
        "reward_at_200": episode_return, "captured_at_200": captured,
        "reward_at_400": reward_at_400, "captured_at_400": captured_at_400,
        "terminal_action": terminal_action, "hit_step_limit": hit_step_limit,
    }


# ---- pure utilities ---------------------------------------------------------

def test_bootstrap_ci_single_value_collapses_to_that_value():
    mean, lo, hi = analyze.bootstrap_ci([3.5])
    assert mean == lo == hi == 3.5


def test_bootstrap_ci_empty_is_nan():
    mean, lo, hi = analyze.bootstrap_ci([])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_contains_the_true_mean_for_constant_values():
    mean, lo, hi = analyze.bootstrap_ci([5.0] * 20, n_resamples=500, seed=0)
    assert mean == pytest.approx(5.0)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)


def test_bootstrap_ci_widens_with_more_variance():
    rng = np.random.default_rng(1)
    tight = rng.normal(0, 0.1, size=200)
    wide = rng.normal(0, 5.0, size=200)
    _, lo_t, hi_t = analyze.bootstrap_ci(tight, n_resamples=2000, seed=0)
    _, lo_w, hi_w = analyze.bootstrap_ci(wide, n_resamples=2000, seed=0)
    assert (hi_w - lo_w) > (hi_t - lo_t)


def test_bootstrap_ci_is_reproducible_with_a_fixed_seed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]
    a = analyze.bootstrap_ci(values, n_resamples=1000, seed=42)
    b = analyze.bootstrap_ci(values, n_resamples=1000, seed=42)
    assert a == b


def test_median_iqr_basic():
    median, q1, q3 = analyze.median_iqr([1, 2, 3, 4, 5, 6, 7, 8])
    assert median == pytest.approx(4.5)
    assert q1 < median < q3


def test_median_iqr_empty_is_nan():
    median, q1, q3 = analyze.median_iqr([])
    assert np.isnan(median) and np.isnan(q1) and np.isnan(q3)


def test_group_by_groups_on_the_given_keys():
    rows = [_row(1, condition="a"), _row(2, condition="a"), _row(3, condition="b")]
    groups = analyze.group_by(rows, keys=("condition",))
    assert set(groups.keys()) == {("a",), ("b",)}
    assert len(groups[("a",)]) == 2
    assert len(groups[("b",)]) == 1


def test_summarize_group_reports_all_requested_metrics():
    rows = [_row(i, episode_return=float(i), episode_len=10) for i in range(1, 6)]
    summary = analyze.summarize_group(rows, {"episode_return": analyze.METRICS["episode_return"]}, seed=0)
    assert summary["episode_return"]["n"] == 5
    assert summary["episode_return"]["mean"] == pytest.approx(3.0)
    assert summary["episode_return"]["median"] == pytest.approx(3.0)


def test_reward_per_step_divides_return_by_length():
    row = _row(1, episode_return=40.0, episode_len=20)
    assert analyze.METRICS["reward_per_step"](row) == pytest.approx(2.0)


def test_reward_per_step_zero_length_does_not_divide_by_zero():
    row = _row(1, episode_return=0.0, episode_len=0)
    assert analyze.METRICS["reward_per_step"](row) == 0.0


def test_stop_rate_and_step_limit_rate_are_booleans_as_floats():
    row = _row(1, terminal_action=True, hit_step_limit=False)
    assert analyze.METRICS["stop_rate"](row) == 1.0
    assert analyze.METRICS["step_limit_rate"](row) == 0.0


# ---- paired differences ------------------------------------------------------

def test_paired_difference_only_uses_shared_seeds():
    rows_a = [_row(1, episode_return=10.0), _row(2, episode_return=20.0), _row(3, episode_return=30.0)]
    rows_b = [_row(2, episode_return=15.0), _row(3, episode_return=25.0), _row(4, episode_return=99.0)]
    diff = analyze.paired_difference(rows_a, rows_b, "episode_return", n_resamples=500, seed=0)
    assert diff["n_pairs"] == 2  # only seeds 2 and 3 are shared
    # (20-15) and (30-25) -> both +5
    assert diff["mean_diff"] == pytest.approx(5.0)
    assert diff["ci_lo"] == pytest.approx(5.0)
    assert diff["ci_hi"] == pytest.approx(5.0)


def test_paired_difference_zero_when_groups_are_identical_on_shared_seeds():
    rows_a = [_row(s, episode_return=float(s) * 2) for s in range(1, 6)]
    rows_b = [_row(s, episode_return=float(s) * 2) for s in range(1, 6)]
    diff = analyze.paired_difference(rows_a, rows_b, "episode_return", n_resamples=500, seed=0)
    assert diff["n_pairs"] == 5
    assert diff["mean_diff"] == pytest.approx(0.0)
    assert diff["ci_lo"] == pytest.approx(0.0)
    assert diff["ci_hi"] == pytest.approx(0.0)


def test_paired_difference_no_shared_seeds_is_empty():
    rows_a = [_row(1)]
    rows_b = [_row(2)]
    diff = analyze.paired_difference(rows_a, rows_b, "episode_return")
    assert diff["n_pairs"] == 0
    assert np.isnan(diff["mean_diff"])


# ---- table rendering ----------------------------------------------------------

def test_build_summary_table_and_markdown_rendering():
    rows = [_row(i, condition="a", episode_return=float(i)) for i in range(1, 4)]
    rows += [_row(i, condition="b", episode_return=float(i) * 10) for i in range(4, 7)]
    table = analyze.build_summary_table(rows, keys=("condition",),
                                         metrics=["episode_return"], n_resamples=200, seed=0)
    assert {e["condition"] for e in table} == {"a", "b"}
    for e in table:
        assert e["n_episodes"] == 3
        assert "episode_return" in e["metrics"]

    md = analyze.to_markdown_table(table, keys=("condition",), metrics=["episode_return"])
    assert "| condition | n | episode_return (95% CI) |" in md
    assert "| a |" in md and "| b |" in md


def test_latex_table_escapes_underscores_and_is_balanced():
    rows = [_row(1, condition="no_ids", scenario="corp_100hosts")]
    table = analyze.build_summary_table(rows, keys=("condition", "scenario"),
                                         metrics=["episode_return"], n_resamples=100, seed=0)
    tex = analyze.to_latex_table(table, keys=("condition", "scenario"), metrics=["episode_return"])
    assert r"no\_ids" in tex
    assert r"corp\_100hosts" in tex
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1


def test_apply_filter_matches_on_stringified_field_values():
    rows = [_row(1, condition="full_dynamics"), _row(2, condition="no_ids")]
    filtered = analyze._apply_filter(rows, ["condition=no_ids"])
    assert len(filtered) == 1
    assert filtered[0]["condition"] == "no_ids"


def test_apply_filter_none_returns_all_rows():
    rows = [_row(1), _row(2)]
    assert analyze._apply_filter(rows, None) == rows


# ---- end-to-end against a real eval_harness.py CSV, untrained net -----------

def test_load_rows_and_summarize_a_real_harness_csv(tmp_path):
    net = build_net("NASimNetDHRL", SCENARIO, step_limit=30, checkpoint_path=None)
    csv_path = str(tmp_path / "eval.csv")
    run_eval(net, "NASimNetDHRL", SCENARIO, "full_dynamics", step_limit=30,
              n_episodes=6, csv_path=csv_path, verbose=False)

    rows = analyze.load_rows([csv_path])
    assert len(rows) == 6
    assert all(r["model"] == "NASimNetDHRL" for r in rows)
    assert all(isinstance(r["terminal_action"], bool) for r in rows)

    table = analyze.build_summary_table(rows, n_resamples=200, seed=0)
    assert len(table) == 1
    entry = table[0]
    assert entry["n_episodes"] == 6
    for m in analyze.SUMMARY_TABLE_METRICS:
        assert m in entry["metrics"]
        assert not np.isnan(entry["metrics"][m]["mean"])

    md = analyze.to_markdown_table(table)
    assert "NASimNetDHRL" in md
    tex = analyze.to_latex_table(table)
    assert tex.count(r"\begin{tabular}") == 1


# ---- additions for the detailed report (unpaired diff, forest plot, milestones) ----

def test_unpaired_difference_of_constant_groups_is_exact():
    rows_a = [_row(s, episode_return=10.0) for s in range(1, 6)]
    rows_b = [_row(s + 100, episode_return=4.0) for s in range(1, 9)]
    d = analyze.unpaired_difference(rows_a, rows_b, "episode_return", n_resamples=300, seed=0)
    assert (d["n_a"], d["n_b"]) == (5, 8)
    assert d["mean_diff"] == pytest.approx(6.0)
    assert d["ci_lo"] == pytest.approx(6.0) and d["ci_hi"] == pytest.approx(6.0)


def test_unpaired_difference_ci_brackets_a_noisy_difference():
    rng = np.random.default_rng(3)
    rows_a = [_row(i, episode_return=float(v)) for i, v in enumerate(rng.normal(50, 5, 200))]
    rows_b = [_row(i + 1000, episode_return=float(v)) for i, v in enumerate(rng.normal(40, 5, 200))]
    d = analyze.unpaired_difference(rows_a, rows_b, "episode_return", n_resamples=2000, seed=0)
    assert d["ci_lo"] < d["mean_diff"] < d["ci_hi"]
    assert d["ci_lo"] > 5.0  # a true gap of 10 with n=200 is well clear of 5


def test_unpaired_difference_empty_group_is_nan():
    d = analyze.unpaired_difference([_row(1)], [], "episode_return")
    assert np.isnan(d["mean_diff"]) and d["n_b"] == 0


def test_milestone_metrics_read_the_right_columns():
    row = _row(1)
    row.update({"reward_at_100": 1.5, "reward_at_200": 2.5, "captured_at_100": 3, "captured_at_200": 4})
    assert analyze.METRICS["return_at_100"](row) == 1.5
    assert analyze.METRICS["return_at_200"](row) == 2.5
    assert analyze.METRICS["captured_at_100"](row) == 3.0
    assert analyze.METRICS["captured_at_200"](row) == 4.0


def test_forest_plot_writes_a_png(tmp_path):
    out = str(tmp_path / "forest.png")
    analyze.plot_paired_forest([("a", -1.0, -2.0, 0.5), ("b", 3.0, 2.0, 4.0)], "captured", out)
    assert os.path.getsize(out) > 0


# ---- optional IDS / scenario-size columns (follow-ups B and F) ------------------------

def _raw_csv_row(**extra):
    base = {"model": "m", "scenario": "s", "condition": "c", "episode_idx": "0", "seed": "1", "episode_return": "10.0",
            "episode_len": "400", "captured": "30", "reward_at_100": "5", "captured_at_100": "20", "reward_at_200": "6",
            "captured_at_200": "25", "reward_at_400": "10", "captured_at_400": "30", "terminal_action": "False",
            "hit_step_limit": "True"}
    base.update({k: str(v) for k, v in extra.items()})
    return base


def test_optional_columns_are_parsed_only_when_present():
    plain = analyze._parse_row(_raw_csv_row())
    assert "ids_detections" not in plain and "n_sensitive" not in plain
    full = analyze._parse_row(_raw_csv_row(ids_detections=7, ids_quarantine=1, ids_patch=2, ids_monitor=4,
                                           ids_first_step=12, ids_max_level=1.5, ids_increases=90, n_sensitive=34, n_hosts=78))
    assert full["ids_detections"] == 7.0 and full["n_sensitive"] == 34.0


def test_ids_penalty_uses_the_response_penalties_and_detected_any_is_binary():
    row = analyze._parse_row(_raw_csv_row(ids_detections=7, ids_quarantine=1, ids_patch=2, ids_monitor=4,
                                          ids_first_step=12, ids_max_level=1.5, ids_increases=90, n_sensitive=34, n_hosts=78))
    assert analyze.METRICS["ids_penalty"](row) == pytest.approx((50 + 2 * 20 + 4 * 5) / 10.0)  # env divides rewards by 10
    assert analyze.METRICS["detected_any"](row) == 1.0
    none = analyze._parse_row(_raw_csv_row(ids_detections=0, ids_quarantine=0, ids_patch=0, ids_monitor=0,
                                           ids_first_step=-1, ids_max_level=0.2, ids_increases=3, n_sensitive=34, n_hosts=78))
    assert analyze.METRICS["detected_any"](none) == 0.0 and analyze.METRICS["ids_penalty"](none) == 0.0


def test_captured_fraction_normalizes_by_the_networks_own_sensitive_host_count():
    row = analyze._parse_row(_raw_csv_row(captured=30, n_sensitive=40, n_hosts=80,
                                          ids_detections=0, ids_quarantine=0, ids_patch=0, ids_monitor=0,
                                          ids_first_step=-1, ids_max_level=0, ids_increases=0))
    assert analyze.METRICS["captured_fraction"](row) == pytest.approx(0.75)
