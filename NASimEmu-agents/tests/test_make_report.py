"""Tests for experiments/make_report.py, the per-experiment report generator.

They build a synthetic results directory with known, hand-set effects (for
example "episode_return is exactly 5 lower than baseline on every seed") and
check that the report says exactly that. Also covers partial data, seed
alignment, generalization labelling from the manifest, and the variant
verification pass/fail logic.
"""
import csv
import json
import os

import numpy as np
import pytest

from experiments import make_report
from experiments.eval_harness import CSV_FIELDS

REAL_VARIANTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios", "variants")

SEEDS_A = list(range(1_000_000, 1_000_010))
SEEDS_B = list(range(1_000_500, 1_000_510))


def _rows(seeds, condition="full_dynamics", scenario="s.yaml", shift=0.0):
    out = []
    for s in seeds:
        ret = 100.0 + (s % 7) + shift
        out.append({
            "model": "m", "scenario": scenario, "condition": condition,
            "episode_idx": s - 1_000_000, "seed": s,
            "episode_return": ret, "episode_len": 400, "captured": 10 + (s % 3),
            "reward_at_100": ret / 4, "captured_at_100": 5,
            "reward_at_200": ret / 2, "captured_at_200": 8,
            "reward_at_400": ret, "captured_at_400": 10 + (s % 3),
            "terminal_action": False, "hit_step_limit": True,
        })
    return out


def _write(results_dir, name, rows):
    with open(os.path.join(results_dir, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _exp7_json(results_dir, variant, rows, n_episodes=50):
    with open(os.path.join(results_dir, f"exp7_verify_{variant}.json"), "w") as f:
        json.dump({"scenario": variant, "n_episodes": n_episodes, "step_limit": 200, "rows": rows}, f)


def _v_row(label, configured, k, n, in_ci=True):
    rate = k / n if n else float("nan")
    return {"label": label, "configured": configured, "measured_k": k, "measured_n": n,
            "measured_rate": rate, "ci_95_low": max(0.0, rate - 0.01) if n else float("nan"),
            "ci_95_high": rate + 0.01 if n else float("nan"), "configured_in_ci": in_ci}


@pytest.fixture
def results(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    _write(str(d), "exp1_base_c0", _rows(SEEDS_A))
    _write(str(d), "exp1_base_c1", _rows(SEEDS_B))
    _write(str(d), "exp1c_fulldyn_variant", _rows(SEEDS_A, scenario="variant.yaml"))
    for c in ("zeroed", "shuffled", "alert_only", "delayed"):
        _write(str(d), f"exp2_{c}", _rows(SEEDS_A, condition=c, shift=-5.0))
    for c in ("no_ids_branch", "no_recurrent_memory", "no_goal_persistence", "no_goal_conditioning", "no_learned_stopping"):
        _write(str(d), f"exp3_{c}", _rows(SEEDS_A, condition=c, shift=2.0))
    for c in ("no_ids", "no_scan_noise", "no_churn", "no_timeouts", "all_off"):
        _write(str(d), f"exp4_{c}", _rows(SEEDS_A, condition=c, shift=10.0))
    for c in ("intensity_off", "intensity_low", "intensity_harder"):
        _write(str(d), f"exp5_{c}", _rows(SEEDS_A, condition=c, shift=1.0))
    for name in ("varA", "varB", "bridge", "test"):
        _write(str(d), f"exp6_{name}_c0", _rows(SEEDS_A, scenario=f"{name}.yaml", shift=-20.0))
    _write(str(d), "exp8_plumb_normal", _rows(SEEDS_A, condition="normal"))
    _write(str(d), "exp8_plumb_full_checkpoint", _rows(SEEDS_A, condition="full_checkpoint"))
    _write(str(d), "exp8_ctrl_zeroed_on_no_ids", _rows(SEEDS_A, condition="zeroed_on_no_ids", shift=10.0))
    _write(str(d), "exp8_ctrl_no_ids_branch_on_no_ids", _rows(SEEDS_A, condition="no_ids_branch_on_no_ids", shift=10.0))
    _write(str(d), "exp8_shuffled_seed1", _rows(SEEDS_A, condition="shuffled_seed1", shift=-3.0))
    _write(str(d), "exp8_shuffled_seed2", _rows(SEEDS_A, condition="shuffled_seed2", shift=-3.0))
    h = _rows(SEEDS_A, condition="horizon_800")
    for r in h:
        r["episode_len"] = 800
        r["episode_return"] = r["reward_at_400"] * 2.0
    _write(str(d), "exp8_horizon800", h)
    _exp7_json(str(d), "no_churn", [_v_row("service_dynamics.churn_probability", 0.0, 0, 5000),
                                     _v_row("scan_noise.service_scan.false_positive_rate", 0.03, 150, 5000)])
    return str(d)


@pytest.fixture
def manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"checkpoints": [{"scenario_roles": {
        "corp_100hosts_dynamic": {"held_out": False},
        "corp_100hosts_dynamic_varA": {"held_out": False},
        "corp_100hosts_dynamic_varB": {"held_out": False},
        "corp_100hosts_dynamic_bridge": {"held_out": True},
        "corp_100hosts_dynamic_test": {"held_out": True}}}]}))
    return str(p)


@pytest.fixture
def trainer_log(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"eval_tst": {"reward_avg": 0.5, "reward_avg_episodes": 200.0,
                                          "captured_avg": 29.0, "eplen_avg": 400.0}}) + "\n")
    return str(p)


def _read(out, *parts):
    with open(os.path.join(out, *parts)) as f:
        return f.read()


def test_report_has_one_folder_per_experiment_with_expected_files(results, manifest, trainer_log, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, manifest_path=manifest, trainer_log=trainer_log,
                             variants_dir=REAL_VARIANTS, n_resamples=100)
    for folder in ("exp1_baseline", "exp2_ids_observation", "exp3_component_interventions",
                   "exp4_mechanism_knockouts", "exp5_intensity_sweep", "exp6_generalization"):
        assert os.path.isfile(os.path.join(out, folder, "results.md")), folder
        assert os.path.isfile(os.path.join(out, folder, "summary_table.tex")), folder
        assert os.path.isfile(os.path.join(out, folder, "summary.json")), folder
        assert any(p.endswith(".png") for p in os.listdir(os.path.join(out, folder, "plots"))), folder
    assert os.path.isfile(os.path.join(out, "exp7_variant_verification", "results.md"))
    readme = _read(out, "README.md")
    assert "exp2_ids_observation" in readme and "Headline" in readme
    assert "one trained checkpoint per architecture" in readme


def test_known_paired_effect_is_reported_exactly(results, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp2_ids_observation", "results.md")
    # every seed's return is exactly 5 lower -> paired diff -5.0 with a zero-width CI, which excludes 0
    assert "-5.0 [-5.0, -5.0]*" in md
    md3 = _read(out, "exp3_component_interventions", "results.md")
    assert "+2.0 [+2.0, +2.0]*" in md3


def test_baseline_rows_are_restricted_to_the_seeds_being_compared(results, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp2_ids_observation", "results.md")
    # the baseline has 20 episodes (two chunks) but exp2 only ran 10 of the same seeds
    assert "Episodes per condition:** 10" in md


def test_missing_experiments_are_reported_as_no_data(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    _write(str(d), "exp1_base_c0", _rows(SEEDS_A))
    out = str(tmp_path / "docs")
    make_report.build_report(str(d), out, n_resamples=100)
    assert "No data yet" in _read(out, "exp2_ids_observation", "results.md")
    assert "No data yet" in _read(out, "exp7_variant_verification", "results.md")
    assert os.path.isfile(os.path.join(out, "exp1_baseline", "results.md"))


def test_exp1_reconciliation_and_identical_pairing_check(results, trainer_log, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, trainer_log=trainer_log, n_resamples=100)
    md = _read(out, "exp1_baseline", "results.md")
    assert "trainer eval_tst" in md and "29.00" in md
    assert "Every episode matches exactly" in md


def test_exp1_pairing_check_flags_a_variant_that_differs(results, tmp_path):
    _write(results, "exp1c_fulldyn_variant", _rows(SEEDS_A, scenario="variant.yaml", shift=1.0))
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp1_baseline", "results.md")
    assert "do NOT produce identical episodes" in md


def test_generalization_is_unpaired_and_labelled_from_the_manifest(results, manifest, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, manifest_path=manifest, n_resamples=100)
    md = _read(out, "exp6_generalization", "results.md")
    assert "bridge (held-out)" in md and "test (held-out)" in md
    assert "varA (trained on)" in md and "varB (trained on)" in md
    assert "unpaired" in md
    assert "10 vs 20" in md or "10 vs 10" in md


def test_intensity_parameter_table_uses_the_real_variant_files(results, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, variants_dir=REAL_VARIANTS, n_resamples=100)
    md = _read(out, "exp5_intensity_sweep", "results.md")
    assert "Full parameter values per intensity level" in md
    assert "intensity_harder" in md and "0.0225" in md  # 1.5x the final timeout probability 0.015
    assert any(p.endswith("_vs_intensity.png") for p in os.listdir(os.path.join(out, "exp5_intensity_sweep", "plots")))


def test_exp7_passes_when_knocked_out_mechanism_has_zero_events(results, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp7_variant_verification", "results.md")
    assert "| no_churn |" in md and "PASS" in md and "FAIL" not in md


def test_exp7_fails_when_a_zero_configured_mechanism_produced_events(results, tmp_path):
    _exp7_json(results, "no_timeouts", [_v_row("network_reliability.timeout_probability", 0.0, 3, 4000, in_ci=False)])
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp7_variant_verification", "results.md")
    assert "FAIL: network_reliability.timeout_probability" in md


def test_json_safe_turns_nan_into_null():
    safe = make_report._json_safe({"a": float("nan"), "b": [np.float64(1.5), np.int64(3)], "c": float("inf")})
    assert safe == {"a": None, "b": [1.5, 3], "c": None}
    json.dumps(safe, allow_nan=False)


def test_experiment_8_subfolders_are_written(results, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    for sub in ("8a_plumbing_controls", "8b_ids_attribution_controls", "8c_shuffle_seed_robustness", "8d_longer_horizon"):
        p = os.path.join(out, "exp8_controls_and_robustness", sub)
        assert os.path.isfile(os.path.join(p, "results.md")), sub
        assert os.path.isfile(os.path.join(p, "summary.json")), sub


def test_plumbing_controls_report_full_identity_and_zero_difference(results, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp8_controls_and_robustness", "8a_plumbing_controls", "results.md")
    assert "10 (100.0%)" in md
    diff_section = md.split("## Differences vs baseline")[1].split("## Plots")[0]
    assert "+0.0 [+0.0, +0.0]" in diff_section
    assert "]*" not in diff_section  # no difference cell is flagged as excluding zero


def test_ids_attribution_control_pairs_against_the_no_ids_variant(results, tmp_path):
    out = str(tmp_path / "docs")
    make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp8_controls_and_robustness", "8b_ids_attribution_controls", "results.md")
    assert "no_ids variant, no intervention (Experiment 4)" in md
    assert "10 (100.0%)" in md  # the fixture's controls equal the no_ids baseline exactly


def test_horizon_identity_compares_step_400_values_and_stays_out_of_the_headline(results, tmp_path):
    out = str(tmp_path / "docs")
    headline = make_report.build_report(results, out, n_resamples=100)
    md = _read(out, "exp8_controls_and_robustness", "8d_longer_horizon", "results.md")
    assert "10 (100.0%)" in md
    assert "not an effect" in md
    assert not any("8d" in h["experiment"] for h in headline)
    assert "8d longer horizon" in _read(out, "README.md")
