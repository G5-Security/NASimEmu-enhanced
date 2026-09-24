"""Tests for experiments/make_followup_report.py on a synthetic results directory with planted effects."""
import csv
import os

import pytest

from experiments import make_followup_report as mfr
from experiments.eval_harness import CSV_FIELDS, IDS_FIELDS, META_FIELDS

SEEDS = list(range(1_000_000, 1_000_020))


def _row(seed, ret=100.0, captured=30, quarantine=0, patch=0, monitor=0, n_sensitive=35, cond="c"):
    det = quarantine + patch + monitor
    return {"model": "m", "scenario": "s", "condition": cond, "episode_idx": seed - 1_000_000, "seed": seed,
            "episode_return": ret, "episode_len": 400, "captured": captured,
            "reward_at_100": ret / 4, "captured_at_100": 5, "reward_at_200": ret / 2, "captured_at_200": 8,
            "reward_at_400": ret, "captured_at_400": captured, "terminal_action": False, "hit_step_limit": True,
            "ids_detections": det, "ids_quarantine": quarantine, "ids_patch": patch, "ids_monitor": monitor,
            "ids_first_step": 10 if det else -1, "ids_max_level": 1.0, "ids_increases": 50,
            "n_sensitive": n_sensitive, "n_hosts": 78}


def _write(d, name, rows, extras=True):
    fields = CSV_FIELDS + (IDS_FIELDS + META_FIELDS if extras else [])
    with open(os.path.join(d, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _series(seeds, **kw):
    return [_row(s, ret=100.0 + (s % 5), **kw) for s in seeds]


@pytest.fixture
def dirs(tmp_path):
    main, follow = tmp_path / "main", tmp_path / "follow"
    main.mkdir(), follow.mkdir()
    _write(str(main), "exp1_base_c0", _series(SEEDS), extras=False)
    _write(str(main), "exp2_delayed", _series(SEEDS, ret=None) if False else [_row(s, ret=98.0 + (s % 5)) for s in SEEDS], extras=False)
    _write(str(follow), "B_baseline", _series(SEEDS, patch=1, monitor=2))
    # zeroed: return 10 lower, and exactly 10 more IDS penalty in return units (2 extra quarantines x 5)
    _write(str(follow), "B_zeroed", [_row(s, ret=100.0 + (s % 5) - 10.0, quarantine=2, patch=1, monitor=2) for s in SEEDS])
    _write(str(follow), "B_ctrl_no_ids_variant", _series(SEEDS[:5]))
    _write(str(follow), "A_mask_reduce_detection", _series(SEEDS, patch=1, monitor=3))
    _write(str(follow), "D_ids_thr_x0.6", [_row(s, ret=90.0 + (s % 5)) for s in SEEDS], extras=False)
    _write(str(follow), "E_noiseobs_x0", _series(SEEDS), extras=False)
    _write(str(follow), "E_noiseobs_x2", [_row(s, ret=95.0 + (s % 5)) for s in SEEDS], extras=False)
    _write(str(follow), "F_dynamic_mesh", _series(SEEDS, n_sensitive=35))
    _write(str(follow), "F_dynamic_chain", _series(SEEDS, captured=15, n_sensitive=30))
    _write(str(follow), "F_ref_dynamic", _series(SEEDS[:5], n_sensitive=35))
    _write(str(follow), "G_delayed_5", [_row(s, ret=99.0 + (s % 5)) for s in SEEDS], extras=False)
    _write(str(follow), "H2_best_c0", [_row(s, ret=101.0 + (s % 5)) for s in SEEDS], extras=False)
    return str(main), str(follow)


def _read(out, *parts):
    return open(os.path.join(out, *parts)).read()


def test_every_experiment_folder_is_written(dirs, tmp_path):
    out = str(tmp_path / "docs")
    mfr.build_followup(*dirs, out, n_resamples=100)
    for rel in ("B_ids_detections", "A_goal_analysis/A1_goal_masks", "D_parameter_sensitivity", "E_scan_noise_observed",
                "F_generated_scenarios", "G_ids_delay", "H_training_and_checkpoint/H2_best_vs_final"):
        assert os.path.isfile(os.path.join(out, rel, "results.md")), rel
    assert os.path.isfile(os.path.join(out, "A_goal_analysis", "A2_goal_traces", "trace_analysis.md"))
    assert "FINDINGS_FOLLOWUP.md" in _read(out, "README.md")


def test_b_reports_identity_and_the_share_of_the_return_change_the_penalties_explain(dirs, tmp_path):
    out = str(tmp_path / "docs")
    mfr.build_followup(*dirs, out, n_resamples=100)
    md = _read(out, "B_ids_detections", "results.md")
    assert "20 (100.0%)" in md                     # the counter does not change any episode
    assert "100%" in md.split("penalty change / return change")[1]   # -10 return, +10 penalty
    assert "Control: IDS disabled in the environment" in md


def test_f_is_unpaired_and_uses_the_exact_sensitive_host_counts(dirs, tmp_path):
    out = str(tmp_path / "docs")
    mfr.build_followup(*dirs, out, n_resamples=100)
    md = _read(out, "F_generated_scenarios", "results.md")
    assert "unpaired" in md and "captured_fraction" in md
    assert "Hand-made scenarios with their exact sensitive-host counts" in md
    assert "dynamic profile, chain" in md


def test_e_x0_control_reports_bit_identity_and_g_has_an_intensity_style_curve(dirs, tmp_path):
    out = str(tmp_path / "docs")
    mfr.build_followup(*dirs, out, n_resamples=100)
    assert "20 (100.0%)" in _read(out, "E_scan_noise_observed", "results.md")
    assert any(p.endswith("_vs_intensity.png") for p in os.listdir(os.path.join(out, "G_ids_delay", "plots")))
    assert any(p.startswith("dose_response") for p in os.listdir(os.path.join(out, "D_parameter_sensitivity", "plots")))


def test_missing_data_is_reported_not_fatal(tmp_path):
    main, follow = tmp_path / "m", tmp_path / "f"
    main.mkdir(), follow.mkdir()
    _write(str(main), "exp1_base_c0", _series(SEEDS), extras=False)
    out = str(tmp_path / "docs")
    mfr.build_followup(str(main), str(follow), out, n_resamples=50)
    assert "No data yet" in _read(out, "B_ids_detections", "results.md")
    assert "No data yet" in _read(out, "A_goal_analysis", "A2_goal_traces", "trace_analysis.md")
