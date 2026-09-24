"""Tests for experiments/make_latex_tables.py: the LaTeX tables in the results
PDF are generated from summary.json, so these check the numbers, the
significance marks, escaping, and that the real report folder produces
well-formed tables."""
import json
import os
import re

import pytest

from experiments import make_latex_tables as mlt

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "docs", "eval_results_2026-09-24")
VARIANTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios", "variants")


def _stat(mean, lo, hi, n=500):
    return {"n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi, "median": mean, "q1": lo, "q3": hi}


def _diff(mean, lo, hi, n=500, label=None):
    return {"mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "n": n, "n_label": label or str(n)}


def _summary(baseline="normal (= Experiment 1)", cond="no_ids_branch", diff=None):
    metrics = ["reward_per_step", "episode_return", "captured"]
    return {
        "baseline": baseline, "pairing": "paired",
        "stats": {baseline: {m: _stat(0.609, 0.598, 0.619) for m in metrics},
                  cond: {m: _stat(0.583, 0.569, 0.596) for m in metrics}},
        "differences_vs_baseline": {cond: {m: diff or _diff(-0.026, -0.033, -0.019) for m in metrics}},
    }


def test_esc_escapes_latex_specials_once():
    assert mlt.esc("no_ids_branch") == r"no\_ids\_branch"
    assert mlt.esc("a & b % c # d") == r"a \& b \% c \# d"
    assert mlt.esc("x^2") == r"x\textasciicircum{}2"
    assert mlt.esc("a\\b") == r"a\textbackslash{}b"


def test_values_that_round_to_zero_print_without_a_sign():
    assert mlt._num(-0.0004, 3, signed=True) == "0.000"
    assert mlt._num(0.0004, 3, signed=True) == "0.000"
    assert mlt._num(-0.0004, 1) == "0.0"


def test_negative_numbers_use_a_math_minus():
    assert mlt._num(-0.655, 3, signed=True) == "$-$0.655"
    assert mlt._num(0.057, 3, signed=True) == "+0.057"


def test_diff_cell_marks_significance_only_when_the_ci_excludes_zero():
    assert mlt.diff_cell(_diff(-0.026, -0.033, -0.019), 3).endswith(r"$^{*}$")
    assert not mlt.diff_cell(_diff(-0.001, -0.004, 0.002), 3).endswith(r"$^{*}$")
    assert not mlt.diff_cell(_diff(0.0, 0.0, 0.0), 3).endswith(r"$^{*}$")
    assert mlt.diff_cell({"mean_diff": None, "ci_lo": None, "ci_hi": None, "n": 0}, 3) == "n/a"


def test_abs_and_diff_tables_contain_the_exact_numbers_and_escape_labels():
    s = _summary()
    a = mlt.abs_table(s)
    assert r"normal (= Experiment 1)" in a  # the experiment's own reference label is kept
    assert r"no\_ids\_branch" in a
    assert "0.609 (0.598, 0.619)" in a
    d = mlt.diff_table(s)
    assert "$-$0.026 ($-$0.033, $-$0.019)$^{*}$" in d
    for t in (a, d):
        assert t.count(r"\begin{tabular}") == t.count(r"\end{tabular}") == 1
        assert r"\toprule" in t and r"\bottomrule" in t


def test_unpaired_n_label_is_shown_as_two_sample_sizes():
    s = _summary(cond="varA (trained on)", diff=_diff(-0.371, -0.382, -0.359, n=1000, label="1000 vs 1000"))
    assert r"1000\,/\,1000" in mlt.diff_table(s)


def test_generalization_table_adds_expected_hosts_and_the_fraction():
    metrics = ["reward_per_step", "episode_return", "captured"]
    summary = {"baseline": "dynamic (trained on, selection scenario)", "pairing": "unpaired",
               "stats": {"dynamic (trained on, selection scenario)": {m: _stat(30.165, 29.9, 30.4) for m in metrics},
                         "test (held-out)": {m: _stat(11.107, 10.9, 11.3) for m in metrics}},
               "differences_vs_baseline": {}}
    t = mlt.abs_table(summary, extra=mlt.exp6_extra())
    assert "33.6" in t and "23.1" in t
    assert "0.90" in t   # 30.165 / 33.6
    assert "0.48" in t   # 11.107 / 23.1


def test_build_writes_tables_from_summary_json(tmp_path):
    report = tmp_path / "report"
    (report / "exp2_ids_observation").mkdir(parents=True)
    (report / "exp2_ids_observation" / "summary.json").write_text(json.dumps(_summary()))
    written = mlt.build(str(report), str(tmp_path / "out"))
    assert set(written) == {"exp2_abs.tex", "exp2_diff.tex"}
    assert (tmp_path / "out" / "exp2_diff.tex").exists()


@pytest.mark.skipif(not os.path.exists(os.path.join(REPORT_DIR, "exp2_ids_observation", "summary.json")),
                    reason="the generated report folder is not present")
def test_real_report_folder_produces_well_formed_tables(tmp_path):
    written = mlt.build(REPORT_DIR, str(tmp_path), VARIANTS_DIR)
    assert {"exp1_abs.tex", "exp1_milestones.tex", "exp2_diff.tex", "exp6_abs.tex", "exp5_params.tex", "exp7_verify.tex"} <= set(written)
    for name in written:
        text = (tmp_path / name).read_text()
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1, name
        assert not re.search(r"(?<!\\)_", text), f"unescaped underscore in {name}"
        assert not re.search(r"(?<!\\)%", text), f"unescaped percent in {name}"
        assert "nan" not in text.lower(), name
    assert "$-$0.655" in (tmp_path / "exp2_diff.tex").read_text()


# ---- follow-up tables --------------------------------------------------------------------

def test_explained_table_reports_the_share_of_the_return_change_that_penalties_account_for():
    stats = {"baseline": {"ids_detections": _stat(13.5, 12, 15), "ids_quarantine": _stat(1.3, 1, 2)},
             "zeroed": {"ids_detections": _stat(185.7, 180, 191), "ids_quarantine": _stat(18.7, 18, 19)}}
    diffs = {"zeroed": {"episode_return": _diff(-262.1, -269.2, -254.9), "ids_penalty": _diff(254.1, 245.3, 262.4)}}
    t = mlt.explained_table({"stats": stats, "differences_vs_baseline": diffs, "baseline": "baseline"})
    assert "185.7" in t and "97\\%" in t
    assert "$-$262.1" in t and t.count("\\begin{tabular}") == 1


def test_ids_and_generation_column_sets_render_their_headers():
    metrics = ["reward_per_step", "episode_return", "ids_detections", "ids_penalty", "ids_quarantine"]
    s = {"baseline": "b", "stats": {"b": {m: _stat(1.0, 0.5, 1.5) for m in metrics},
                                     "c": {m: _stat(2.0, 1.5, 2.5) for m in metrics}},
         "differences_vs_baseline": {"c": {m: _diff(1.0, 0.5, 1.5) for m in metrics}}}
    assert "IDS detections" in mlt.abs_table(s, columns=mlt.IDS_COLS)
    assert "IDS penalty" in mlt.diff_table(s, columns=mlt.IDS_COLS)
