"""Generate the LaTeX tables for the results PDF from the report's summary.json files.

Every number in the PDF's tables comes from the per-experiment summary.json
that experiments/make_report.py wrote, so nothing is retyped by hand. Output is
one booktabs `tabular` per file, meant to be \\input inside a \\footnotesize
group:

    <report_dir>/latex/tables/exp2_abs.tex     mean (95% CI) per condition
    <report_dir>/latex/tables/exp2_diff.tex    difference vs baseline, `*` = CI excludes 0
    ...
    <report_dir>/latex/tables/exp1_milestones.tex
    <report_dir>/latex/tables/exp5_params.tex  full parameter values per intensity level
    <report_dir>/latex/tables/exp7_verify.tex  variant verification summary
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.make_report import exp7_summary_rows  # noqa: E402

AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (short name, folder relative to the report dir)
EXPERIMENTS = [
    ("exp2", "exp2_ids_observation"),
    ("exp3", "exp3_component_interventions"),
    ("exp4", "exp4_mechanism_knockouts"),
    ("exp5", "exp5_intensity_sweep"),
    ("exp6", "exp6_generalization"),
    ("exp8a", "exp8_controls_and_robustness/8a_plumbing_controls"),
    ("exp8b", "exp8_controls_and_robustness/8b_ids_attribution_controls"),
    ("exp8c", "exp8_controls_and_robustness/8c_shuffle_seed_robustness"),
    ("exp8d", "exp8_controls_and_robustness/8d_longer_horizon"),
]

# metric -> (column header, decimals)
MAIN = [("reward_per_step", r"reward / step", 3), ("episode_return", r"return", 1), ("captured", r"captured", 2)]
MILESTONES = [("return_at_100", r"return@100", 1), ("return_at_200", r"return@200", 1), ("return_at_400", r"return@400", 1),
              ("captured_at_100", r"captured@100", 2), ("captured_at_200", r"captured@200", 2),
              ("captured_at_400", r"captured@400", 2)]

# follow-up column sets (metric, header, decimals)
IDS_COLS = [("reward_per_step", r"reward / step", 3), ("episode_return", r"return", 1),
            ("ids_detections", r"IDS detections", 1), ("ids_penalty", r"IDS penalty", 1),
            ("ids_quarantine", r"quarantines", 1)]
GEN_COLS = [("reward_per_step", r"reward / step", 3), ("captured", r"captured", 2),
            ("captured_fraction", r"captured / sensitive", 3), ("n_sensitive", r"sensitive hosts", 1)]

# (short name, folder under the follow-up report dir, column set)
FOLLOWUP = [
    ("fuB", "B_ids_detections", IDS_COLS),
    ("fuA1", "A_goal_analysis/A1_goal_masks", IDS_COLS),
    ("fuD", "D_parameter_sensitivity", MAIN),
    ("fuE", "E_scan_noise_observed", MAIN),
    ("fuF", "F_generated_scenarios", GEN_COLS),
    ("fuG", "G_ids_delay", MAIN),
    ("fuH2", "H_training_and_checkpoint/H2_best_vs_final", MAIN),
]

# expected number of capturable (sensitive) hosts per scenario: per-subnet
# sensitive-host probability times mean hosts per subnet, from each scenario's YAML
EXPECTED_SENSITIVE = {"dynamic": 33.6, "varA": 30.1, "varB": 25.9, "bridge": 28.3, "test": 23.1}

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    return "".join(_ESC.get(ch, ch) for ch in str(s))


def _num(x, nd, signed=False):
    if round(x, nd) == 0:  # avoid "-0.000" / "+0.000" for values that round to zero
        return f"{0:.{nd}f}"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def ci_cell(s, nd):
    if s["mean"] is None:
        return "n/a"
    return f"{_num(s['mean'], nd)} ({_num(s['ci_lo'], nd)}, {_num(s['ci_hi'], nd)})"


def diff_cell(d, nd):
    if d["mean_diff"] is None or d["n"] == 0:
        return "n/a"
    star = r"$^{*}$" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else ""
    return f"{_num(d['mean_diff'], nd, True)} ({_num(d['ci_lo'], nd, True)}, {_num(d['ci_hi'], nd, True)}){star}"


def tabular(colspec, header, rows):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _display(label, baseline):
    return esc(label)


def abs_table(summary, extra=None, columns=None):
    columns = columns or MAIN
    stats, baseline = summary["stats"], summary["baseline"]
    header = ["condition", "$n$"] + [h for _, h, _ in columns]
    colspec = "@{}lr" + "c" * len(columns)
    if extra:
        header += [h for h, _ in extra]
        colspec += "c" * len(extra)
    rows = []
    for label, s in stats.items():
        row = [_display(label, baseline), str(s["reward_per_step"]["n"])] + [ci_cell(s[m], nd) for m, _, nd in columns]
        if extra:
            row += [fn(label, s) for _, fn in extra]
        rows.append(row)
    return tabular(colspec + "@{}", header, rows)


def diff_table(summary, columns=None):
    columns = columns or MAIN
    diffs = summary["differences_vs_baseline"]
    header = ["condition", "$n$"] + [h for _, h, _ in columns]
    rows = [[esc(label), d["reward_per_step"]["n_label"].replace(" vs ", r"\,/\,")] + [diff_cell(d[m], nd) for m, _, nd in columns]
            for label, d in diffs.items()]
    return tabular("@{}lr" + "c" * len(columns) + "@{}", header, rows)


def _scenario_key(label):
    return label.split()[0]


def exp6_extra():
    def expected(label, _s):
        return f"{EXPECTED_SENSITIVE[_scenario_key(label)]:.1f}"

    def fraction(label, s):
        return f"{s['captured']['mean'] / EXPECTED_SENSITIVE[_scenario_key(label)]:.2f}"
    return [("expected capturable", expected), ("captured / expected", fraction)]


def milestones_table(summary):
    label, s = next(iter(summary["stats"].items()))
    header = ["condition", "$n$"] + [h for _, h, _ in MILESTONES]
    row = [esc(label), str(s["reward_per_step"]["n"])] + [ci_cell(s[m], nd) for m, _, nd in MILESTONES]
    return tabular("@{}lr" + "c" * len(MILESTONES) + "@{}", header, [row])


def params_table(variants_dir):
    from experiments.make_variants import final_stage, load_scenario
    rows = []
    for name in ("off", "low", "final", "harder"):
        path = os.path.join(variants_dir, f"corp_100hosts_dynamic__intensity_{name}.v2.yaml")
        st = final_stage(load_scenario(path))
        ids, noise = st.get("ids", {}), st.get("scan_noise", {})
        types = ("service_scan", "os_scan", "process_scan")
        fp = "/".join(f"{noise.get(t, {}).get('false_positive_rate', 0):g}" for t in types)
        fn = "/".join(f"{noise.get(t, {}).get('false_negative_rate', 0):g}" for t in types)
        thr = ", ".join(f"{x:.3f}" for x in ids.get("base_thresholds", [])) or "--"
        rows.append([esc(f"intensity_{name}"), "yes" if ids.get("enabled") else "no", thr, fp, fn,
                     f"{st.get('service_dynamics', {}).get('churn_probability', 0):g}",
                     f"{st.get('network_reliability', {}).get('timeout_probability', 0):g}"])
    header = ["level", "IDS", "IDS thresholds", "scan FP", "scan FN", "churn", "timeout"]
    return tabular("@{}llcccrr@{}", header, rows)


def explained_table(summary):
    """Follow-up B: IDS detections per episode, and the share of each return change that the
    change in IDS penalties accounts for (paired differences from the baseline)."""
    stats, diffs, base = summary["stats"], summary["differences_vs_baseline"], summary["baseline"]
    header = ["condition", "IDS detections", "quarantines", "change in return", "change in IDS penalty", "penalty / return"]
    rows = [["baseline", f"{stats[base]['ids_detections']['mean']:.1f}", f"{stats[base]['ids_quarantine']['mean']:.1f}", "--", "--", "--"]]
    for label, d in diffs.items():
        dr, dp = d["episode_return"]["mean_diff"], d["ids_penalty"]["mean_diff"]
        ratio = "--" if abs(dr) < 1e-9 else f"{-100 * dp / dr:.0f}\\%"
        rows.append([esc(label), f"{stats[label]['ids_detections']['mean']:.1f}", f"{stats[label]['ids_quarantine']['mean']:.1f}",
                     diff_cell(d["episode_return"], 1), diff_cell(d["ids_penalty"], 1), ratio])
    return tabular("@{}lcccc c@{}".replace(" ", ""), header, rows)


def mi_table(trace_summary):
    """Follow-up A2: association between the held goal and the IDS level bin, raw and controlled for time."""
    header = ["condition", "episodes", "MI(goal; level)", "MI(goal; step)", "MI(goal; level $\\mid$ step)", "null 95th pct"]
    rows = []
    for label, t in trace_summary.items():
        rows.append([esc(label), str(t["n_episodes"]), f"{t['mi']:.3f}", f"{t['mi_goal_time']:.3f}",
                     f"{t['mi_given_time']:.3f}", f"{t['mi_given_time_null_p95']:.4f}"])
    return tabular("@{}lrcccc@{}", header, rows)


def h1_table(h1_summary):
    """Follow-up H1: goal usage and evaluation by curriculum stage."""
    goals = ("REDUCE_DETECTION", "RECOVER_OR_REPLAN", "ESCALATE_PRIVILEGE")
    header = ["stage", "reduce det.", "recover/replan", "escalate priv.", "manager entropy", "eval reward / step", "eval captured"]
    rows = []
    for stage, w in h1_summary["windows"].items():
        rows.append([esc(stage)] + [f"{100 * w['goal_share'][g]:.1f}\\%" for g in goals] +
                    [f"{w['manager_entropy']:.3f}", f"{w['eval_tst_reward_per_step']:.3f}", f"{w['eval_tst_captured']:.2f}"])
    return tabular("@{}lcccccc@{}", header, rows)


def exp7_table(machine):
    header = ["variant", "rates in 95\\% CI", "zero-config.\\ rows with 0 events", "check", "IDS detections"]
    rows = [[esc(r[0]), esc(r[1]), esc(r[2]), esc(r[3]), esc(r[4])] for r in exp7_summary_rows(machine)]
    return tabular("@{}lcccr@{}", header, rows)


def build(report_dir, out_dir, variants_dir=None, followup_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def emit(name, text):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
        written.append(name)

    exp1_path = os.path.join(report_dir, "exp1_baseline", "summary.json")
    if os.path.exists(exp1_path):
        with open(exp1_path) as f:
            e1 = json.load(f)
        emit("exp1_abs.tex", abs_table({"stats": e1["stats"], "baseline": None}))
        emit("exp1_milestones.tex", milestones_table(e1))

    for short, folder in EXPERIMENTS:
        path = os.path.join(report_dir, folder, "summary.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            summary = json.load(f)
        # in the generalization table the reference row is a scenario, not a control condition
        abs_summary = {**summary, "baseline": None} if short == "exp6" else summary
        emit(f"{short}_abs.tex", abs_table(abs_summary, extra=exp6_extra() if short == "exp6" else None))
        emit(f"{short}_diff.tex", diff_table(summary))

    exp7_path = os.path.join(report_dir, "exp7_variant_verification", "summary.json")
    if os.path.exists(exp7_path):
        with open(exp7_path) as f:
            emit("exp7_verify.tex", exp7_table(json.load(f)))
    if variants_dir and os.path.isdir(variants_dir):
        emit("exp5_params.tex", params_table(variants_dir))

    if followup_dir and os.path.isdir(followup_dir):
        for short, folder, columns in FOLLOWUP:
            path = os.path.join(followup_dir, folder, "summary.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                summary = json.load(f)
            emit(f"{short}_abs.tex", abs_table(summary, columns=columns))
            emit(f"{short}_diff.tex", diff_table(summary, columns=columns))
            if short == "fuB":
                emit("fuB_explained.tex", explained_table(summary))
        mi_path = os.path.join(followup_dir, "A_goal_analysis", "A2_goal_traces", "trace_summary.json")
        if os.path.exists(mi_path):
            with open(mi_path) as f:
                emit("fuA2_mi.tex", mi_table(json.load(f)))
        h1_path = os.path.join(followup_dir, "H1_training_log", "training_log_summary.json")
        if os.path.exists(h1_path):
            with open(h1_path) as f:
                emit("fuH1_windows.tex", h1_table(json.load(f)))
        c_path = os.path.join(followup_dir, "C_ids_semantics", "ids_parameters.tex")
        if os.path.exists(c_path):
            with open(c_path) as f:
                emit("fuC_ids_parameters.tex", f.read())
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report_dir", default=os.path.join(AGENTS_DIR, "..", "docs", "eval_results_2026-09-24"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--followup_dir", default=os.path.join(AGENTS_DIR, "..", "docs", "eval_followup_2026-09-24"),
                     help="Follow-up report folder (make_followup_report.py output); its tables are added when present")
    ap.add_argument("--variants_dir", default=os.path.join(AGENTS_DIR, "..", "scenarios", "variants"))
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.report_dir, "latex", "tables")
    written = build(args.report_dir, out_dir, args.variants_dir, args.followup_dir)
    print(f"wrote {len(written)} tables to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
