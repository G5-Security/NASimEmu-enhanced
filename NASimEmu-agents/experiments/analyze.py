"""Step 8: statistics, claims (docs/eval_revision_plan.tex).

Reads per-episode CSVs written by experiments/eval_harness.py and reports,
per (model, scenario, condition) group, the length-fair metrics with a 95%
bootstrap CI (10,000 resamples by default) and median/IQR, plus paired
differences between two conditions/models over shared evaluation seeds
(also bootstrap CIs). Emits a summary table in Markdown and LaTeX, and can
plot one metric across conditions with CI error bars.

Checkpoint-independence, and how this was tested
--------------------------------------------------
This script only ever reads already-written CSVs -- it has no idea whether
they came from a trained checkpoint or not. So, following the same "tests
before large runs" discipline as Steps 1-7, its statistics engine (bootstrap
CI, paired differences, table/plot rendering) is fully built and tested here
against CSVs that experiments/eval_harness.py already produces from an
*untrained* NASimNetDHRL (see tests/test_analyze.py) -- no trained
checkpoint needed to validate that the numbers this script reports are
computed correctly. What genuinely cannot happen yet is producing headline
*results*: those numbers are only meaningful once real trained-checkpoint
CSVs exist (docs/eval_revision_plan.tex, Step 0/8 "Blocked").

Deviation from the plan's sketch: known and explicit
------------------------------------------------------
The plan's Step 8 summary table lists "IDS triggers" as a column. That is
not implemented here, because it is not in eval_harness.py's CSV schema
(CSV_FIELDS) -- IDS-trigger counts exist only as DynamicsRecorder events
(experiments/dynamics_recorder.py), which are recorded per-simulator-call,
not wired into eval_harness.py's one-row-per-episode output. Adding it would
mean touching eval_harness.py itself (a Step 1 file, already built and
tested) to plumb a per-episode IDS-trigger count through -- out of scope for
Step 8, which reads what Step 1 already writes. Documented here rather than
silently dropped, the same way earlier steps corrected the plan against the
running code instead of matching it by construction.
"""
import argparse
import csv
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


# Optional columns written by eval_harness.py's --record_ids / --record_meta; absent from older CSVs.
OPTIONAL_NUMERIC = ("ids_detections", "ids_quarantine", "ids_patch", "ids_monitor", "ids_first_step",
                    "ids_max_level", "ids_increases", "n_sensitive", "n_hosts")


def _parse_row(raw):
    row = _parse_base_row(raw)
    for k in OPTIONAL_NUMERIC:
        if raw.get(k) not in (None, ""):
            row[k] = float(raw[k])
    return row


def _parse_base_row(raw):
    return {
        "model": raw["model"], "scenario": raw["scenario"], "condition": raw["condition"],
        "episode_idx": int(raw["episode_idx"]), "seed": int(raw["seed"]),
        "episode_return": float(raw["episode_return"]), "episode_len": int(raw["episode_len"]),
        "captured": int(float(raw["captured"])),
        "reward_at_100": float(raw["reward_at_100"]), "captured_at_100": int(float(raw["captured_at_100"])),
        "reward_at_200": float(raw["reward_at_200"]), "captured_at_200": int(float(raw["captured_at_200"])),
        "reward_at_400": float(raw["reward_at_400"]), "captured_at_400": int(float(raw["captured_at_400"])),
        "terminal_action": raw["terminal_action"] == "True",
        "hit_step_limit": raw["hit_step_limit"] == "True",
    }


def load_rows(csv_paths):
    """Load and concatenate one or more eval_harness.py CSVs."""
    rows = []
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(_parse_row(r) for r in reader)
    return rows


# ---- Length-fair per-episode metrics ---------------------------------------
# Each takes one parsed row and returns a scalar. reward_per_step and
# return_at_400/captured_at_400 are fair across episodes of different length
# (see docs/eval_revision_plan.tex, Step 8's "metrics that are fair across
# episode lengths"); raw episode_return/captured are included too since
# they're still meaningful within a single fixed step_limit run.
METRICS = {
    "reward_per_step": lambda r: r["episode_return"] / r["episode_len"] if r["episode_len"] else 0.0,
    "episode_return": lambda r: r["episode_return"],
    "captured": lambda r: float(r["captured"]),
    "return_at_100": lambda r: r["reward_at_100"],
    "return_at_200": lambda r: r["reward_at_200"],
    "return_at_400": lambda r: r["reward_at_400"],
    "captured_at_100": lambda r: float(r["captured_at_100"]),
    "captured_at_200": lambda r: float(r["captured_at_200"]),
    "captured_at_400": lambda r: float(r["captured_at_400"]),
    # need eval_harness.py --record_ids / --record_meta columns
    "ids_detections": lambda r: r["ids_detections"],
    "ids_quarantine": lambda r: r["ids_quarantine"],
    "ids_patch": lambda r: r["ids_patch"],
    "ids_monitor": lambda r: r["ids_monitor"],
    # Reward the IDS took directly, in the units of the reported returns. The raw penalties are
    # -50 quarantine, -20 patch, -5 monitor, but NASimEmuEnv divides every reward by 10
    # (src/nasimemu/env.py, "reward scaling"), so returns see -5, -2 and -0.5.
    "ids_penalty": lambda r: (50.0 * r["ids_quarantine"] + 20.0 * r["ids_patch"] + 5.0 * r["ids_monitor"]) / 10.0,
    "detected_any": lambda r: 1.0 if r["ids_detections"] > 0 else 0.0,
    "ids_max_level": lambda r: r["ids_max_level"],
    "captured_fraction": lambda r: r["captured"] / r["n_sensitive"] if r["n_sensitive"] else float("nan"),
    "n_sensitive": lambda r: r["n_sensitive"],
    "n_hosts": lambda r: r["n_hosts"],
    "stop_rate": lambda r: 1.0 if r["terminal_action"] else 0.0,
    "step_limit_rate": lambda r: 1.0 if r["hit_step_limit"] else 0.0,
}

SUMMARY_TABLE_METRICS = ["reward_per_step", "episode_return", "captured", "step_limit_rate", "stop_rate"]


def bootstrap_ci(values, n_resamples=10000, ci=0.95, seed=None):
    """Percentile bootstrap CI for the mean. (mean, lo, hi); (nan, nan, nan)
    for an empty sample, (v, v, v) for a single-element sample (nothing to
    resample)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(values) == 1:
        v = float(values[0])
        return (v, v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    resample_means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(resample_means, [alpha, 1.0 - alpha])
    return (float(values.mean()), float(lo), float(hi))


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    median = float(np.median(values))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return (median, float(q1), float(q3))


def group_by(rows, keys=("model", "scenario", "condition")):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    return groups


def summarize_group(rows, metrics, n_resamples=10000, seed=None):
    """metrics: dict of name -> row->float callable."""
    out = {}
    for name, fn in metrics.items():
        values = [fn(r) for r in rows]
        mean, lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        median, q1, q3 = median_iqr(values)
        out[name] = {"n": len(values), "mean": mean, "ci_lo": lo, "ci_hi": hi,
                     "median": median, "q1": q1, "q3": q3}
    return out


def paired_difference(rows_a, rows_b, metric, n_resamples=10000, seed=None):
    """Bootstrap CI on the mean (A - B) difference, paired by seed -- only
    seeds present in both groups contribute (the harness's fixed seed bank
    guarantees a shared starting network per seed, see eval_harness.py's
    module docstring)."""
    fn = METRICS[metric] if isinstance(metric, str) else metric
    by_seed_a = {r["seed"]: r for r in rows_a}
    by_seed_b = {r["seed"]: r for r in rows_b}
    shared = sorted(set(by_seed_a) & set(by_seed_b))
    diffs = [fn(by_seed_a[s]) - fn(by_seed_b[s]) for s in shared]
    mean, lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)
    return {"n_pairs": len(shared), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi}


def unpaired_difference(rows_a, rows_b, metric, n_resamples=10000, seed=None):
    """Bootstrap CI on mean(A) - mean(B) for two independent samples, each
    resampled on its own. For groups whose episodes cannot be paired by seed
    (different scenarios generate different networks for the same seed)."""
    fn = METRICS[metric] if isinstance(metric, str) else metric
    a = np.asarray([fn(r) for r in rows_a], dtype=float)
    b = np.asarray([fn(r) for r in rows_b], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"n_a": len(a), "n_b": len(b), "mean_diff": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), size=(n_resamples, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(n_resamples, len(b)))].mean(axis=1)
    lo, hi = np.quantile(ma - mb, [0.025, 0.975])
    return {"n_a": len(a), "n_b": len(b), "mean_diff": float(a.mean() - b.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi)}


def plot_paired_forest(diffs, metric, out_png, title=None):
    """diffs: list of (label, mean_diff, ci_lo, ci_hi). Horizontal CI bars
    around each mean difference with a reference line at zero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.6 * len(diffs) + 1.2)))
    y = np.arange(len(diffs))
    means = [d[1] for d in diffs]
    los = [max(0.0, d[1] - d[2]) for d in diffs]
    his = [max(0.0, d[3] - d[1]) for d in diffs]
    ax.errorbar(means, y, xerr=[los, his], fmt="o", capsize=4)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in diffs])
    ax.invert_yaxis()
    ax.set_xlabel(f"difference in {metric} (condition - baseline), 95% bootstrap CI")
    ax.set_title(textwrap.fill(title or f"Paired difference: {metric}", 60), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_summary_table(rows, keys=("model", "scenario", "condition"),
                         metrics=SUMMARY_TABLE_METRICS, n_resamples=10000, seed=None):
    groups = group_by(rows, keys)
    metric_fns = {m: METRICS[m] for m in metrics}
    table = []
    for key in sorted(groups):
        group_rows = groups[key]
        summary = summarize_group(group_rows, metric_fns, n_resamples=n_resamples, seed=seed)
        entry = dict(zip(keys, key))
        entry["n_episodes"] = len(group_rows)
        entry["metrics"] = summary
        table.append(entry)
    return table


def to_markdown_table(table, keys=("model", "scenario", "condition"), metrics=SUMMARY_TABLE_METRICS):
    header = list(keys) + ["n"] + [f"{m} (95% CI)" for m in metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for entry in table:
        row = [str(entry[k]) for k in keys] + [str(entry["n_episodes"])]
        for m in metrics:
            s = entry["metrics"][m]
            row.append(f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_escape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def to_latex_table(table, keys=("model", "scenario", "condition"), metrics=SUMMARY_TABLE_METRICS):
    ncols = len(keys) + 1 + len(metrics)
    header = list(keys) + ["n"] + metrics
    lines = [r"\begin{tabular}{" + "l" * ncols + "}", r"\hline",
             " & ".join(_latex_escape(h) for h in header) + r" \\", r"\hline"]
    for entry in table:
        row = [_latex_escape(entry[k]) for k in keys] + [str(entry["n_episodes"])]
        for m in metrics:
            s = entry["metrics"][m]
            row.append(f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def plot_metric_by_condition(table, metric, out_png, keys=("condition",), title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [" / ".join(str(entry[k]) for k in keys) for entry in table]
    means = [entry["metrics"][metric]["mean"] for entry in table]
    los = [max(0.0, entry["metrics"][metric]["mean"] - entry["metrics"][metric]["ci_lo"]) for entry in table]
    his = [max(0.0, entry["metrics"][metric]["ci_hi"] - entry["metrics"][metric]["mean"]) for entry in table]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[los, his], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(textwrap.fill(title or metric, 60), fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _apply_filter(rows, filters):
    if not filters:
        return rows
    parsed = dict(f.split("=", 1) for f in filters)
    return [r for r in rows if all(str(r.get(k)) == v for k, v in parsed.items())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True, help="One or more eval_harness.py CSVs to combine")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_tex", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--n_resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0,
                     help="Bootstrap RNG seed, for reproducible CIs across re-runs of the same CSVs")
    ap.add_argument("--plot_metric", default=None, choices=list(METRICS))
    ap.add_argument("--plot_out_png", default=None)
    ap.add_argument("--plot_group_keys", nargs="+", default=["condition"])
    ap.add_argument("--paired_metric", default=None, choices=list(METRICS))
    ap.add_argument("--paired_filter_a", nargs="+", default=None, help="e.g. condition=full_dynamics")
    ap.add_argument("--paired_filter_b", nargs="+", default=None, help="e.g. condition=no_ids")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("no rows loaded from --csv")

    table = build_summary_table(rows, n_resamples=args.n_resamples, seed=args.seed)

    md = to_markdown_table(table)
    print(md)
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(md + "\n")
    if args.out_tex:
        with open(args.out_tex, "w") as f:
            f.write(to_latex_table(table) + "\n")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(table, f, indent=2)

    if args.plot_metric:
        if not args.plot_out_png:
            raise SystemExit("--plot_metric requires --plot_out_png")
        plot_table = build_summary_table(rows, keys=tuple(args.plot_group_keys),
                                          metrics=[args.plot_metric],
                                          n_resamples=args.n_resamples, seed=args.seed)
        plot_metric_by_condition(plot_table, args.plot_metric, args.plot_out_png,
                                  keys=tuple(args.plot_group_keys))
        print(f"\nwrote {args.plot_out_png}")

    if args.paired_metric:
        rows_a = _apply_filter(rows, args.paired_filter_a)
        rows_b = _apply_filter(rows, args.paired_filter_b)
        diff = paired_difference(rows_a, rows_b, args.paired_metric,
                                  n_resamples=args.n_resamples, seed=args.seed)
        print(f"\npaired difference ({args.paired_metric}, A - B): "
              f"{diff['mean_diff']:.4f} [{diff['ci_lo']:.4f}, {diff['ci_hi']:.4f}] "
              f"over {diff['n_pairs']} paired seeds")


if __name__ == "__main__":
    main()
