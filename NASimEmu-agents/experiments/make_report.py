"""Turn a results directory of eval_harness.py CSVs into a detailed report,
one subfolder per experiment (docs/eval_revision_plan.tex, Steps 1-8).

Raw per-episode CSVs stay where the harness wrote them (results/<date>/).
Everything generated here goes to a separate docs folder:

    <out_dir>/README.md                  index, method, headline table, caveats
    <out_dir>/exp1_baseline/             results.md, summary_table.tex, summary.json, plots/
    <out_dir>/exp2_ids_observation/      ...
    <out_dir>/exp3_component_interventions/
    <out_dir>/exp4_mechanism_knockouts/
    <out_dir>/exp5_intensity_sweep/
    <out_dir>/exp6_generalization/
    <out_dir>/exp7_variant_verification/ results.md, summary.json

Per experiment: absolute results for every metric (mean with 95% bootstrap CI,
median with IQR), paired differences against the baseline on shared seeds
(unpaired for generalization, where different scenarios generate different
networks for the same seed), and plots. A '*' on a paired difference means its
95% CI excludes zero. These intervals are not adjusted for the number of
comparisons made, so '*' is descriptive.

Missing CSVs are skipped and the affected experiment is reported as "no data",
so the script can run on a partially finished results directory.
"""
import argparse
import glob
import json
import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import analyze  # noqa: E402
from experiments.analyze import METRICS  # noqa: E402

AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ABS_METRICS = ["reward_per_step", "episode_return", "captured",
               "return_at_100", "return_at_200", "return_at_400",
               "captured_at_100", "captured_at_200", "captured_at_400",
               "stop_rate", "step_limit_rate"]
MAIN_METRICS = ["reward_per_step", "episode_return", "captured", "stop_rate", "step_limit_rate"]
MILESTONE_METRICS = ["return_at_100", "return_at_200", "return_at_400",
                     "captured_at_100", "captured_at_200", "captured_at_400"]
MEDIAN_METRICS = ["reward_per_step", "episode_return", "captured"]
HEADLINE = ["reward_per_step", "captured"]

DEFAULT_METRIC_SETS = {"abs": ABS_METRICS, "main": MAIN_METRICS, "milestones": MILESTONE_METRICS,
                       "medians": MEDIAN_METRICS, "headline": HEADLINE, "tex": analyze.SUMMARY_TABLE_METRICS}

WORDING = ("Results are based on one trained checkpoint per architecture. Confidence "
           "intervals quantify evaluation-environment variability, not variability "
           "across independent training runs.")


# ---- small helpers ----------------------------------------------------------

def _nd(metric):
    if metric.startswith("return") or metric in ("episode_return", "ids_penalty"):
        return 1
    return 2 if metric.startswith("ids_") else 3


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return None if (np.isnan(o) or np.isinf(o)) else o
    return o


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _files(results_dir, name):
    return (sorted(glob.glob(os.path.join(results_dir, f"{name}.csv")))
            + sorted(glob.glob(os.path.join(results_dir, f"{name}_c[0-9].csv"))))


def _rows(results_dir, name):
    paths = _files(results_dir, name)
    return analyze.load_rows(paths) if paths else []


def _align(baseline, groups):
    """Restrict the baseline and every group to the seeds they all share, so
    the baseline row in each table is on exactly the episodes being compared."""
    seed_sets = [{r["seed"] for r in baseline}] + [{r["seed"] for r in rows} for rows in groups.values()]
    shared = set.intersection(*seed_sets) if all(seed_sets) else set()
    keep = lambda rows: [r for r in rows if r["seed"] in shared]  # noqa: E731
    return keep(baseline), {k: keep(v) for k, v in groups.items()}


def _summ(groups, metrics, n_resamples, seed):
    fns = {m: METRICS[m] for m in metrics}
    return {label: analyze.summarize_group(rows, fns, n_resamples=n_resamples, seed=seed)
            for label, rows in groups.items()}


def _diff(rows, base, metric, pairing, n_resamples, seed):
    """condition - baseline; unified result shape for paired and unpaired."""
    if pairing == "paired":
        d = analyze.paired_difference(rows, base, metric, n_resamples=n_resamples, seed=seed)
        return {"mean_diff": d["mean_diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                "n": d["n_pairs"], "n_label": str(d["n_pairs"])}
    d = analyze.unpaired_difference(rows, base, metric, n_resamples=n_resamples, seed=seed)
    return {"mean_diff": d["mean_diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
            "n": min(d["n_a"], d["n_b"]), "n_label": f"{d['n_a']} vs {d['n_b']}"}


def _cell_ci(s, metric):
    nd = _nd(metric)
    if np.isnan(s["mean"]):
        return "n/a"
    return f"{s['mean']:.{nd}f} [{s['ci_lo']:.{nd}f}, {s['ci_hi']:.{nd}f}]"


def _cell_median(s, metric):
    nd = _nd(metric)
    return f"{s['median']:.{nd}f} [{s['q1']:.{nd}f}, {s['q3']:.{nd}f}]"


def _cell_diff(d, metric):
    nd = _nd(metric)
    if d["n"] == 0 or np.isnan(d["mean_diff"]):
        return "n/a"
    star = "*" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else ""
    return f"{d['mean_diff']:+.{nd}f} [{d['ci_lo']:+.{nd}f}, {d['ci_hi']:+.{nd}f}]{star}"


def _stats_md(stats, labels, metrics, cell):
    header = ["condition", "n"] + metrics
    rows = [[lab, str(stats[lab][metrics[0]]["n"])] + [cell(stats[lab][m], m) for m in metrics]
            for lab in labels]
    return _table(header, rows)


def _diff_md(diffs, labels, metrics):
    header = ["condition", "n"] + metrics
    rows = [[lab, diffs[lab][metrics[0]]["n_label"]] + [_cell_diff(diffs[lab][m], m) for m in metrics]
            for lab in labels]
    return _table(header, rows)


def _plot_bars(stats, labels, metric, out_png, title):
    table = [{"label": lab, "metrics": {metric: stats[lab][metric]}} for lab in labels]
    analyze.plot_metric_by_condition(table, metric, out_png, keys=("label",), title=title)


def _plot_curve(stats, order, metric, out_png, title, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [o for o in order if o in stats]
    means = [stats[o][metric]["mean"] for o in order]
    los = [max(0.0, stats[o][metric]["mean"] - stats[o][metric]["ci_lo"]) for o in order]
    his = [max(0.0, stats[o][metric]["ci_hi"] - stats[o][metric]["mean"]) for o in order]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(range(len(order)), means, yerr=[los, his], fmt="-o", capsize=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(textwrap.fill(title, 60), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_hist(rows, metric, out_png, title, bins=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [METRICS[metric](r) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(values, bins=bins)
    ax.set_xlabel(metric)
    ax.set_ylabel("episodes")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---- generic experiment writer -----------------------------------------------

def write_experiment(out_dir, key, title, purpose, baseline_label, baseline_rows, groups, *,
                     pairing="paired", notes=(), extra_md="", checkpoint="", n_resamples=10000,
                     seed=0, curve_order=None, curve_xlabel="", extra_plots=None, align=True, metric_sets=None):
    """Write one experiment folder. Returns headline rows for the README, or
    None when there is no data."""
    d = os.path.join(out_dir, key)
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    groups = {k: v for k, v in groups.items() if v}
    if not groups or not baseline_rows:
        with open(os.path.join(d, "results.md"), "w") as f:
            f.write(f"# {title}\n\nNo data yet: the CSVs for this experiment were not found in the "
                    f"results directory (or the baseline was missing).\n")
        return None

    ms = {**DEFAULT_METRIC_SETS, **(metric_sets or {})}
    base, grp = (_align(baseline_rows, groups) if align else (baseline_rows, groups))
    all_groups = {baseline_label: base, **grp}
    stats = _summ(all_groups, ms["abs"], n_resamples, seed)
    labels = list(all_groups)
    cond_labels = list(grp)
    diffs = {lab: {m: _diff(grp[lab], base, m, pairing, n_resamples, seed) for m in ms["abs"]}
             for lab in cond_labels}

    seeds = sorted({r["seed"] for r in base})
    md = [f"# {title}", "",
          f"- **Checkpoint:** {checkpoint}",
          f"- **Baseline:** {baseline_label}",
          f"- **Episodes per condition:** {len(base)} (seeds {seeds[0]} to {seeds[-1]})",
          f"- **Comparison:** {'paired by seed (same starting network per seed)' if pairing == 'paired' else 'unpaired (independent samples; different scenarios generate different networks)'}",
          f"- **Interval:** 95% percentile bootstrap, {n_resamples:,} resamples, bootstrap seed {seed}",
          "", "## What this measures", "", purpose, ""]
    md += ["## Absolute results: mean [95% CI]", "", _stats_md(stats, labels, ms["main"], _cell_ci), ""]
    if ms["milestones"]:
        md += [ms.get("second_heading", "Cumulative reward and captures at fixed action counts (fair across episode lengths):"), "",
               _stats_md(stats, labels, ms["milestones"], _cell_ci), ""]
    if ms["medians"]:
        md += ["## Medians [Q1, Q3]", "", _stats_md(stats, labels, ms["medians"], _cell_median), ""]
    diff_word = "paired" if pairing == "paired" else "unpaired"
    md += [f"## Differences vs baseline ({diff_word}; condition minus baseline)", "",
           "`*` = the 95% CI excludes zero (descriptive, not adjusted for multiple comparisons).", "",
           _diff_md(diffs, cond_labels, ms["main"]), ""]
    if ms["milestones"]:
        md += [_diff_md(diffs, cond_labels, ms["milestones"]), ""]
    if extra_md:
        md += [extra_md, ""]

    plots = []
    for m in ms["headline"]:
        p = f"plots/{m}_by_condition.png"
        _plot_bars(stats, labels, m, os.path.join(d, p), f"{title}: {m}")
        plots.append(p)
        p = f"plots/paired_forest_{m}.png" if pairing == "paired" else f"plots/diff_forest_{m}.png"
        analyze.plot_paired_forest([(lab, diffs[lab][m]["mean_diff"], diffs[lab][m]["ci_lo"], diffs[lab][m]["ci_hi"])
                                    for lab in cond_labels], m, os.path.join(d, p),
                                   title=f"{title}: {m} vs baseline")
        plots.append(p)
        if curve_order:
            p = f"plots/{m}_vs_intensity.png"
            _plot_curve(stats, curve_order, m, os.path.join(d, p), f"{title}: {m}", curve_xlabel)
            plots.append(p)
    for name, fn in (extra_plots or {}).items():
        fn(os.path.join(d, f"plots/{name}.png"))
        plots.append(f"plots/{name}.png")
    md += ["## Plots", ""] + [f"![{p}]({p})" for p in plots] + [""]
    if notes:
        md += ["## Notes and caveats", ""] + [f"- {n}" for n in notes] + [""]
    md += ["---", WORDING, ""]
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("\n".join(md))

    tex_entries = [{"label": lab, "n_episodes": len(all_groups[lab]), "metrics": stats[lab]} for lab in labels]
    with open(os.path.join(d, "summary_table.tex"), "w") as f:
        f.write(analyze.to_latex_table(tex_entries, keys=("label",), metrics=ms["tex"]) + "\n")
    _write_json(os.path.join(d, "summary.json"), {"title": title, "baseline": baseline_label, "pairing": pairing,
                                                  "n_resamples": n_resamples, "seed": seed,
                                                  "stats": stats, "differences_vs_baseline": diffs})

    return [{"experiment": title, "condition": lab, "pairing": pairing,
             "d_rps": diffs[lab]["reward_per_step"], "d_cap": diffs[lab]["captured"]} for lab in cond_labels]


# ---- experiment 1: baseline, reconciliation, pairing check ---------------------

def _trainer_reference(trainer_log):
    if not trainer_log or not os.path.exists(trainer_log):
        return None
    last = None
    with open(trainer_log) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def write_exp1(out_dir, results_dir, trainer_log, checkpoint, n_resamples, seed):
    base = _rows(results_dir, "exp1_base")
    variant = _rows(results_dir, "exp1c_fulldyn_variant")
    d = os.path.join(out_dir, "exp1_baseline")
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    if not base:
        with open(os.path.join(d, "results.md"), "w") as f:
            f.write("# Experiment 1: baseline\n\nNo data yet.\n")
        return None

    label = "full_dynamics (dynamic scenario)"
    stats = _summ({label: base}, ABS_METRICS, n_resamples, seed)
    chunks = {}
    for r in base:
        chunks.setdefault("seeds " + ("1,000,000-1,000,499" if r["seed"] < 1_000_500 else "1,000,500-1,000,999"), []).append(r)
    chunk_stats = _summ(chunks, ["reward_per_step", "captured", "episode_return"], n_resamples, seed)

    extra = ["## Stability across the two 500-episode halves", "",
             "The 1,000 episodes were run as two 500-episode jobs. Agreement between the halves is a check "
             "that the estimate is not driven by a few episodes.", "",
             _stats_md(chunk_stats, list(chunks), ["reward_per_step", "captured", "episode_return"], _cell_ci), ""]

    ref = _trainer_reference(trainer_log)
    if ref:
        s = stats[label]
        extra += ["## Reconciliation with the trainer's own evaluation", "",
                  "The trainer evaluated the final step with 64 parallel environments, stopping after about 100 "
                  "finished episodes. That loop over-represents short episodes and is not seed-paired (flaws 1 and 2 "
                  "in the plan). It is shown for comparison only.", "",
                  _table(["source", "reward per step", "episode return", "captured", "episode length"], [
                      ["trainer eval_tst, final step (dynamic scenario)", f"{ref['eval_tst']['reward_avg']:.3f}",
                       f"{ref['eval_tst']['reward_avg_episodes']:.1f}", f"{ref['eval_tst']['captured_avg']:.2f}",
                       f"{ref['eval_tst']['eplen_avg']:.1f}"],
                      ["harness, this report (1,000 episodes)", _cell_ci(s["reward_per_step"], "reward_per_step"),
                       _cell_ci(s["episode_return"], "episode_return"), _cell_ci(s["captured"], "captured"),
                       f"{np.mean([r['episode_len'] for r in base]):.1f}"]]), ""]

    if variant:
        b_al, g_al = _align(base, {"v": variant})
        v_al = g_al["v"]
        by_seed = {r["seed"]: r for r in b_al}
        identical = sum(1 for r in v_al if r["episode_return"] == by_seed[r["seed"]]["episode_return"]
                        and r["captured"] == by_seed[r["seed"]]["captured"]
                        and r["episode_len"] == by_seed[r["seed"]]["episode_len"])
        pd_ret = analyze.paired_difference(v_al, b_al, "episode_return", n_resamples=n_resamples, seed=seed)
        extra += ["## Pairing check: base YAML vs regenerated `full_dynamics` variant file", "",
                  "Experiments 4 and 5 use variant YAMLs written by `make_variants.py`. The `full_dynamics` variant "
                  "should be the same environment as the base scenario. Same seeds, same checkpoint:", "",
                  _table(["episodes compared", "episodes with identical return, captured and length", "mean return difference [95% CI]"],
                         [[str(len(v_al)), f"{identical} ({100.0 * identical / max(1, len(v_al)):.1f}%)",
                           _cell_diff({"mean_diff": pd_ret["mean_diff"], "ci_lo": pd_ret["ci_lo"], "ci_hi": pd_ret["ci_hi"],
                                       "n": pd_ret["n_pairs"]}, "episode_return")]]), ""]
        if identical == len(v_al):
            extra += ["Every episode matches exactly: the variant file is behaviourally identical to the base "
                      "scenario, so Experiments 4 and 5 baselines are valid for the base scenario too.", ""]
        else:
            extra += ["The two files do NOT produce identical episodes. Experiments 4 and 5 are therefore paired "
                      "against the variant file's own `full_dynamics` run, not against the base-scenario run.", ""]

    md = ["# Experiment 1: baseline on the dynamic scenario", "",
          f"- **Checkpoint:** {checkpoint}", f"- **Scenario:** corp_100hosts_dynamic (full dynamics, hardest curriculum stage)",
          f"- **Episodes:** {len(base)}, seeds 1,000,000 to 1,000,000+N-1, step limit 400, actions sampled",
          f"- **Interval:** 95% percentile bootstrap, {n_resamples:,} resamples, bootstrap seed {seed}", "",
          "## What this measures", "",
          "The reference every other experiment is compared against: the checkpoint on the environment it was "
          "trained on, with exactly N complete episodes per seed (no length bias) and a fixed seed bank.", "",
          "## Absolute results: mean [95% CI]", "", _stats_md(stats, [label], MAIN_METRICS, _cell_ci), "",
          "Cumulative reward and captures at fixed action counts:", "",
          _stats_md(stats, [label], MILESTONE_METRICS, _cell_ci), "",
          "## Medians [Q1, Q3]", "", _stats_md(stats, [label], MEDIAN_METRICS, _cell_median), ""] + extra
    plots = []
    for m in ("episode_return", "captured", "reward_per_step"):
        p = f"plots/{m}_hist.png"
        _plot_hist(base, m, os.path.join(d, p), f"Baseline: distribution of {m}")
        plots.append(p)
    md += ["## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", "---", WORDING, ""]
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("\n".join(md))
    tex = [{"label": label, "n_episodes": len(base), "metrics": stats[label]}]
    with open(os.path.join(d, "summary_table.tex"), "w") as f:
        f.write(analyze.to_latex_table(tex, keys=("label",), metrics=analyze.SUMMARY_TABLE_METRICS) + "\n")
    _write_json(os.path.join(d, "summary.json"), {"stats": stats, "halves": chunk_stats, "trainer_reference": ref})
    return []


# ---- identity check used by the experiment 8 controls ----------------------------------

def _identity_md(baseline, groups, base_key=None, group_key=None, note=""):
    """How many episodes are bit-for-bit the same as the baseline's on the same
    seed. A control that should change nothing must score 100%."""
    base_key = base_key or (lambda r: (round(r["episode_return"], 6), r["captured"], r["episode_len"]))
    group_key = group_key or base_key
    b, g = _align(baseline, groups)
    by_seed = {r["seed"]: base_key(r) for r in b}
    rows = []
    for label, rs in g.items():
        same = sum(1 for r in rs if group_key(r) == by_seed[r["seed"]])
        rows.append([label, str(len(rs)), f"{same} ({100.0 * same / max(1, len(rs)):.1f}%)"])
    out = ["## Bit-for-bit identity check", "",
           note or "Episodes whose return, captured count and length all equal the baseline's on the same seed.", "",
           _table(["condition", "episodes compared", "identical to baseline"], rows), ""]
    return "\n".join(out)


# ---- experiment 5 parameter table ------------------------------------------------

def _intensity_parameter_table(variants_dir):
    from experiments.make_variants import final_stage, load_scenario
    rows = []
    for name in ("off", "low", "final", "harder"):
        path = os.path.join(variants_dir, f"corp_100hosts_dynamic__intensity_{name}.v2.yaml")
        if not os.path.exists(path):
            continue
        st = final_stage(load_scenario(path))
        ids = st.get("ids", {})
        noise = st.get("scan_noise", {})
        fp = "/".join(f"{noise.get(t, {}).get('false_positive_rate', 0):g}" for t in ("service_scan", "os_scan", "process_scan"))
        fn = "/".join(f"{noise.get(t, {}).get('false_negative_rate', 0):g}" for t in ("service_scan", "os_scan", "process_scan"))
        thr = ", ".join(f"{x:.3f}" for x in ids.get("base_thresholds", []))
        rows.append([f"intensity_{name}", str(ids.get("enabled")), thr or "-", fp, fn,
                     f"{st.get('service_dynamics', {}).get('churn_probability', 0):g}",
                     f"{st.get('network_reliability', {}).get('timeout_probability', 0):g}"])
    if not rows:
        return ""
    return "\n".join(["## Full parameter values per intensity level", "",
                      _table(["level", "IDS enabled", "IDS base thresholds", "scan false-positive (service/os/process)",
                              "scan false-negative (service/os/process)", "churn probability", "timeout probability"], rows), ""])


# ---- experiment 6: generalization --------------------------------------------------

def _scenario_roles(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        m = json.load(f)
    return m["checkpoints"][0].get("scenario_roles", {})


# ---- experiment 7: variant verification ----------------------------------------------

def exp7_summary_rows(machine):
    """One summary row per variant: [name, 'k/n in CI', zero-config check, verdict, IDS detections]."""
    summary = []
    for name, data in machine.items():
        rows = data["rows"]
        tested = [r for r in rows if r["measured_n"]]
        in_ci = sum(1 for r in tested if r["configured_in_ci"])
        zero_rows = [r for r in tested if r["configured"] == 0.0]
        zero_fail = [r["label"] for r in zero_rows if r["measured_k"] != 0]
        ids_rows = [r for r in rows if r["label"].startswith("ids.response_types.")]
        detections = ids_rows[0]["measured_n"] if ids_rows else 0
        summary.append([name, f"{in_ci}/{len(tested)}",
                        f"{len(zero_rows) - len(zero_fail)}/{len(zero_rows)}" if zero_rows else "-",
                        "PASS" if not zero_fail else "FAIL: " + ", ".join(zero_fail),
                        str(detections)])
    return summary


def write_exp7(out_dir, results_dir):
    d = os.path.join(out_dir, "exp7_variant_verification")
    os.makedirs(d, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(results_dir, "exp7_verify_*.json")))
    if not paths:
        with open(os.path.join(d, "results.md"), "w") as f:
            f.write("# Experiment 7: variant verification\n\nNo data yet.\n")
        return
    machine = {}
    for path in paths:
        name = os.path.basename(path)[len("exp7_verify_"):-len(".json")]
        with open(path) as f:
            machine[name] = json.load(f)
    summary = exp7_summary_rows(machine)
    detail = []
    for name, data in machine.items():
        rows = data["rows"]
        detail += [f"### {name}", "",
                   _table(["mechanism", "configured", "measured", "trials", "95% CI", "configured in CI"],
                          [[r["label"], f"{r['configured']:.4f}", "n/a" if not r["measured_n"] else f"{r['measured_rate']:.4f}",
                            str(r["measured_n"]),
                            "n/a" if not r["measured_n"] else f"[{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}]",
                            "n/a" if not r["measured_n"] else ("yes" if r["configured_in_ci"] else "NO")] for r in rows]), ""]
    md = ["# Experiment 7: variant verification", "",
          "Each generated scenario variant was run with scripted actions (no policy) and every configured rate "
          "compared to the rate the simulator actually produced. A knocked-out mechanism must produce zero events; "
          "an incorrect YAML key would show up here as a non-zero count.", "",
          f"Episodes per variant: {next(iter(machine.values()))['n_episodes']}, step limit {next(iter(machine.values()))['step_limit']}, "
          "scripted action cycle (see `validate_dynamics.py`).", "",
          "## Summary", "",
          _table(["variant", "configured rates inside 95% CI", "zero-configured mechanisms with zero events", "zero-event check",
                  "IDS detections observed"], summary), "",
          "Reading the table: 'zero-configured mechanisms' are rows whose configured rate is 0 (scan noise, churn, "
          "timeouts in the knock-out variants). 'IDS detections observed' must be 0 for variants with the IDS "
          "disabled and positive for variants with it enabled. Rates inside the CI are expected to miss for about "
          "1 in 20 rows by chance alone.", "",
          "## Per-variant detail", ""] + detail + ["---", WORDING, ""]
    with open(os.path.join(d, "results.md"), "w") as f:
        f.write("\n".join(md))
    _write_json(os.path.join(d, "summary.json"), machine)


# ---- top-level driver ----------------------------------------------------------------

def build_report(results_dir, out_dir, manifest_path=None, trainer_log=None, variants_dir=None,
                 checkpoint="pendhrl_c0_seed1_final (Pen-DHRL baseline, no LLM shaping/distillation, seed 1, final step 20,000)",
                 n_resamples=10000, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(checkpoint=checkpoint, n_resamples=n_resamples, seed=seed)
    headline = []

    exp1_base = _rows(results_dir, "exp1_base")
    variant_base = _rows(results_dir, "exp1c_fulldyn_variant")

    write_exp1(out_dir, results_dir, trainer_log, checkpoint, n_resamples, seed)

    def add(res):
        if res:
            headline.extend(res)

    add(write_experiment(
        out_dir, "exp2_ids_observation", "Experiment 2: IDS observation sensitivity",
        "The same checkpoint on the same environment, with only the three IDS features (detection level, threshold, "
        "multiplier) that reach the network edited. `zeroed` sets them to 0, `shuffled` permutes the (level, multiplier) "
        "pairs among the hosts of one observation, `alert_only` keeps only a detected/not-detected signal, `delayed` "
        "shows the previous step's values. The environment and the model weights are unchanged. A large drop means "
        "the checkpoint relies on the IDS observations.",
        "normal (= Experiment 1 on the same seeds)", exp1_base,
        {c: _rows(results_dir, f"exp2_{c}") for c in ("zeroed", "shuffled", "alert_only", "delayed")},
        notes=["These are observation interventions on a fixed checkpoint. They show functional reliance on the "
               "features, not what a policy trained without them would learn."], **kw))

    add(write_experiment(
        out_dir, "exp3_component_interventions", "Experiment 3: post-hoc Pen-DHRL component interventions",
        "The same checkpoint with one inference-time mechanism switched off at a time: the IDS-bias branch, the recurrent "
        "memory, subgoal persistence, goal conditioning of the worker, and the learned stopping decision. Weights and "
        "tensor shapes are unchanged. This measures how much the trained checkpoint functionally depends on each "
        "component.",
        "full_checkpoint (= Experiment 1 on the same seeds)", exp1_base,
        {c: _rows(results_dir, f"exp3_{c}") for c in ("no_ids_branch", "no_recurrent_memory", "no_goal_persistence",
                                                       "no_goal_conditioning", "no_learned_stopping")},
        notes=["Post-hoc interventions do NOT establish the learning contribution of a component: a model trained without "
               "it might learn as well. Do not call these retrained ablations.",
               "If the checkpoint never chooses to stop (stop_rate 0 in the baseline), `no_learned_stopping` is expected "
               "to match the baseline; that outcome checks the intervention rather than reporting an effect."], **kw))

    add(write_experiment(
        out_dir, "exp4_mechanism_knockouts", "Experiment 4: environmental mechanism knock-outs",
        "The same checkpoint evaluated in variants of the scenario where one dynamic mechanism is removed from the final "
        "curriculum stage: no IDS, no scan noise, no service churn, no network timeouts, and all four off. Each variant "
        "is verified in Experiment 7. This estimates the checkpoint's sensitivity to each environmental mechanism.",
        "full_dynamics (variant file)", variant_base,
        {c: _rows(results_dir, f"exp4_{c}") for c in ("no_ids", "no_scan_noise", "no_churn", "no_timeouts", "all_off")},
        notes=["Environmental ablations of an evaluation environment for a fixed policy; not retrained ablations."], **kw))

    intensity_groups = {c: _rows(results_dir, f"exp5_{c}") for c in ("intensity_off", "intensity_low", "intensity_harder")}
    intensity_groups["intensity_final (= full_dynamics)"] = variant_base
    order = ["intensity_off", "intensity_low", "intensity_final (= full_dynamics)", "intensity_harder"]
    ordered = {k: intensity_groups[k] for k in order}
    add(write_experiment(
        out_dir, "exp5_intensity_sweep", "Experiment 5: defense-intensity sweep",
        "The same checkpoint at four defense intensities: everything off, the authored low stage, the final stage "
        "(the one the checkpoint is normally evaluated on), and a synthetic harder setting (1.5x scan-noise, churn and "
        "timeout rates, IDS thresholds x0.8). Reading left to right shows how performance degrades as defenses strengthen.",
        "intensity_final (= full_dynamics)", variant_base,
        {k: v for k, v in ordered.items() if k != "intensity_final (= full_dynamics)"},
        extra_md=_intensity_parameter_table(variants_dir) if variants_dir else "",
        curve_order=order, curve_xlabel="defense intensity",
        notes=["Only three stages are authored (baseline, medium, full difficulty); the fourth level is synthetic and is "
               "labelled with its full parameter values above.",
               "intensity_final is byte-identical to the full_dynamics variant, so it reuses that run."], **kw))

    # Experiment 6: generalization, unpaired against the dynamic scenario
    roles = _scenario_roles(manifest_path)
    tag = lambda short: ("held-out" if roles.get(short, {}).get("held_out") else "trained on") if roles else "role unknown"  # noqa: E731
    gen_groups = {}
    for short, name in (("corp_100hosts_dynamic_varA", "exp6_varA"), ("corp_100hosts_dynamic_varB", "exp6_varB"),
                        ("corp_100hosts_dynamic_bridge", "exp6_bridge"), ("corp_100hosts_dynamic_test", "exp6_test")):
        gen_groups[f"{short.replace('corp_100hosts_dynamic_', '')} ({tag(short)})"] = _rows(results_dir, name)
    add(write_experiment(
        out_dir, "exp6_generalization", "Experiment 6: generalization across scenarios",
        "The checkpoint evaluated on scenarios other than its base training scenario. Each scenario is labelled "
        "held-out or trained-on from the checkpoint's provenance manifest. Only held-out scenarios can support "
        "a generalization claim; the trained-on ones show how the checkpoint behaves on scenario variants it saw.",
        f"dynamic ({tag('corp_100hosts_dynamic')}, selection scenario)", exp1_base, gen_groups, pairing="unpaired", align=False,
        notes=["Different scenarios generate different networks even for the same seed, so episodes cannot be paired "
               "across scenarios; differences are unpaired bootstrap differences of means.",
               "The base dynamic scenario is in the training list. varA and varB were training scenarios. bridge and test "
               "were not in the training list under this run's command line (the plan text is stricter and names only "
               "`test`)."], **kw))

    write_exp7(out_dir, results_dir)

    # Experiment 8: controls and robustness checks (paired on the same seed bank)
    e8 = "exp8_controls_and_robustness"
    plumb = {"normal (ids_condition normal)": _rows(results_dir, "exp8_plumb_normal"),
             "full_checkpoint (net_condition full_checkpoint)": _rows(results_dir, "exp8_plumb_full_checkpoint")}
    add(write_experiment(
        out_dir, f"{e8}/8a_plumbing_controls", "Experiment 8a: plumbing controls",
        "The two reference conditions of Experiments 2 and 3 (`normal` and `full_checkpoint`) are meant to change nothing. "
        "This run pushes them through the intervention code paths anyway. If the wiring is correct, every episode is "
        "identical to the plain baseline on the same seed; anything else would mean the intervention machinery itself alters "
        "behaviour, which would contaminate every other comparison.",
        "Experiment 1 (no intervention)", exp1_base, plumb,
        extra_md=_identity_md(exp1_base, {k: v for k, v in plumb.items() if v}) if any(plumb.values()) and exp1_base else "",
        notes=["Expected result: 100% identical episodes and all differences exactly zero."], **kw))

    noids_groups = {"zeroed IDS observation, IDS disabled": _rows(results_dir, "exp8_ctrl_zeroed_on_no_ids"),
                    "no_ids_branch, IDS disabled": _rows(results_dir, "exp8_ctrl_no_ids_branch_on_no_ids")}
    no_ids_rows = _rows(results_dir, "exp4_no_ids")
    add(write_experiment(
        out_dir, f"{e8}/8b_ids_attribution_controls", "Experiment 8b: IDS attribution controls",
        "Experiments 2 and 3 found that removing the IDS signal changes behaviour. If the effect really comes from the IDS "
        "information, it should disappear when the environment has no IDS. Here the two IDS-removing interventions are "
        "applied in the `no_ids` variant, where the IDS features carry no information. Baseline: the same checkpoint in "
        "the `no_ids` variant with no intervention (Experiment 4).",
        "no_ids variant, no intervention (Experiment 4)", no_ids_rows, noids_groups,
        extra_md=_identity_md(no_ids_rows, {k: v for k, v in noids_groups.items() if v}) if any(noids_groups.values()) and no_ids_rows else "",
        notes=["A difference near zero here supports reading the Experiment 2/3 effects as reliance on IDS information. "
               "A large difference would mean the intervention disturbs the network for a reason other than losing IDS information "
               "(for example an out-of-distribution input)."], **kw))

    shuffle = {"shuffled, shuffle seed 0 (Experiment 2)": _rows(results_dir, "exp2_shuffled"),
               "shuffled, shuffle seed 1": _rows(results_dir, "exp8_shuffled_seed1"),
               "shuffled, shuffle seed 2": _rows(results_dir, "exp8_shuffled_seed2")}
    add(write_experiment(
        out_dir, f"{e8}/8c_shuffle_seed_robustness", "Experiment 8c: shuffled-IDS robustness to the shuffle draw",
        "The `shuffled` condition permutes IDS values among the hosts of each observation using a random generator. "
        "Experiment 2 used one shuffle seed. Repeating with two more shows whether the result depends on that particular "
        "random draw.", "normal (Experiment 1 on the same seeds)", exp1_base, shuffle,
        notes=["Similar differences across the three shuffle seeds mean the Experiment 2 result is not an artifact of one draw."], **kw))

    horizon = {"horizon_800 (step limit 800)": _rows(results_dir, "exp8_horizon800")}
    h_extra = ""
    if any(horizon.values()) and exp1_base:
        h_extra = _identity_md(
            exp1_base, {k: v for k, v in horizon.items() if v},
            base_key=lambda r: (round(r["episode_return"], 6), r["captured"]),
            group_key=lambda r: (round(r["reward_at_400"], 6), r["captured_at_400"]),
            note="The 800-step run's cumulative reward and captures at step 400, against the baseline's final values "
                 "(its episodes end at step 400). If the step limit does not influence the policy, these are identical.")
    write_experiment(  # not in the headline table: its reward per step has a different denominator
        out_dir, f"{e8}/8d_longer_horizon", "Experiment 8d: longer horizon (800 steps)",
        "The checkpoint was trained and evaluated with a 400-step limit and never chooses to stop. Running the same seeds to 800 "
        "steps shows whether it keeps collecting reward and captures after step 400 or has plateaued.",
        "baseline, 400-step limit (Experiment 1)", exp1_base, horizon, extra_md=h_extra,
        notes=["Only `return_at_400` and `captured_at_400` are comparable with the baseline. `reward_per_step`, `episode_return` "
               "and `captured` in the 800-step row are measured over 800 steps and differ from the 400-step baseline by "
               "construction; the differences in those columns are not an effect.",
               "The trained step limit is 400, so behaviour after step 400 is out of the training distribution."], **kw)

    lines = ["# Evaluation results: Pen-DHRL baseline (C0), single checkpoint", "",
             f"Checkpoint: {checkpoint}", "",
             "Generated by `NASimEmu-agents/experiments/make_report.py` from the raw per-episode CSVs in "
             f"`{results_dir}`. One folder per experiment; each has `results.md` (tables and plots), `summary_table.tex`, "
             "`summary.json` and `plots/`.", "",
             "**Start with [FINDINGS.md](FINDINGS.md)**: the readings, the caveats, and two simulator findings.", "",
             "| folder | experiment |", "|---|---|",
             "| [exp1_baseline](exp1_baseline/results.md) | baseline on the dynamic scenario, reconciliation with the trainer's eval, pairing check |",
             "| [exp2_ids_observation](exp2_ids_observation/results.md) | IDS observation sensitivity |",
             "| [exp3_component_interventions](exp3_component_interventions/results.md) | post-hoc Pen-DHRL component interventions |",
             "| [exp4_mechanism_knockouts](exp4_mechanism_knockouts/results.md) | environmental mechanism knock-outs |",
             "| [exp5_intensity_sweep](exp5_intensity_sweep/results.md) | defense-intensity sweep |",
             "| [exp6_generalization](exp6_generalization/results.md) | generalization across scenarios |",
             "| [exp7_variant_verification](exp7_variant_verification/results.md) | verification that each variant does what its name says |",
             "| [8a plumbing controls](exp8_controls_and_robustness/8a_plumbing_controls/results.md) | `normal` and `full_checkpoint` must equal the baseline exactly |",
             "| [8b IDS attribution controls](exp8_controls_and_robustness/8b_ids_attribution_controls/results.md) | IDS interventions applied where the IDS is disabled |",
             "| [8c shuffle-seed robustness](exp8_controls_and_robustness/8c_shuffle_seed_robustness/results.md) | shuffled IDS with three different shuffle draws |",
             "| [8d longer horizon](exp8_controls_and_robustness/8d_longer_horizon/results.md) | the same seeds run to 800 steps |", ""]
    if headline:
        rows = [[h["experiment"].split(":")[0], h["condition"], _cell_diff(h["d_rps"], "reward_per_step"),
                 _cell_diff(h["d_cap"], "captured"), h["d_rps"]["n_label"]] for h in headline]
        lines += ["## Headline: difference from baseline", "",
                  "Reward per step and hosts captured, condition minus baseline, 95% bootstrap CI. `*` = CI excludes zero.", "",
                  _table(["experiment", "condition", "reward per step", "captured", "n"], rows), ""]
    lines += ["## How to read these results", "",
              f"- {WORDING}",
              "- Actions are sampled from the policy (`net.eval()` still samples the host/action choice); the torch seed is set "
              "per episode so reruns are bit-reproducible. Deterministic best-action evaluation is not reported.",
              "- Each seed contributes exactly one complete episode, so there is no length bias. Episode k starts from the same "
              "generated network for every condition; later randomness depends on the actions taken, so the pairing is "
              "'same starting network', not 'identical episodes'.",
              "- Confidence intervals are unadjusted. Many conditions and metrics are compared, so a `*` is descriptive.",
              "- Interventions and knock-outs on a fixed checkpoint measure sensitivity of this policy. They are not retrained "
              "ablations and say nothing about what a differently trained policy would learn.",
              "- Not available with one checkpoint: comparison with other architectures, comparison with Pen-DHRL-TD, and any "
              "statement about variability across training seeds.", ""]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    _write_json(os.path.join(out_dir, "report_meta.json"),
                {"results_dir": results_dir, "n_resamples": n_resamples, "bootstrap_seed": seed, "checkpoint": checkpoint})
    return headline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=os.path.join(AGENTS_DIR, "results", "2026-09-24"))
    ap.add_argument("--out_dir", default=os.path.join(AGENTS_DIR, "..", "docs", "eval_results_2026-09-24"))
    ap.add_argument("--manifest", default=os.path.join(AGENTS_DIR, "eval_checkpoints", "manifest.json"))
    ap.add_argument("--trainer_log", default=os.path.join(AGENTS_DIR, "training_data", "runs", "g9t68f9h.json"))
    ap.add_argument("--variants_dir", default=os.path.join(AGENTS_DIR, "..", "scenarios", "variants"))
    ap.add_argument("--n_resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    headline = build_report(args.results_dir, args.out_dir, manifest_path=args.manifest,
                            trainer_log=args.trainer_log, variants_dir=args.variants_dir,
                            n_resamples=args.n_resamples, seed=args.seed)
    print(f"wrote report to {os.path.abspath(args.out_dir)} ({len(headline)} headline rows)")


if __name__ == "__main__":
    main()
