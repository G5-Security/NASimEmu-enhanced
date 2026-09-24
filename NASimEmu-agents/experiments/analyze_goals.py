"""Follow-up A: does the IDS signal drive the manager's goal choice?

Reads per-step goal traces written by `eval_harness.py --trace_goals` (one row per
step: the manager's held goal, the entropy of its goal distribution, and the IDS
detection level as the network sees it) for several conditions, and reports:

  * goal shares per detection-level bin (does the mix shift as detection rises?)
  * mutual information I(goal; level bin), in nats, with a permutation null (what
    plug-in MI looks like when goal and level are independent) and an
    episode-clustered bootstrap CI -- steps within an episode are not independent
  * how often the held goal changes, and the manager's mean goal entropy

Conditions are meant to be compared: the normal run, the run with IDS observations
zeroed, the run with the IDS-bias branch disabled, and a run in an environment with
the IDS turned off (where the level is always 0). If the IDS signal drives goal
choice, MI should be clearly above the null for the normal run and fall toward the
null when the signal is removed.
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_teacher.goal_ontology import GOAL_NAMES  # noqa: E402

LEVEL_EDGES = (0.0, 0.25, 0.5, 0.75)
BIN_LABELS = ("0", "(0, 0.25]", "(0.25, 0.5]", "(0.5, 0.75]", "> 0.75")
IDS_GOALS = ("REDUCE_DETECTION", "RECOVER_OR_REPLAN")
TIME_BIN = 50  # steps per stratum when controlling for time within the episode


def load_trace(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def level_bin(level):
    if level <= LEVEL_EDGES[0]:
        return 0
    for i, edge in enumerate(LEVEL_EDGES[1:], start=1):
        if level <= edge:
            return i
    return len(LEVEL_EDGES)


def mutual_information(goals, bins):
    """Plug-in MI (nats) between two equal-length integer sequences."""
    goals, bins = np.asarray(goals, dtype=int), np.asarray(bins, dtype=int)
    n = len(goals)
    if n == 0:
        return float("nan")
    joint = np.zeros((goals.max() + 1, bins.max() + 1))
    np.add.at(joint, (goals, bins), 1.0)
    joint /= n
    pg, pb = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pg @ pb)[nz])).sum())


def permutation_null(goals, bins, n_perm=200, seed=0):
    """MI values obtained when the goal sequence is shuffled against the bins."""
    rng = np.random.default_rng(seed)
    goals = np.asarray(goals)
    return np.array([mutual_information(rng.permutation(goals), bins) for _ in range(n_perm)])


def conditional_mutual_information(goals, bins, strata):
    """I(goal; level bin | stratum): the stratum-weighted average of the plug-in MI computed inside
    each stratum. Used with the step number as the stratum, because both the IDS level and the goal
    change over an episode, so a raw goal-level association can just reflect time."""
    goals, bins, strata = np.asarray(goals), np.asarray(bins), np.asarray(strata)
    total = 0.0
    for st in np.unique(strata):
        sel = strata == st
        total += sel.mean() * mutual_information(goals[sel], bins[sel])
    return float(total)


def stratified_permutation_null(goals, bins, strata, n_perm=100, seed=0):
    """Conditional-MI values when goals are shuffled only within each stratum."""
    rng = np.random.default_rng(seed)
    goals, strata = np.asarray(goals), np.asarray(strata)
    out = []
    for _ in range(n_perm):
        permuted = goals.copy()
        for st in np.unique(strata):
            idx = np.where(strata == st)[0]
            permuted[idx] = rng.permutation(goals[idx])
        out.append(conditional_mutual_information(permuted, bins, strata))
    return np.array(out)


def cluster_bootstrap_mi(rows, n_boot=300, seed=0):
    """95% CI for MI, resampling whole episodes (steps inside one are correlated)."""
    rng = np.random.default_rng(seed)
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    keys = list(by_seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        sample = [r for i in pick for r in by_seed[keys[i]]]
        vals.append(mutual_information([int(r["goal_idx"]) for r in sample], [level_bin(r["obs_max_level"]) for r in sample]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def switch_rate(rows):
    """Fraction of steps (after each episode's first) on which the held goal changed."""
    changes = total = 0
    prev = {}
    for r in rows:
        s = r["seed"]
        if s in prev:
            total += 1
            changes += int(prev[s] != r["goal_idx"])
        prev[s] = r["goal_idx"]
    return changes / total if total else float("nan")


def summarize(rows, n_perm=200, n_boot=300, seed=0):
    goals = [int(r["goal_idx"]) for r in rows]
    bins = [level_bin(r["obs_max_level"]) for r in rows]
    n_goals = len(GOAL_NAMES)
    share_by_bin = {}
    for b in range(len(BIN_LABELS)):
        sel = [g for g, bb in zip(goals, bins) if bb == b]
        share_by_bin[b] = {"n": len(sel), "share": [sel.count(k) / len(sel) if sel else float("nan") for k in range(n_goals)]}
    steps = [int(r["step"]) // TIME_BIN for r in rows]
    mi = mutual_information(goals, bins)
    mi_time = mutual_information(goals, steps)
    mi_given_time = conditional_mutual_information(goals, bins, steps)
    cond_null = stratified_permutation_null(goals, bins, steps, max(20, n_perm // 2), seed)
    null = permutation_null(goals, bins, n_perm, seed)
    lo, hi = cluster_bootstrap_mi(rows, n_boot, seed)
    ids_idx = [GOAL_NAMES.index(g) for g in IDS_GOALS]
    return {
        "n_steps": len(rows), "n_episodes": len({r["seed"] for r in rows}),
        "overall_share": [goals.count(k) / len(goals) for k in range(n_goals)],
        "share_by_bin": share_by_bin,
        "ids_goal_share_by_bin": {b: (sum(v["share"][k] for k in ids_idx) if v["n"] else float("nan")) for b, v in share_by_bin.items()},
        "mi": mi, "mi_null_mean": float(null.mean()), "mi_null_p95": float(np.quantile(null, 0.95)),
        "mi_goal_time": mi_time, "mi_given_time": mi_given_time,
        "mi_given_time_null_mean": float(cond_null.mean()), "mi_given_time_null_p95": float(np.quantile(cond_null, 0.95)),
        "mi_ci": (lo, hi), "mi_excess": mi - float(null.mean()),
        "mean_goal_entropy": float(np.mean([r["goal_entropy"] for r in rows])),
        "mean_goal_maxprob": float(np.mean([r["goal_maxprob"] for r in rows])),
        "switch_rate": switch_rate(rows),
        "frac_steps_level_above_0": float(np.mean([b > 0 for b in bins])),
    }


def write_report(traces, out_dir, n_perm=200, n_boot=300):
    """traces: {label: path}. Writes results.md, summary.json and plots into out_dir."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    summaries = {label: summarize(load_trace(path), n_perm, n_boot) for label, path in traces.items()}

    md = ["# A2: does the IDS signal drive the manager's goal choice?", "",
          "Per-step traces of the manager's held goal, with the IDS detection level as the network sees it "
          "(the highest level among the hosts in the observation). Steps within an episode are correlated, so "
          "the mutual-information interval resamples whole episodes.", "",
          "## Summary", "",
          "| condition | episodes | steps | MI(goal; level bin), nats [95% CI] | MI(goal; step) | MI(goal; level bin \\| step) (null 95th pct) | share of steps with level > 0 | goal switches per step | mean goal entropy (max 2.079) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for label, s in summaries.items():
        md.append(f"| {label} | {s['n_episodes']} | {s['n_steps']} | {s['mi']:.4f} [{s['mi_ci'][0]:.4f}, {s['mi_ci'][1]:.4f}] | "
                  f"{s['mi_goal_time']:.4f} | {s['mi_given_time']:.4f} ({s['mi_given_time_null_p95']:.4f}) | {100 * s['frac_steps_level_above_0']:.1f}% | "
                  f"{s['switch_rate']:.3f} | {s['mean_goal_entropy']:.3f} |")
    md += ["", "MI is in nats; the largest possible value here is ln(8) = 2.079. **The raw goal-level association is confounded by "
           f"time**: the IDS level rises over an episode, and so does the goal mix (MI(goal; step) is larger than MI(goal; level bin)). "
           f"The conditional column recomputes the association inside {TIME_BIN}-step blocks of the episode; its parenthesis is the "
           "95th percentile of the same quantity when goals are shuffled within each block, so only the excess over that means "
           "anything.", "",
           "## Share of goal choices by detection-level bin", ""]
    header = "| condition | bin | steps | " + " | ".join(g.lower() for g in GOAL_NAMES) + " |"
    md += [header, "|---|---|---|" + "---|" * len(GOAL_NAMES)]
    for label, s in summaries.items():
        for b, v in s["share_by_bin"].items():
            if v["n"]:
                md.append(f"| {label} | {BIN_LABELS[b]} | {v['n']} | " + " | ".join(f"{100 * x:.1f}%" for x in v["share"]) + " |")
    md += ["", "## Share of the two IDS-related goals (REDUCE_DETECTION + RECOVER_OR_REPLAN) by bin", "",
           "| condition | " + " | ".join(BIN_LABELS) + " |", "|---|" + "---|" * len(BIN_LABELS)]
    for label, s in summaries.items():
        md.append(f"| {label} | " + " | ".join("n/a" if math.isnan(s["ids_goal_share_by_bin"][b]) else f"{100 * s['ids_goal_share_by_bin'][b]:.1f}%"
                                              for b in range(len(BIN_LABELS))) + " |")

    n = len(summaries)
    cols = 2 if n > 2 else n
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.6 * cols, 3.3 * rows_n), sharey=True, squeeze=False)
    handles = None
    for ax, (label, s) in zip(axes.flat, summaries.items()):
        bottoms = np.zeros(len(BIN_LABELS))
        for k, name in enumerate(GOAL_NAMES):
            vals = np.array([0.0 if math.isnan(s["share_by_bin"][b]["share"][k]) else s["share_by_bin"][b]["share"][k] for b in range(len(BIN_LABELS))])
            ax.bar(range(len(BIN_LABELS)), vals, bottom=bottoms, label=name.lower())
            bottoms += vals
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, fontsize=8)
        ax.set_xlabel("IDS detection level seen by the network", fontsize=8)
        ax.set_title(label, fontsize=10)
        handles = ax.get_legend_handles_labels()
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of goal choices", fontsize=9)
    fig.legend(*handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Goal mix by IDS detection level", fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "plots", "goal_mix_by_level.png"), dpi=150)
    plt.close(fig)
    md += ["", "![goal mix by level](plots/goal_mix_by_level.png)", ""]
    with open(os.path.join(out_dir, "trace_analysis.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(out_dir, "trace_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=float)
    return summaries
