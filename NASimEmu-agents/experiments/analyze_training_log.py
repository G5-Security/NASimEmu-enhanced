"""Follow-up H1: analysis of the training log alone (no evaluation compute).

Reads training_data/runs/<run>.json, a JSONL log with one line per epoch
(100 training steps): trainer-side evaluation on the training mixture and on the
selection scenario, the manager's goal histogram (a fixed sample of goal choices
per epoch) and mean subgoal entropy. It splits the run at the curriculum-stage
boundaries read from the scenario YAML and reports what changed there.

Caveats stated in every output: this is ONE run, so the spread across epochs is
within-run variation and NOT variability across training seeds; trainer-side
evaluation uses a small number of episodes.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAVEAT = ("One training run: spread across epochs is within-run variation, not variability across training seeds. "
          "Trainer-side evaluation uses a small number of episodes per epoch.")


def load_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stage_windows(rows, stages):
    """[(name, lo, hi)] as epoch index ranges; `stages` = [(name, start_frac, end_frac)]."""
    n = len(rows)
    return [(name, int(round(a * n)), int(round(b * n))) for name, a, b in stages]


def _mean(xs):
    return st.mean(xs) if xs else float("nan")


def summarize_window(rows, lo, hi):
    rs = rows[lo:hi]
    goals = list(rs[0]["goal_hist"])
    total = sum(sum(r["goal_hist"].values()) for r in rs)
    return {
        "epochs": (lo, hi),
        "goal_share": {g: sum(r["goal_hist"][g] for r in rs) / total for g in goals},
        "manager_entropy": _mean([r["manager_entropy"] for r in rs]),
        "eval_tst_reward_per_step": _mean([r["eval_tst"]["reward_avg"] for r in rs]),
        "eval_tst_captured": _mean([r["eval_tst"]["captured_avg"] for r in rs]),
        "eval_trn_reward_per_step": _mean([r["eval_trn"]["reward_avg"] for r in rs]),
        "eval_trn_captured": _mean([r["eval_trn"]["captured_avg"] for r in rs]),
    }


def plateau_spread(rows, start):
    """Spread of trainer-side eval across epochs from `start` on (within one run)."""
    out = {}
    for key, getter in (("eval_tst reward/step", lambda r: r["eval_tst"]["reward_avg"]),
                        ("eval_tst captured", lambda r: r["eval_tst"]["captured_avg"]),
                        ("eval_trn reward/step", lambda r: r["eval_trn"]["reward_avg"]),
                        ("eval_trn captured", lambda r: r["eval_trn"]["captured_avg"])):
        xs = [getter(r) for r in rows[start:]]
        out[key] = {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs), "n": len(xs)}
    return out


def collapsed_epochs(rows, start, frac=0.8):
    """Epochs from `start` on whose trainer-side eval_tst captured count fell below
    `frac` of the median over those epochs: transient collapses during training."""
    med = st.median(r["eval_tst"]["captured_avg"] for r in rows[start:])
    return [{"epoch": i, "captured": r["eval_tst"]["captured_avg"], "reward_per_step": r["eval_tst"]["reward_avg"]}
            for i, r in enumerate(rows) if i >= start and r["eval_tst"]["captured_avg"] < frac * med], med


def _bounds(windows):
    return [lo for _n, lo, _hi in windows[1:]]


def plot_learning_curves(rows, windows, out_png):
    ep = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, label in ((axes[0], "reward_avg", "reward per step"), (axes[1], "captured_avg", "hosts captured")):
        ax.plot(ep, [r["eval_tst"][key] for r in rows], label="eval_tst (selection scenario)")
        ax.plot(ep, [r["eval_trn"][key] for r in rows], label="eval_trn (training mixture)")
        for b in _bounds(windows):
            ax.axvline(b, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("epoch (100 steps each)")
        ax.set_ylabel(label)
    axes[0].legend(loc="lower right", fontsize=8)
    for name, lo, hi in windows:
        axes[0].text((lo + hi) / 2, axes[0].get_ylim()[1], name.replace("_difficulty", ""), ha="center", va="top", fontsize=8)
    fig.suptitle("Trainer-side evaluation over training, with curriculum stage boundaries", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_goal_usage(rows, windows, out_png):
    goals = list(rows[0]["goal_hist"])
    shares = [[r["goal_hist"][g] / sum(r["goal_hist"].values()) for r in rows] for g in goals]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(range(len(rows)), shares, labels=goals)
    for b in _bounds(windows):
        ax.axvline(b, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("share of goal choices")
    ax.set_xlim(0, len(rows) - 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    ax.set_title("Manager goal usage per epoch", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_entropy(rows, windows, out_png, n_goals):
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.plot([r["manager_entropy"] for r in rows], label="mean manager subgoal entropy")
    ax.axhline(math.log(n_goals), color="red", linestyle=":", label=f"uniform over {n_goals} goals (ln {n_goals})")
    for b in _bounds(windows):
        ax.axvline(b, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("entropy (nats)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build(log_path, out_dir, stages):
    rows = load_log(log_path)
    windows = stage_windows(rows, stages)
    os.makedirs(out_dir, exist_ok=True)
    win = {name: summarize_window(rows, lo, hi) for name, lo, hi in windows}
    plateau_start = windows[1][1] + 10 if len(windows) > 1 else len(rows) // 2
    spread = plateau_spread(rows, plateau_start)
    collapses, collapse_median = collapsed_epochs(rows, plateau_start)
    n_goals = len(rows[0]["goal_hist"])
    plot_learning_curves(rows, windows, os.path.join(out_dir, "learning_curves.png"))
    plot_goal_usage(rows, windows, os.path.join(out_dir, "goal_usage.png"))
    plot_entropy(rows, windows, os.path.join(out_dir, "manager_entropy.png"), n_goals)

    goals = list(rows[0]["goal_hist"])
    md = ["# Training-log analysis", "", f"Log: `{os.path.basename(log_path)}`, {len(rows)} epochs. {CAVEAT}", "",
          "## Curriculum stages", "",
          "| stage | epochs | IDS |", "|---|---|---|"]
    for (name, lo, hi), (_n, _a, _b, ids_on) in zip(windows, stages_with_ids(stages)):
        md.append(f"| {name} | {lo}-{hi - 1} | {'on' if ids_on else 'off'} |")
    md += ["", "## Behaviour by stage", "",
           "| stage | " + " | ".join(g.lower() for g in goals) + " | manager entropy | eval_tst reward/step | eval_tst captured |",
           "|---|" + "---|" * (len(goals) + 3)]
    for name, w in win.items():
        md.append(f"| {name} | " + " | ".join(f"{100 * w['goal_share'][g]:.1f}%" for g in goals) +
                  f" | {w['manager_entropy']:.3f} | {w['eval_tst_reward_per_step']:.3f} | {w['eval_tst_captured']:.2f} |")
    md += ["", f"Maximum possible manager entropy for {n_goals} goals is ln({n_goals}) = {math.log(n_goals):.3f}. "
           "Goal shares are shares of a fixed-size sample of goal choices per epoch.", "",
           f"## Spread across epochs {plateau_start}-{len(rows) - 1} (within one run)", "",
           "| metric | mean | sd | min | max |", "|---|---|---|---|---|"]
    for k, v in spread.items():
        md.append(f"| {k} | {v['mean']:.3f} | {v['sd']:.3f} | {v['min']:.3f} | {v['max']:.3f} |")
    md += ["", "This is variation across epochs of a single run, **not** variability across training seeds.", ""]
    md += [f"## Transient collapses (eval_tst captured below 80% of the median {collapse_median:.1f})", ""]
    if collapses:
        stage_starts = ", ".join(f"{n} at epoch {lo}" for n, lo, _hi in windows[1:])
        md += ["| epoch | captured | reward/step |", "|---|---|---|"]
        md += [f"| {c['epoch']} | {c['captured']:.2f} | {c['reward_per_step']:.3f} |" for c in collapses]
        md += ["", f"{len(collapses)} of {len(rows) - plateau_start} epochs. Stage boundaries: {stage_starts}."]
    else:
        md += ["None."]
    md += [""]
    md += [
           "![learning curves](learning_curves.png)", "", "![goal usage](goal_usage.png)", "", "![manager entropy](manager_entropy.png)", ""]
    with open(os.path.join(out_dir, "training_log_summary.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(out_dir, "training_log_summary.json"), "w") as f:
        json.dump({"windows": win, "plateau_spread": spread, "plateau_start_epoch": plateau_start,
                   "collapses": collapses, "collapse_median_captured": collapse_median,
                   "max_entropy": math.log(n_goals)}, f, indent=2, default=list)
    return {"windows": win, "plateau_spread": spread, "collapses": collapses}


def stages_with_ids(stages):
    return [(n, a, b, ids) for (n, a, b), ids in zip(stages, _STAGE_IDS[: len(stages)])]


_STAGE_IDS = [False, True, True]


def stages_from_scenario(path):
    from experiments.make_variants import load_scenario
    global _STAGE_IDS
    stages = load_scenario(path)["curriculum"]["stages"]
    _STAGE_IDS = [bool(s.get("ids", {}).get("enabled")) for s in stages]
    return [(s["name"], s["start_frac"], s["end_frac"]) for s in stages]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=os.path.join(AGENTS_DIR, "training_data", "runs", "g9t68f9h.json"))
    ap.add_argument("--scenario", default=os.path.join(AGENTS_DIR, "..", "scenarios", "corp_100hosts_dynamic.v2.yaml"))
    ap.add_argument("--out_dir", default=os.path.join(AGENTS_DIR, "..", "docs", "eval_followup_2026-09-24", "H1_training_log"))
    args = ap.parse_args()
    build(args.log, args.out_dir, stages_from_scenario(args.scenario))
    print("wrote", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
