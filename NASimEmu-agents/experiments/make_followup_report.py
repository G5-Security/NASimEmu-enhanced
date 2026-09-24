"""Report generator for the follow-up experiments A-H (reviewer-driven, no retraining).

Reads two results directories -- the main evaluation (results/2026-09-24: Experiment 1
is the baseline every follow-up pairs against) and the follow-up runs
(results/2026-09-24_followup) -- and writes one folder per experiment under
docs/eval_followup_2026-09-24/, in the same format as the main report.

    A_goal_analysis/    A1 goal masks; A2 per-step goal traces vs IDS level
    B_ids_detections/   IDS detections per episode, and how much of the return change they explain
    C_ids_semantics/    written by make_ids_semantics.py
    D_parameter_sensitivity/
    E_scan_noise_observed/
    F_generated_scenarios/
    G_ids_delay/
    H_training_and_checkpoint/   H1 from analyze_training_log.py; H2 best vs final checkpoint

Missing CSVs are skipped and reported as "no data yet", so it runs on partial results.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import analyze, analyze_goals  # noqa: E402
from experiments import make_report as mr  # noqa: E402
from experiments.make_variants import SENSITIVITY_SPECS  # noqa: E402

AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = "pendhrl_c0_seed1_final (Pen-DHRL baseline, seed 1, final step 20,000)"

IDS_SETS = {"abs": ["reward_per_step", "episode_return", "captured", "ids_detections", "ids_quarantine", "ids_patch",
                    "ids_monitor", "ids_penalty", "detected_any", "ids_max_level"],
            "main": ["reward_per_step", "episode_return", "captured", "ids_detections", "ids_penalty", "detected_any"],
            "milestones": ["ids_quarantine", "ids_patch", "ids_monitor", "ids_max_level"],
            "second_heading": "IDS responses per episode and the highest detection level reached:",
            "medians": ["episode_return", "ids_detections"], "headline": ["reward_per_step", "ids_detections"],
            "tex": ["reward_per_step", "episode_return", "captured", "ids_detections", "ids_penalty"]}
PLAIN_SETS = {"abs": ["reward_per_step", "episode_return", "captured"], "main": ["reward_per_step", "episode_return", "captured"],
              "milestones": [], "medians": [], "headline": ["reward_per_step", "captured"],
              "tex": ["reward_per_step", "episode_return", "captured"]}
GEN_SETS = {"abs": ["reward_per_step", "episode_return", "captured", "captured_fraction", "n_sensitive", "n_hosts"],
            "main": ["reward_per_step", "captured", "captured_fraction", "n_sensitive", "n_hosts"],
            "milestones": [], "medians": [], "headline": ["captured_fraction", "reward_per_step"],
            "tex": ["reward_per_step", "captured", "captured_fraction"]}

# dose-response families for D: (title, [(x label, variant name or None for the baseline)], categorical?)
D_FAMILIES = [
    ("IDS threshold scale", [("x0.6", "ids_thr_x0.6"), ("x0.8", "ids_thr_x0.8"), ("x1 (trained)", None), ("x1.25", "ids_thr_x1.25"), ("x1.5", "ids_thr_x1.5")]),
    ("Detection increase scale", [("x0.5", "ids_inc_x0.5"), ("x1 (trained)", None), ("x2", "ids_inc_x2")]),
    ("Detection decay per action", [("0.90", "ids_decay_0.90"), ("0.95", "ids_decay_0.95"), ("0.98 (trained)", None), ("0.995", "ids_decay_0.995")]),
    ("Response mix", [("mix (trained)", None), ("all monitor", "resp_all_monitor"), ("all patch", "resp_all_patch"), ("all quarantine", "resp_all_quarantine")]),
    ("Failed-exploit multiplier", [("0", "fem_0"), ("1.25", "fem_x0.5"), ("2.5 (trained)", None)]),
    ("Churn / timeout scale", [("churn x1 (trained)", None), ("churn x2", "churn_x2"), ("churn x3", "churn_x3"),
                               ("timeout x2", "timeout_x2"), ("timeout x4", "timeout_x4")]),
]


def _load(results_dir, prefix, names):
    return {n: mr._rows(results_dir, f"{prefix}{n}") for n in names}


def _present(d):
    return {k: v for k, v in d.items() if v}


def penalty_explained_md(baseline, groups, n_resamples, seed):
    """How much of each condition's return change is the IDS penalties themselves."""
    b, g = mr._align(baseline, groups)
    rows = []
    for label, rs in g.items():
        dr = analyze.paired_difference(rs, b, "episode_return", n_resamples=n_resamples, seed=seed)
        dp = analyze.paired_difference(rs, b, "ids_penalty", n_resamples=n_resamples, seed=seed)
        ratio = (-dp["mean_diff"] / dr["mean_diff"]) if abs(dr["mean_diff"]) > 1e-9 else float("nan")
        rows.append([label, mr._cell_diff({**dr, "n": dr["n_pairs"]}, "episode_return"),
                     mr._cell_diff({**dp, "n": dp["n_pairs"]}, "ids_penalty"),
                     "n/a" if np.isnan(ratio) else f"{100 * ratio:.0f}%"])
    return "\n".join(["## How much of the return change is the IDS penalties themselves", "",
                      "Paired differences from the baseline. `ids penalty` is the reward the IDS took directly "
                      "(5 x quarantines + 2 x patches + 0.5 x monitors in return units: the raw penalties are 50, 20 and 5, "
                      "and the environment divides every reward by 10); the last column is the change in penalty "
                      "(sign-flipped) divided by the change in return. Quarantine also removes access, so penalties are "
                      "a lower bound on the IDS-related loss.", "",
                      mr._table(["condition", "change in return", "change in ids penalty", "penalty change / return change"], rows), ""])


def ids_control_md(rows):
    if not rows:
        return ""
    det = [r["ids_detections"] for r in rows]
    return "\n".join(["## Control: IDS disabled in the environment", "",
                      f"{len(rows)} episodes in the `no_ids` variant: {int(sum(det))} detections in total "
                      f"(maximum {int(max(det))} in one episode). Expected: 0.", ""])


def reference_table_md(ref_rows):
    if not ref_rows:
        return ""
    fns = {m: analyze.METRICS[m] for m in ("captured", "n_sensitive", "captured_fraction", "reward_per_step")}
    body = []
    for name, rows in ref_rows.items():
        s = analyze.summarize_group(rows, fns, n_resamples=2000, seed=0)
        body.append([name, str(len(rows))] + [mr._cell_ci(s[m], m) for m in fns])
    return "\n".join(["## Hand-made scenarios with their exact sensitive-host counts (200 episodes each)", "",
                      "The same fixed checkpoint on the five hand-made scenarios, with the number of capturable hosts "
                      "recorded per episode, so the captured fraction is exact rather than estimated from the YAML.", "",
                      mr._table(["scenario", "n", "captured", "sensitive hosts", "captured / sensitive", "reward / step"], body), ""])


def dose_response_plot(baseline_rows, variant_rows):
    """One panel per parameter family: reward per step (paired baseline restricted to the same seeds)."""
    def plot(path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        b, g = mr._align(baseline_rows, variant_rows)
        base_stat = mr._summ({"b": b}, ["reward_per_step"], 2000, 0)["b"]["reward_per_step"]
        stats = mr._summ(g, ["reward_per_step"], 2000, 0)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True)
        for ax, (title, points) in zip(axes.flat, D_FAMILIES):
            xs, ys, lo, hi, labels = [], [], [], [], []
            for i, (lab, name) in enumerate(points):
                s = base_stat if name is None else stats.get(name, {}).get("reward_per_step")
                if s is None:
                    continue
                xs.append(i); ys.append(s["mean"]); lo.append(s["mean"] - s["ci_lo"]); hi.append(s["ci_hi"] - s["mean"]); labels.append(lab)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", capsize=3)
            ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9)
        axes[0][0].set_ylabel("reward per step")
        axes[1][0].set_ylabel("reward per step")
        fig.suptitle("Parameter sensitivity of the fixed checkpoint (reward per step, 95% CI)", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return plot


def build_followup(main_results, follow_results, out_dir, n_resamples=10000, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(checkpoint=CHECKPOINT, n_resamples=n_resamples, seed=seed)
    headline = []

    def add(res):
        if res:
            headline.extend(res)

    exp1 = mr._rows(main_results, "exp1_base")
    exp2_delayed = mr._rows(main_results, "exp2_delayed")

    # ---- B: IDS detections per episode
    B = _load(follow_results, "B_", ["baseline", "zeroed", "shuffled", "alert_only", "delayed", "no_ids_branch", "no_recurrent_memory"])
    b_groups = {k: v for k, v in _present(B).items() if k != "baseline"}
    extra = ""
    if B["baseline"] and exp1:
        extra += mr._identity_md(exp1, {"baseline with the IDS counter attached": B["baseline"]},
                                 note="Attaching the counter must not change any episode. Return, captured and length against "
                                      "the plain Experiment 1 run on the same seeds:") + "\n"
    if B["baseline"] and b_groups:
        extra += penalty_explained_md(B["baseline"], b_groups, n_resamples, seed) + "\n"
    extra += ids_control_md(mr._rows(follow_results, "B_ctrl_no_ids_variant"))
    add(mr.write_experiment(
        out_dir, "B_ids_detections", "B: IDS detections per episode",
        "The main evaluation recorded return and captures but not how often the IDS fired. These runs repeat the baseline and "
        "the IDS-related interventions on the same seeds with a counter attached, so each condition's reward change can be "
        "compared with how many detections (by response type) it caused. This tests whether the collapse when IDS observations "
        "are removed is a stealth failure, and shows what the IDS signal costs the agent in the baseline.",
        "baseline (no intervention)", B["baseline"], b_groups, metric_sets=IDS_SETS, extra_md=extra,
        notes=["Detections count IDS responses. The level is not reset by an alert, so a host that stays above its threshold "
               "alerts again on every further action against it (see C_ids_semantics).",
               "`ids_max_level` is the highest post-decay detection level of any host in the episode."], **kw))

    # ---- A1: goal masks (baseline = the recorded baseline, which has the IDS columns)
    goal_names = ["mask_reduce_detection", "mask_recover_or_replan", "mask_both_ids_goals", "only_discover_subnet", "only_enumerate_host",
                  "only_gain_initial_access", "only_escalate_privilege", "only_pivot", "only_capture_sensitive_host",
                  "only_reduce_detection", "only_recover_or_replan"]
    A1 = _present(_load(follow_results, "A_", goal_names))
    add(mr.write_experiment(
        out_dir, "A_goal_analysis/A1_goal_masks", "A1: restricting the manager's goals",
        "The manager picks one of eight named subgoals. `mask_*` blocks the IDS-related goals (REDUCE_DETECTION, RECOVER_OR_REPLAN) "
        "so the manager cannot choose them; `only_*` forces a single goal for the whole episode. If the IDS-informed goals matter "
        "for stealth, blocking them should raise detections. Post-hoc interventions on a fixed checkpoint: they show what this "
        "checkpoint depends on, not what a differently trained manager would learn.",
        "baseline (all eight goals)", B["baseline"], A1, metric_sets=IDS_SETS,
        notes=["Forcing one goal is a stress test, not a policy anyone would deploy; the useful readings are the two `mask_*` "
               "conditions and the spread across the single-goal runs."], **kw))

    # ---- A2: per-step goal traces
    trace_files = {"normal": "A_trace_normal_steps.csv", "IDS observations zeroed": "A_trace_zeroed_steps.csv",
                   "IDS-bias branch off": "A_trace_no_ids_branch_steps.csv", "IDS disabled in the environment": "A_trace_no_ids_env_steps.csv"}
    traces = {k: os.path.join(follow_results, v) for k, v in trace_files.items() if os.path.exists(os.path.join(follow_results, v))}
    a2_dir = os.path.join(out_dir, "A_goal_analysis", "A2_goal_traces")
    if traces:
        analyze_goals.write_report(traces, a2_dir, n_perm=200, n_boot=300)
    else:
        os.makedirs(a2_dir, exist_ok=True)
        open(os.path.join(a2_dir, "trace_analysis.md"), "w").write("# A2\n\nNo data yet.\n")

    # ---- D: parameter sensitivity
    D = _present(_load(follow_results, "D_", list(SENSITIVITY_SPECS)))
    add(mr.write_experiment(
        out_dir, "D_parameter_sensitivity", "D: sensitivity to the chosen IDS and dynamics parameters",
        "Reviewers asked how the parameter values were chosen and what evidence supports them. Each variant changes one parameter "
        "family of the final stage (IDS thresholds, detection increments, decay, response mix, the failed-exploit compounding "
        "factor, churn, timeouts) and evaluates the same fixed checkpoint on the same seeds. This shows how much the outcome depends "
        "on each value. It cannot say what the values 'should' be, and the checkpoint was trained at the default values.",
        "as trained (Experiment 1, same seeds)", exp1, D, metric_sets=PLAIN_SETS,
        extra_plots={"dose_response": dose_response_plot(exp1, D)} if D and exp1 else None,
        notes=["`fem_0` removes the compounding of failed-exploit detection increments, which otherwise grows without bound on a "
               "host that keeps failing (found earlier, not fixed).",
               "Scan-noise rates are not in this sweep: scan noise did not reach the agent in the trained configuration (see E)."], **kw))

    # ---- E: scan noise reaching the observation
    E = _present({f"scan noise x{f} (observed)": mr._rows(follow_results, f"E_noiseobs_x{f}") for f in (0, 1, 2, 4)})
    e_extra = ""
    x0 = E.get("scan noise x0 (observed)")
    if x0 and exp1:
        e_extra = mr._identity_md(exp1, {"scan noise x0, observed": x0},
                                  note="With no noise, letting scan results reach the observation must change nothing. Against the "
                                       "Experiment 1 baseline on the same seeds:")
    add(mr.write_experiment(
        out_dir, "E_scan_noise_observed", "E: scan false positives and negatives that the agent actually sees",
        "In the trained configuration, scan noise is computed but never reaches the agent (the observation copies the true host "
        "state). These runs enable an opt-in flag so the noisy scan result is what the agent observes, at the configured rates "
        "(x1) and at 2x and 4x, on the fixed checkpoint. The checkpoint was trained without noisy observations, so this is a "
        "distribution-shift test, not a result for an agent trained under noise. Baseline: the trained configuration, where the "
        "noise is not observed.",
        "trained configuration (noise not observed)", exp1, E, metric_sets=PLAIN_SETS, extra_md=e_extra,
        notes=["Rates at x1: service scan FP/FN 0.03/0.05, OS scan 0.015/0.03, process scan 0.04/0.07.",
               "The x0 control is expected to be identical except when a service-churn event fires on a host in the same step it "
               "is scanned: the scan result is taken at action time, while the unflagged observation reads the post-step state "
               "(one of 300 episodes; traced to a churn event on the scanned host).",
               "Noise that reaches the agent changes a share of episodes (about 13% at x1, 18% at x2, 22% at x4 against the "
               "baseline) but leaves the mean unchanged."], **kw))

    # ---- F: generated scenarios (unpaired)
    fam = {"dynamic profile, mesh": "dynamic_mesh", "dynamic profile, chain": "dynamic_chain", "dynamic profile, random topology": "dynamic_random",
           "dynamic profile, sensitive-host jitter 0.5": "dynamic_jitter", "dynamic profile, 8 subnets (small)": "dynamic_small",
           "varB profile, mesh": "profile_varB", "bridge profile, mesh": "profile_bridge", "test profile, mesh": "profile_test"}
    F = {label: mr._rows(follow_results, f"F_{key}") for label, key in fam.items()}
    F = _present(F)
    ref = _present({name: mr._rows(follow_results, f"F_ref_{name}") for name in ("dynamic", "varA", "varB", "bridge", "test")})
    ref_label = "dynamic profile, mesh"
    f_groups = {k: v for k, v in F.items() if k != ref_label}
    add(mr.write_experiment(
        out_dir, "F_generated_scenarios", "F: freshly generated networks",
        "Each episode gets a new network from the scenario generator, seeded by the episode seed, with the host configuration, "
        "firewall and topology all regenerated. The families vary the topology, the network size, and which subnets hold sensitive "
        "hosts (borrowed from the dynamic, varB, bridge and test scenarios). This tests the earlier finding that performance "
        "depends on where the sensitive hosts are, on many networks rather than five hand-made ones. Captured / sensitive is "
        "exact (each episode records its own sensitive-host count).",
        ref_label + " (reference)", F.get(ref_label, []), f_groups, pairing="unpaired", align=False, metric_sets=GEN_SETS,
        extra_md=reference_table_md(ref),
        notes=["Groups are independent samples of networks, so differences are unpaired.",
               "The generator draws 6-8 hosts per subnet, so network size is only steerable through the subnet count."], **kw))

    # ---- G: IDS delay sweep
    G = {"delay 1": exp2_delayed}
    G.update({f"delay {k}": mr._rows(follow_results, f"G_delayed_{k}") for k in (5, 10, 25, 50)})
    G = _present(G)
    order = ["delay 0 (normal)"] + list(G)
    add(mr.write_experiment(
        out_dir, "G_ids_delay", "G: how far back does the checkpoint's use of IDS information reach?",
        "The IDS values shown to the network lag the truth by k steps (k = 1 was Experiment 2; the sweep adds 5, 10, 25, 50). If "
        "the checkpoint only needs a recent snapshot, a long delay changes little; if it tracks the IDS state over a long "
        "horizon, reward falls as k grows. Hosts not observed k steps earlier show neutral values.",
        "delay 0 (normal)", exp1, G, metric_sets=PLAIN_SETS, curve_order=order, curve_xlabel="IDS observation delay (steps)",
        notes=["The detection level itself decays only when an action targets that host, so a host's IDS state can stay "
               "informative for many steps."], **kw))

    # ---- H2: best checkpoint vs final
    H2 = _present({"best checkpoint (step 12,900)": mr._rows(follow_results, "H2_best")})
    add(mr.write_experiment(
        out_dir, "H_training_and_checkpoint/H2_best_vs_final", "H2: the step-12,900 checkpoint against the final one",
        "The run saved two checkpoints: the best by trainer-side evaluation (step 12,900) and the final (step 20,000). Evaluating "
        "both on the same 1,000 seeds shows how much two checkpoints of the same run differ. This is a within-run comparison; it is "
        "NOT a measure of variability across training seeds.",
        "final checkpoint (step 20,000)", exp1, H2, metric_sets=PLAIN_SETS,
        notes=["Only one training run exists. Two checkpoints of one run are strongly correlated, so agreement here understates "
               "how much independent runs would differ."], **kw))

    lines = ["# Follow-up experiments (no retraining)", "",
             f"Checkpoint: {CHECKPOINT}. Driven by the AISec review comments; every experiment here evaluates the existing "
             "checkpoint or analyses the existing training log. One folder per experiment.", "",
             "**Start with [FINDINGS_FOLLOWUP.md](FINDINGS_FOLLOWUP.md)**, then the folders below.", "",
             "| folder | experiment |", "|---|---|",
             "| [A_goal_analysis/A1_goal_masks](A_goal_analysis/A1_goal_masks/results.md) | blocking the IDS-related goals, forcing single goals |",
             "| [A_goal_analysis/A2_goal_traces](A_goal_analysis/A2_goal_traces/trace_analysis.md) | does the IDS level drive goal choice? (per-step traces, mutual information) |",
             "| [B_ids_detections](B_ids_detections/results.md) | IDS detections per episode; how much of the return change they explain |",
             "| [C_ids_semantics](C_ids_semantics/ids_semantics.md) | the IDS parameter table and exact semantics |",
             "| [D_parameter_sensitivity](D_parameter_sensitivity/results.md) | sensitivity to the chosen parameter values |",
             "| [E_scan_noise_observed](E_scan_noise_observed/results.md) | scan noise that reaches the agent |",
             "| [F_generated_scenarios](F_generated_scenarios/results.md) | freshly generated networks; sensitive-host placement |",
             "| [G_ids_delay](G_ids_delay/results.md) | IDS observation delay sweep |",
             "| [H1_training_log](H1_training_log/training_log_summary.md) | learning curves, goal usage, collapses (training log only) |",
             "| [H_training_and_checkpoint/H2_best_vs_final](H_training_and_checkpoint/H2_best_vs_final/results.md) | step-12,900 vs final checkpoint |", ""]
    if headline:
        rows = [[h["experiment"].split(":")[0], h["condition"], mr._cell_diff(h["d_rps"], "reward_per_step"),
                 mr._cell_diff(h["d_cap"], "captured"), h["d_rps"]["n_label"]] for h in headline]
        lines += ["## Headline: difference from each experiment's baseline", "",
                  "Reward per step and hosts captured, condition minus baseline (paired unless the row shows two sample "
                  "sizes), 95% bootstrap CI; `*` = CI excludes zero. Intervals are not adjusted for the many comparisons.", "",
                  mr._table(["experiment", "condition", "reward per step", "captured", "n"], rows), ""]
    lines += ["## How to read these results", "",
              f"- {mr.WORDING}",
              "- All conditions are post-hoc changes to how one fixed checkpoint is evaluated; none is a retrained ablation.",
              "- Not possible without retraining: variability across training seeds, other architectures, retrained ablations, "
              "and the IDS-disabled-training regime.", ""]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    mr._write_json(os.path.join(out_dir, "report_meta.json"),
                   {"main_results": main_results, "follow_results": follow_results, "n_resamples": n_resamples, "bootstrap_seed": seed})
    return headline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main_results", default=os.path.join(AGENTS_DIR, "results", "2026-09-24"))
    ap.add_argument("--follow_results", default=os.path.join(AGENTS_DIR, "results", "2026-09-24_followup"))
    ap.add_argument("--out_dir", default=os.path.join(AGENTS_DIR, "..", "docs", "eval_followup_2026-09-24"))
    ap.add_argument("--n_resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    h = build_followup(args.main_results, args.follow_results, args.out_dir, args.n_resamples, args.seed)
    print(f"wrote {os.path.abspath(args.out_dir)} ({len(h)} headline rows)")


if __name__ == "__main__":
    main()
