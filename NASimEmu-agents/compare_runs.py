"""
Evaluate/compare a training_data/runs/<wandb_run_id>.json log against either
another such log (a freshly-run C0) or, by default, this project's own
published Table 5 numbers for Pen-DHRL (no-IDS/no-curriculum dynamic
scenario) -- used when no local baseline checkpoint/log is available.

Usage:
    python compare_runs.py <run.json>                     # vs paper Table 5
    python compare_runs.py <run.json> --baseline <c0.json>  # vs a local C0 run

Reads only the LAST logged record per file (the final, most-trained eval).
"""
import argparse
import json


# Paper Table 5, "Baseline Performance on Dynamic Auto Generated Scenarios
# Without IDS or Curriculum", Pen-DHRL row (Training / Testing columns).
PAPER_TABLE5_PENDHRL = {
    "reward_avg": (0.49, 0.48),
    "reward_avg_episodes": (49.41, 48.43),
    "eplen_avg": (103.40, 106.40),
    "captured_avg": (36.13, 36.68),
}


def load_last_record(path):
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        raise ValueError(f"{path} has no logged records yet")
    return last


def fmt_row(name, this_trn, this_tst, ref_trn, ref_tst):
    d_trn = this_trn - ref_trn
    d_tst = this_tst - ref_tst
    return f"{name:<22}{this_trn:>10.4f}{this_tst:>10.4f}{ref_trn:>12.4f}{ref_tst:>12.4f}{d_trn:>+10.4f}{d_tst:>+10.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json", help="path to the run's training_data/runs/<id>.json")
    ap.add_argument("--baseline", default=None, help="optional path to a local C0 run.json; defaults to paper Table 5 Pen-DHRL numbers")
    args = ap.parse_args()

    rec = load_last_record(args.run_json)
    trn, tst = rec["eval_trn"], rec["eval_tst"]

    if args.baseline:
        base_rec = load_last_record(args.baseline)
        base_trn, base_tst = base_rec["eval_trn"], base_rec["eval_tst"]
        ref_label = f"local C0 ({args.baseline})"
        ref = {k: (base_trn.get(k, float('nan')), base_tst.get(k, float('nan'))) for k in trn}
    else:
        ref_label = "paper Table 5 (Pen-DHRL, no IDS/curriculum)"
        ref = PAPER_TABLE5_PENDHRL

    print(f"Run:      {args.run_json}  (train_step={rec['train_step']}, env_steps_total={rec['env_steps_total']})")
    print(f"Baseline: {ref_label}")
    print()
    header = f"{'metric':<22}{'this_trn':>10}{'this_tst':>10}{'ref_trn':>12}{'ref_tst':>12}{'d_trn':>10}{'d_tst':>10}"
    print(header)
    print("-" * len(header))
    for metric in ["reward_avg", "reward_avg_episodes", "eplen_avg", "captured_avg"]:
        if metric in trn and metric in ref:
            print(fmt_row(metric, trn[metric], tst[metric], ref[metric][0], ref[metric][1]))

    if "llm_shaping_mean_abs" in rec:
        print()
        print("LLM-shaping diagnostics (this run):")
        print(f"  mean |F_shaping|   = {rec['llm_shaping_mean_abs']:.4f}")
        print(f"  mean |r_env|       = {rec['llm_shaping_env_r_mean_abs']:.4f}")
        print(f"  lambda (final)     = {rec['llm_shaping_lambda']:.4f}")
        ratio = rec['llm_shaping_mean_abs'] / max(rec['llm_shaping_env_r_mean_abs'], 1e-9)
        print(f"  |F|/|r_env| ratio  = {ratio:.3f}  (near 0 = negligible shaping; near/over 1 = comparable-to-dominant)")


if __name__ == "__main__":
    main()
