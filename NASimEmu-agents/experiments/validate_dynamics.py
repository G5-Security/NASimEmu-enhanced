"""Step 7: simulator validation (docs/eval_revision_plan.tex).

Runs many scripted (non-policy) interactions with a DynamicsRecorder
attached, then compares each measured rate against the scenario's
*configured* rate (its final/full_difficulty curriculum stage -- the only
stage evaluation ever reads, see experiments/make_variants.py's docstring)
with a 95% Wilson confidence interval. No policy, no checkpoint: this
validates the simulator itself, independent of any trained model.

Scripted interaction, not a random policy: every step performs, in turn, a
service scan, an OS scan, a process scan, an exploit attempt and a privilege
escalation attempt against whichever host address happens to occupy the
first observation row -- cycling through action kinds so all the
detection-increase paths in HostVector.update_detection() get real activity,
not just whichever kind a random policy happens to sample most. This is
about exercising the simulator's own dynamics, not evaluating a policy, so
scripted, deterministic coverage is preferable to a random walk here.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.dynamics_recorder import DynamicsRecorder, wilson_interval  # noqa: E402
from experiments.make_variants import final_stage, load_scenario  # noqa: E402
from nasimemu.env import NASimEmuEnv  # noqa: E402
from nasimemu.nasim.envs.host_vector import HostVector  # noqa: E402
from nasimemu.nasim.envs.network import Network  # noqa: E402


def run_scripted_episodes(scenario, n_episodes, step_limit, seed_base, recorder):
    HostVector.set_event_recorder(recorder)
    Network.set_event_recorder(recorder)
    try:
        for i in range(n_episodes):
            env = NASimEmuEnv(scenario_name=scenario, step_limit=step_limit,
                               observation_format="graph_v2", seed=seed_base + i,
                               training_mode=False)
            env.reset()
            done = False
            t = 0
            while t < step_limit and not done:
                host_addr = HostVector(env.s_raw[0]).address
                action_id = t % 6
                _s, r, done, info = env.step((np.array(host_addr), action_id))
                t += 1
    finally:
        HostVector.set_event_recorder(None)
        Network.set_event_recorder(None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "scenarios",
        "corp_100hosts_dynamic.v2.yaml"))
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--step_limit", type=int, default=200)
    ap.add_argument("--seed_base", type=int, default=2_000_000)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    recorder = DynamicsRecorder()
    run_scripted_episodes(args.scenario, args.n_episodes, args.step_limit, args.seed_base, recorder)

    scenario_dict = load_scenario(args.scenario)
    stage = final_stage(scenario_dict)

    rows = []

    def add_row(label, measured, configured):
        k, n, rate = measured
        lo, hi = wilson_interval(k, n)
        rows.append({
            "label": label, "configured": configured,
            "measured_k": k, "measured_n": n, "measured_rate": rate,
            "ci_95_low": lo, "ci_95_high": hi,
            "configured_in_ci": bool(n > 0 and lo <= configured <= hi),
        })

    add_row("service_dynamics.churn_probability", recorder.measured_rate("churn_trial"),
            stage.get("service_dynamics", {}).get("churn_probability", 0.0))
    add_row("network_reliability.timeout_probability", recorder.measured_rate("timeout_trial"),
            stage.get("network_reliability", {}).get("timeout_probability", 0.0))

    for scan_type in ("service_scan", "os_scan", "process_scan"):
        noise = stage.get("scan_noise", {}).get(scan_type, {})
        type_events = recorder.filtered("scan_noise_trial", scan_type=scan_type)
        sub = DynamicsRecorder()
        sub.events = type_events
        add_row(f"scan_noise.{scan_type}.false_positive_rate", sub.measured_flip_rate("fp"),
                noise.get("false_positive_rate", 0.0))
        add_row(f"scan_noise.{scan_type}.false_negative_rate", sub.measured_flip_rate("fn"),
                noise.get("false_negative_rate", 0.0))

    response_types = stage.get("ids", {}).get("response_types", {})
    counts = recorder.response_type_counts()
    total_detections = sum(counts.values())
    for rt, configured_p in response_types.items():
        k = counts.get(rt, 0)
        rate = k / total_detections if total_detections else float("nan")
        lo, hi = wilson_interval(k, total_detections)
        rows.append({
            "label": f"ids.response_types.{rt}", "configured": configured_p,
            "measured_k": k, "measured_n": total_detections, "measured_rate": rate,
            "ci_95_low": lo, "ci_95_high": hi,
            "configured_in_ci": bool(total_detections > 0 and lo <= configured_p <= hi),
        })

    print(f"{'label':45s} {'configured':>10s} {'measured':>10s} {'n':>6s} "
          f"{'95% CI':>20s}  ok")
    for row in rows:
        ci = f"[{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]" if row["measured_n"] else "n/a"
        print(f"{row['label']:45s} {row['configured']:10.4f} {row['measured_rate']:10.4f} "
              f"{row['measured_n']:6d} {ci:>20s}  {'OK' if row['configured_in_ci'] else 'MISMATCH'}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"scenario": args.scenario, "n_episodes": args.n_episodes,
                       "step_limit": args.step_limit, "rows": rows}, f, indent=2)
        print(f"\nWrote {args.out_json}")

    n_mismatch = sum(1 for r in rows if r["measured_n"] and not r["configured_in_ci"])
    if n_mismatch:
        print(f"\n{n_mismatch}/{len(rows)} measured rates fell outside their 95% CI "
              f"of the configured value.")


if __name__ == "__main__":
    main()
