"""Step 7: one-host IDS detection-level trace plot (docs/eval_revision_plan.tex).

Runs one scripted episode with a DynamicsRecorder attached, picks the host
with the most detection-level activity, and plots its detection_level
(post-decay, i.e. what HostVector._vector and the agent's observation
actually hold -- see host_vector.py's `observe()`) against step number,
with the host's own detection_threshold as a reference line and each
threshold crossing marked. No policy, no checkpoint: this is a simulator
validation artifact, independent of any trained model.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.dynamics_recorder import DynamicsRecorder  # noqa: E402
from experiments.validate_dynamics import run_scripted_episodes  # noqa: E402
from nasimemu.nasim.envs.host_vector import HostVector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "scenarios",
        "corp_100hosts_dynamic.v2.yaml"))
    ap.add_argument("--step_limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4_000_000)
    ap.add_argument("--out_png", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "training_data", "ids_trace.png"))
    args = ap.parse_args()

    rec = DynamicsRecorder()
    run_scripted_episodes(args.scenario, n_episodes=1, step_limit=args.step_limit,
                           seed_base=args.seed, recorder=rec)

    hosts = {e["host"] for e in rec.events if e["kind"] == "ids_increase"}
    if not hosts:
        raise SystemExit("no ids_increase events recorded -- is IDS enabled for this scenario's final stage?")
    host = max(hosts, key=lambda h: len(rec.level_trace(h)))
    trace = rec.level_trace(host)
    steps, levels = zip(*trace)

    threshold_events = [e for e in rec.events if e["kind"] == "ids_increase" and e["host"] == host]
    # detection_threshold isn't itself in the recorded ids_increase fields
    # (only level_before/level_after_decay are -- see the module docstring
    # in dynamics_recorder.py), so recover it from where the level first
    # crosses it: HostVector.update_detection() fires 'DETECTED' exactly
    # when level_after_decay > threshold, so at a detection, threshold <
    # level_after_decay <= threshold + one step's worth of noise; report the
    # crossing steps instead of a numeric threshold line, which needs no
    # such approximation.
    crossings = [e["step"] for e in rec.filtered("ids_detected", host=host)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, levels, marker="o", markersize=3, linewidth=1, label="detection_level (post-decay)")
    for c in crossings:
        ax.axvline(c, color="red", alpha=0.3, linestyle="--")
    if crossings:
        ax.axvline(crossings[0], color="red", alpha=0.3, linestyle="--", label="threshold crossing (DETECTED)")
    ax.set_xlabel("step")
    ax.set_ylabel("detection_level")
    ax.set_title(f"IDS detection-level trace, host {host} (scenario: {os.path.basename(args.scenario)})")
    ax.legend(loc="upper left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f"host {host}: {len(trace)} steps recorded, {len(crossings)} threshold crossings")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
