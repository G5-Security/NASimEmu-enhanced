"""
Dataset audit report -- master plan Sec 15.13 repo layout.

Reads dataset.jsonl (every record, valid and rejected) and reports the
numbers a QA pass needs before spending a training run against the dataset:
validity rate and rejection-reason breakdown, per-class balance and
confidence stats (Sec 15.8's own worry -- "labeling only successful expert
trajectories would produce a narrow dataset" shows up here as a starved
class), trigger-type mix, episode/split bookkeeping, which teacher
backend(s)/models produced the data, and label latency.

Run from NASimEmu-agents/:
    python -m llm_teacher.audit_dataset
    python -m llm_teacher.audit_dataset --out_dir training_data/llm_teacher_dataset --min_class_count 20
"""
import argparse
import json
import os
from collections import Counter, defaultdict

from .dataset_writer import DATASET_JSONL_NAME
from .goal_ontology import GOAL_NAMES


def _load_records(dataset_dir):
    path = os.path.join(dataset_dir, DATASET_JSONL_NAME)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records, path


def audit(dataset_dir, min_class_count=20):
    records, path = _load_records(dataset_dir)
    n = len(records)
    valid = [r for r in records if r.get("valid")]
    invalid = [r for r in records if not r.get("valid")]

    print(f"[audit_dataset] {path}")
    print(f"  total records:   {n}")
    print(f"  valid:           {len(valid)} ({100.0 * len(valid) / max(n, 1):.1f}%)")
    print(f"  rejected:        {len(invalid)} ({100.0 * len(invalid) / max(n, 1):.1f}%)")

    if invalid:
        print("\n  rejection reasons:")
        reasons = Counter(r.get("reject_reason") for r in invalid)
        for reason, count in reasons.most_common():
            print(f"    {reason:<40} {count:>5}")

    print("\n  class balance (valid records only):")
    class_counts = Counter()
    class_conf = defaultdict(list)
    for r in valid:
        parsed = r.get("parsed_output") or {}
        goal = parsed.get("goal")
        if goal is None:
            continue
        class_counts[goal] += 1
        conf = parsed.get("confidence")
        if conf is not None:
            class_conf[goal].append(float(conf))

    starved = []
    for name in GOAL_NAMES:
        count = class_counts.get(name, 0)
        confs = class_conf.get(name, [])
        conf_str = f"conf mean={sum(confs)/len(confs):.2f}" if confs else "conf n/a"
        flag = "  <-- BELOW min_class_count" if count < min_class_count else ""
        print(f"    {name:<24} {count:>5}  {conf_str}{flag}")
        if count < min_class_count:
            starved.append(name)

    if starved:
        print(f"\n  WARNING: {len(starved)} class(es) below --min_class_count={min_class_count}: {', '.join(starved)}")
        print("  Training the distillation head against these classes now will mostly be noise.")
        print("  Fix: collect more with a policy/backend mix that actually reaches these goals")
        print("  (see llm_teacher/label_states.py --policy checkpoint and --teacher_backend).")

    triggers = Counter(r.get("trigger") for r in records if r.get("trigger") is not None)
    n_no_trigger = sum(1 for r in records if r.get("trigger") is None)
    if triggers:
        print("\n  trigger mix:")
        for trig, count in triggers.most_common():
            print(f"    {trig:<28} {count:>5}")
    if n_no_trigger:
        print(f"    (legacy, no trigger field)   {n_no_trigger:>5}")

    episode_ids = {r.get("episode_id") for r in records if r.get("episode_id") is not None}
    n_no_episode = sum(1 for r in records if r.get("episode_id") is None)
    print(f"\n  episodes: {len(episode_ids)} distinct episode_id(s)"
          + (f", {n_no_episode} legacy record(s) with no episode_id" if n_no_episode else ""))

    splits = Counter(r.get("split") for r in records if "split" in r)
    if splits:
        print(f"  split: {dict(splits)}")
    else:
        print("  split: NOT SPLIT YET -- run `python -m llm_teacher.split_dataset` before training with a held-out set")

    models = Counter(r.get("model") for r in records)
    print(f"\n  teacher backend/model mix: {dict(models)}")

    latencies = [r.get("latency_s") for r in records if r.get("latency_s") is not None]
    if latencies:
        total_min = sum(latencies) / 60.0
        print(f"  total labeling time: {total_min:.1f} min ({sum(latencies)/len(latencies):.2f}s/record mean)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "training_data", "llm_teacher_dataset"))
    ap.add_argument("--min_class_count", type=int, default=20,
                     help="Flag any goal class with fewer valid records than this")
    args = ap.parse_args()
    audit(args.out_dir, min_class_count=args.min_class_count)


if __name__ == "__main__":
    main()
