"""
Train/test split at the scenario-instance (episode) level -- master plan Sec
15.8: "Split at the concrete scenario-instance level -- never place two steps
of the same episode across a train/test boundary."

Operates on an existing dataset.jsonl in place, stamping a "split":
"train"|"test" field onto every record (valid or not, so audit-time analysis
can still see the full picture) based on which episode_id it belongs to.
Idempotent: re-running with the same --seed reproduces the same split even as
more records are appended later, since the split is computed by hashing each
episode_id rather than by shuffling the whole episode-id list positionally
(shuffling would reassign existing episodes to different folds every time a
new episode_id is added to the pool).

Records written before label_states.py started stamping episode_id (this
repo's original 300-record pilot) have no episode_id at all -- those fall
back to per-record splitting (each such record treated as its own singleton
episode), which is a known imprecision for that legacy slice only, not
silently hidden: see the printed counts this script reports.

Run from NASimEmu-agents/:
    python -m llm_teacher.split_dataset --test_frac 0.2
"""
import argparse
import hashlib
import json
import os

from .dataset_writer import DATASET_JSONL_NAME


def _episode_fraction(episode_id, seed):
    """Deterministic, seed-salted hash of episode_id into [0, 1) -- used
    instead of shuffle-and-slice so the split is stable under dataset growth
    (see module docstring)."""
    h = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_dataset(dataset_dir, test_frac=0.2, seed=0):
    path = os.path.join(dataset_dir, DATASET_JSONL_NAME)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    legacy_count = 0
    n_train = n_test = 0
    episodes_seen = set()

    for i, r in enumerate(records):
        eid = r.get("episode_id")
        if eid is None:
            eid = f"__legacy_record_{i}"
            legacy_count += 1
        episodes_seen.add(eid)
        r["split"] = "test" if _episode_fraction(eid, seed) < test_frac else "train"
        if r["split"] == "train":
            n_train += 1
        else:
            n_test += 1

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_test_episodes = sum(1 for e in episodes_seen if _episode_fraction(e, seed) < test_frac)
    print(f"[split_dataset] {len(episodes_seen)} episode groups "
          f"({legacy_count} legacy records with no episode_id, split at record granularity) "
          f"-> {n_test_episodes} test / {len(episodes_seen) - n_test_episodes} train")
    print(f"[split_dataset] {len(records)} records -> {n_train} train / {n_test} test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "training_data", "llm_teacher_dataset"))
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    split_dataset(args.out_dir, test_frac=args.test_frac, seed=args.seed)


if __name__ == "__main__":
    main()
