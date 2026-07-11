"""Reads back a dataset written by dataset_writer.py, keeping only
`valid=True` records (Sec 15.7 -- rejected records stay on disk for audit but
must never be trained against)."""
import json
import os
import pickle

from .goal_ontology import GOAL_INDEX
from .dataset_writer import DATASET_JSONL_NAME, DATASET_STATES_NAME


def load_dataset(dataset_dir, split=None):
    """
    Returns a list of dicts: {"s_true": <graph observation>, "goal_idx": int,
    "confidence": float, "goal_name": str}, one per valid record.

    split: None (all valid records, default -- unchanged behavior) or
    "train"/"test" to filter by the field llm_teacher/split_dataset.py stamps
    onto every record. Requesting a split on a dataset that was never run
    through split_dataset.py raises, rather than silently returning an
    unsplit set that looks split (Sec 15.8: never let train/test evidence be
    ambiguous about whether a real split happened).
    """
    jsonl_path = os.path.join(dataset_dir, DATASET_JSONL_NAME)
    states_path = os.path.join(dataset_dir, DATASET_STATES_NAME)
    if not os.path.exists(jsonl_path) or not os.path.exists(states_path):
        raise FileNotFoundError(
            f"No LLM-teacher dataset found under {dataset_dir} "
            f"(expected {DATASET_JSONL_NAME} + {DATASET_STATES_NAME}). "
            f"Run llm_teacher/label_states.py first."
        )

    with open(states_path, "rb") as f:
        states = pickle.load(f)

    records = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("valid"):
                continue
            if split is not None:
                if "split" not in rec:
                    raise ValueError(
                        f"{jsonl_path} has no 'split' field -- run "
                        f"`python -m llm_teacher.split_dataset` before requesting split={split!r}."
                    )
                if rec["split"] != split:
                    continue
            parsed = rec["parsed_output"]
            goal_name = parsed["goal"]
            records.append({
                "s_true": states[idx],
                "goal_idx": GOAL_INDEX[goal_name],
                "goal_name": goal_name,
                "confidence": float(parsed.get("confidence", 1.0)),
            })
    return records
