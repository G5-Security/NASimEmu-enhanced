"""
Dataset construction -- master plan Sec 15.8/15.14.

Two files are written per dataset, kept in lock-step by append order:
  - `dataset.jsonl`: one line per labeled state, teacher-client's full
    provenance record MINUS the raw graph tensor -- human-auditable, and
    matches Sec 15.14's provenance requirement (model/version, prompt
    version, input hash, raw+parsed output, validation status).
  - `dataset_states.pkl`: a parallel list of the raw graph-format
    observation (`info['s_true']`, exactly what NASimNetDHRL.prepare_batch
    consumes) for every record, valid or not, at the same index as its
    `dataset.jsonl` line -- kept out of the JSONL because a graph tensor
    isn't meaningfully human-readable and would bloat the audit file.

Only the JSONL is meant to be opened and read by a person; dataset_reader.py
is the only code that reads dataset_states.pkl.
"""
import json
import os
import pickle

DATASET_JSONL_NAME = "dataset.jsonl"
DATASET_STATES_NAME = "dataset_states.pkl"


class DatasetWriter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, DATASET_JSONL_NAME)
        self.states_path = os.path.join(out_dir, DATASET_STATES_NAME)
        self._states = []
        if os.path.exists(self.states_path):
            with open(self.states_path, "rb") as f:
                self._states = pickle.load(f)

    def __len__(self):
        return len(self._states)

    def add(self, record, s_true):
        """record: teacher_client.label_one_state(...)'s return dict.
        s_true: the raw graph observation for that same state."""
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._states.append(s_true)
        with open(self.states_path, "wb") as f:
            pickle.dump(self._states, f)

    def count_valid(self):
        if not os.path.exists(self.jsonl_path):
            return 0
        n = 0
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line and json.loads(line).get("valid"):
                    n += 1
        return n
