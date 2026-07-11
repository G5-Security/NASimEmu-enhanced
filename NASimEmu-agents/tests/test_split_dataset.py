"""Master plan Sec 15.8: split at the scenario-instance level, and the split
must be stable as the dataset grows (see split_dataset.py's module docstring
for why hashing, not shuffle-and-slice, is used)."""
import json
import os

from llm_teacher.split_dataset import split_dataset


def _write_dataset(tmp_path, records):
    (tmp_path / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (tmp_path / "dataset_states.pkl").write_bytes(b"")  # split_dataset.py never reads this file


def _make_records(episode_ids_per_record):
    return [{"episode_id": eid, "valid": True, "parsed_output": {"goal": "DISCOVER_SUBNET"}}
            for eid in episode_ids_per_record]


def test_two_steps_of_same_episode_never_split(tmp_path):
    records = _make_records(["ep1", "ep1", "ep1", "ep2", "ep2", "ep3"])
    _write_dataset(tmp_path, records)
    split_dataset(str(tmp_path), test_frac=0.5, seed=0)

    written = [json.loads(l) for l in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    by_episode = {}
    for r in written:
        by_episode.setdefault(r["episode_id"], set()).add(r["split"])
    for eid, splits in by_episode.items():
        assert len(splits) == 1, f"episode {eid} spans both splits: {splits}"


def test_split_is_stable_as_dataset_grows(tmp_path):
    initial = _make_records([f"ep{i}" for i in range(20)])
    _write_dataset(tmp_path, initial)
    split_dataset(str(tmp_path), test_frac=0.3, seed=0)
    first_pass = {json.loads(l)["episode_id"]: json.loads(l)["split"]
                  for l in (tmp_path / "dataset.jsonl").read_text().splitlines()}

    # simulate more data being appended for the same episodes plus new ones
    grown = initial + _make_records([f"ep{i}" for i in range(20, 30)])
    _write_dataset(tmp_path, grown)
    split_dataset(str(tmp_path), test_frac=0.3, seed=0)
    second_pass = {json.loads(l)["episode_id"]: json.loads(l)["split"]
                   for l in (tmp_path / "dataset.jsonl").read_text().splitlines()}

    for eid in first_pass:
        assert first_pass[eid] == second_pass[eid], f"{eid} changed split after dataset grew"


def test_legacy_records_without_episode_id_still_get_a_split(tmp_path):
    records = [{"valid": True, "parsed_output": {"goal": "DISCOVER_SUBNET"}} for _ in range(10)]
    _write_dataset(tmp_path, records)
    split_dataset(str(tmp_path), test_frac=0.5, seed=0)
    written = [json.loads(l) for l in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    assert all(r["split"] in ("train", "test") for r in written)


def test_split_fraction_is_roughly_respected(tmp_path):
    records = _make_records([f"ep{i}" for i in range(200)])
    _write_dataset(tmp_path, records)
    split_dataset(str(tmp_path), test_frac=0.2, seed=0)
    written = [json.loads(l) for l in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    test_frac_actual = sum(1 for r in written if r["split"] == "test") / len(written)
    assert 0.1 < test_frac_actual < 0.3
