"""dagger_loop.py's collection step passes label_states.py --target_records
an ABSOLUTE floor on total valid records, not "how many new records to add
this round". A prior version of run_round() passed --records_per_round
directly, which silently collected zero new records in every round after
the first, once the dataset already held that many valid records from prior
rounds. Fixed by computing existing_valid + records_per_round explicitly
(_compute_target_records, dagger_loop.py). This locks that fix in so it
can't silently regress back to the flat-records_per_round bug."""
from types import SimpleNamespace

from experiments.dagger_loop import _compute_target_records, _build_collect_cmd
from llm_teacher.dataset_writer import DatasetWriter


def _collect_args(hardest_stage=True, teacher_backend="heuristic", model=None):
    return SimpleNamespace(
        scenario="../scenarios/corp_100hosts_dynamic.v2.yaml",
        out_dir="training_data/llm_teacher_dataset",
        teacher_backend=teacher_backend,
        hardest_stage=hardest_stage,
        model=model,
    )


def _seed_dataset(out_dir, n_valid, n_invalid=0):
    writer = DatasetWriter(out_dir)
    for _ in range(n_valid):
        writer.add({"valid": True, "reject_reason": None}, s_true=None)
    for _ in range(n_invalid):
        writer.add({"valid": False, "reject_reason": "test"}, s_true=None)
    return writer


def test_empty_dataset_target_equals_records_per_round(tmp_path):
    out_dir = str(tmp_path / "dataset")
    existing_valid, target = _compute_target_records(out_dir, records_per_round=50)
    assert existing_valid == 0
    assert target == 50  # round 0: no prior data, matches the old (correct-by-accident) behavior


def test_nonempty_dataset_target_adds_on_top_of_existing(tmp_path):
    """The actual bug: round 1+ must not silently collect zero new records."""
    out_dir = str(tmp_path / "dataset")
    _seed_dataset(out_dir, n_valid=50)

    existing_valid, target = _compute_target_records(out_dir, records_per_round=50)
    assert existing_valid == 50
    assert target == 100, (
        "target_records did not account for existing valid records -- this is exactly the "
        "accumulation bug that made DAgger round 2+ collect zero new records"
    )


def test_invalid_records_do_not_count_toward_existing_valid(tmp_path):
    out_dir = str(tmp_path / "dataset")
    _seed_dataset(out_dir, n_valid=30, n_invalid=20)

    existing_valid, target = _compute_target_records(out_dir, records_per_round=50)
    assert existing_valid == 30  # rejected records must not inflate the floor
    assert target == 80


def test_three_rounds_accumulate_monotonically(tmp_path):
    """Simulates what run_round() does across rounds without running real
    collection/training subprocesses: each round's target must exceed the
    previous round's actual valid count."""
    out_dir = str(tmp_path / "dataset")
    records_per_round = 20

    _, target_round_0 = _compute_target_records(out_dir, records_per_round)
    assert target_round_0 == 20
    _seed_dataset(out_dir, n_valid=target_round_0)  # simulate round 0's collection reaching its target

    _, target_round_1 = _compute_target_records(out_dir, records_per_round)
    assert target_round_1 == 40
    _seed_dataset(out_dir, n_valid=20)  # round 1 adds 20 more, reaching 40 total

    existing_valid, target_round_2 = _compute_target_records(out_dir, records_per_round)
    assert existing_valid == 40
    assert target_round_2 == 60


# --- Phase 3: DAgger must relabel IDS-on states (--hardest_stage) on-policy ---

def test_collect_cmd_includes_hardest_stage_by_default():
    cmd = _build_collect_cmd(_collect_args(hardest_stage=True), prev_checkpoint=None, target_records=100)
    assert "--hardest_stage" in cmd, (
        "DAgger collection must pin the labeling env to the IDS-on stage, else the aggregated "
        "dataset never gains the REDUCE_DETECTION examples curriculum-aligned distillation needs"
    )


def test_collect_cmd_omits_hardest_stage_when_disabled():
    cmd = _build_collect_cmd(_collect_args(hardest_stage=False), prev_checkpoint=None, target_records=100)
    assert "--hardest_stage" not in cmd


def test_collect_cmd_round0_is_random_policy():
    cmd = _build_collect_cmd(_collect_args(), prev_checkpoint=None, target_records=100)
    assert "random" in cmd
    assert "--checkpoint_path" not in cmd


def test_collect_cmd_later_rounds_roll_out_the_checkpoint_on_policy():
    cmd = _build_collect_cmd(_collect_args(), prev_checkpoint="/tmp/round_0.pt", target_records=200)
    assert "checkpoint" in cmd
    assert "--checkpoint_path" in cmd
    assert "/tmp/round_0.pt" in cmd
    # target floor is absolute, passed straight through
    assert "200" in cmd
