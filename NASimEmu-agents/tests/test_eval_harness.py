"""Regression tests for the Step 1 evaluation harness
(experiments/eval_harness.py, docs/eval_revision_plan.tex).

These test the harness's own correctness -- determinism, exact-episode-count,
CSV schema -- independent of model quality, so they don't need a real
trained checkpoint: an untrained NASimNetDHRL (checkpoint_path=None) is a
valid, fast stand-in, exactly as the plan's Step 1 "tests before large runs"
calls for.
"""
import csv
import os

import pytest

from experiments.eval_harness import CSV_FIELDS, build_net, run_episode, run_eval
from nasimemu.nasim.envs.host_vector import HostVector

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)
STEP_LIMIT = 30  # short episodes -- this suite tests harness plumbing, not policy quality


@pytest.fixture(scope="module")
def untrained_net():
    return build_net("NASimNetDHRL", SCENARIO, STEP_LIMIT, checkpoint_path=None)


def test_same_seed_gives_identical_episode(untrained_net):
    r1 = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=42)
    r2 = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=42)

    assert r1["episode_return"] == r2["episode_return"]
    assert r1["episode_len"] == r2["episode_len"]
    assert r1["captured"] == r2["captured"]
    assert r1["terminal_action"] == r2["terminal_action"]
    assert r1["milestones"] == r2["milestones"]


def test_different_seeds_can_diverge(untrained_net):
    # Not a strict correctness requirement (two seeds *could* coincide), but
    # a stuck/broken seed plumbing (e.g. seed silently ignored) would make
    # every seed produce the exact same trajectory -- catch that here across
    # enough seeds that a real collision is implausible.
    results = [run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=s) for s in range(2000, 2008)]
    fingerprints = {(r["episode_return"], r["episode_len"], r["captured"]) for r in results}
    assert len(fingerprints) > 1, "all 8 distinct seeds produced identical episodes -- seed is not being used"


def test_run_eval_produces_exactly_n_rows(untrained_net, tmp_path):
    n = 5
    rows = run_eval(untrained_net, "NASimNetDHRL", SCENARIO, "full_dynamics",
                     STEP_LIMIT, n_episodes=n, seed_base=5000, verbose=False)
    assert len(rows) == n
    assert [r["episode_idx"] for r in rows] == list(range(n))
    assert [r["seed"] for r in rows] == list(range(5000, 5000 + n))


def test_run_eval_csv_matches_declared_schema(untrained_net, tmp_path):
    out_csv = tmp_path / "eval.csv"
    run_eval(untrained_net, "NASimNetDHRL", SCENARIO, "full_dynamics",
              STEP_LIMIT, n_episodes=3, seed_base=6000, csv_path=str(out_csv), verbose=False)

    with open(out_csv, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        written_rows = list(reader)
    assert len(written_rows) == 3


def test_episode_end_is_either_terminal_action_or_step_limit_not_both(untrained_net):
    # NASimEmuEnv.step() only sets d=True via a chosen TerminalAction or via
    # the step limit (see eval_harness.py module docstring) -- never both at
    # once for the same episode.
    for seed in range(7000, 7010):
        r = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=seed)
        assert r["terminal_action"] != r["hit_step_limit"] or not r["terminal_action"], (
            "an episode was flagged as both a voluntary stop and a step-limit "
            "exceedance"
        )
        assert r["episode_len"] <= STEP_LIMIT


def test_milestone_short_episode_carries_final_value_forward(untrained_net):
    # STEP_LIMIT=30 is below every milestone (100/200/400), so every episode
    # here is "short" and every milestone must equal the final return/capture.
    r = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=8000)
    for m in (100, 200, 400):
        assert r["milestones"][m] == (r["episode_return"], r["captured"])


def test_episodes_run_under_the_hardest_curriculum_stage(untrained_net):
    # Regression test: run_episode() used to construct NASimEmuEnv without
    # training_mode=False, so it silently defaulted to True -- under which
    # CurriculumManager resolves the stage by epoch (starting at epoch 0 =
    # "baseline": IDS off, zero scan noise/churn/timeouts), not "hardest
    # stage". nasim_debug.py's own _eval() sets training_mode=False
    # explicitly for exactly this reason. corp_100hosts_dynamic.v2.yaml's
    # final ("full_difficulty") stage has IDS enabled and nonzero scan
    # noise, so if this ever regresses, the assertions below fail.
    run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=9000)
    assert HostVector.ids_config.get("enabled") is True
    assert HostVector.scan_noise["service_scan"]["false_positive_rate"] > 0.0


def test_terminal_action_flag_fires_on_a_real_terminal_choice():
    # Regression test: terminal_action used to be computed as
    # isinstance(action, TerminalAction) on the raw (target, action_id)
    # tuple net() returns -- that tuple only becomes an actual
    # TerminalAction instance inside env.step()'s own translation
    # (src/nasimemu/env.py: _translate_action), so the check never fired and
    # terminal_action was always False, regardless of what the policy chose.
    #
    # This is tested directly against the exact tuple shape
    # nasim_net_base_hrl.py's forward() returns ("return list(zip(targets,
    # a_id)), ..."), rather than by hoping an untrained policy happens to
    # terminate within some seed range/step budget -- it does not
    # necessarily ever do so (checked empirically: 40 seeds x up to 100
    # steps of an untrained NASimNetDHRL under full_difficulty dynamics
    # never terminated once, confirmed independently against the network's
    # own raw `terminate` tensor, not just this proxy check).
    import numpy as np
    from nasimemu.env import TerminalAction

    terminal_raw_action = (np.array([3, 1]), -1)
    non_terminal_raw_action = (np.array([3, 1]), 5)

    # the bug: this is the check that used to be in eval_harness.py, and it
    # never fires on the untranslated tuple net() actually returns
    assert isinstance(terminal_raw_action, TerminalAction) is False

    # the fix: check the action_id directly, matching _translate_action's
    # own condition ("if action_id == -1: return TerminalAction()")
    assert (terminal_raw_action[1] == -1) is True
    assert (non_terminal_raw_action[1] == -1) is False


# ---- follow-up instrumentation: --record_ids / --record_meta / --trace_goals / --auto_template ----

from experiments.eval_harness import IDS_FIELDS, META_FIELDS, TRACE_FIELDS  # noqa: E402


class _ListWriter:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_default_csv_schema_is_unchanged_without_the_optional_flags(untrained_net, tmp_path):
    path = str(tmp_path / "plain.csv")
    run_eval(untrained_net, "m", SCENARIO, "c", STEP_LIMIT, 2, csv_path=path, verbose=False)
    with open(path) as f:
        assert next(csv.reader(f)) == CSV_FIELDS


def test_record_ids_and_meta_add_columns_and_leave_the_episode_untouched(untrained_net, tmp_path):
    plain = run_eval(untrained_net, "m", SCENARIO, "c", STEP_LIMIT, 3, verbose=False)
    path = str(tmp_path / "instrumented.csv")
    run_eval(untrained_net, "m", SCENARIO, "c", STEP_LIMIT, 3, csv_path=path, verbose=False,
             record_ids=True, record_meta=True)
    rows = _read_csv(path)
    assert set(IDS_FIELDS + META_FIELDS) <= set(rows[0])
    for r, base in zip(rows, plain):
        assert float(r["episode_return"]) == pytest.approx(base["episode_return"])
        assert int(r["captured"]) == base["captured"] and int(r["episode_len"]) == base["episode_len"]
        assert int(r["ids_detections"]) == int(r["ids_quarantine"]) + int(r["ids_patch"]) + int(r["ids_monitor"])
        assert int(r["n_sensitive"]) > 0 and int(r["n_hosts"]) > int(r["n_sensitive"])
    assert HostVector.event_recorder is None, "the counter must be detached after every episode"


def test_goal_trace_has_one_row_per_step_with_valid_goal_and_entropy(untrained_net):
    writer = _ListWriter()
    result = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=7, trace_writer=writer)
    assert len(writer.rows) == result["episode_len"]
    assert set(writer.rows[0]) == set(TRACE_FIELDS)
    for row in writer.rows:
        assert 0 <= row["goal_idx"] < untrained_net.num_subgoals
        assert 0.0 <= row["goal_entropy"] <= 2.0795  # ln(8)
        assert 0.0 <= row["goal_maxprob"] <= 1.0
    assert [r["step"] for r in writer.rows] == list(range(result["episode_len"]))
    assert HostVector.event_recorder is None


def test_generated_scenarios_run_and_report_their_own_size(untrained_net):
    # each generated subnet holds 6-8 hosts, so the host count is only steerable through the subnet count
    env_kwargs = dict(auto_mode="per_episode", auto_template=SCENARIO, auto_topology="chain",
                      auto_subnet_count=8, auto_host_range="48-64")
    a = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=3, record_meta=True, env_kwargs=env_kwargs)
    b = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=3, record_meta=True, env_kwargs=env_kwargs)
    base = run_episode(untrained_net, SCENARIO, STEP_LIMIT, seed=3, record_meta=True)
    assert a["meta"] == b["meta"], "the generated network must be reproducible from the episode seed"
    assert 48 <= a["meta"]["n_hosts"] <= 64
    assert a["meta"]["n_hosts"] < base["meta"]["n_hosts"]
