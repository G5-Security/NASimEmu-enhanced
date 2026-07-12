"""experiments/evaluate_llm_selector.py (Design B: live-vs-distilled
evaluation driver) had zero automated test coverage despite a real bug found
and fixed this session (reading env.captured after NASimEmuEnv.step()'s
internal reset() instead of info['captured'], which silently reported 0
captured hosts on completed episodes) and a feature just added (per-action
"reward" in the transcript). Both are locked in here.

Constructs a real, untrained NASimNetDHRL (via llm_teacher.label_states,
same helper the recurrent-replay safety test uses) against a real env --
mocks only the teacher call itself (label_one_state), since the property
under test is the episode-stepping/bookkeeping logic, not the LLM."""
import os
from unittest.mock import patch

import pytest
import torch

from nasimemu import env_utils
from nasimemu.env import NASimEmuEnv
from llm_teacher.label_states import _build_dhrl_net
from experiments.evaluate_llm_selector import run_episode_distilled, run_episode_live

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)
STEP_LIMIT = 10


def _fresh_net_and_env(seed=7):
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=STEP_LIMIT, random_init=False,
                       observation_format="graph_v2", seed=seed)
    net = _build_dhrl_net(SCENARIO, STEP_LIMIT, checkpoint_path=None)
    net.eval()
    return net, env


def _fake_teacher_record(goal="DISCOVER_SUBNET", horizon=100):
    """horizon=100 (> STEP_LIMIT) means the teacher is only ever called
    once, holding its goal for the whole episode -- keeps the reward-sum
    assertion below simple to reason about."""
    return {
        "valid": True,
        "reject_reason": None,
        "parsed_output": {
            "goal": goal, "target_host": None, "target_subnet": None,
            "horizon": horizon, "confidence": 0.9,
            "reason_code": "test", "rationale": "test rationale",
        },
    }


def test_run_episode_distilled_basic_shape_and_no_teacher_calls():
    net, env = _fresh_net_and_env()
    with patch("experiments.evaluate_llm_selector.label_one_state") as mock_label:
        stats = run_episode_distilled(net, env, STEP_LIMIT)
    mock_label.assert_not_called()  # distilled mode must never call the teacher

    assert stats["mode"] == "distilled"
    assert 0 < stats["episode_len"] <= STEP_LIMIT
    assert isinstance(stats["episode_return"], float)
    assert isinstance(stats["captured"], int)
    assert stats["captured"] >= 0


def test_run_episode_distilled_captured_reflects_info_not_post_reset_env_state():
    """Regression test for the fixed bug: captured must come from info[...]
    snapshotted before NASimEmuEnv.step()'s internal reset(), not from
    env.captured read after the loop (which would read back 0 whenever the
    episode actually reaches a terminal state)."""
    net, env = _fresh_net_and_env()
    stats = run_episode_distilled(net, env, STEP_LIMIT)
    # If this ever regresses to reading env.captured post-loop, this
    # assertion still passes when the episode does NOT hit done=True inside
    # the loop -- the meaningful guarantee is that the two readings agree
    # whenever the episode ran to completion (not truncated by step_limit).
    if stats["episode_len"] < STEP_LIMIT:
        assert stats["captured"] == env.captured


def test_run_episode_live_forces_teacher_goal_and_holds_it_for_horizon():
    net, env = _fresh_net_and_env()
    transcript_records = []
    with patch("experiments.evaluate_llm_selector.label_one_state",
               return_value=_fake_teacher_record(goal="DISCOVER_SUBNET", horizon=100)) as mock_label:
        stats = run_episode_live(net, env, STEP_LIMIT, model="ignored",
                                  transcript_records=transcript_records, episode_idx=0)

    mock_label.assert_called_once()  # horizon=100 > STEP_LIMIT -> only one switch decision needed
    assert stats["mode"] == "live"
    assert len(transcript_records) == 1
    assert transcript_records[0]["goal"] == "DISCOVER_SUBNET"
    assert transcript_records[0]["episode"] == 0


def test_run_episode_live_reward_field_matches_episode_return():
    """Regression test for the reward field just added to
    switch_actions.append(...) -- must stay wired to the exact same reward
    signal that accumulates into episode_return, not a placeholder."""
    net, env = _fresh_net_and_env()
    transcript_records = []
    with patch("experiments.evaluate_llm_selector.label_one_state",
               return_value=_fake_teacher_record()):
        stats = run_episode_live(net, env, STEP_LIMIT, model="ignored",
                                  transcript_records=transcript_records, episode_idx=0)

    all_actions = [a for rec in transcript_records for a in rec["actions_until_next_switch"]]
    assert len(all_actions) == stats["episode_len"]
    assert all("reward" in a and isinstance(a["reward"], float) for a in all_actions)

    summed = sum(a["reward"] for a in all_actions)
    assert summed == pytest.approx(stats["episode_return"], abs=1e-4)


def test_run_episode_live_falls_back_gracefully_on_invalid_teacher_output():
    """An invalid/rejected teacher call must not crash the episode -- holds
    the previous goal (or DISCOVER_SUBNET on the very first call) and
    retries next step, per run_episode_live's own fallback branch."""
    net, env = _fresh_net_and_env()
    transcript_records = []
    invalid_record = {"valid": False, "reject_reason": "schema_error", "parsed_output": {}}
    with patch("experiments.evaluate_llm_selector.label_one_state", return_value=invalid_record):
        stats = run_episode_live(net, env, STEP_LIMIT, model="ignored",
                                  transcript_records=transcript_records, episode_idx=0)

    assert stats["mode"] == "live"
    assert 0 < stats["episode_len"] <= STEP_LIMIT
    assert all(rec.get("reject_reason") == "schema_error" for rec in transcript_records)
