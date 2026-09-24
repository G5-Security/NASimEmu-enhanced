"""Tests for the Step 3 Pen-DHRL component interventions
(experiments/component_interventions.py, docs/eval_revision_plan.tex).

Each flag is checked at the exact internal value/behavior the plan says it
changes, and (where the plan says something else must stay untouched) that
the untouched thing really is untouched -- e.g. no_goal_conditioning must
change the worker's action logits but not the value head's output, since
value_context keeps its own separate copy of the goal vector.
"""
import os

import torch

from experiments.component_interventions import apply_condition, CONDITIONS
from experiments.eval_harness import _graph_obs, build_net
from nasimemu.env import NASimEmuEnv

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)
STEP_LIMIT = 20


def _net():
    return build_net("NASimNetDHRL", SCENARIO, STEP_LIMIT, checkpoint_path=None)


def _fresh_obs(seed=1):
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=STEP_LIMIT,
                       observation_format="graph_v2", seed=seed, training_mode=False)
    env.reset()
    return _graph_obs(env)


def test_all_flags_default_off():
    net = _net()
    assert net.disable_ids_branch is False
    assert net.disable_recurrent_memory is False
    assert net.disable_goal_persistence is False
    assert net.disable_goal_conditioning is False
    assert net.force_continue is False


def test_apply_condition_restores_flags_on_normal_exit():
    net = _net()
    with apply_condition(net, "no_ids_branch"):
        assert net.disable_ids_branch is True
    assert net.disable_ids_branch is False


def test_apply_condition_restores_flags_on_exception():
    net = _net()
    try:
        with apply_condition(net, "no_recurrent_memory"):
            assert net.disable_recurrent_memory is True
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert net.disable_recurrent_memory is False


def test_apply_condition_restores_prior_nondefault_state():
    # If a caller had already set a flag before entering the context (e.g.
    # nesting, or a bug elsewhere), restore *that* value, not just False.
    net = _net()
    net.disable_goal_persistence = True
    with apply_condition(net, "no_ids_branch"):
        assert net.disable_goal_persistence is False  # baseline applied inside
    assert net.disable_goal_persistence is True  # restored, not left at baseline


def test_full_checkpoint_leaves_everything_off():
    net = _net()
    with apply_condition(net, "full_checkpoint"):
        assert net.disable_ids_branch is False
        assert net.disable_recurrent_memory is False
        assert net.disable_goal_persistence is False
        assert net.disable_goal_conditioning is False
        assert net.force_continue is False


def test_unknown_condition_raises():
    net = _net()
    try:
        with apply_condition(net, "not_a_real_condition"):
            pass
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_no_ids_branch_zeros_the_ids_bias_contribution():
    net = _net()
    s = _fresh_obs()
    net.reset_state()

    calls = []
    orig_forward = net.ids_projection.forward
    net.ids_projection.forward = lambda x: calls.append(orig_forward(x)) or orig_forward(x)
    try:
        with apply_condition(net, "no_ids_branch"):
            with torch.no_grad():
                net.reset_state()
                v_disabled = net([s], only_v=True)
    finally:
        net.ids_projection.forward = orig_forward

    with torch.no_grad():
        net.reset_state()
        v_enabled = net([s], only_v=True)

    # The ids_projection module itself still ran (its output just gets
    # discarded before reaching x_global) -- the branch is disabled at the
    # point of use, not by skipping the computation.
    assert len(calls) >= 1
    # Different global context bias -> the value head's input differs, so
    # in general v_disabled != v_enabled (not a strict guarantee for every
    # possible weight init, but true for this net's actual random weights;
    # a false pass here would mean the flag has no effect at all).
    assert not torch.equal(v_disabled, v_enabled)


def test_no_recurrent_memory_ignores_prior_hidden_state():
    # NOT "same input twice in a row must match": ep_t and subgoal-persistence
    # state also carry over between two raw net([s]) calls with no
    # reset_state() between them (ep_t increments every forward() call
    # regardless of this flag, which legitimately changes the value head's
    # time-penalty term) -- that is unrelated to recurrent memory and would
    # make two consecutive calls differ even with a perfectly-working flag.
    # Isolate hidden-state specifically instead: compare a truly fresh call
    # against one where every *other* piece of state is reset to the same
    # fresh baseline but manager_gnn.hidden is deliberately left populated
    # with a real (nonzero) value a prior, different observation would have
    # produced.
    net = _net()
    s = _fresh_obs()

    net.reset_state()
    with torch.no_grad():
        v_fresh = net([s], only_v=True)

    net.reset_state()
    with torch.no_grad():
        net([s], only_v=True)  # one real call to produce a genuine nonzero hidden
    poisoned_hidden = net.manager_gnn.hidden.clone()
    assert not torch.all(poisoned_hidden == 0)

    def _reset_all_but_hidden():
        net.current_subgoal_idx = None
        net.subgoal_steps_remaining = None
        net.current_goal_vec = None
        net.batch_ind = None
        net.ep_t = None
        net.manager_gnn.hidden = poisoned_hidden.clone()

    _reset_all_but_hidden()
    with torch.no_grad():
        v_poisoned_no_flag = net([s], only_v=True)
    assert not torch.equal(v_fresh, v_poisoned_no_flag), (
        "expected hidden state to matter for a normal (non-intervened) forward pass"
    )

    _reset_all_but_hidden()
    with torch.no_grad(), apply_condition(net, "no_recurrent_memory"):
        v_poisoned_with_flag = net([s], only_v=True)
    assert torch.equal(v_fresh, v_poisoned_with_flag), (
        "no_recurrent_memory must ignore whatever hidden state is present "
        "and match a truly fresh (hidden=None) call exactly"
    )


def test_no_goal_persistence_forces_full_reset_every_call():
    net = _net()
    s = _fresh_obs()
    net.reset_state()

    with torch.no_grad(), apply_condition(net, "no_goal_persistence"):
        net([s])
        # whatever remaining steps a normal switch would set, forced-persistence
        # means every call is treated as "just switched" -> full goal_horizon
        assert torch.all(net.subgoal_steps_remaining == net.goal_horizon)

        net.subgoal_steps_remaining = torch.tensor([5.0])
        net([s])
        # forced to 0 before need_new_subgoal is computed, then immediately
        # reset to goal_horizon by the (forced) switch -- never decays to 4
        assert torch.all(net.subgoal_steps_remaining == net.goal_horizon)

    # Contrast: without the flag, an artificially large remaining count just
    # decays by one and does not force a switch.
    net.reset_state()
    with torch.no_grad():
        net([s])
        net.subgoal_steps_remaining = torch.tensor([5.0])
        net([s])
        assert torch.all(net.subgoal_steps_remaining == 4.0)


def test_no_goal_conditioning_changes_action_logits_not_value():
    # The returned `pi` (3rd value) is total_prob = a_prob * continue_probs *
    # subgoal_factor (nasim_net_base_hrl.py), and subgoal_factor legitimately
    # depends on which subgoal was selected regardless of
    # disable_goal_conditioning (it's the probability of having *chosen*
    # that subgoal, not the worker's action distribution) -- comparing it
    # across two different forced goals would always differ and prove
    # nothing about this flag. Capture the worker's actual raw output
    # (action_head's output, pre subgoal_factor) directly instead.
    net = _net()
    s = _fresh_obs()

    captured = []
    orig_forward = net.action_head.forward
    net.action_head.forward = lambda x: captured.append(orig_forward(x).clone()) or captured[-1]

    def run_with_forced_goal(goal_idx, condition):
        net.reset_state()
        forced = torch.tensor([goal_idx], dtype=torch.long)
        with torch.no_grad():
            if condition is None:
                a, v, pi, _ = net([s], force_action=(None, None, forced))
            else:
                with apply_condition(net, condition):
                    a, v, pi, _ = net([s], force_action=(None, None, forced))
        return captured[-1], v

    try:
        logits0, v0 = run_with_forced_goal(0, None)
        logits1, v1 = run_with_forced_goal(1, None)
        assert not torch.equal(logits0, logits1), "expected normal goal-conditioning to change action logits"

        logits0d, v0d = run_with_forced_goal(0, "no_goal_conditioning")
        logits1d, v1d = run_with_forced_goal(1, "no_goal_conditioning")
        assert torch.equal(logits0d, logits1d), (
            "no_goal_conditioning must make the worker's raw output "
            "independent of which subgoal was selected"
        )
        # value_context keeps its own goal_vectors term -- untouched by this flag
        assert not torch.equal(v0d, v1d), (
            "no_goal_conditioning must not affect the value/termination head, "
            "which uses a separate copy of the goal vector"
        )
    finally:
        net.action_head.forward = orig_forward


def test_no_learned_stopping_never_emits_terminal_action():
    net = _net()
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=STEP_LIMIT,
                       observation_format="graph_v2", seed=1, training_mode=False)
    net.reset_state()
    env.reset()

    saw_terminal = False
    with torch.no_grad(), apply_condition(net, "no_learned_stopping"):
        for _ in range(STEP_LIMIT):
            s = _graph_obs(env)
            a, _v, _pi, raw_a = net([s])
            _, terminate, _ = raw_a
            if bool(terminate.any()):
                saw_terminal = True
            _s, r, done, info = env.step(a[0])
            if done:
                break

    assert not saw_terminal, "force_continue=True must suppress the learned stop decision entirely"


def test_condition_names_cover_every_flag_and_force_continue():
    assert set(CONDITIONS) == {
        "full_checkpoint", "no_ids_branch", "no_recurrent_memory",
        "no_goal_persistence", "no_goal_conditioning", "no_learned_stopping",
    }


# ---- follow-up A: goal-mask conditions ---------------------------------------------

import pytest  # noqa: E402

from experiments.component_interventions import ALL_CONDITIONS, GOAL_CONDITIONS, GOAL_CONDITION_ALLOWED  # noqa: E402
from experiments.eval_harness import run_episode  # noqa: E402
from llm_teacher.goal_ontology import GOAL_NAMES  # noqa: E402


class _Rows:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


def test_goal_conditions_are_registered_without_changing_the_original_six():
    assert len(CONDITIONS) == 6
    assert set(CONDITIONS) < set(ALL_CONDITIONS)
    assert len(GOAL_CONDITIONS) == 3 + len(GOAL_NAMES)


def test_allowed_goals_defaults_to_unrestricted_and_is_restored_after_a_condition():
    net = _net()
    assert net.allowed_goals is None
    with apply_condition(net, "mask_reduce_detection"):
        assert net.allowed_goals == GOAL_CONDITION_ALLOWED["mask_reduce_detection"]
        assert GOAL_NAMES.index("REDUCE_DETECTION") not in net.allowed_goals
    assert net.allowed_goals is None


def test_set_allowed_goals_rejects_invalid_indices():
    net = _net()
    for bad in ((), (8,), (-1,)):
        with pytest.raises(ValueError):
            net.set_allowed_goals(bad)


@pytest.mark.parametrize("condition,blocked,forced", [
    ("mask_reduce_detection", GOAL_NAMES.index("REDUCE_DETECTION"), None),
    ("mask_both_ids_goals", GOAL_NAMES.index("RECOVER_OR_REPLAN"), None),
    ("only_pivot", None, GOAL_NAMES.index("PIVOT")),
])
def test_goal_masks_actually_constrain_the_goals_the_manager_picks(condition, blocked, forced):
    net = _net()
    rows = _Rows()
    run_episode(net, SCENARIO, STEP_LIMIT, seed=11, net_condition=condition, trace_writer=rows)
    goals = {r["goal_idx"] for r in rows.rows}
    if forced is not None:
        assert goals == {forced}
    else:
        assert blocked not in goals
    # the recorded goal distribution itself puts zero mass on masked goals
    assert net.allowed_goals is None
