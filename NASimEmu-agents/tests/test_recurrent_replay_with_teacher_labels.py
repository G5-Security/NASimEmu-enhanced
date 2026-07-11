"""Master plan Sec 15.10: 'the teacher must never live inside forward()' in a
way that corrupts recurrent replay -- forward() is invoked during rollout,
target-value computation, PPO replay, and (per this project's --llm_distill
addition) interleaved distillation warm-start/joint-training steps. This test
proves the only_subgoal_logits=True path (the one the distillation loss
actually calls) is truly side-effect-free on the ongoing rollout: same
recurrent hidden state, same persisted subgoal state, and -- critically --
the exact same action distribution on the next ordinary forward() call,
whether or not a distillation call was interleaved in between.

Constructs a real NASimNetDHRL (via llm_teacher.label_states._build_dhrl_net,
already exercised by that module's own smoke path) against a real env
observation, since the property under test is about actual persisted
nn.Module state, not something mockable without losing the point.
"""
import os

import torch

from nasimemu import env_utils
from nasimemu.env import NASimEmuEnv
from llm_teacher.label_states import _build_dhrl_net

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


def _graph_obs(env):
    return env_utils.convert_to_graph(env.s_raw, env.subnet_graph, version=2)


def _rollout_two_steps(net, env):
    """Two ordinary forward() calls, as PPO's rollout loop would make them."""
    s1 = [_graph_obs(env)]
    with torch.no_grad():
        a1, v1, pi1, raw1 = net(s1)
    env.step(a1[0])
    s2 = [_graph_obs(env)]
    with torch.no_grad():
        a2, v2, pi2, raw2 = net(s2)
    return a2, v2, pi2, raw2


def _fresh_net_and_env(checkpoint_path=None):
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=50, random_init=True,
                       observation_format="graph_v2", seed=42)
    env.reset()
    net = _build_dhrl_net(SCENARIO, 50, checkpoint_path=checkpoint_path)
    net.eval()  # deterministic (argmax) sampling so the two runs are directly comparable
    net.reset_state()
    return net, env


def test_only_subgoal_logits_call_does_not_consume_rng_or_change_hidden_state():
    torch.manual_seed(0)
    net_a, env_a = _fresh_net_and_env()
    a2_no_interleave, v2_no_interleave, _pi, raw2_no_interleave = _rollout_two_steps(net_a, env_a)
    hidden_no_interleave = net_a.manager_gnn.hidden.clone()

    torch.manual_seed(0)
    net_b, env_b = _fresh_net_and_env()
    # simulate main.py's warm-start/joint-distillation step landing between
    # the two rollout steps
    s_mid = [_graph_obs(env_b)]
    with torch.no_grad():
        subgoal_logits = net_b(s_mid, only_subgoal_logits=True, reset_hidden=True)
    assert subgoal_logits.shape[-1] == net_b.num_subgoals

    # replay the identical two rollout steps net_a took (same seed -> same env
    # draws, since both envs were constructed with seed=42)
    a2_interleaved, v2_interleaved, _pi, raw2_interleaved = _rollout_two_steps(net_b, env_b)
    hidden_interleaved = net_b.manager_gnn.hidden.clone()

    assert torch.allclose(hidden_no_interleave, hidden_interleaved, atol=1e-6)
    assert torch.equal(raw2_no_interleave[0], raw2_interleaved[0])  # action_selected
    assert torch.equal(raw2_no_interleave[1], raw2_interleaved[1])  # terminate
    assert torch.equal(raw2_no_interleave[2], raw2_interleaved[2])  # selected_subgoals
    assert torch.allclose(v2_no_interleave, v2_interleaved, atol=1e-6)


def test_only_subgoal_logits_restores_caller_batch_size_state():
    """reset_hidden=True's save/restore must hand back the exact persistent
    state the caller had, even though the only_subgoal_logits call itself may
    run at a different (e.g. distillation minibatch) batch size."""
    net, env = _fresh_net_and_env()
    s1 = [_graph_obs(env)]
    with torch.no_grad():
        net(s1)  # establishes batch-size-1 persistent state

    subgoal_idx_before = net.current_subgoal_idx.clone()
    steps_remaining_before = net.subgoal_steps_remaining.clone()

    # distillation minibatch of size 4 -- a different batch size than the
    # ongoing rollout's batch size of 1
    fake_batch = [s1[0]] * 4
    with torch.no_grad():
        net(fake_batch, only_subgoal_logits=True, reset_hidden=True)

    assert net.current_subgoal_idx.shape == subgoal_idx_before.shape
    assert torch.equal(net.current_subgoal_idx, subgoal_idx_before)
    assert torch.equal(net.subgoal_steps_remaining, steps_remaining_before)


def test_only_subgoal_logits_gradients_flow_only_through_shared_encoder_and_subgoal_head():
    """goal_distillation_loss backprops through only_subgoal_logits's output --
    confirm that stays confined to subgoal_head + upstream shared layers, and
    does not touch worker/value/termination-only parameters (which would mean
    the distillation step is silently nudging unrelated behavior)."""
    from llm_teacher.distillation_loss import goal_distillation_loss

    net, env = _fresh_net_and_env()
    net.train()
    s1 = [_graph_obs(env)]

    subgoal_logits = net(s1, only_subgoal_logits=True, reset_hidden=True)
    loss = goal_distillation_loss(subgoal_logits, torch.tensor([0]), torch.tensor([1.0]))
    loss.backward()

    assert net.subgoal_head[0].weight.grad is not None
    assert net.subgoal_head[0].weight.grad.abs().sum().item() > 0

    worker_only_params = [net.action_head.weight, net.value_head[0].weight, net.termination_head[0].weight]
    for p in worker_only_params:
        assert p.grad is None or p.grad.abs().sum().item() == 0.0
