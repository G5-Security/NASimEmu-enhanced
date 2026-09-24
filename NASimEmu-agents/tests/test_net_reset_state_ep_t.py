"""Regression test for the ep_t leak in NASimNetDHRL.reset_state(batch_mask=None)
(nasim_problem/nasim_net_base_hrl.py), found while building the Step 1
evaluation harness (docs/eval_revision_plan.tex, experiments/eval_harness.py).

reset_state(batch_mask=None) is the "start a fresh episode on this net
object" call used by every serial episode loop in this repo
(experiments/evaluate_llm_selector.py, experiments/eval_harness.py). It used
to clear every piece of manager state (current subgoal, goal vector, GRU
hidden state) except self.ep_t, which forward() only re-initializes lazily
when it is None or the wrong batch size. So a second episode run on the same
net object silently inherited the first episode's step count, corrupting the
`too_early` termination gate (forward(): "too_early = self.ep_t.flatten() <
min_steps") from the second episode onward -- a real, silent, non-obvious
cross-episode state leak, not a training or environment bug.
"""
import torch

from experiments.eval_harness import build_net

SCENARIO = "../scenarios/corp_100hosts_dynamic.v2.yaml"


def test_full_reset_clears_ep_t():
    net = build_net("NASimNetDHRL", SCENARIO, step_limit=30, checkpoint_path=None)

    # Simulate having already stepped through part of an episode.
    net.ep_t = torch.tensor([17])

    net.reset_state()  # batch_mask=None: "start a fresh episode"

    assert net.ep_t is None, (
        "reset_state(batch_mask=None) left a stale ep_t from a previous "
        "episode/step in place -- this leaks the previous episode's step "
        "count into the next episode's termination gating"
    )


def test_masked_reset_still_zeros_ep_t_for_masked_envs():
    # The masked (per-environment) branch was already correct; this just
    # pins that behavior so a future change to the unmasked branch can't
    # regress it by accident.
    net = build_net("NASimNetDHRL", SCENARIO, step_limit=30, checkpoint_path=None)
    net.ep_t = torch.tensor([5, 9])

    net.reset_state(batch_mask=[True, False])

    assert net.ep_t.tolist() == [0, 9]
