import torch

from rl import _replay_batch


class FeedForwardPolicy:
    """Deliberately has no recurrent-only ``reset_hidden`` argument."""

    def __call__(self, states, force_action=None):
        assert states == ['state']
        assert force_action == ['action']
        return None, torch.tensor([[2.0]]), torch.tensor([[0.25]]), None


class ResetAwarePolicy:
    def __init__(self):
        self.reset_hidden = None

    def __call__(self, states, force_action=None, reset_hidden=False):
        self.reset_hidden = reset_hidden
        return None, torch.tensor([[3.0]]), torch.tensor([[0.5]]), None


def test_feed_forward_replay_does_not_receive_recurrent_keyword():
    value, probability = _replay_batch(
        FeedForwardPolicy(), ['state'], ['action'], reset_hidden=False,
    )

    assert value.tolist() == [2.0]
    assert probability.tolist() == [0.25]


def test_replay_forwards_explicit_hidden_reset_to_aware_policy():
    policy = ResetAwarePolicy()

    _replay_batch(policy, ['state'], ['action'], reset_hidden=True)

    assert policy.reset_hidden is True
