"""Master plan Sec 15.9, Lgoal term. Confirms the confidence-weighted CE
behaves as specified: low-confidence labels are clipped away from zero (still
contribute) rather than silently dropped, and the loss actually rewards
matching the teacher's label."""
import torch

from llm_teacher.distillation_loss import goal_distillation_loss


def _one_hot_logits(idx, n=8, correct_logit=10.0):
    logits = torch.full((n,), -correct_logit / (n - 1))
    logits[idx] = correct_logit
    return logits


def test_confident_correct_prediction_has_near_zero_loss():
    logits = _one_hot_logits(3).unsqueeze(0)
    idx = torch.tensor([3])
    conf = torch.tensor([1.0])
    loss = goal_distillation_loss(logits, idx, conf)
    assert loss.item() < 1e-3


def test_confident_wrong_prediction_has_large_loss():
    logits = _one_hot_logits(3).unsqueeze(0)  # net is confident about class 3
    idx = torch.tensor([5])  # teacher says 5
    conf = torch.tensor([1.0])
    loss = goal_distillation_loss(logits, idx, conf)
    assert loss.item() > 5.0


def test_zero_confidence_label_still_contributes_via_wmin_clip():
    logits = torch.zeros(1, 8)  # uniform prediction
    idx = torch.tensor([2])
    zero_conf_loss = goal_distillation_loss(logits, idx, torch.tensor([0.0]), w_min=0.05)
    unclipped_equiv = torch.log(torch.tensor(8.0)) * 0.05  # nll for uniform dist * w_min
    assert zero_conf_loss.item() > 0.0
    assert abs(zero_conf_loss.item() - unclipped_equiv.item()) < 1e-4


def test_higher_confidence_label_weighted_more_in_batch_mean():
    # two examples, both equally "wrong" in log-prob terms, but one has much
    # higher teacher confidence -- the batch-mean loss must be closer to the
    # high-confidence example's per-example loss scaled by its own weight,
    # i.e. confidence actually changes the gradient magnitude, not just cosmetically
    logits = torch.zeros(2, 8)
    idx = torch.tensor([0, 0])
    conf_low = torch.tensor([0.05, 0.05])
    conf_high = torch.tensor([1.0, 1.0])
    loss_low = goal_distillation_loss(logits, idx, conf_low)
    loss_high = goal_distillation_loss(logits, idx, conf_high)
    assert loss_high.item() > loss_low.item()


def test_loss_is_nonnegative_and_scalar():
    logits = torch.randn(4, 8)
    idx = torch.tensor([0, 1, 2, 3])
    conf = torch.tensor([0.5, 0.6, 0.7, 0.8])
    loss = goal_distillation_loss(logits, idx, conf)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_confidence_above_one_is_clamped_not_amplified():
    logits = torch.zeros(1, 8)
    idx = torch.tensor([0])
    loss_over = goal_distillation_loss(logits, idx, torch.tensor([5.0]))  # should clamp to 1.0
    loss_at_one = goal_distillation_loss(logits, idx, torch.tensor([1.0]))
    assert abs(loss_over.item() - loss_at_one.item()) < 1e-6


# --- Phase 1 anti-collapse additions: class weighting + label smoothing ---

def test_defaults_are_backward_compatible():
    # class_weights=None, label_smoothing=0.0 must reproduce the original NLL
    # exactly (an omitted-argument call must not change any existing run).
    torch.manual_seed(0)
    logits = torch.randn(6, 8)
    idx = torch.tensor([0, 1, 2, 3, 4, 5])
    conf = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    base = goal_distillation_loss(logits, idx, conf)
    explicit = goal_distillation_loss(
        logits, idx, conf, class_weights=None, label_smoothing=0.0
    )
    assert abs(base.item() - explicit.item()) < 1e-9


def test_class_weights_reweight_minority_class():
    # Two identically-wrong samples for classes 0 and 1; up-weighting class 1
    # must raise the batch-mean loss vs an unweighted call.
    logits = torch.zeros(2, 8)
    idx = torch.tensor([0, 1])
    conf = torch.tensor([1.0, 1.0])
    unweighted = goal_distillation_loss(logits, idx, conf)
    weights = torch.tensor([1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weighted = goal_distillation_loss(logits, idx, conf, class_weights=weights)
    assert weighted.item() > unweighted.item()


def test_label_smoothing_softens_confident_correct_target():
    # A perfectly-confident correct prediction has ~0 NLL, but with label
    # smoothing the target is no longer one-hot, so a strictly positive loss
    # remains -- exactly the "never push to a hard peak" property we want.
    logits = _one_hot_logits(3, correct_logit=20.0).unsqueeze(0)
    idx = torch.tensor([3])
    conf = torch.tensor([1.0])
    hard = goal_distillation_loss(logits, idx, conf, label_smoothing=0.0)
    smoothed = goal_distillation_loss(logits, idx, conf, label_smoothing=0.1)
    assert hard.item() < 1e-3
    assert smoothed.item() > hard.item()


def test_label_smoothing_stays_scalar_and_finite():
    torch.manual_seed(1)
    logits = torch.randn(5, 8)
    idx = torch.tensor([0, 2, 4, 6, 7])
    conf = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    loss = goal_distillation_loss(logits, idx, conf, label_smoothing=0.05)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.0
