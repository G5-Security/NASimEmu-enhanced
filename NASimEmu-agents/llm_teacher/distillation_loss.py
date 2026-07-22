"""
Confidence-weighted cross-entropy distillation loss -- master plan Sec 15.9,
Lgoal term only.

LKD (the temperature-scaled soft-label KL term Sec 15.9 also defines) is
deliberately NOT implemented: it requires a full teacher probability
distribution over the 8-way ontology, and Sec 15.9 explicitly warns "do not
fabricate soft probabilities if the teacher only exposes a discrete choice."
teacher_client.py's local model returns exactly one discrete goal plus a
scalar confidence -- never a distribution -- so LKD has no valid input here.

Anti-collapse additions (Phase 1): the teacher dataset is class-imbalanced
(mostly GAIN_INITIAL_ACCESS/ESCALATE_PRIVILEGE), and a plain NLL toward the
argmax label peaks subgoal_head on the majority class during the Stage-1
warm-start -- a direct driver of the "manager converges to one goal" failure.
Two optional, target-side regularizers address that without fabricating a
teacher distribution:
  * ``class_weights`` -- per-goal multipliers (e.g. inverse frequency),
    re-weighting the SAME discrete labels so rare goals aren't drowned out.
  * ``label_smoothing`` -- a small uniform floor spread over the ontology, so
    the target is never a hard one-hot the head can collapse onto.
Both default off, so an omitted-argument call is byte-for-byte the original
confidence-weighted NLL.
"""
import torch
import torch.nn.functional as F


def goal_distillation_loss(
    subgoal_logits,
    teacher_goal_idx,
    teacher_confidence,
    w_min=0.05,
    class_weights=None,
    label_smoothing=0.0,
):
    """
    subgoal_logits: [N, num_subgoals] raw logits, e.g. from
        NASimNetDHRL.forward(..., only_subgoal_logits=True).
    teacher_goal_idx: LongTensor [N], values in [0, num_subgoals).
    teacher_confidence: FloatTensor [N] in [0, 1] -- clipped away from 0 so a
        low-confidence label still contributes rather than being silently
        dropped (Sec 15.9: "wt = clip(ct, wmin, 1)").
    class_weights: optional FloatTensor [num_subgoals]. Multiplies each
        sample's loss by the weight of its teacher class (anti-collapse
        class balancing). None -> all-ones (original behavior).
    label_smoothing: optional float in [0, 1). Replaces the hard one-hot
        target with (1 - eps) on the teacher class + eps/num_subgoals spread
        uniformly, so the head is never pushed to a degenerate one-hot peak.
        0.0 -> exact NLL (original behavior).
    """
    num_subgoals = subgoal_logits.size(-1)
    w = teacher_confidence.clamp(min=w_min, max=1.0)
    if class_weights is not None:
        # weight each sample by the (frequency-corrected) weight of its own
        # teacher label; keep it on the logits' device/dtype
        cw = class_weights.to(device=subgoal_logits.device, dtype=w.dtype)
        w = w * cw[teacher_goal_idx]

    log_p = F.log_softmax(subgoal_logits, dim=-1)

    if label_smoothing and label_smoothing > 0.0:
        eps = float(label_smoothing)
        with torch.no_grad():
            target = torch.full_like(log_p, eps / num_subgoals)
            target.scatter_(
                1,
                teacher_goal_idx.unsqueeze(1),
                1.0 - eps + eps / num_subgoals,
            )
        per_sample = -(target * log_p).sum(dim=-1)
    else:
        per_sample = F.nll_loss(log_p, teacher_goal_idx, reduction="none")

    return (w * per_sample).mean()
