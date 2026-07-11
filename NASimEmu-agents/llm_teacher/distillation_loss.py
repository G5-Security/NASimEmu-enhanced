"""
Confidence-weighted cross-entropy distillation loss -- master plan Sec 15.9,
Lgoal term only.

LKD (the temperature-scaled soft-label KL term Sec 15.9 also defines) is
deliberately NOT implemented: it requires a full teacher probability
distribution over the 8-way ontology, and Sec 15.9 explicitly warns "do not
fabricate soft probabilities if the teacher only exposes a discrete choice."
teacher_client.py's local model returns exactly one discrete goal plus a
scalar confidence -- never a distribution -- so LKD has no valid input here.
"""
import torch
import torch.nn.functional as F


def goal_distillation_loss(subgoal_logits, teacher_goal_idx, teacher_confidence, w_min=0.05):
    """
    subgoal_logits: [N, num_subgoals] raw logits, e.g. from
        NASimNetDHRL.forward(..., only_subgoal_logits=True).
    teacher_goal_idx: LongTensor [N], values in [0, num_subgoals).
    teacher_confidence: FloatTensor [N] in [0, 1] -- clipped away from 0 so a
        low-confidence label still contributes rather than being silently
        dropped (Sec 15.9: "wt = clip(ct, wmin, 1)").
    """
    w = teacher_confidence.clamp(min=w_min, max=1.0)
    log_p = F.log_softmax(subgoal_logits, dim=-1)
    nll = F.nll_loss(log_p, teacher_goal_idx, reduction="none")
    return (w * nll).mean()
