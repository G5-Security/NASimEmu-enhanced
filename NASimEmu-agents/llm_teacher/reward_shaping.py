"""
Corrected potential-based reward shaping over the augmented (state,
active-subgoal) space. Implements docs/llm_integration_plan.pdf Sec 4 exactly:

    F_t = gamma * lambda_{t+1} * Phi(x_{t+1}) - lambda_t * Phi(x_t),
    x_t = (s_t, g_t), using the REALIZED next subgoal g_{t+1} (never a frozen
    copy of g_t), with terminal potentials fixed to 0.

lambda is annealed across GLOBAL training steps but held constant *within* a
single PPO rollout window (main.py's config.ppo_t-step inner loop), so
lambda_t == lambda_{t+1} for every non-boundary step in a window and the
dynamic form above reduces to lambda * (gamma*Phi(x_{t+1}) - Phi(x_t)) -- this
is an exact instance of the corrected formula, not an approximation of it,
because lambda's own update granularity is defined at the window level.

Everything here runs under torch.no_grad() and is computed once at rollout
collection time, then cached into the reward before main.py calls
net.update(...) -- net() itself never calls into this module, satisfying the
"never recompute during PPO replay" constraint from Sec 4.2/ELLM (Sec 3.3) of
the plan.
"""

import numpy as np
import torch

from .potential_functions import compute_all_potentials, select_potential


def anneal_lambda(step, lambda_start=1.0, lambda_min=0.0, rate=8000.0):
    """Exponential decay of the shaping weight across global training steps."""
    return lambda_min + (lambda_start - lambda_min) * float(np.exp(-step / rate))


def compute_shaping_terms(trace, net, gamma, lambda_t, device):
    """
    trace: list of (s_orig, raw_a, a_cnt, r, s_true, d_true) tuples of length
           config.ppo_t, exactly as built by main.py's inner rollout loop.
           raw_a[2] is g_t, the active-subgoal index batch that conditioned
           the worker when this step's action was produced.
    net:   the live net instance, used only for its static prepare_batch
           parser (no gradients, no side effects on net's persistent state).
    Returns: list of length ppo_t of numpy float32 arrays, shape [config.batch]
             each -- the shaping term F_t to ADD to that step's env reward.

    Window-boundary handling (Sec 4.3, "what this does and does not
    guarantee"): g_{tau+1} is only known once trace[tau+1] exists, so the last
    step in the window can't be shaped without peeking into the next window
    (an extra forward pass that would also perturb the recurrent hidden
    state). If that boundary step is itself terminal, Phi(x_{t+1})=0 is exact
    and used; otherwise F=0 is emitted for that one step -- neutral, not
    biased, and affects at most 1/ppo_t of all steps.
    """
    ppo_t = len(trace)
    phis = []
    dones = []

    with torch.no_grad():
        for tau in range(ppo_t):
            s_orig, raw_a, a_cnt, r, s_true, d_true = trace[tau]
            g_tau = raw_a[2].to(device)
            _, data_lens, batch, batch_ind, _, _ = net.prepare_batch(s_orig)
            num_graphs = len(data_lens)
            phi_dict = compute_all_potentials(batch, batch_ind, num_graphs)
            phis.append(select_potential(phi_dict, g_tau))
            dones.append(np.asarray(d_true))

        shaping = []
        for tau in range(ppo_t):
            done_mask = torch.tensor(dones[tau], dtype=torch.bool, device=phis[tau].device)
            if tau < ppo_t - 1:
                phi_next = torch.where(done_mask, torch.zeros_like(phis[tau + 1]), phis[tau + 1])
                f_tau = lambda_t * (gamma * phi_next - phis[tau])
            else:
                f_tau = torch.where(
                    done_mask,
                    lambda_t * (0.0 - phis[tau]),
                    torch.zeros_like(phis[tau]),
                )
            shaping.append(f_tau.cpu().numpy().astype(np.float32))

    return shaping


def apply_shaping_to_trace(trace, shaping):
    """Returns a new trace list with each step's r replaced by r + F_t."""
    new_trace = []
    for tau, (s_orig, raw_a, a_cnt, r, s_true, d_true) in enumerate(trace):
        shaped_r = np.asarray(r) + shaping[tau]
        new_trace.append((s_orig, raw_a, a_cnt, shaped_r, s_true, d_true))
    return new_trace
