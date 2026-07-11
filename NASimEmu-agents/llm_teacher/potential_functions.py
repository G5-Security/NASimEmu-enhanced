"""
Offline-authored potential functions phi_g(state) -> [0, 1], one per semantic
goal type in goal_ontology.GOAL_NAMES. Per docs/llm_integration_plan.pdf Sec
5.2 (Phase 2): the plan's full design has an LLM generate and validate several
candidates per goal against sample-rollout correlation. For this first,
smoke-testable pass the functions below are hand-authored directly against
the same fields the existing baseline network already reads from the
observation tensor (see _aggregate_ids_features in nasim_net_base_hrl.py for
the identical "+1 column offset" convention) -- no hidden/privileged state
(true sensitive-host identity, true IDS internals) is used anywhere here, per
the partial-observability constraint in Sec 4.1 of the plan.

7 of the 8 goals have a real, non-constant proxy; RECOVER_OR_REPLAN is left as
an explicit neutral placeholder (phi==0 everywhere) -- documented here rather
than silently faked, to be replaced in a later refinement pass (the plan's
Sec 5.3 explicitly budgets 2-3 refinement rounds, not a single-pass function).
"""

import torch
from torch_scatter import scatter

from nasimemu.nasim.envs.host_vector import HostVector
from .goal_ontology import GOAL_NAMES


def _host_feature(batch_x, idx):
    """Column 0 of batch.x is the node-type flag (0=host, 1=subnet); all
    HostVector fields are offset by +1 -- the same convention already used in
    NASimNetDHRL._aggregate_ids_features for the IDS slice."""
    return batch_x[:, 1 + idx]


def _per_graph_fraction(mask_num, mask_den, batch_ind, num_graphs):
    """Per-graph sum(mask_num)/sum(mask_den), 0 where the denominator is 0
    (e.g. no hosts discovered yet)."""
    num = scatter(mask_num.float(), batch_ind, dim=0, dim_size=num_graphs, reduce="sum")
    den = scatter(mask_den.float(), batch_ind, dim=0, dim_size=num_graphs, reduce="sum")
    return torch.where(den > 0, num / den.clamp_min(1.0), torch.zeros_like(num))


def compute_all_potentials(batch, batch_ind, num_graphs):
    """
    Returns {goal_name: tensor[num_graphs]} of phi_g values in [0, 1].
    `batch`/`batch_ind` are exactly what NASimNetDHRL.prepare_batch produces.
    """
    x = batch.x
    host_mask = (x[:, 0] == 0).float()

    discovered = _host_feature(x, HostVector._discovered_idx) * host_mask
    access = _host_feature(x, HostVector._access_idx)
    has_user_or_root = (access >= 1).float() * host_mask
    has_root = (access >= 2).float() * host_mask
    reachable = _host_feature(x, HostVector._reachable_idx) * host_mask
    detection_level = _host_feature(x, HostVector._detection_level_idx) * host_mask

    svc_proc_lo = 1 + HostVector._service_start_idx
    svc_proc_hi = 1 + HostVector._process_start_idx + HostVector.num_processes
    scanned = (x[:, svc_proc_lo:svc_proc_hi].sum(dim=1) > 0).float() * host_mask

    host_count = host_mask

    phi = {}
    phi["DISCOVER_SUBNET"] = _per_graph_fraction(discovered, host_count, batch_ind, num_graphs)
    phi["ENUMERATE_HOST"] = _per_graph_fraction(scanned, discovered, batch_ind, num_graphs)
    phi["GAIN_INITIAL_ACCESS"] = _per_graph_fraction(has_user_or_root, discovered, batch_ind, num_graphs)
    phi["ESCALATE_PRIVILEGE"] = _per_graph_fraction(has_root, has_user_or_root, batch_ind, num_graphs)
    phi["PIVOT"] = _per_graph_fraction(reachable, host_count, batch_ind, num_graphs)
    # Proxy for sensitive-host capture progress that avoids touching the hidden
    # true-value field: root access is the same threshold env.py's corrected
    # capture-metric fix uses (AccessLevel.ROOT), just aggregated per-graph here
    # instead of counted as a distinct-host set.
    phi["CAPTURE_SENSITIVE_HOST"] = _per_graph_fraction(has_root, discovered, batch_ind, num_graphs)
    mean_detection = _per_graph_fraction(detection_level, discovered, batch_ind, num_graphs)
    phi["REDUCE_DETECTION"] = 1.0 - mean_detection
    phi["RECOVER_OR_REPLAN"] = torch.zeros(num_graphs, device=x.device)  # placeholder, see module docstring

    assert set(phi.keys()) == set(GOAL_NAMES)
    return phi


def select_potential(phi_dict, goal_indices):
    """
    phi_dict: output of compute_all_potentials.
    goal_indices: LongTensor[num_graphs], the active semantic-goal index per
    graph (raw_a[2] from NASimNetDHRL.forward, i.e. g_t).
    Returns tensor[num_graphs] = phi_{g}(state) for each graph's active goal.
    """
    stacked = torch.stack([phi_dict[name] for name in GOAL_NAMES], dim=1)  # [num_graphs, 8]
    return torch.gather(stacked, 1, goal_indices.view(-1, 1).long()).squeeze(1)
