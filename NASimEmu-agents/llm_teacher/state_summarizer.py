"""
Sanitized, bounded state summarizer -- master plan Sec 15.6.

Builds the JSON object the teacher actually sees. Operates directly on the
raw pre-graph observation rows (env.py's `info['s_raw']`, i.e. what
PartiallyObservableWrapper already accumulates across the episode) using
HostVector's own property accessors -- not the padded/batched torch_geometric
tensor used by the network, so there is no "+1 column offset" bookkeeping
here and the field names are the same ones HostVector itself exposes.

Hard constraint (Sec 15.6 / 15.14): only fields the agent has actually
observed. Concretely this means:
  - only rows with `discovered == 1` are ever included;
  - `HostVector.value` / `.discovery_value` are never read here, mirroring
    the same restraint already documented in potential_functions.py -- they
    encode the "true" sensitive-host label, which is exactly the kind of
    simulator-only fact Sec 15.6 forbids handing to the teacher;
  - no firewall rules, no undiscovered hosts/subnets, no raw banner/filename
    text (there isn't any in this codebase's HostVector -- services/os/
    processes are already enumerated, schema-defined booleans, not strings
    pulled from a simulated banner).
"""
import numpy as np

from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel
from .goal_ontology import GOAL_NAMES, GOAL_ONTOLOGY_VERSION

ACCESS_NAMES = {AccessLevel.NONE: "none", AccessLevel.USER: "user", AccessLevel.ROOT: "root"}

MAX_RECENT_ACTIONS = 16


def _addr_str(addr):
    return f"{int(addr[0])}.{int(addr[1])}"


def _known_names(name_val_dict):
    return sorted(name for name, val in name_val_dict.items() if val)


def summarize_state(
    s_raw,
    ep_t,
    ep_limit,
    active_goal_name=None,
    active_goal_remaining_horizon=None,
    recent_actions=None,
):
    """
    s_raw: the raw observation rows for the CURRENT episode, i.e.
        NASimEmuEnv.s_raw / info['s_raw'] -- a list/array of HostVector rows
        plus a trailing action-result row (mirrors PartiallyObservableWrapper,
        which is exactly what this function must NOT depend on more than that
        wrapper already exposes to the agent).
    recent_actions: optional list of {"action": str, "target": "s.h", "success": bool},
        oldest first; only the last MAX_RECENT_ACTIONS are kept.
    Returns a JSON-serializable dict (Sec 15.6 schema).
    """
    host_rows = s_raw[:-1] if len(s_raw) > 0 else s_raw

    subnets = {}
    hosts = []
    for row in host_rows:
        hv = HostVector(np.asarray(row, dtype=np.float32))
        addr = hv.address
        if addr == (0, 0):
            continue
        if not bool(hv.discovered):
            continue

        subnet_id = int(addr[0])
        entry = subnets.setdefault(subnet_id, {"id": str(subnet_id), "discovered_hosts": 0, "compromised_hosts": 0})
        entry["discovered_hosts"] += 1
        if bool(hv.compromised):
            entry["compromised_hosts"] += 1

        access_val = int(hv.access)
        hosts.append({
            "addr": _addr_str(addr),
            "reachable": bool(hv.reachable),
            "compromised": bool(hv.compromised),
            "access": ACCESS_NAMES.get(access_val, "none"),
            "os": _known_names(hv.os),
            "services": _known_names(hv.services),
            "processes": _known_names(hv.processes),
            "ids_level": float(hv.vector[hv._detection_level_idx]) if hv._detection_level_idx is not None else 0.0,
            "ids_threshold": float(hv.vector[hv._detection_threshold_idx]) if hv._detection_threshold_idx is not None else 0.0,
        })

    recent = list(recent_actions or [])[-MAX_RECENT_ACTIONS:]

    return {
        "ontology_version": GOAL_ONTOLOGY_VERSION,
        "step": int(ep_t),
        "remaining_frac": float(max(0.0, min(1.0, 1.0 - (ep_t / max(1, ep_limit))))),
        "subnets": [subnets[k] for k in sorted(subnets.keys())],
        "hosts": hosts,
        "recent_actions": recent,
        "active_goal": active_goal_name,
        "active_goal_remaining_horizon": (
            int(active_goal_remaining_horizon) if active_goal_remaining_horizon is not None else None
        ),
        "available_goals": list(GOAL_NAMES),
    }


def known_addrs(summary):
    """Every host/subnet id actually present in the summary -- the validator's
    anti-hallucination check rejects any teacher-named target not in this set."""
    host_addrs = {h["addr"] for h in summary["hosts"]}
    subnet_ids = {s["id"] for s in summary["subnets"]}
    return host_addrs, subnet_ids
