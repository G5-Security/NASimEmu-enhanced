"""Step 2: IDS observation sensitivity analysis (docs/eval_revision_plan.tex).

Corrected against the actual code, not the plan's first sketch
--------------------------------------------------------------
`HostVector.observe()` (src/nasimemu/nasim/envs/host_vector.py) reveals
`detection_level` and `detection_multiplier` to the agent whenever a host is
observed at all, by an explicit, deliberate design choice documented in its
own comment ("The hidden per-host detection *threshold* is deliberately NOT
exposed -- that uncertainty is what makes stealth non-trivial"). Confirmed
empirically: `detection_threshold` is 0.0 in every agent-facing observation
this session constructed, while the true per-host threshold (sampled in
`HostVector.vectorize()`, ~U(0.65, 0.88) under `full_difficulty`) is real and
nonzero in the environment's internal `current_state` tensor.

This means:
  - `detection_threshold` is *already* always 0 in the observation. Any
    "zero it" intervention on that column is close to a no-op; the
    meaningful, genuinely-observed IDS signal is only `detection_level` and
    `detection_multiplier`.
  - An "alert" flag (`level >= threshold`) is not something the agent could
    ever construct for itself from its own observation, since it never sees
    the threshold. Injecting one (this module's `alert_only` condition) is a
    real test-time intervention -- it is allowed to differ from anything the
    policy saw during training, that is the point of a sensitivity analysis
    -- but it hands the policy *more* privileged information than training
    ever did, not less, and should not be described as an ablation of
    existing information.

Serial-harness correction
--------------------------
The plan sketched "shuffled" as swapping a host's IDS values with those of a
random host *in another parallel environment*. experiments/eval_harness.py
runs episodes serially, one env at a time, by design (Step 1's determinism
guarantee), so there is no second parallel environment to draw from at the
point this hook runs. `shuffle` here instead permutes (level, multiplier)
pairs *among the hosts present in the same observation* -- it still breaks
the correspondence between a host's true IDS state and what the policy sees
for that host, which is the property the condition is meant to test.

Interventions
-------------
  normal       unmodified observation (the reference condition)
  zeroed       detection_level, detection_threshold and detection_multiplier
               all set to 0.0 for every host row. Note detection_multiplier's
               real range is >=1.0 (1.0 = baseline, 2.0 after a "monitor"
               response) and is never 0 in a real observation, so this
               specific column's edit is out-of-distribution, unlike
               level/threshold which do take 0 naturally -- interpret a
               "zeroed" effect that seems to come from the multiplier column
               with that in mind.
  shuffled     (detection_level, detection_multiplier) pairs permuted among
               host rows within the current observation (paired per host, so
               a real level+multiplier combination just gets misattributed
               to a different host); detection_threshold untouched (already
               always 0). Deterministic per episode via a local RNG seeded
               from the episode seed, independent of the env/policy RNG
               streams so it cannot perturb their determinism.
  alert_only   detection_level replaced by 1[true_level >= true_threshold]
               (computed from privileged ground truth read off
               env.env.current_state, not from the observation -- see above);
               detection_threshold forced to 0.0 (neutral, matches its
               always-0 normal value); detection_multiplier forced to 1.0
               (neutral/no-extra-monitoring baseline, in-distribution unlike
               `zeroed`'s literal 0 for this column).
  delayed      detection_level/detection_multiplier show the *previous*
               step's true values for each host (looked up by host address,
               since row order is not guaranteed stable across steps as new
               hosts are discovered); both are forced to (0.0, 1.0) -- the
               same neutral baseline as episode start -- on an episode's
               first step and for any host not seen in the previous step.

Sanity check (belongs in Step 1's "tests before large runs" for this
module): every condition above must actually change at least one of the
three IDS columns on a host row that has nonzero ids -- if a condition
already used above ever produces literally the same tensor as `normal`,
the intervention is not doing anything and the run should not be trusted.
"""
import collections

import numpy as np

from nasimemu.nasim.envs.host_vector import HostVector

NEUTRAL_LEVEL = 0.0
NEUTRAL_MULTIPLIER = 1.0  # HostVector default / no-extra-monitoring baseline


def _columns():
    """Column indices into graph_v2 node_feats for the three IDS features.

    Lazy, not module-level constants: HostVector._detection_level_idx etc.
    are None until HostVector._initialize() has run (first triggered by
    constructing/resetting a NASimEmuEnv), so these can't be computed at
    import time. env_utils.convert_to_graph places node_type in column 0
    and the raw HostVector columns (unpermuted) starting at column 1 --
    verified directly against a real observation's host rows this session,
    not assumed from reading convert_to_graph alone.
    """
    assert HostVector._detection_level_idx is not None, (
        "HostVector is not initialized yet -- construct/reset a NASimEmuEnv "
        "(or call build_net(...)) before using ids_interventions"
    )
    return (
        1 + HostVector._detection_level_idx,
        1 + HostVector._detection_threshold_idx,
        1 + HostVector._detection_multiplier_idx,
    )


def _host_rows(node_feats):
    return np.where(node_feats[:, 0] == 0)[0]


def _true_level_and_threshold(env, address):
    """Privileged ground truth for one host, read off the environment's
    internal state tensor (env.env.current_state), not the agent's own
    observation -- detection_threshold is never in the observation (see
    module docstring)."""
    state = env.env.current_state
    row = state.host_num_map[tuple(int(a) for a in address)]
    level = state.tensor[row, HostVector._detection_level_idx]
    threshold = state.tensor[row, HostVector._detection_threshold_idx]
    return float(level), float(threshold)


def _edit(s, mutate):
    node_feats, edge_index, node_index, pos_index = s
    node_feats = node_feats.copy()
    mutate(node_feats)
    return node_feats, edge_index, node_index, pos_index


def make_normal():
    return lambda env, s: s


def make_zeroed():
    def observe(env, s):
        level_col, thresh_col, mult_col = _columns()

        def mutate(node_feats):
            rows = _host_rows(node_feats)
            node_feats[rows, level_col] = 0.0
            node_feats[rows, thresh_col] = 0.0
            node_feats[rows, mult_col] = 0.0
        return _edit(s, mutate)
    return observe


def make_shuffled(seed):
    rng = np.random.default_rng(seed)

    def observe(env, s):
        level_col, thresh_col, mult_col = _columns()

        def mutate(node_feats):
            rows = _host_rows(node_feats)
            if len(rows) < 2:
                return
            perm = rng.permutation(len(rows))
            node_feats[rows, level_col] = node_feats[rows, level_col][perm]
            node_feats[rows, mult_col] = node_feats[rows, mult_col][perm]
        return _edit(s, mutate)
    return observe


def make_alert_only():
    def observe(env, s):
        level_col, thresh_col, mult_col = _columns()

        def mutate(node_feats):
            rows = _host_rows(node_feats)
            _, _, node_index, _ = s
            for row in rows:
                address = node_index[row]
                true_level, true_threshold = _true_level_and_threshold(env, address)
                node_feats[row, level_col] = 1.0 if true_level >= true_threshold else 0.0
                node_feats[row, thresh_col] = 0.0
                node_feats[row, mult_col] = NEUTRAL_MULTIPLIER
        return _edit(s, mutate)
    return observe


def make_delayed(delay=1):
    """IDS values lag the truth by `delay` steps (delay=1 is the original
    `delayed` condition). A host not yet observed `delay` steps ago, and every
    host during an episode's first `delay` steps, shows the neutral baseline
    (level 0.0, multiplier 1.0). Follow-up G sweeps the delay to measure how far
    back the checkpoint's use of IDS information reaches."""
    if delay < 1:
        raise ValueError("delay must be >= 1")
    history = collections.deque(maxlen=delay)  # oldest first; one {address: (level, mult)} per past step

    def observe(env, s):
        level_col, thresh_col, mult_col = _columns()
        node_feats, edge_index, node_index, pos_index = s
        current_by_address = {}
        lagged_by_address = history[0] if len(history) == delay else {}

        def mutate(node_feats):
            rows = _host_rows(node_feats)
            for row in rows:
                address = tuple(int(a) for a in node_index[row])
                current_by_address[address] = (
                    float(node_feats[row, level_col]), float(node_feats[row, mult_col])
                )
                delayed_level, delayed_mult = lagged_by_address.get(
                    address, (NEUTRAL_LEVEL, NEUTRAL_MULTIPLIER)
                )
                node_feats[row, level_col] = delayed_level
                node_feats[row, thresh_col] = 0.0
                node_feats[row, mult_col] = delayed_mult

        edited = _edit(s, mutate)
        history.append(current_by_address)
        return edited
    return observe


def make_condition_factory(name, seed=0):
    """Dispatch by condition name; `seed` only matters for `shuffled`."""
    if name == "normal":
        return make_normal
    if name == "zeroed":
        return make_zeroed
    if name == "shuffled":
        return lambda: make_shuffled(seed)
    if name == "alert_only":
        return make_alert_only
    if name == "delayed":
        return make_delayed
    if name.startswith("delayed_"):
        delay = int(name[len("delayed_"):])
        return lambda: make_delayed(delay)
    raise ValueError(f"unknown IDS condition: {name!r}")


CONDITIONS = ("normal", "zeroed", "shuffled", "alert_only", "delayed",
              "delayed_5", "delayed_10", "delayed_25", "delayed_50")
