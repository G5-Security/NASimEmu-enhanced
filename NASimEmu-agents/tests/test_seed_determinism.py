"""NASimEmuEnv.__init__ used to call bare random.seed()/np.random.seed()
unconditionally, silently discarding any requested -seed -- this defeated
reproducibility across SubprocVecEnv's forked workers (main.py seeds each
worker's env with config.seed + worker_index). Fixed by only reseeding from
fresh OS entropy when seed is None (env.py). This was verified once by hand
this session via an ad-hoc fingerprint script; formalized here as a
permanent regression test so the fix can't silently regress.

Exercises NASimEmuEnv directly (in-process, no real subprocess workers) --
the fix itself lives entirely in NASimEmuEnv.__init__, so this is a faithful
and much faster/less flaky test than spinning up real SubprocVecEnv workers."""
import os

import numpy as np

from nasimemu.env import NASimEmuEnv

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


def _scenario_fingerprint(seed):
    """The initial observation right after reset() is fully determined by
    the random scenario generation seeded in __init__ -- a faithful
    fingerprint of "which scenario got generated". graph_v2 observations are
    a tuple of ragged-shape arrays (node features, edges, ...), not one flat
    array -- np.array_equal mishandles that directly (it tries np.asarray()
    on the whole tuple first), so compare element-wise instead."""
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=40,
                       observation_format="graph_v2", seed=seed)
    s = env.reset()
    return s


def _fingerprints_equal(fp1, fp2):
    if len(fp1) != len(fp2):
        return False
    return all(np.array_equal(a, b) for a, b in zip(fp1, fp2))


def test_same_seed_produces_identical_scenario_across_independent_constructions():
    fp1 = _scenario_fingerprint(seed=100)
    fp2 = _scenario_fingerprint(seed=100)
    assert _fingerprints_equal(fp1, fp2), (
        "same -seed produced different scenarios across two independent "
        "NASimEmuEnv constructions -- seed is being silently discarded again"
    )


def test_different_seeds_produce_different_scenarios():
    fp1 = _scenario_fingerprint(seed=100)
    fp2 = _scenario_fingerprint(seed=200)
    assert not _fingerprints_equal(fp1, fp2), (
        "two different seeds produced identical scenarios -- seed has no "
        "effect on scenario generation (or generation isn't seed-driven at all)"
    )


def test_unseeded_construction_still_works_and_need_not_be_deterministic():
    """seed=None must remain valid (the pre-existing exploratory-run
    behavior, still used when the user omits -seed) -- just confirms it
    doesn't crash and returns a usable observation, not that it's random
    (that would make the test itself flaky)."""
    env = NASimEmuEnv(scenario_name=SCENARIO, step_limit=40,
                       observation_format="graph_v2", seed=None)
    s = env.reset()
    assert s is not None
