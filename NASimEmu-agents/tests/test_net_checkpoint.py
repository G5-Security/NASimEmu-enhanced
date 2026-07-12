"""Master plan Sec 15.13: an old checkpoint must never be silently loaded
into a model with a mismatched semantic ontology. net.py stamps every
checkpoint with the saving class's ONTOLOGY_VERSION and refuses to load a
mismatched one -- this was part of the original plan's verification list but
never got a permanent regression test, so the exact bug it fixes (silent
wrong-ontology loads) could come back unnoticed."""
import warnings

import pytest
import torch
from torch.nn import Linear

from config import config
from net import Net


def _ensure_config_initialized():
    """Net.__init__ reads config.device/opt_lr/alpha_h -- config is a
    module-global singleton that only has those attributes after .init() is
    called once with a real args object (see main.py/evaluate_llm_selector.py
    _build_net). Safe to call more than once across the test session."""
    if not hasattr(config, "device"):
        from types import SimpleNamespace
        config.init(SimpleNamespace(
            batch=1, epoch=10, alpha_h=0.1, force_continue_epochs=0,
            emb_dim=8, mp_iterations=1, seed=None, device="cpu", cpus=1,
            lr=1e-3, max_norm=3., sched_lr_factor=None, sched_lr_min=None,
            sched_lr_rate=None, sched_alpha_h_factor=None, sched_alpha_h_min=None,
            sched_alpha_h_rate=None, max_epochs=None, load_model=None,
        ))


class _NetV5(Net):
    ONTOLOGY_VERSION = 5

    def __init__(self):
        super().__init__()
        self.lin = Linear(4, 4)


class _NetV6(Net):
    ONTOLOGY_VERSION = 6

    def __init__(self):
        super().__init__()
        self.lin = Linear(4, 4)


class _NetUnversioned(Net):
    # ONTOLOGY_VERSION left as the base class's None -- "this architecture
    # has no ontology", per net.py's own docstring.
    def __init__(self):
        super().__init__()
        self.lin = Linear(4, 4)


def setup_module():
    _ensure_config_initialized()


def test_round_trip_preserves_ontology_version_and_weights(tmp_path):
    ckpt = str(tmp_path / "model.pt")
    net_a = _NetV5()
    net_a.save(ckpt)

    net_b = _NetV5()
    net_b.load(ckpt)  # must not raise -- same ONTOLOGY_VERSION on both sides

    for p_a, p_b in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_mismatched_ontology_version_raises(tmp_path):
    ckpt = str(tmp_path / "model.pt")
    _NetV5().save(ckpt)

    with pytest.raises(ValueError, match="ontology_version"):
        _NetV6().load(ckpt)


def test_unversioned_legacy_checkpoint_warns_and_loads_anyway(tmp_path):
    """Pre-existing checkpoints (trained_models/*.pt) are a bare state_dict,
    not the {'state_dict', 'ontology_version'} wrapper -- must still load,
    with a warning, not raise."""
    ckpt = str(tmp_path / "legacy_model.pt")
    legacy_net = _NetV5()
    torch.save(legacy_net.state_dict(), ckpt)  # bare state_dict, old-style

    loader = _NetV5()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader.load(ckpt)  # must not raise
    assert any("unversioned checkpoint" in str(w.message) for w in caught)

    for p_a, p_b in zip(legacy_net.parameters(), loader.parameters()):
        assert torch.equal(p_a, p_b)


def test_none_ontology_version_on_either_side_never_raises(tmp_path):
    """The mismatch check only fires when BOTH sides declare a non-None
    ONTOLOGY_VERSION (net.py: 'ckpt_version is not None and
    self.ONTOLOGY_VERSION is not None') -- an unversioned architecture on
    either side must be able to load without raising."""
    unversioned_ckpt = str(tmp_path / "model_unversioned.pt")
    _NetUnversioned().save(unversioned_ckpt)
    _NetV5().load(unversioned_ckpt)  # ckpt_version is None -> no mismatch check, no raise
    _NetUnversioned().load(unversioned_ckpt)  # both None -> no mismatch check, no raise

    versioned_ckpt = str(tmp_path / "model_versioned.pt")
    _NetV5().save(versioned_ckpt)
    _NetUnversioned().load(versioned_ckpt)  # self.ONTOLOGY_VERSION is None -> no mismatch check, no raise
