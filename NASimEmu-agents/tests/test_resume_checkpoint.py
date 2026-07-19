from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import main
from config import config
from net import Net
from training_lock import acquire_training_lock


class TinyNet(Net):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.01)


def test_legacy_resume_schedule_uses_last_completed_boundary():
    lr = main.scheduled_value_at_step(
        11600, 0.0007, 0.0003, 0.8, 10000, main.decay_exp,
    )
    alpha_h = main.scheduled_value_at_step(
        11600, 0.02, 0.005, 0.5, 15000, main.decay_time,
    )

    assert lr == pytest.approx(0.00062)
    assert alpha_h == pytest.approx(0.02)


def test_rng_state_round_trip():
    main.init_seed(123)
    state = main.capture_rng_state()

    expected_python = main.random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)

    main.init_seed(999)
    main.restore_rng_state(state)

    assert main.random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_resume_config_rejects_training_drift():
    saved = {
        'scenario': 'scenario.yaml',
        'net_class': 'NASimNetDHRL',
        'batch': 128,
        'epoch': 100,
        'max_epochs': 200,
    }
    args = SimpleNamespace(
        scenario='scenario.yaml', test_scenario=None,
        net_class='NASimNetDHRL', llm_shaping=False, llm_distill=False,
    )
    runtime = SimpleNamespace(
        batch=16, epoch=100, max_epochs=200, seed=1,
        opt_lr=0.0007, alpha_h=0.02, step_limit=400, use_a_t=True,
        observation_format='graph_v2', sched_lr_rate=10000,
        sched_lr_factor=0.8, sched_lr_min=0.0003,
        sched_alpha_h_rate=15000, sched_alpha_h_factor=0.5,
        sched_alpha_h_min=0.005,
    )

    with pytest.raises(SystemExit, match='batch: checkpoint=128, command=16'):
        main.validate_resume_run_config(saved, args, runtime)


def test_checkpoint_round_trip_with_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'device', 'cpu', raising=False)
    monkeypatch.setattr(config, 'opt_lr', 0.01, raising=False)
    monkeypatch.setattr(config, 'alpha_h', 0.02, raising=False)

    source = TinyNet()
    training_state = {
        'format_version': 1,
        'step': 7,
        'optimizer_state_dict': source.opt.state_dict(),
    }
    checkpoint = tmp_path / 'model.pt'
    source.save(checkpoint, training_state=training_state)

    restored = TinyNet()
    loaded_training_state = restored.load(checkpoint)

    assert loaded_training_state['step'] == 7
    assert not (tmp_path / 'model.pt.tmp').exists()
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_weights_only_versioned_checkpoint_remains_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'device', 'cpu', raising=False)
    monkeypatch.setattr(config, 'opt_lr', 0.01, raising=False)
    monkeypatch.setattr(config, 'alpha_h', 0.02, raising=False)

    source = TinyNet()
    checkpoint = tmp_path / 'legacy.pt'
    torch.save(
        {
            'state_dict': source.state_dict(),
            'ontology_version': source.ONTOLOGY_VERSION,
        },
        checkpoint,
    )

    restored = TinyNet()
    assert restored.load(checkpoint) is None


def test_training_mode_detection_excludes_read_only_commands():
    training = SimpleNamespace(
        calc_baseline=False, trace=False, eval=False, debug=False,
    )
    assert main.command_starts_training(training)

    for mode in ('calc_baseline', 'trace', 'eval', 'debug'):
        command = SimpleNamespace(
            calc_baseline=False, trace=False, eval=False, debug=False,
        )
        setattr(command, mode, True)
        assert not main.command_starts_training(command)


def test_training_lock_rejects_second_process_and_releases(tmp_path):
    lock_path = tmp_path / 'training.lock'
    first = acquire_training_lock(lock_path)
    code = (
        "from training_lock import acquire_training_lock; "
        f"acquire_training_lock({str(lock_path)!r})"
    )

    try:
        blocked = subprocess.run(
            [sys.executable, '-c', code],
            cwd=os.path.dirname(main.__file__),
            capture_output=True, text=True,
        )
        assert blocked.returncode != 0
        assert 'Another NASimEmu training run is already active' in blocked.stderr
    finally:
        first.release()

    released = subprocess.run(
        [sys.executable, '-c', code],
        cwd=os.path.dirname(main.__file__),
        capture_output=True, text=True,
    )
    assert released.returncode == 0, released.stderr
