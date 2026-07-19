import pytest

from config import config
from nasim_problem.nasim_debug import NASimDebug


def test_test_only_evaluation_skips_training_scenario(monkeypatch):
    monkeypatch.setattr(config, 'scenario_name', 'train.yaml', raising=False)
    monkeypatch.setattr(config, 'test_scenario_name', 'test-a.yaml:test-b.yaml', raising=False)
    monkeypatch.setattr(config, 'eval_split', 'test', raising=False)
    calls = []

    def fake_eval(self, net, scenario_name):
        calls.append(scenario_name)
        return {'reward_avg': 1.25}

    monkeypatch.setattr(NASimDebug, '_eval', fake_eval)

    result = NASimDebug().evaluate(object())

    assert calls == ['test-a.yaml:test-b.yaml']
    assert result == {'eval_tst': {'reward_avg': 1.25}}


def test_test_only_evaluation_requires_test_scenario(monkeypatch):
    monkeypatch.setattr(config, 'scenario_name', 'train.yaml', raising=False)
    monkeypatch.setattr(config, 'test_scenario_name', None, raising=False)
    monkeypatch.setattr(config, 'eval_split', 'test', raising=False)

    with pytest.raises(SystemExit, match='requires --test_scenario'):
        NASimDebug().evaluate(object())


def test_default_evaluation_keeps_both_splits(monkeypatch):
    monkeypatch.setattr(config, 'scenario_name', 'train.yaml', raising=False)
    monkeypatch.setattr(config, 'test_scenario_name', 'test.yaml', raising=False)
    monkeypatch.setattr(config, 'eval_split', 'both', raising=False)

    def fake_eval(self, net, scenario_name):
        return {'scenario': scenario_name}

    monkeypatch.setattr(NASimDebug, '_eval', fake_eval)

    assert NASimDebug().evaluate(object()) == {
        'eval_trn': {'scenario': 'train.yaml'},
        'eval_tst': {'scenario': 'test.yaml'},
    }
