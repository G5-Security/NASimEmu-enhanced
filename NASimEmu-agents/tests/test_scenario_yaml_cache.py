import os
import time

import yaml

from nasimemu.nasim.scenarios import utils


def test_load_yaml_caches_parse_and_returns_independent_copy(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("subnets: [2, 3]\n", encoding="utf-8")

    utils._YAML_CACHE.clear()
    real_load = yaml.load
    parse_count = 0

    def counting_load(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(utils.yaml, "load", counting_load)

    first = utils.load_yaml(scenario_path)
    first["subnets"].append(99)
    second = utils.load_yaml(scenario_path)

    assert parse_count == 1
    assert second == {"subnets": [2, 3]}


def test_load_yaml_invalidates_cache_when_file_changes(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("value: 1\n", encoding="utf-8")

    utils._YAML_CACHE.clear()
    assert utils.load_yaml(scenario_path) == {"value": 1}

    scenario_path.write_text("value: 200\n", encoding="utf-8")
    # Some filesystems have coarse timestamp resolution. Force a distinct
    # nanosecond mtime while keeping this test fast.
    new_mtime_ns = time.time_ns() + 1_000_000_000
    os.utime(scenario_path, ns=(new_mtime_ns, new_mtime_ns))

    assert utils.load_yaml(scenario_path) == {"value": 200}
