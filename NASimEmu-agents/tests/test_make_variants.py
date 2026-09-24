"""Tests for the Step 4-5 scenario variant generator (experiments/make_variants.py,
docs/eval_revision_plan.tex).

Two layers, matching the plan's own "each variant is checked ... so an
incorrect YAML key cannot go unnoticed":
  1. Unit-level: the generated dict has exactly the expected values.
  2. End-to-end: a real NASimEmuEnv constructed from a *written* variant file,
     with training_mode=False (what evaluation always uses -- see the module
     docstring), reports the expected runtime dynamics via
     HostVector.scan_noise / HostVector.churn_config /
     env.env.network.timeout_config. This is what actually catches a wrong
     YAML key -- a typo'd field name would silently no-op at the dict level
     but show up here as the base scenario's value leaking through.
"""
import os

import pytest
import yaml

from experiments.make_variants import (
    load_scenario, make_mechanism_knockouts, make_intensity_sweep,
    write_variants, final_stage, RATE_LEAVES,
)
from experiments import make_variants
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector

SCENARIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "scenarios",
    "corp_100hosts_dynamic.v2.yaml",
)


def _construct(path):
    env = NASimEmuEnv(scenario_name=path, step_limit=20, observation_format="graph_v2",
                       seed=1, training_mode=False)
    env.reset()
    return env


def test_full_dynamics_matches_source_final_stage():
    scenario_dict = load_scenario(SCENARIO)
    variants = make_mechanism_knockouts(scenario_dict)
    generated_stage = variants["full_dynamics"]["curriculum"]["stages"][-1]
    original_stage = final_stage(scenario_dict)

    for field in ("ids", "scan_noise", "network_reliability", "service_dynamics"):
        assert generated_stage[field] == original_stage[field]


def test_no_ids_only_touches_ids_enabled():
    scenario_dict = load_scenario(SCENARIO)
    variants = make_mechanism_knockouts(scenario_dict)
    stage = variants["no_ids"]["curriculum"]["stages"][-1]
    original = final_stage(scenario_dict)

    assert stage["ids"]["enabled"] is False
    # everything else about the ids block (thresholds, response_types, ...)
    # is untouched, not deleted
    assert stage["ids"]["base_thresholds"] == original["ids"]["base_thresholds"]
    assert stage["scan_noise"] == original["scan_noise"]


def test_no_scan_noise_zeros_all_three_scan_types():
    scenario_dict = load_scenario(SCENARIO)
    stage = make_mechanism_knockouts(scenario_dict)["no_scan_noise"]["curriculum"]["stages"][-1]

    for scan_type in ("service_scan", "os_scan", "process_scan"):
        assert stage["scan_noise"][scan_type]["false_positive_rate"] == 0.0
        assert stage["scan_noise"][scan_type]["false_negative_rate"] == 0.0
    # untouched
    assert stage["ids"]["enabled"] is True


def test_no_churn_and_no_timeouts_are_independent():
    scenario_dict = load_scenario(SCENARIO)
    variants = make_mechanism_knockouts(scenario_dict)

    no_churn = variants["no_churn"]["curriculum"]["stages"][-1]
    assert no_churn["service_dynamics"]["churn_probability"] == 0.0
    assert no_churn["network_reliability"]["timeout_probability"] != 0.0  # untouched

    no_timeouts = variants["no_timeouts"]["curriculum"]["stages"][-1]
    assert no_timeouts["network_reliability"]["timeout_probability"] == 0.0
    assert no_timeouts["service_dynamics"]["churn_probability"] != 0.0  # untouched


def test_all_off_zeros_every_rate_leaf_and_disables_ids():
    scenario_dict = load_scenario(SCENARIO)
    stage = make_mechanism_knockouts(scenario_dict)["all_off"]["curriculum"]["stages"][-1]

    assert stage["ids"]["enabled"] is False
    for path in RATE_LEAVES:
        cur = stage
        for k in path:
            cur = cur[k]
        assert cur == 0.0, f"{'.'.join(path)} was not zeroed"


def test_intensity_levels_use_the_scenarios_own_authored_stages():
    scenario_dict = load_scenario(SCENARIO)
    variants = make_intensity_sweep(scenario_dict)

    baseline = next(s for s in scenario_dict["curriculum"]["stages"] if s["name"] == "baseline")
    medium = next(s for s in scenario_dict["curriculum"]["stages"] if s["name"] == "medium_difficulty")

    off_stage = variants["intensity_off"]["curriculum"]["stages"][-1]
    low_stage = variants["intensity_low"]["curriculum"]["stages"][-1]
    final = variants["intensity_final"]["curriculum"]["stages"][-1]

    for field in ("ids", "scan_noise", "network_reliability", "service_dynamics"):
        assert off_stage[field] == baseline[field]
        assert low_stage[field] == medium[field]
    assert final == final_stage(scenario_dict) | {
        "name": final["name"], "start_frac": final["start_frac"], "end_frac": final["end_frac"],
        "start_epoch": final["start_epoch"], "end_epoch": final["end_epoch"],
    }


def test_intensity_harder_scales_rates_up_and_thresholds_down():
    scenario_dict = load_scenario(SCENARIO)
    base = final_stage(scenario_dict)
    harder = make_intensity_sweep(scenario_dict)["intensity_harder"]["curriculum"]["stages"][-1]

    assert harder["service_dynamics"]["churn_probability"] == base["service_dynamics"]["churn_probability"] * 1.5
    assert harder["network_reliability"]["timeout_probability"] == base["network_reliability"]["timeout_probability"] * 1.5
    assert harder["ids"]["base_thresholds"] == [t * 0.8 for t in base["ids"]["base_thresholds"]]
    # non-rate fields (response_types, detection_increase, ...) untouched
    assert harder["ids"]["response_types"] == base["ids"]["response_types"]


def test_written_variant_files_are_valid_yaml_with_single_stage():
    scenario_dict = load_scenario(SCENARIO)
    variants = make_mechanism_knockouts(scenario_dict)
    variants.update(make_intensity_sweep(scenario_dict))

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        written = write_variants(variants, tmp, "corp_100hosts_dynamic")
        assert set(written) == set(variants)
        for condition, path in written.items():
            with open(path) as f:
                reloaded = yaml.safe_load(f)
            assert len(reloaded["curriculum"]["stages"]) == 1
            assert reloaded["curriculum"]["stages"][0]["start_frac"] == 0.0
            assert reloaded["curriculum"]["stages"][0]["end_frac"] == 1.0


def test_end_to_end_no_ids_variant_disables_ids_at_runtime(tmp_path):
    scenario_dict = load_scenario(SCENARIO)
    variants = make_mechanism_knockouts(scenario_dict)
    written = write_variants(variants, str(tmp_path), "corp_100hosts_dynamic")

    env = _construct(written["no_ids"])
    assert HostVector.ids_config.get("enabled") is False


def test_end_to_end_no_churn_variant_zeros_churn_at_runtime(tmp_path):
    scenario_dict = load_scenario(SCENARIO)
    variants = make_mechanism_knockouts(scenario_dict)
    written = write_variants(variants, str(tmp_path), "corp_100hosts_dynamic")

    env = _construct(written["no_churn"])
    assert HostVector.churn_config.get("churn_probability") == 0.0


def test_end_to_end_intensity_off_matches_baseline_stage_at_runtime(tmp_path):
    scenario_dict = load_scenario(SCENARIO)
    variants = make_intensity_sweep(scenario_dict)
    written = write_variants(variants, str(tmp_path), "corp_100hosts_dynamic")

    env = _construct(written["intensity_off"])
    assert HostVector.ids_config.get("enabled") is False
    assert env.env.network.timeout_config.get("timeout_probability", 0.0) == 0.0
    assert HostVector.churn_config.get("churn_probability", 0.0) == 0.0


def test_end_to_end_full_dynamics_matches_unmodified_scenario_at_runtime(tmp_path):
    # HostVector.scan_noise is a class-level singleton mutated by each
    # construction below, so each snapshot must be taken immediately after
    # its own env is built, before the next construction overwrites it.
    scenario_dict = load_scenario(SCENARIO)
    variants = make_mechanism_knockouts(scenario_dict)
    written = write_variants(variants, str(tmp_path), "corp_100hosts_dynamic")

    _construct(SCENARIO)
    ref_scan_noise = dict(HostVector.scan_noise)

    var_env = _construct(written["full_dynamics"])
    var_scan_noise = dict(HostVector.scan_noise)

    assert var_scan_noise == ref_scan_noise
    assert var_env.env.network.timeout_config["timeout_probability"] == \
        final_stage(scenario_dict)["network_reliability"]["timeout_probability"]


# ---- follow-up D/E: parameter-sensitivity and scan-noise-scaled variants ----

def _flatten(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def _stage_diff_paths(base_stage, variant_stage):
    a, b = _flatten(base_stage), _flatten(variant_stage)
    return {p for p in set(a) | set(b) if a.get(p) != b.get(p) and p[0] not in ("name", "start_frac", "end_frac", "start_epoch", "end_epoch")}


@pytest.fixture(scope="module")
def _dyn_scenario():
    return make_variants.load_scenario(SCENARIO)


def test_each_sensitivity_variant_changes_only_its_own_parameter_family(_dyn_scenario):
    base = make_variants.final_stage(_dyn_scenario)
    variants = make_variants.make_sensitivity_variants(_dyn_scenario)
    assert set(variants) == set(make_variants.SENSITIVITY_SPECS)
    family = {
        "ids_thr": {("ids", "base_thresholds")}, "ids_inc": None, "ids_decay": {("ids", "detection_decay")},
        "resp_": {("ids", "response_types", "quarantine"), ("ids", "response_types", "patch"), ("ids", "response_types", "monitor")},
        "fem_": {("ids", "failed_exploit_multiplier")}, "churn_": {("service_dynamics", "churn_probability")},
        "timeout_": {("network_reliability", "timeout_probability")},
    }
    for name, scenario in variants.items():
        changed = _stage_diff_paths(base, make_variants.final_stage(scenario))
        assert changed, f"{name} changed nothing"
        key = next(k for k in family if name.startswith(k))
        if family[key] is None:  # ids_inc: every detection_increase leaf, nothing else
            assert all(p[:2] == ("ids", "detection_increase") for p in changed), (name, changed)
        else:
            assert changed <= family[key], (name, changed)
        assert len(scenario["curriculum"]["stages"]) == 1


def test_scan_noise_scaled_variants_scale_only_scan_noise_and_clip_at_one(_dyn_scenario):
    base = make_variants.final_stage(_dyn_scenario)
    variants = make_variants.make_scan_noise_scaled(_dyn_scenario, factors=(0.0, 2.0, 100.0))
    for name, scenario in variants.items():
        assert all(p[0] == "scan_noise" for p in _stage_diff_paths(base, make_variants.final_stage(scenario)))
    zero = make_variants.final_stage(variants["scan_noise_x0"])["scan_noise"]
    assert all(v == 0.0 for scan in zero.values() for v in scan.values())
    big = make_variants.final_stage(variants["scan_noise_x100"])["scan_noise"]
    assert all(v == 1.0 for scan in big.values() for v in scan.values())
    two = make_variants.final_stage(variants["scan_noise_x2"])["scan_noise"]["service_scan"]["false_positive_rate"]
    assert two == pytest.approx(2 * base["scan_noise"]["service_scan"]["false_positive_rate"])


@pytest.mark.parametrize("name,check", [
    ("ids_decay_0.90", lambda c: c["detection_decay"] == 0.90),
    ("fem_0", lambda c: c["failed_exploit_multiplier"] == 0.0),
    ("resp_all_monitor", lambda c: c["response_types"] == {"quarantine": 0.0, "patch": 0.0, "monitor": 1.0}),
    ("ids_thr_x0.6", lambda c: c["base_thresholds"] == pytest.approx([0.39, 0.528])),
    ("ids_inc_x2", lambda c: c["detection_increase"]["exploit_failed"] == pytest.approx(0.18)),
])
def test_sensitivity_variant_values_reach_the_running_simulator(tmp_path, _dyn_scenario, name, check):
    scenario = make_variants.make_sensitivity_variants(_dyn_scenario)[name]
    path = make_variants.write_variants({name: scenario}, str(tmp_path), "dyn")[name]
    _construct(path)
    assert check(HostVector.ids_config), (name, HostVector.ids_config)
