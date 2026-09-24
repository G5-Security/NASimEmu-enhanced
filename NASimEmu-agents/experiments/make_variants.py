"""Steps 4-5 scenario variant generator (docs/eval_revision_plan.tex,
"Mechanism knock-outs and intensity sweep").

Generates evaluation-only scenario YAML variants that each change exactly
one thing about the curriculum's *final* stage -- the only stage evaluation
ever reads (see below) -- while leaving everything else (topology, hosts,
services, the earlier curriculum stages) untouched. No environment code
changes; this is a pure YAML transform.

Why only the final stage matters for evaluation
-------------------------------------------------
CurriculumManager.get_current_stage() (src/nasimemu/nasim/envs/curriculum.py)
returns `self.config['stages'][-1]` whenever `training_mode=False`
unconditionally, regardless of epoch bounds -- and nasim_debug.py's own
_eval() constructs every evaluation env with `training_mode=False`
specifically so evaluation always sees the hardest stage. Environment.
_apply_curriculum_settings() (nasim/envs/environment.py) then reads exactly
four sub-keys off that one stage dict -- `ids`, `scan_noise`,
`network_reliability`, `service_dynamics` -- and nothing else from the
scenario file affects runtime dynamics. So a variant only needs to replace
those four sub-keys (or the whole stage) on `curriculum.stages[-1]`; nothing
elsewhere in the YAML needs to change, and no other stage in the list is
ever consulted at evaluation time. Each variant file therefore keeps
`curriculum.stages` as a *single*-element list holding just the transformed
final stage, spanning start_frac 0.0-1.0 -- functionally identical to the
original multi-stage list under `training_mode=False`, and additionally
safe (picks the intended stage for the whole run) if a variant file were
ever accidentally loaded with `training_mode=True`.

Mechanism knock-outs (Step 4)
------------------------------
Each is the scenario's own final ("full_difficulty") stage with exactly one
mechanism disabled:
  full_dynamics    unmodified final stage (the reference condition)
  no_ids           ids.enabled = False
  no_scan_noise    every scan_noise.*.false_positive_rate/false_negative_rate = 0.0
  no_churn         service_dynamics.churn_probability = 0.0
  no_timeouts      network_reliability.timeout_probability = 0.0
  all_off          all four of the above at once

Intensity sweep (Step 5) -- corrected against the plan text
--------------------------------------------------------------
The plan sketched four non-off levels ("low", "medium", the final stage, and
one harder setting), assuming two distinct pre-existing intermediate
stages. Every corp_100hosts_dynamic*.v2.yaml scenario in this repo (dynamic,
varA, varB, bridge, test -- checked directly) in fact defines exactly three
authored stages: "baseline" (index 0, structurally identical in effect to
`all_off` above -- zero rates, ids disabled), "medium_difficulty" (index 1),
and "full_difficulty" (index -1, the reference condition). There is no
second, distinct authored intermediate stage to call "medium". So the
intensity levels generated here are the three real authored stages plus one
synthetic harder-than-final level, not four independently-authored ones:
  intensity_off      = the scenario's own "baseline" stage verbatim
                        (behaviorally identical to all_off; not regenerated)
  intensity_low       = the scenario's own "medium_difficulty" stage verbatim
  intensity_final      = the scenario's own "full_difficulty" stage (alias of
                        full_dynamics, kept as a separate name for the
                        intensity table)
  intensity_harder    = full_difficulty with every rate (scan FP/FN, churn,
                        timeout) multiplied by 1.5 (clipped to at most 1.0)
                        and `ids.base_thresholds` multiplied by 0.8 (lower
                        thresholds = detection triggers more easily = harder
                        for the attacker), exactly the multipliers named in
                        the plan text
"""
import argparse
import copy
import os

import yaml

RATE_LEAVES = (
    ("scan_noise", "service_scan", "false_positive_rate"),
    ("scan_noise", "service_scan", "false_negative_rate"),
    ("scan_noise", "os_scan", "false_positive_rate"),
    ("scan_noise", "os_scan", "false_negative_rate"),
    ("scan_noise", "process_scan", "false_positive_rate"),
    ("scan_noise", "process_scan", "false_negative_rate"),
    ("network_reliability", "timeout_probability"),
    ("service_dynamics", "churn_probability"),
)

HARDER_RATE_MULTIPLIER = 1.5
HARDER_THRESHOLD_MULTIPLIER = 0.8


def _get_path(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set_path(d, path, value):
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def _wrap_single_stage(stage, name):
    """Build a curriculum.stages list with exactly one stage, spanning the
    whole run, as described in the module docstring."""
    stage = copy.deepcopy(stage)
    stage["name"] = name
    stage["start_frac"] = 0.0
    stage["end_frac"] = 1.0
    stage["start_epoch"] = 0
    stage["end_epoch"] = 999999
    return [stage]


def load_scenario(path):
    with open(path) as f:
        return yaml.safe_load(f)


def final_stage(scenario_dict):
    stages = scenario_dict["curriculum"]["stages"]
    if not stages:
        raise ValueError("scenario has an empty curriculum.stages list")
    return stages[-1]


def stage_by_name(scenario_dict, name):
    for stage in scenario_dict["curriculum"]["stages"]:
        if stage.get("name") == name:
            return stage
    raise KeyError(f"no curriculum stage named {name!r} in this scenario")


def make_mechanism_knockouts(scenario_dict):
    """Step 4: dict of {condition_name: scenario_dict} sharing everything
    except curriculum.stages, each disabling exactly one mechanism (or none,
    for the reference condition) on top of the scenario's own final stage."""
    base = final_stage(scenario_dict)
    variants = {}

    def make(name, mutate):
        stage = copy.deepcopy(base)
        mutate(stage)
        out = copy.deepcopy(scenario_dict)
        out["curriculum"]["stages"] = _wrap_single_stage(stage, base.get("name", "full_difficulty"))
        variants[name] = out

    make("full_dynamics", lambda s: None)
    make("no_ids", lambda s: s.setdefault("ids", {}).update(enabled=False))
    make("no_scan_noise", lambda s: [
        _set_path(s, p, 0.0) for p in RATE_LEAVES if p[0] == "scan_noise"
    ])
    make("no_churn", lambda s: _set_path(s, ("service_dynamics", "churn_probability"), 0.0))
    make("no_timeouts", lambda s: _set_path(s, ("network_reliability", "timeout_probability"), 0.0))

    def all_off(s):
        s.setdefault("ids", {}).update(enabled=False)
        for p in RATE_LEAVES:
            _set_path(s, p, 0.0)
    make("all_off", all_off)

    return variants


def make_intensity_sweep(scenario_dict):
    """Step 5: dict of {condition_name: scenario_dict}. See module docstring
    for why this is 3 authored stages + 1 synthetic "harder" level, not the
    4 independently-authored levels the plan text first sketched."""
    variants = {}

    def wrap(name, stage):
        out = copy.deepcopy(scenario_dict)
        out["curriculum"]["stages"] = _wrap_single_stage(stage, stage.get("name", name))
        variants[name] = out

    wrap("intensity_off", stage_by_name(scenario_dict, "baseline"))
    wrap("intensity_low", stage_by_name(scenario_dict, "medium_difficulty"))
    wrap("intensity_final", final_stage(scenario_dict))

    harder = copy.deepcopy(final_stage(scenario_dict))
    for path in RATE_LEAVES:
        val = _get_path(harder, path)
        if val is not None:
            _set_path(harder, path, min(1.0, val * HARDER_RATE_MULTIPLIER))
    thresholds = _get_path(harder, ("ids", "base_thresholds"))
    if thresholds is not None:
        _set_path(harder, ("ids", "base_thresholds"),
                  [max(0.0, t * HARDER_THRESHOLD_MULTIPLIER) for t in thresholds])
    wrap("intensity_harder", harder)

    return variants


# Parameter-sensitivity sweep (follow-up D): each variant changes ONE parameter
# family of the final stage and leaves everything else alone. Answers "how were
# these values chosen / how much does the outcome depend on them" empirically
# by evaluating the fixed checkpoint across a range, not by retraining.
# name -> (parameter path or None, description, mutation(stage))
def _scale_thresholds(f):
    return lambda st: _set_path(st, ("ids", "base_thresholds"),
                                [t * f for t in _get_path(st, ("ids", "base_thresholds"))])


def _scale_increases(f):
    def mutate(st):
        inc = _get_path(st, ("ids", "detection_increase"))
        _set_path(st, ("ids", "detection_increase"), {k: v * f for k, v in inc.items()})
    return mutate


def _scale_leaf(path, f, cap=1.0):
    return lambda st: _set_path(st, path, min(cap, _get_path(st, path) * f))


def _set_response(quarantine, patch, monitor):
    return lambda st: _set_path(st, ("ids", "response_types"),
                                {"quarantine": quarantine, "patch": patch, "monitor": monitor})


SENSITIVITY_SPECS = {
    "ids_thr_x0.6": _scale_thresholds(0.6),
    "ids_thr_x0.8": _scale_thresholds(0.8),
    "ids_thr_x1.25": _scale_thresholds(1.25),
    "ids_thr_x1.5": _scale_thresholds(1.5),
    "ids_inc_x0.5": _scale_increases(0.5),
    "ids_inc_x2": _scale_increases(2.0),
    "ids_decay_0.90": lambda st: _set_path(st, ("ids", "detection_decay"), 0.90),
    "ids_decay_0.95": lambda st: _set_path(st, ("ids", "detection_decay"), 0.95),
    "ids_decay_0.995": lambda st: _set_path(st, ("ids", "detection_decay"), 0.995),
    "resp_all_monitor": _set_response(0.0, 0.0, 1.0),
    "resp_all_patch": _set_response(0.0, 1.0, 0.0),
    "resp_all_quarantine": _set_response(1.0, 0.0, 0.0),
    "fem_0": lambda st: _set_path(st, ("ids", "failed_exploit_multiplier"), 0.0),
    "fem_x0.5": _scale_leaf(("ids", "failed_exploit_multiplier"), 0.5, cap=1e9),
    "churn_x2": _scale_leaf(("service_dynamics", "churn_probability"), 2.0),
    "churn_x3": _scale_leaf(("service_dynamics", "churn_probability"), 3.0),
    "timeout_x2": _scale_leaf(("network_reliability", "timeout_probability"), 2.0),
    "timeout_x4": _scale_leaf(("network_reliability", "timeout_probability"), 4.0),
}


def make_sensitivity_variants(scenario_dict):
    """{name: scenario_dict}: the final stage with exactly one parameter family
    changed, per SENSITIVITY_SPECS."""
    base = final_stage(scenario_dict)
    variants = {}
    for name, mutate in SENSITIVITY_SPECS.items():
        stage = copy.deepcopy(base)
        mutate(stage)
        out = copy.deepcopy(scenario_dict)
        out["curriculum"]["stages"] = _wrap_single_stage(stage, base.get("name", "full_difficulty"))
        variants[name] = out
    return variants


def make_scan_noise_scaled(scenario_dict, factors=(0.0, 1.0, 2.0, 4.0)):
    """{name: scenario_dict}: every scan-noise rate multiplied by a factor
    (clipped to 1.0), used by follow-up E once scan noise reaches the agent."""
    base = final_stage(scenario_dict)
    variants = {}
    for f in factors:
        stage = copy.deepcopy(base)
        for path in RATE_LEAVES:
            if path[0] == "scan_noise":
                v = _get_path(stage, path)
                if v is not None:
                    _set_path(stage, path, min(1.0, v * f))
        out = copy.deepcopy(scenario_dict)
        out["curriculum"]["stages"] = _wrap_single_stage(stage, base.get("name", "full_difficulty"))
        variants[f"scan_noise_x{f:g}"] = out
    return variants


def write_variants(variants, out_dir, base_name):
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for condition, scenario_dict in variants.items():
        out_path = os.path.join(out_dir, f"{base_name}__{condition}.v2.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(scenario_dict, f, sort_keys=False)
        written[condition] = out_path
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", help="Path to a source scenario .yaml")
    ap.add_argument("--out_dir", default=None,
                     help="Defaults to a 'variants' subdirectory next to the source scenario")
    ap.add_argument("--sensitivity", action="store_true",
                     help="Generate the follow-up parameter-sensitivity and scan-noise-scaled variants "
                          "instead of the Step 4/5 set (default out_dir: 'variants_sensitivity')")
    args = ap.parse_args()

    scenario_dict = load_scenario(args.scenario)
    base_name = os.path.splitext(os.path.splitext(os.path.basename(args.scenario))[0])[0]
    default_sub = "variants_sensitivity" if args.sensitivity else "variants"
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.scenario)), default_sub)

    all_variants = {}
    if args.sensitivity:
        all_variants.update(make_sensitivity_variants(scenario_dict))
        all_variants.update(make_scan_noise_scaled(scenario_dict))
    else:
        all_variants.update(make_mechanism_knockouts(scenario_dict))
        all_variants.update(make_intensity_sweep(scenario_dict))

    written = write_variants(all_variants, out_dir, base_name)
    for condition, path in written.items():
        print(f"[make_variants] {condition}: {path}")


if __name__ == "__main__":
    main()
