"""Follow-up C: the IDS parameter table and semantics write-up.

Reviewer B: "The semantics of the IDS, one of the core contributions, are not
given. What happens when the sampled threshold is exceeded? (Action blocked?
Penalized? Episode terminated?) How do alerts accumulate? I suggest to provide
the IDS full parameter table."

Parameter values come from the scenario YAML's final curriculum stage (the
stage evaluation uses); response penalties and effects are read by *calling*
HostVector._handle_detection(), not retyped; the mechanics are the behaviours
pinned by tests/test_ids_semantics.py. Output: ids_semantics.md and
ids_parameters.tex in a separate docs folder.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.make_variants import final_stage, load_scenario  # noqa: E402
from nasimemu.env import NASimEmuEnv  # noqa: E402
from nasimemu.nasim.envs.host_vector import HostVector  # noqa: E402

AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES = ("quarantine", "patch", "monitor")


def measure_responses(scenario_path):
    """{response: {'penalty': int, 'effect': str}}, obtained by forcing each response type."""
    env = NASimEmuEnv(scenario_name=scenario_path, step_limit=10, observation_format="graph_v2",
                      seed=1, training_mode=False)
    env.reset()
    state = env.env.current_state
    host = next(state.get_host(a) for a in state.host_num_map if any(state.get_host(a).services.values()))
    original = dict(HostVector.ids_config)
    out = {}
    try:
        for r in RESPONSES:
            HostVector.ids_config = {**original, "response_types": {k: float(k == r) for k in RESPONSES}}
            result = host._handle_detection()
            effect = {"quarantine": "the host is quarantined: the attacker's access to it is set to none",
                      "patch": "1-2 of the host's running services are patched: exploits against them then fail",
                      "monitor": "the host's detection multiplier becomes 2.0: every later increase on it is doubled"}[r]
            out[r] = {"penalty": result["penalty"], "effect": effect, "type": result["type"]}
            assert result["type"] == r
    finally:
        HostVector.ids_config = original
    return out


def build(scenario_path, out_dir):
    ids = final_stage(load_scenario(scenario_path))["ids"]
    responses = measure_responses(scenario_path)
    inc = ids["detection_increase"]
    lo, hi = ids["base_thresholds"]
    os.makedirs(out_dir, exist_ok=True)

    inc_rows = [("subnet_scan", "subnet scan"), ("service_scan", "service scan"), ("os_scan", "OS scan"),
                ("process_scan", "process scan"), ("exploit_failed", "failed exploit"),
                ("exploit_success", "successful exploit"), ("privesc_failed", "failed privilege escalation"),
                ("privesc_success", "successful privilege escalation")]
    md = ["# IDS semantics and parameters", "",
          f"Values are the final curriculum stage of `{os.path.basename(scenario_path)}` (the stage evaluation uses). "
          "Every mechanic below is pinned by `tests/test_ids_semantics.py`.", "",
          "## What the IDS keeps and what the agent sees", "",
          "- Each host has its own **detection level**, a non-negative number that starts at 0.",
          f"- Each host also has a hidden **threshold**, drawn once when the host is created, uniformly from [{lo}, {hi}]. "
          "The agent never observes the threshold.",
          "- Whenever a host is observed, the agent sees its detection level and its **monitoring multiplier**; it does not see the threshold.", "",
          "## How the level changes (each time an action targets the host)", "",
          "1. The action adds a base amount that depends on its type (table below).",
          "2. Scans (service, OS, process) that follow another scan of the same host by fewer than 5 steps count double.",
          f"3. A failed exploit is multiplied by 1 + 0.3 x (failed-exploit count on that host) x {ids['failed_exploit_multiplier']}. "
          "The count includes failed privilege escalations, is never reset, and grows for the rest of the episode.",
          "4. The amount is multiplied by the host's monitoring multiplier (1.0 normally, 2.0 after a *monitor* response).",
          f"5. The level is then multiplied by the decay factor {ids['detection_decay']}. "
          "**Decay happens only when an action targets that host**, not every environment step.", "",
          "## What happens when the level exceeds the threshold", "",
          "The IDS draws one response for the host, on that step:", "",
          "| response | probability | reward penalty | effect |", "|---|---|---|---|"]
    for r in RESPONSES:
        md.append(f"| {r} | {ids['response_types'][r]:.2f} | {responses[r]['penalty']} | {responses[r]['effect']} |")
    md += ["",
           "- Penalties are raw reward points. The environment divides every reward by 10 before it is reported or trained on, "
           "so they appear as -5, -2 and -0.5 in the returns quoted elsewhere in this project.",
           "- The action itself is **not** blocked, and the **episode does not end**.",
           "- **The level is not reset by an alert.** While a host stays above its threshold, every further action on it triggers another response and another penalty, until decay (which needs actions on that host) brings the level back down.",
           "- Alerts on different hosts are independent.", "",
           "## Parameters", "",
           "| parameter | value |", "|---|---|",
           f"| IDS enabled | {ids['enabled']} |",
           f"| per-host threshold | uniform [{lo}, {hi}] (hidden) |",
           f"| decay factor (per action on the host) | {ids['detection_decay']} |",
           f"| failed-exploit multiplier | {ids['failed_exploit_multiplier']} |"]
    md += [f"| base increase: {label} | {inc[key]} |" for key, label in inc_rows]
    md += [f"| response probability: {r} | {ids['response_types'][r]} |" for r in RESPONSES]
    md += ["", "Scan false-positive and false-negative rates, churn and timeouts are separate mechanisms; see the "
           "intensity table in the main report.", ""]
    with open(os.path.join(out_dir, "ids_semantics.md"), "w") as f:
        f.write("\n".join(md))

    tex = [r"\begin{tabular}{@{}ll@{}}", r"\toprule", r"parameter & value \\", r"\midrule",
           f"IDS enabled & {'yes' if ids['enabled'] else 'no'} \\\\",
           f"per-host threshold (hidden) & uniform [{lo}, {hi}] \\\\",
           f"decay factor (per action on the host) & {ids['detection_decay']} \\\\",
           f"failed-exploit multiplier & {ids['failed_exploit_multiplier']} \\\\"]
    tex += [f"base increase: {label} & {inc[key]} \\\\" for key, label in inc_rows]
    tex += [f"response probability / penalty: {r} & {ids['response_types'][r]:.2f} / {responses[r]['penalty']} \\\\".replace("$-$", "-")
            for r in RESPONSES]
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(out_dir, "ids_parameters.tex"), "w") as f:
        f.write("\n".join(tex))
    return responses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=os.path.join(AGENTS_DIR, "..", "scenarios", "corp_100hosts_dynamic.v2.yaml"))
    ap.add_argument("--out_dir", default=os.path.join(AGENTS_DIR, "..", "docs", "eval_followup_2026-09-24", "C_ids_semantics"))
    args = ap.parse_args()
    responses = build(args.scenario, args.out_dir)
    print("wrote", os.path.abspath(args.out_dir), {k: v["penalty"] for k, v in responses.items()})


if __name__ == "__main__":
    main()
