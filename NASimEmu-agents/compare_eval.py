import os
import re
import json
import argparse
import subprocess
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional, Tuple, Any
import shlex

# ========================= USER EDITABLE SECTION ==============================
# Number of times to repeat each eval (can be overridden by --runs)
NUM_RUNS = 3

# Define scenarios with their complete commands
SCENARIO_COMMANDS = {
    "Training Template": (
        "python main.py ../scenarios/corp_100hosts_dynamic.v2.yaml --eval "
        "-load_model wandb/latest-run/files/model.pt "
        "-device cpu -cpus 8 "
        "-net_class NASimNetGNN_LSTM "
        "-use_a_t -episode_step_limit 400 "
        "-observation_format graph_v2 -mp_iterations 3 "
        "--feature_dropout_p 0.0 --dr_prob_jitter 0.0 --dr_cost_jitter 0.0 --dr_scan_cost_jitter 0.0 "
        "--auto_mode per_episode --auto_template ../scenarios/corp_100hosts_dynamic.v2.yaml "
        "--auto_host_range 72-96 --auto_subnet_count 12 --auto_topology mesh "
        "--auto_sensitive_policy deep --auto_sensitive_jitter 0.0"
    ),
    "Bridge Template": (
        "python main.py ../scenarios/corp_100hosts_dynamic_bridge.v2.yaml --eval "
        "-load_model wandb/latest-run/files/model.pt "
        "-device cpu -cpus 8 "
        "-net_class NASimNetGNN_LSTM "
        "-use_a_t -episode_step_limit 400 "
        "-observation_format graph_v2 -mp_iterations 3 "
        "--feature_dropout_p 0.0 --dr_prob_jitter 0.0 --dr_cost_jitter 0.0 --dr_scan_cost_jitter 0.0 "
        "--auto_mode per_episode --auto_template ../scenarios/corp_100hosts_dynamic_bridge.v2.yaml "
        "--auto_host_range 72-96 --auto_subnet_count 12 --auto_topology mesh "
        "--auto_sensitive_policy deep --auto_sensitive_jitter 0.0"
    ),
    "Test Template": (
        "python main.py ../scenarios/corp_100hosts_dynamic_test.v2.yaml --eval "
        "-load_model wandb/latest-run/files/model.pt "
        "-device cpu -cpus 8 "
        "-net_class NASimNetGNN_LSTM "
        "-use_a_t -episode_step_limit 400 "
        "-observation_format graph_v2 -mp_iterations 3 "
        "--feature_dropout_p 0.0 --dr_prob_jitter 0.0 --dr_cost_jitter 0.0 --dr_scan_cost_jitter 0.0 "
        "--auto_mode per_episode --auto_template ../scenarios/corp_100hosts_dynamic_test.v2.yaml "
        "--auto_host_range 72-96 --auto_subnet_count 12 --auto_topology mesh "
        "--auto_sensitive_policy deep --auto_sensitive_jitter 0.0"
    ),
    # Add more scenarios as needed:
    # "Variant A": (
    #     "python main.py ../scenarios/corp_100hosts_dynamic_varA.v2.yaml --eval "
    #     "-load_model wandb/latest-run/files/model.pt "
    #     # ... rest of command
    # ),
}

# Output filename for JSON results (can be overridden by --out)
OUTPUT_FILENAME = "multi_scenario_eval.json"
# Optional run description (can be overridden by --desc)
DESCRIPTION = "Multi-scenario evaluation comparison"
# =============================================================================

# Regexes to extract metrics; handles np.float64(…) or plain numeric literals
METRIC_PATTERNS = {
    "reward_avg": re.compile(r"'reward_avg'\s*:\s*(?:np\.float64\()?([-+\deE\.]+)\)?"),
    "reward_avg_episodes": re.compile(r"'reward_avg_episodes'\s*:\s*(?:np\.float64\()?([-+\deE\.]+)\)?"),
    # Accept both singular/plural from logs
    "reward_avg_episode": re.compile(r"'reward_avg_episode'\s*:\s*(?:np\.float64\()?([-+\deE\.]+)\)?"),
    "eplen_avg": re.compile(r"'eplen_avg'\s*:\s*(?:np\.float64\()?([-+\deE\.]+)\)?"),
    "captured_avg": re.compile(r"'captured_avg'\s*:\s*(?:np\.float64\()?([-+\deE\.]+)\)?"),
}

# Optional extra info from stdout
PARAM_COUNT_RE = re.compile(r"Number of parameters:\s*(\d+)")
MODEL_LOADED_RE = re.compile(r"Model loaded:\s*(\S+)")


def run_once(command: str) -> Tuple[int, str, str]:
    """Run a shell command once, capture return code, stdout, stderr."""
    proc = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_metrics_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse metrics and optional extras from evaluation output text. Returns a dict or None."""
    found: Dict[str, Any] = {}

    # Prefer 'reward_avg_episodes' but fall back to 'reward_avg_episode'
    for key, pattern in METRIC_PATTERNS.items():
        m = pattern.search(text)
        if m:
            try:
                found[key] = float(m.group(1))
            except ValueError:
                pass

    # Optional: parameter count and model path
    m_params = PARAM_COUNT_RE.search(text)
    if m_params:
        try:
            found["num_parameters"] = int(m_params.group(1))
        except ValueError:
            pass
    m_model = MODEL_LOADED_RE.search(text)
    if m_model:
        found["model_loaded_path"] = m_model.group(1)

    # Normalize to a consistent key set
    if "reward_avg_episodes" not in found and "reward_avg_episode" in found:
        found["reward_avg_episodes"] = found["reward_avg_episode"]

    required = {"reward_avg", "reward_avg_episodes", "eplen_avg", "captured_avg"}
    if not required.issubset(found.keys()):
        return None
    return {
        "reward_avg": float(found["reward_avg"]),
        "reward_avg_episodes": float(found["reward_avg_episodes"]),
        "eplen_avg": float(found["eplen_avg"]),
        "captured_avg": float(found["captured_avg"]),
        # optional
        **({"num_parameters": found["num_parameters"]} if "num_parameters" in found else {}),
        **({"model_loaded_path": found["model_loaded_path"]} if "model_loaded_path" in found else {}),
    }


def parse_cmd_params(cmd: str) -> Dict[str, Any]:
    """Parse key hyperparameters and flags from a command string."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    parsed: Dict[str, Any] = {
        "raw": cmd,
        "python": None,
        "entry": None,
        "scenario_arg": None,
        "flags": {},
        "bool_flags": [],
    }

    if tokens:
        parsed["python"] = tokens[0]
    if len(tokens) > 1:
        parsed["entry"] = tokens[1]

    # find first non-flag token after entry as scenario argument
    idx = 2
    while idx < len(tokens):
        t = tokens[idx]
        if t.startswith("-"):
            break
        parsed["scenario_arg"] = t
        idx += 1
        break

    # collect flags and values
    i = idx
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("-"):
            key = t.lstrip("-")
            val: Any = True  # default for boolean flags
            # if next token exists and is not a flag, treat as value
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                val = tokens[i + 1]
                i += 1
            else:
                parsed["bool_flags"].append(key)
            parsed["flags"][key] = val
        i += 1

    # convenient promoted keys
    for k in [
        "net_class",
        "episode_step_limit",
        "mp_iterations",
        "observation_format",
        "device",
        "cpus",
        "lr",
        "feature_dropout_p",
        "dr_prob_jitter",
        "dr_cost_jitter",
        "dr_scan_cost_jitter",
        "load_model",
    ]:
        if k in parsed["flags"]:
            parsed[k] = parsed["flags"][k]

    # boolean convenience
    for bk in ["use_a_t", "augment_with_action", "eval", "no_debug"]:
        if bk in parsed["flags"] or bk in parsed["bool_flags"]:
            parsed[bk] = True

    return parsed


def run_n_times(label: str, command: str, n: int) -> List[Dict[str, Any]]:
    print(f"\n=== Running {label} command {n} times (cwd={os.getcwd()}) ===")
    results: List[Dict[str, Any]] = []
    for i in range(1, n + 1):
        print(f"\n[{label} run {i}] Command:\n{command}")
        code, out, err = run_once(command)
        if code != 0:
            print(f"[{label} run {i}] ERROR: return code {code}")
            if err:
                print(f"[{label} run {i}] stderr:\n{err}")
        if out:
            print(f"[{label} run {i}] stdout (tail):\n" + "\n".join(out.strip().splitlines()[-10:]))
        parsed = parse_metrics_from_text(out)
        if parsed is None:
            print(f"[{label} run {i}] Could not parse metrics from output.")
        else:
            print(
                f"[{label} run {i}] Parsed metrics: "
                f"reward_avg={parsed['reward_avg']:.6f}, "
                f"reward_avg_episodes={parsed['reward_avg_episodes']:.6f}, "
                f"eplen_avg={parsed['eplen_avg']:.6f}, "
                f"captured_avg={parsed['captured_avg']:.6f}"
            )
            # add debugging tails for export
            parsed_with_io: Dict[str, Any] = dict(parsed)
            parsed_with_io["stdout_tail"] = "\n".join(out.strip().splitlines()[-10:]) if out else ""
            parsed_with_io["return_code"] = code
            results.append(parsed_with_io)
    return results


def summarize(label: str, runs: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    numeric_runs = [
        {k: v for k, v in r.items() if isinstance(v, (int, float))}
        for r in runs
        if all(mk in r for mk in ("reward_avg", "reward_avg_episodes", "eplen_avg", "captured_avg"))
    ]
    if not numeric_runs:
        print(f"\n[SUMMARY] {label}: No successful runs to summarize.")
        return None
    avg = {
        "reward_avg": mean(r["reward_avg"] for r in numeric_runs),
        "reward_avg_episodes": mean(r["reward_avg_episodes"] for r in numeric_runs),
        "eplen_avg": mean(r["eplen_avg"] for r in numeric_runs),
        "captured_avg": mean(r["captured_avg"] for r in numeric_runs),
        "num_successful_runs": len(numeric_runs),
    }
    print(
        f"\n[SUMMARY] {label} averages over {len(numeric_runs)} runs: "
        f"reward_avg={avg['reward_avg']:.6f}, "
        f"reward_avg_episodes={avg['reward_avg_episodes']:.6f}, "
        f"eplen_avg={avg['eplen_avg']:.6f}, "
        f"captured_avg={avg['captured_avg']:.6f}"
    )
    return avg


def compare_scenarios(scenario_averages: Dict[str, Optional[Dict[str, float]]], baseline_scenario: str = None) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
    """Compare all scenarios against each other or against a baseline scenario."""
    print("\n=== Multi-Scenario Comparison ===")
    
    # Filter out None values
    valid_scenarios = {k: v for k, v in scenario_averages.items() if v is not None}
    if len(valid_scenarios) < 2:
        print("Cannot compare; need at least 2 valid scenarios.")
        return None
    
    # If no baseline specified, use the first scenario as baseline
    if baseline_scenario is None or baseline_scenario not in valid_scenarios:
        baseline_scenario = list(valid_scenarios.keys())[0]
        print(f"Using '{baseline_scenario}' as baseline scenario")
    
    baseline_avg = valid_scenarios[baseline_scenario]
    comparison: Dict[str, Dict[str, Dict[str, float]]] = {"vs_baseline": {}, "pairwise": {}}
    
    # Compare each scenario against baseline
    print(f"\n--- Comparisons vs Baseline ({baseline_scenario}) ---")
    for scenario_name, avg in valid_scenarios.items():
        if scenario_name == baseline_scenario:
            continue
        comparison["vs_baseline"][scenario_name] = {}
        print(f"\n{scenario_name} vs {baseline_scenario}:")
        for key in ["reward_avg", "reward_avg_episodes", "eplen_avg", "captured_avg"]:
            baseline_val = baseline_avg[key]
            scenario_val = avg[key]
            delta = scenario_val - baseline_val
            ratio = (scenario_val / baseline_val) if baseline_val != 0 else float('inf')
            print(f"  {key}: {scenario_name}={scenario_val:.6f}, baseline={baseline_val:.6f}, delta={delta:.6f}, ratio={ratio:.3f}")
            comparison["vs_baseline"][scenario_name][key] = {"delta": delta, "ratio": ratio}
    
    # Pairwise comparisons
    print(f"\n--- Pairwise Comparisons ---")
    scenario_names = list(valid_scenarios.keys())
    for i, scenario_a in enumerate(scenario_names):
        for j, scenario_b in enumerate(scenario_names):
            if i >= j:  # Skip same scenario and avoid duplicates
                continue
            pair_key = f"{scenario_a}_vs_{scenario_b}"
            comparison["pairwise"][pair_key] = {}
            print(f"\n{scenario_a} vs {scenario_b}:")
            avg_a = valid_scenarios[scenario_a]
            avg_b = valid_scenarios[scenario_b]
            for key in ["reward_avg", "reward_avg_episodes", "eplen_avg", "captured_avg"]:
                val_a = avg_a[key]
                val_b = avg_b[key]
                delta = val_b - val_a
                ratio = (val_b / val_a) if val_a != 0 else float('inf')
                print(f"  {key}: {scenario_a}={val_a:.6f}, {scenario_b}={val_b:.6f}, delta={delta:.6f}, ratio={ratio:.3f}")
                comparison["pairwise"][pair_key][key] = {"delta": delta, "ratio": ratio}
    
    return comparison


def export_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to: {os.path.abspath(path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NASimEmu eval results across multiple scenarios")
    parser.add_argument("--scenarios", type=str, nargs="*", help="Override scenarios (space-separated scenario names)")
    parser.add_argument("--baseline", type=str, help="Baseline scenario for comparisons")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs per command")
    parser.add_argument("--out", type=str, default=OUTPUT_FILENAME, help="Output JSON filename")
    parser.add_argument("--desc", type=str, default=DESCRIPTION, help="Optional description of this run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine scenarios to evaluate
    if args.scenarios:
        # Use command line provided scenario names
        selected_scenarios = {name: SCENARIO_COMMANDS[name] for name in args.scenarios if name in SCENARIO_COMMANDS}
        if not selected_scenarios:
            print(f"Error: None of the specified scenarios {args.scenarios} found in SCENARIO_COMMANDS")
            print(f"Available scenarios: {list(SCENARIO_COMMANDS.keys())}")
            exit(1)
        scenarios = selected_scenarios
    else:
        # Use all default scenarios from SCENARIO_COMMANDS dict
        scenarios = SCENARIO_COMMANDS

    print(f"Evaluating {len(scenarios)} scenarios:")
    for name, command in scenarios.items():
        print(f"  {name}: {command[:100]}...")

    # Run evaluations for all scenarios
    scenario_runs = {}
    scenario_averages = {}
    
    for scenario_name, command in scenarios.items():
        print(f"\n=== Running command for scenario: {scenario_name} ===")
        runs = run_n_times(scenario_name, command, args.runs)
        scenario_runs[scenario_name] = runs
        avg = summarize(scenario_name, runs)
        scenario_averages[scenario_name] = avg

    # Comparison
    comparison = compare_scenarios(scenario_averages, args.baseline)

    # Parse command flags for metadata (using first scenario as example)
    first_scenario_name = list(scenarios.keys())[0]
    first_command = scenarios[first_scenario_name]
    cmd_parsed = parse_cmd_params(first_command)

    # Metadata and export
    payload: Dict[str, Any] = {
        "metadata": {
            "script_version": 4,
            "cwd": os.getcwd(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "num_runs": args.runs,
            "description": args.desc,
            "num_scenarios": len(scenarios),
            "scenario_names": list(scenarios.keys()),
            "baseline_scenario": args.baseline,
            "command_parsed": cmd_parsed,
        },
        "scenarios": {},
        "comparison": comparison,
    }

    # Add results for each scenario
    for scenario_name, command in scenarios.items():
        payload["scenarios"][scenario_name] = {
            "command": command,
            "runs": scenario_runs[scenario_name],
            "average": scenario_averages[scenario_name]
        }

    export_json(args.out, payload) 
