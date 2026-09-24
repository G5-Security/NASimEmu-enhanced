"""Step 1 core evaluation harness (docs/eval_revision_plan.tex, "Core harness").

Runs a fixed bank of episode seeds against one trained checkpoint, scenario
and condition, and writes exactly one CSV row per completed episode. This is
the foundation every later step in the plan (IDS sensitivity, component
interventions, mechanism ablations, generalization) builds on, so it is
built and tested first, independent of which trained checkpoints are
available.

Design notes
------------
Fixed, out-of-band seed bank
  Episode k uses seed ``SEED_BASE + k`` (default base 1_000_000), an offset
  no training run in this repo has used (main.py training seeds are small
  integers via `-seed`, and SubprocVecEnv workers use `config.seed +
  worker_index`).

Pairing across models/conditions
  ``NASimEmuEnv.__init__`` only reseeds Python's `random` / NumPy's global
  RNG when given an explicit `seed` (src/nasimemu/env.py; regression-tested
  by tests/test_seed_determinism.py); `_generate_env()` draws the generated
  network from those same global RNGs. So constructing a *fresh*
  ``NASimEmuEnv(..., seed=SEED_BASE + k)`` per episode reproduces the same
  generated network for every model/scenario/condition that shares seed k --
  this is the exact pattern already used by
  experiments/evaluate_llm_selector.py's paired (live, distilled) episodes,
  reused here instead of the "reseed a persistent env, then reset()"
  approach sketched in the plan doc, because it sidesteps any ambiguity
  about where the global RNG sits mid-episode (service-dynamics/churn draws
  during step() also consume it).

No SubprocVecEnv
  nasim_debug.py's `_eval` runs 64 environments in parallel and stops once
  100 total episodes have finished -- environments running short episodes
  finish (and get counted) multiple times while long-running ones are still
  mid-episode, over-representing short episodes in the average. This harness
  instead hands out seeds from a single counter and runs episodes serially,
  one full episode per seed, so exactly N seeds contribute exactly N
  episodes. Parallelism across models/conditions/scenarios is done at the
  OS-process level (one job per combination) rather than intra-run, which
  keeps the determinism guarantee above simple to reason about.

Episode-end classification
  NASimEmuEnv.step() discards the wrapped environment's own `done` signal
  unless the agent chose `TerminalAction` (env.py: "ignore done flag from
  the environment, the agent has to choose to terminate"); `d_true` is then
  unconditionally overwritten to equal `d` (env.py, "this will disable the
  difference between true termination and step_limit exceedance"). So the
  *only* two ways an episode ends are: the agent chose TerminalAction, or
  the step limit was reached. This harness records `terminal_action` and
  `hit_step_limit` directly instead of relying on `d_true`.
"""
import argparse
import contextlib
import csv
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One thread per evaluation process: parallelism here is one OS process per
# (model, scenario, condition) job. Uncapped, torch/BLAS each spawn a thread
# per core and N parallel jobs oversubscribe the machine -- measured 27 s vs
# 2.4 s per 400-step episode. Same rationale as the block at the top of main.py.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import torch  # noqa: E402

# Defense in depth for reproducibility: this does not fix the actual bug
# that made episodes with an identical seed diverge (that turned out to be
# NASimNetDHRL.reset_state() leaking `self.ep_t` across episodes -- see the
# comment on that method in nasim_problem/nasim_net_base_hrl.py, and
# tests/test_net_reset_state_ep_t.py). With that fixed at the source,
# episodes were bit-reproducible even *without* this call, across 15/15
# tried seeds. It is kept anyway because it is cheap and covers genuine
# ATen-level nondeterminism (e.g. CUDA scatter ops) that the architectures
# this harness hasn't exercised yet (GNN/Invariant/Attention) might hit.
# GPU runs additionally need CUBLAS_WORKSPACE_CONFIG=:4096:8 set in the shell
# *before* the CUDA context is created, which this in-process call cannot
# do -- set it in the environment when evaluating with `-device cuda`.
torch.use_deterministic_algorithms(True)

from config import config as net_config  # noqa: E402
from nasim_problem import NASimRRL as Problem  # noqa: E402
from nasimemu import env_utils  # noqa: E402
from nasimemu.env import NASimEmuEnv  # noqa: E402
from experiments import ids_interventions  # noqa: E402
from experiments import component_interventions  # noqa: E402
from experiments.dynamics_recorder import IdsCounter  # noqa: E402
from nasimemu.nasim.envs.host_vector import HostVector  # noqa: E402
from nasimemu.nasim.envs.network import Network  # noqa: E402

SEED_BASE = 1_000_000

CSV_FIELDS = [
    "model", "scenario", "condition", "episode_idx", "seed",
    "episode_return", "episode_len", "captured",
    "reward_at_100", "captured_at_100",
    "reward_at_200", "captured_at_200",
    "reward_at_400", "captured_at_400",
    "terminal_action", "hit_step_limit",
]

MILESTONES = (100, 200, 400)

# Optional extra columns, written only when the matching flag is on so the
# default schema (and every existing CSV) is unchanged.
IDS_FIELDS = ["ids_detections", "ids_quarantine", "ids_patch", "ids_monitor",
              "ids_first_step", "ids_max_level", "ids_increases"]
META_FIELDS = ["n_sensitive", "n_hosts"]
TRACE_FIELDS = ["seed", "step", "goal_idx", "goal_entropy", "goal_maxprob",
                "obs_max_level", "true_max_level", "target_level", "detections_so_far"]


def build_net(net_class, scenario, step_limit, checkpoint_path, extra_args=None):
    """Construct and load a checkpoint outside of main.py's CLI flow.

    Same recipe as experiments/evaluate_llm_selector.py's `_build_net`,
    generalized to any `-net_class` (GNN/Invariant/Attention/DHRL/...) so the
    same harness serves the model-comparison steps too. `extra_args`
    overrides/extends the SimpleNamespace for architectures needing flags
    NASimNetDHRL doesn't (e.g. `augment_with_action=True`,
    `mp_iterations=2` for NASimNetGNN_MAct -- see
    NASimEmu-agents/trained_models/models.txt).
    """
    net_args = SimpleNamespace(
        device="cpu", cpus="1", batch=1, seed=None, load_model=None,
        epoch=10, max_epochs=None, mp_iterations=3, emb_dim=64,
        net_class=net_class,
        scenario=scenario, test_scenario=None,
        episode_step_limit=step_limit, use_a_t=True, emulate=False,
        fully_obs=False, observation_format="graph_v2", augment_with_action=False,
        auto_mode="off", auto_template=None, auto_host_range=None, auto_subnet_count=None,
        auto_topology=None, auto_sensitive_policy=None, auto_seed_base=None, auto_sensitive_jitter=0.0,
        force_continue_epochs=0, lr=3e-3, alpha_h=0.3, max_norm=3.,
        sched_lr_rate=None, sched_lr_factor=None, sched_lr_min=None,
        sched_alpha_h_rate=None, sched_alpha_h_factor=None, sched_alpha_h_min=None,
    )
    if extra_args:
        for k, v in extra_args.items():
            setattr(net_args, k, v)

    problem = Problem()
    problem_config = problem.make_config()
    net_config.init(net_args)
    problem_config.update_config(net_config, net_args)
    problem.register_gym()

    net = problem.make_net()
    if checkpoint_path is not None:
        net.load(checkpoint_path)
    net.eval()
    return net


def _graph_obs(env):
    return env_utils.convert_to_graph(env.s_raw, env.subnet_graph, version=2)


def _write_trace_row(writer, net, env, s, action, seed, step, counter):
    """One row per step: which goal the manager holds, how peaked its goal
    distribution is, and the IDS state (as the network sees it, and the true
    values). Written before the environment step, so `detections_so_far` counts
    detections up to but not including this step's action."""
    level_col = ids_interventions._columns()[0]
    node_feats = s[0]
    rows = ids_interventions._host_rows(node_feats)
    obs_max = float(node_feats[rows, level_col].max()) if len(rows) else 0.0
    state = env.env.current_state
    idx = [state.host_num_map[a] for a in state.host_num_map]
    true_max = float(state.tensor[idx, HostVector._detection_level_idx].max())
    try:
        target_level = ids_interventions._true_level_and_threshold(env, action[0])[0]
    except (KeyError, IndexError, TypeError, ValueError):
        target_level = float("nan")
    probs = net.last_subgoal_probs[0].clamp_min(1e-12) if getattr(net, "last_subgoal_probs", None) is not None else None
    entropy = float(-(probs * probs.log()).sum()) if probs is not None else float("nan")
    writer.writerow({
        "seed": seed, "step": step, "goal_idx": int(net.current_subgoal_idx[0]),
        "goal_entropy": entropy, "goal_maxprob": float(probs.max()) if probs is not None else float("nan"),
        "obs_max_level": obs_max, "true_max_level": true_max, "target_level": target_level,
        "detections_so_far": counter.detections if counter is not None else 0,
    })


def run_episode(net, scenario, step_limit, seed, observe_fn_factory=None, net_condition=None,
                record_ids=False, record_meta=False, env_kwargs=None, trace_writer=None):
    """Run exactly one episode from a freshly constructed, seeded env.

    `observe_fn_factory` is the Step 2 intervention point (IDS-observation
    edits, experiments/ids_interventions.py): a zero-arg callable returning
    a *fresh* `observe_fn(env, s) -> s` for this episode -- a factory, not a
    single shared function -- because some conditions (one-step IDS delay,
    within-observation shuffle) carry state across steps that must not leak
    between episodes. A condition with no state can ignore this and just
    return the same function every time. Identity by default.

    `net_condition` is the Step 3 intervention point (Pen-DHRL component
    interventions, experiments/component_interventions.py): a condition name
    applied to `net` for the duration of this episode only, via
    `component_interventions.apply_condition`, which restores every flag it
    touches on exit. None (default) skips that context manager entirely, so
    Step 1/2-only usage never touches these flags at all.
    """
    if observe_fn_factory is None:
        observe_fn_factory = lambda: (lambda env, s: s)  # noqa: E731
    observe_fn = observe_fn_factory()

    condition_ctx = (
        contextlib.nullcontext(net) if net_condition is None
        else component_interventions.apply_condition(net, net_condition)
    )

    # Reproducibility covers both halves of the episode: NASimEmuEnv(seed=)
    # seeds scenario generation (Python random / NumPy global state), but the
    # policy's own action sampling draws from torch's global RNG, which is
    # untouched by that -- without this, two runs with the same seed still
    # diverge as soon as the network samples a stochastic action.
    torch.manual_seed(seed)

    # training_mode=False is required, not optional: NASimEmuEnv defaults to
    # True, under which CurriculumManager.get_current_stage() picks a stage
    # by *epoch* (starting at epoch 0 = "baseline" -- IDS off, zero scan
    # noise/churn/timeouts), not by "hardest stage". nasim_debug.py's own
    # _eval() sets this explicitly for exactly this reason (its comment:
    # "Force training_mode=False so the curriculum manager reports its FINAL
    # (hardest) stage during evaluation, regardless of the training epoch").
    # Missing this is a real, previously-silent bug this harness had until
    # tests/test_eval_harness.py::test_episodes_run_under_the_hardest_curriculum_stage
    # caught it -- every episode was evaluated under the easiest stage
    # instead of the intended one; see docs/eval_revision_plan.tex.
    env = NASimEmuEnv(scenario_name=scenario, step_limit=step_limit,
                       observation_format="graph_v2", seed=seed,
                       training_mode=False, **(env_kwargs or {}))
    net.reset_state()
    env.reset()

    # Optional instrumentation. A class-level recorder is inert to the
    # environment (tests/test_dynamics_recorder.py pins that a recorder never
    # changes a trajectory) and is detached again in the finally below.
    counter = IdsCounter() if (record_ids or trace_writer is not None) else None
    if counter is not None:
        HostVector.set_event_recorder(counter)
    meta = None
    if record_meta:
        meta = {"n_sensitive": len(env.env.scenario.sensitive_hosts), "n_hosts": len(env.env.scenario.hosts)}
    if trace_writer is not None and hasattr(net, "record_subgoal_probs"):
        net.record_subgoal_probs = True

    ep_t, ep_return, done = 0, 0.0, False
    captured = 0
    terminal_action = False
    milestone_vals = {m: (None, None) for m in MILESTONES}  # (reward, captured) at that step

    try:
        with condition_ctx, torch.no_grad():
            while ep_t < step_limit and not done:
                s = observe_fn(env, _graph_obs(env))
                a, _v, _pi, _raw_a = net([s])
                action = a[0]
                if trace_writer is not None:
                    _write_trace_row(trace_writer, net, env, s, action, seed, ep_t, counter)
                # net() returns the raw (target, action_id) tuple (nasim_net_base_hrl.py
                # forward(): "return list(zip(targets, a_id)), ..."); action_id==-1 is
                # only turned into an actual TerminalAction instance *inside*
                # env.step() -> _translate_action() (src/nasimemu/env.py). Checking
                # isinstance(action, TerminalAction) here, before that translation,
                # never fires -- action is always the untranslated tuple at this
                # point. Check the action_id directly instead.
                terminal_action = (action[1] == -1)

                _s, r, done, info = env.step(action)
                ep_return += float(r)
                ep_t += 1
                captured = info["captured"]

                if ep_t in milestone_vals:
                    milestone_vals[ep_t] = (ep_return, captured)
    finally:
        if counter is not None:
            HostVector.set_event_recorder(None)

    hit_step_limit = (ep_t >= step_limit) and not terminal_action

    # Carry the last-known value forward for milestones the episode didn't
    # reach (short episode): final return/captured are the right fill-in,
    # since nothing changes after the episode ends.
    for m in MILESTONES:
        if milestone_vals[m] == (None, None):
            milestone_vals[m] = (ep_return, captured)

    return {
        "episode_return": ep_return,
        "episode_len": ep_t,
        "captured": captured,
        "terminal_action": terminal_action,
        "hit_step_limit": hit_step_limit,
        "milestones": milestone_vals,
        "ids": counter.summary() if (record_ids and counter is not None) else None,
        "meta": meta,
    }


def run_eval(net, model_name, scenario, condition, step_limit, n_episodes,
             seed_base=SEED_BASE, observe_fn_factory=None, net_condition=None,
             csv_path=None, verbose=True, record_ids=False, record_meta=False,
             env_kwargs=None, trace_path=None):
    """Run exactly n_episodes (seed_base .. seed_base+n_episodes-1) and
    return the list of per-episode result dicts. Optionally streams rows to
    csv_path as they complete (append mode, so a run can be resumed by
    pointing a fresh seed_base at the same file)."""
    rows = []
    writer = None
    f = None
    if csv_path is not None:
        new_file = not os.path.exists(csv_path)
        f = open(csv_path, "a", newline="")
        fieldnames = CSV_FIELDS + (IDS_FIELDS if record_ids else []) + (META_FIELDS if record_meta else [])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
    trace_f = trace_writer = None
    if trace_path is not None:
        new_trace = not os.path.exists(trace_path)
        trace_f = open(trace_path, "a", newline="")
        trace_writer = csv.DictWriter(trace_f, fieldnames=TRACE_FIELDS)
        if new_trace:
            trace_writer.writeheader()

    t_start = time.time()
    try:
        for i in range(n_episodes):
            seed = seed_base + i
            result = run_episode(net, scenario, step_limit, seed,
                                  observe_fn_factory=observe_fn_factory, net_condition=net_condition,
                                  record_ids=record_ids, record_meta=record_meta,
                                  env_kwargs=env_kwargs, trace_writer=trace_writer)

            row = {
                "model": model_name, "scenario": scenario, "condition": condition,
                "episode_idx": i, "seed": seed,
                "episode_return": result["episode_return"],
                "episode_len": result["episode_len"],
                "captured": result["captured"],
                "terminal_action": result["terminal_action"],
                "hit_step_limit": result["hit_step_limit"],
            }
            for m in MILESTONES:
                r_m, c_m = result["milestones"][m]
                row[f"reward_at_{m}"] = r_m
                row[f"captured_at_{m}"] = c_m

            if result.get("ids"):
                row.update(result["ids"])
            if result.get("meta"):
                row.update(result["meta"])
            rows.append(row)
            if trace_f is not None:
                trace_f.flush()
            if writer is not None:
                writer.writerow(row)
                f.flush()

            if verbose and (i + 1) % max(1, n_episodes // 10) == 0:
                elapsed = time.time() - t_start
                print(f"[eval_harness] {model_name}/{scenario}/{condition}: "
                      f"{i + 1}/{n_episodes} episodes, {elapsed:.1f}s elapsed")
    finally:
        if f is not None:
            f.close()
        if trace_f is not None:
            trace_f.close()

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint_path", default=None,
                     help="Omit to run an untrained net -- useful for dry-running the harness "
                          "itself (plumbing, CSV schema, a condition's wiring) without a "
                          "trained checkpoint; never a substitute for real results")
    ap.add_argument("--net_class", default="NASimNetDHRL")
    ap.add_argument("--model_name", default=None, help="Label for the CSV `model` column; defaults to --net_class")
    ap.add_argument("--scenario", required=True,
                     help="A base scenario, or one of experiments/make_variants.py's "
                          "generated variants for the Step 4/5 mechanism-knockout/intensity conditions")
    ap.add_argument("--condition", default=None,
                     help="CSV `condition` label; defaults to --ids_condition if given, else 'full_dynamics'")
    ap.add_argument("--ids_condition", default=None, choices=ids_interventions.CONDITIONS,
                     help="Step 2: apply one of ids_interventions.py's observation edits")
    ap.add_argument("--ids_shuffle_seed", type=int, default=0,
                     help="Only used when --ids_condition shuffled")
    ap.add_argument("--net_condition", default=None, choices=component_interventions.ALL_CONDITIONS,
                     help="Step 3: apply one of component_interventions.py's inference-time "
                          "flags; only meaningful with --net_class NASimNetDHRL, the only "
                          "architecture with these flags")
    ap.add_argument("--step_limit", type=int, default=400)
    ap.add_argument("--n_episodes", type=int, default=1000)
    ap.add_argument("--seed_base", type=int, default=SEED_BASE)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--record_ids", action="store_true",
                     help="Add per-episode IDS columns (detections by response type, first detection step, max level)")
    ap.add_argument("--record_meta", action="store_true",
                     help="Add per-episode scenario columns (n_sensitive, n_hosts): needed to normalize captured counts across generated networks")
    ap.add_argument("--trace_goals", default=None,
                     help="Also write a per-step trace CSV (manager goal, goal entropy, IDS levels) to this path")
    ap.add_argument("--scan_noise_observed", action="store_true",
                     help="Follow-up E: let scan false-positive/negative noise reach the agent's observation "
                          "(off by default; the simulator otherwise observes the true scan result)")
    ap.add_argument("--auto_template", default=None,
                     help="Follow-up F: generate a fresh network per episode from this template scenario YAML "
                          "(the network is generated from the episode seed)")
    ap.add_argument("--auto_topology", default=None, choices=["mesh", "chain", "random"])
    ap.add_argument("--auto_host_range", default=None, help="Total hosts range for generated networks, e.g. 60-80")
    ap.add_argument("--auto_subnet_count", type=int, default=None)
    ap.add_argument("--auto_sensitive_jitter", type=float, default=None,
                     help="Per-subnet multiplicative jitter on the template's sensitive-host probabilities")
    args = ap.parse_args()

    if args.net_condition is not None and args.net_class != "NASimNetDHRL":
        raise SystemExit(f"--net_condition needs --net_class NASimNetDHRL, got {args.net_class!r}")

    net = build_net(args.net_class, args.scenario, args.step_limit, args.checkpoint_path)
    model_name = args.model_name or args.net_class
    condition = args.condition or args.ids_condition or args.net_condition or "full_dynamics"
    observe_fn_factory = None
    if args.ids_condition is not None:
        observe_fn_factory = ids_interventions.make_condition_factory(
            args.ids_condition, seed=args.ids_shuffle_seed)

    env_kwargs = {}
    if args.auto_template:
        env_kwargs.update(auto_mode="per_episode", auto_template=os.path.abspath(args.auto_template))
        for name in ("auto_topology", "auto_host_range", "auto_subnet_count", "auto_sensitive_jitter"):
            if getattr(args, name) is not None:
                env_kwargs[name] = getattr(args, name)
    if args.scan_noise_observed:
        HostVector.set_scan_noise_observed(True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    run_eval(net, model_name, args.scenario, condition, args.step_limit,
              args.n_episodes, seed_base=args.seed_base, csv_path=args.out_csv,
              observe_fn_factory=observe_fn_factory, net_condition=args.net_condition,
              record_ids=args.record_ids, record_meta=args.record_meta,
              env_kwargs=env_kwargs or None, trace_path=args.trace_goals)


if __name__ == "__main__":
    main()
