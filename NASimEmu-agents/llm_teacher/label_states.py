"""
Triggered-sampling data-collection driver -- master plan Sec 15.8.

Run from NASimEmu-agents/:
    python -m llm_teacher.label_states --target_records 2000 --policy random
    python -m llm_teacher.label_states --target_records 2000 --policy checkpoint --checkpoint_path <ckpt>

Behavior-policy mixture (Sec 15.8: "a deliberately diverse mixture of
behavior policies... labeling only successful expert trajectories would
produce a narrow dataset"): random (--policy random), untrained Pen-DHRL
(--policy untrained_net, zero training cost), and any partially/fully
trained Pen-DHRL checkpoint (--policy checkpoint --checkpoint_path ...).
Run this script multiple times against the same --out_dir with different
--policy values to mix them into one dataset (DatasetWriter appends).
Flat-baseline (GNN/Attention) rollout policies from Sec 15.8's full list are
not implemented -- those architectures don't share NASimNetDHRL's action
head/goal machinery, so reusing the same net-based rollout path here isn't a
small addition.

Trigger set (Sec 15.8's "strategically triggered points"): episode start,
new subnet discovered, first compromise, privilege change, an IDS alert, a
periodic fallback every --periodic_every steps, three consecutive failures
against the same target, and (only under --policy untrained_net/checkpoint,
since it needs a net to query) high student policy entropy. "High
student/teacher disagreement" is not implemented as a trigger: it would
require calling the teacher speculatively just to decide *whether* to call
the teacher, which defeats the point of a trigger (call the teacher only
when a cheap, teacher-free signal says the state is worth it).

Each episode gets a random episode_id, stamped into every record from that
episode -- llm_teacher/split_dataset.py uses it to split at the
scenario-instance level (Sec 15.8: never place two steps of the same episode
across a train/test boundary). Records collected before this field existed
have no episode_id; split_dataset.py falls back to per-record granularity for
those and says so.
"""
import argparse
import os
import random
import sys
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nasimemu.env import NASimEmuEnv  # noqa: E402
from nasimemu import env_utils  # noqa: E402
from llm_teacher.state_summarizer import summarize_state  # noqa: E402
from llm_teacher.teacher_client import label_one_state as _llm_label_one_state, DEFAULT_MODEL  # noqa: E402
from llm_teacher.heuristic_teacher import label_one_state as _heuristic_label_one_state  # noqa: E402
from llm_teacher.cascade import label_one_state_cascade as _cascade_label_one_state  # noqa: E402
from llm_teacher.dataset_writer import DatasetWriter  # noqa: E402

DEFAULT_SCENARIO = (
    "../scenarios/corp_100hosts_dynamic.v2.yaml:"
    "../scenarios/corp_100hosts_dynamic_varA.v2.yaml:"
    "../scenarios/corp_100hosts_dynamic_varB.v2.yaml"
)


def _build_dhrl_net(scenario, step_limit, checkpoint_path=None):
    """
    Constructs a NASimNetDHRL (+ the minimal config it needs) for use as a
    rollout policy during dataset collection -- diversifying label_states.py's
    behavior-policy mixture beyond pure-random actions, per master plan Sec
    15.8 ("labeling only successful expert trajectories would produce a narrow
    dataset"; the inverse problem -- random-only -- starves the same classes
    a diverse mixture is meant to cover, e.g. ESCALATE_PRIVILEGE).

    With checkpoint_path=None this is an UNTRAINED net (random init) -- still
    architecturally biased away from uniform-random, at zero training cost.
    With checkpoint_path set, loads those trained weights via Net.load (which
    enforces the ontology-version check, net.py).

    Deferred import of config/nasim_problem (not torch, imported at module
    level): this module is also imported by other llm_teacher/ code that has
    no reason to pull in the full net-construction machinery, which brings in
    gym registration side effects.
    """
    from types import SimpleNamespace
    from config import config as net_config
    from nasim_problem import NASimRRL as Problem

    net_args = SimpleNamespace(
        device='cpu', cpus='1', batch=1, seed=None, load_model=None,
        epoch=10, max_epochs=None, mp_iterations=3, emb_dim=64,
        net_class='NASimNetDHRL',
        scenario=scenario, test_scenario=None,
        episode_step_limit=step_limit, use_a_t=True, emulate=False,
        fully_obs=False, observation_format='graph_v2', augment_with_action=False,
        auto_mode='off', auto_template=None, auto_host_range=None, auto_subnet_count=None,
        auto_topology=None, auto_sensitive_policy=None, auto_seed_base=None, auto_sensitive_jitter=0.0,
        force_continue_epochs=0, lr=3e-3, alpha_h=0.3, max_norm=3.,
        sched_lr_rate=None, sched_lr_factor=None, sched_lr_min=None,
        sched_alpha_h_rate=None, sched_alpha_h_factor=None, sched_alpha_h_min=None,
    )

    problem = Problem()
    problem_config = problem.make_config()
    net_config.init(net_args)
    problem_config.update_config(net_config, net_args)
    problem.register_gym()

    net = problem.make_net()
    if checkpoint_path is not None:
        net.load(checkpoint_path)
    net.train()  # sample stochastically (Categorical/Bernoulli), not argmax -- see forward()'s self.training branch
    return net


def _net_action(net, env):
    """Runs one NASimNetDHRL forward pass on the current single-env state and
    returns an (addr, action_id) tuple in the same shape _random_action
    returns -- see forward()'s `return list(zip(targets, a_id)), ...`."""
    s_batch = [env_utils.convert_to_graph(env.s_raw, env.subnet_graph, version=2)]
    with torch.no_grad():
        a, _v, _pi, _raw_a = net(s_batch)
    return a[0]


MAX_SUBGOAL_ENTROPY = np.log(8)  # 8-way uniform categorical, nats


def _student_entropy(net, env):
    """Shannon entropy (nats) of the trained student's own subgoal
    distribution at the current state -- Sec 15.8's 'high student policy
    entropy' trigger. Queried via only_subgoal_logits=True, reset_hidden=True
    -- a pure side-channel read verified side-effect-free on the ongoing
    rollout's recurrent state by
    tests/test_recurrent_replay_with_teacher_labels.py -- never used to pick
    an action, only to decide whether this state is worth labeling."""
    s_batch = [env_utils.convert_to_graph(env.s_raw, env.subnet_graph, version=2)]
    with torch.no_grad():
        logits = net(s_batch, only_subgoal_logits=True, reset_hidden=True)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
    return float(entropy.item())


def _graph_obs(env):
    """The graph-format observation NASimNetDHRL.prepare_batch actually
    consumes (4-tuple: node_feats, edge_index, node_index, pos_index) --
    NOT env.s_raw (the flat HostVector rows used only for the sanitized
    summary). Recomputed the same way env.py's step()/reset() already do
    internally (env_utils.convert_to_graph), from the same env.s_raw +
    env.subnet_graph this call site already has in hand."""
    return env_utils.convert_to_graph(env.s_raw, env.subnet_graph, version=2)


def _random_action(env):
    """Sec 15.8's first behavior-policy type: a random valid-action policy.
    "Valid" here means "targets a host the agent has actually discovered" --
    everything else about action validity (e.g. whether an exploit applies to
    that host's OS/services) is left to the environment, matching how a truly
    naive random policy would behave."""
    discovered = [addr for addr in env.host_index if tuple(addr) != (0, 0)]
    # host_index is the fixed node ordering, not "discovered" -- so also
    # allow any address; NASimEnv silently no-ops/penalizes invalid targets
    # rather than crashing, which is fine for a labeling-only rollout.
    addr = tuple(random.choice(list(env.host_index)))
    action_id = random.randrange(len(env.action_list))
    return (addr, action_id)


def collect(args):
    # Phase 2: with --hardest_stage the env is built in evaluation mode
    # (training_mode=False), which makes CurriculumManager pin the final,
    # hardest curriculum stage -- IDS fully on -- from the first reset. That is
    # what lets detection (ids_level) actually accumulate, so the heuristic/LLM
    # teacher's REDUCE_DETECTION rule can fire and the dataset gains stealth
    # labels. Labeling in the default training_mode leaves IDS effectively off
    # (baseline stage), which is why an existing default-built dataset has
    # almost no REDUCE_DETECTION records.
    env = NASimEmuEnv(
        scenario_name=args.scenario,
        step_limit=args.step_limit,
        observation_format="graph_v2",
        training_mode=(not args.hardest_stage),
    )
    if args.hardest_stage:
        # The inner NASim env (and its curriculum application) is created lazily
        # on the first reset, so read IDS state only after one -- reading before
        # would see the pre-reset default and misreport IDS as off. The collect
        # loop resets again per episode, so this throwaway reset is harmless.
        env.reset()
        from nasimemu.nasim.envs.host_vector import HostVector
        ids_enabled = bool((getattr(HostVector, 'ids_config', None) or {}).get('enabled', False))
        print(f"[label_states] --hardest_stage ON: env pinned to final curriculum stage "
              f"(training_mode=False). IDS enabled={ids_enabled}. "
              f"REDUCE_DETECTION becomes labelable once accessed hosts accumulate detection.")
    writer = DatasetWriter(args.out_dir)

    # Teacher backend dispatch. 'llm' (default): single-model calls, unchanged.
    # 'heuristic' (master plan Ch.16 condition 6): rule-based, no network call,
    # same validator, a fair comparison rather than a strawman. 'llm_cascade'
    # (docs/llm_teacher_workstation_upgrade_walkthrough.md): primary model +
    # conditional escalation to a second model -- --model is ignored under
    # this backend (use --primary_model/--escalation_model instead).
    if args.teacher_backend == "llm":
        label_one_state = _llm_label_one_state
    elif args.teacher_backend == "heuristic":
        label_one_state = _heuristic_label_one_state
    else:
        if not args.escalation_model:
            raise SystemExit("--teacher_backend llm_cascade requires --escalation_model")
        primary_model = args.primary_model or args.model

        def label_one_state(summary, model=None):  # noqa: ARG001 -- kept for call-signature parity
            return _cascade_label_one_state(summary, primary_model=primary_model, escalation_model=args.escalation_model)

    # Policy dispatch: 'random' (default, unchanged) vs 'untrained_net' (a
    # freshly-initialized NASimNetDHRL, zero training cost) vs 'checkpoint' (a
    # loaded, at-least-lightly-trained NASimNetDHRL) -- see _build_dhrl_net's
    # docstring for why this exists. Run this script multiple times with the
    # same --out_dir and different --policy values to mix policies into one
    # dataset (DatasetWriter already appends).
    net = None
    if args.policy != "random":
        if args.policy == "checkpoint" and not args.checkpoint_path:
            raise SystemExit("--policy checkpoint requires --checkpoint_path")
        net = _build_dhrl_net(args.scenario, args.step_limit, checkpoint_path=args.checkpoint_path)
        print(f"[label_states] policy={args.policy}" + (f" ({args.checkpoint_path})" if args.checkpoint_path else " (untrained)"))

    n_valid = writer.count_valid()
    print(f"[label_states] existing dataset: {n_valid} valid records already in {args.out_dir}")

    total_steps = 0
    t_start = time.time()

    while n_valid < args.target_records and total_steps < args.max_steps:
        env.reset()
        if net is not None:
            net.reset_state()
        episode_id = uuid.uuid4().hex[:12]
        recent_actions = []
        seen_subnets = set()
        compromised_seen = set()
        prev_access = {}   # addr -> access string, for the privilege_change trigger
        ids_alerted = set()  # addrs that have already crossed their ids_threshold this episode
        ep_t = 0

        def _label(summary, trigger_name):
            record = label_one_state(summary, model=args.model)
            record["episode_id"] = episode_id
            record["trigger"] = trigger_name
            writer.add(record, _graph_obs(env))
            return record

        # trigger: episode start
        summary = summarize_state(env.s_raw, ep_t, args.step_limit, recent_actions=recent_actions)
        record = _label(summary, "episode_start")
        if record["valid"]:
            n_valid += 1
        print(f"[label_states] step={total_steps} valid={n_valid}/{args.target_records} trigger=episode_start")

        for _ in range(args.step_limit):
            if n_valid >= args.target_records or total_steps >= args.max_steps:
                break

            action = _net_action(net, env) if net is not None else _random_action(env)
            s, r, d, info = env.step(action)
            total_steps += 1
            ep_t += 1

            addr, action_id = action
            action_name = env.action_list[action_id][0].__name__
            success = bool(r > 0)
            target_str = f"{int(addr[0])}.{int(addr[1])}"
            recent_actions.append({"action": action_name, "target": target_str, "success": success})

            triggered, trigger_name = False, None

            new_subnets = env.discovered_subnets - seen_subnets
            if new_subnets:
                seen_subnets |= new_subnets
                triggered, trigger_name = True, "new_subnet_discovered"

            # single summary per step, reused by every trigger check below and
            # (if a trigger fires) by the actual label call -- the previous
            # version called summarize_state twice per step (once here, once
            # again only if triggered).
            summary = summarize_state(env.s_raw, ep_t, args.step_limit, recent_actions=recent_actions)
            hosts = summary["hosts"]

            newly_compromised = {h["addr"] for h in hosts if h["compromised"]} - compromised_seen
            if newly_compromised:
                compromised_seen |= newly_compromised
                if not triggered:
                    triggered, trigger_name = True, "first_compromise"

            current_access = {h["addr"]: h["access"] for h in hosts}
            privilege_up = any(
                prev_access.get(a) is not None and current_access[a] != prev_access[a]
                and current_access[a] in ("user", "root")
                for a in current_access
            )
            if privilege_up and not triggered:
                triggered, trigger_name = True, "privilege_change"
            prev_access = current_access

            newly_alerted = {
                h["addr"] for h in hosts
                if h["ids_threshold"] > 0 and h["ids_level"] >= h["ids_threshold"]
            } - ids_alerted
            if newly_alerted:
                ids_alerted |= newly_alerted
                if not triggered:
                    triggered, trigger_name = True, "ids_alert"

            last_three = recent_actions[-3:]
            same_target = len(last_three) == 3 and len({a["target"] for a in last_three}) == 1
            all_failed = len(last_three) == 3 and all(not a["success"] for a in last_three)
            if same_target and all_failed and not triggered:
                triggered, trigger_name = True, "three_consecutive_failures"

            if net is not None and not triggered:
                entropy = _student_entropy(net, env)
                if entropy > 0.8 * MAX_SUBGOAL_ENTROPY:
                    triggered, trigger_name = True, "high_student_entropy"

            if not triggered and (total_steps % args.periodic_every == 0):
                triggered, trigger_name = True, "periodic"

            if d:
                break
            if not triggered:
                continue

            record = _label(summary, trigger_name)
            if record["valid"]:
                n_valid += 1
            print(f"[label_states] step={total_steps} valid={n_valid}/{args.target_records} trigger={trigger_name}")

    elapsed = time.time() - t_start
    print(f"[label_states] done: {n_valid} valid records, {total_steps} env steps, {elapsed/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--step_limit", type=int, default=400)
    ap.add_argument("--target_records", type=int, default=300, help="stop once this many VALID records are collected")
    ap.add_argument("--max_steps", type=int, default=20000, help="hard cap on total env steps regardless of target_records")
    ap.add_argument("--periodic_every", type=int, default=25)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ignored when --teacher_backend is heuristic or llm_cascade")
    ap.add_argument("--teacher_backend", choices=["llm", "heuristic", "llm_cascade"], default="llm",
                     help="'llm' (default, unchanged): single-model calls. 'heuristic' (master plan Ch.16 "
                          "condition 6): rule-based, no network call -- see llm_teacher/heuristic_teacher.py. "
                          "'llm_cascade' (docs/llm_teacher_workstation_upgrade_walkthrough.md, workstation-scale): "
                          "a primary model with conditional escalation to a second model -- see "
                          "--primary_model/--escalation_model and llm_teacher/cascade.py. Run a separate "
                          "collection pass into a separate --out_dir per backend to build a comparable dataset.")
    ap.add_argument("--primary_model", default=None, help="llm_cascade only; defaults to --model")
    ap.add_argument("--escalation_model", default=None, help="llm_cascade only; required if --teacher_backend llm_cascade")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "training_data", "llm_teacher_dataset"))
    ap.add_argument("--policy", choices=["random", "untrained_net", "checkpoint"], default="random",
                     help="Rollout policy for this collection pass (Sec 15.8's behavior-policy mixture). "
                          "Run this script multiple times with the same --out_dir and different --policy "
                          "values to mix policies into one dataset.")
    ap.add_argument("--checkpoint_path", default=None,
                     help="Path to a NASimNetDHRL checkpoint (required for --policy checkpoint)")
    ap.add_argument("--hardest_stage", action="store_true",
                     help="Phase 2: build the env in evaluation mode so the curriculum is pinned to its "
                          "final (hardest) stage -- IDS fully on -- for the whole collection. Use this to "
                          "collect a dataset whose states actually contain detection pressure, so the "
                          "teacher labels REDUCE_DETECTION. Pairs with main.py's --llm_distill_rewarm_scale, "
                          "which revives the teacher weight when the training curriculum turns IDS on.")
    args = ap.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
