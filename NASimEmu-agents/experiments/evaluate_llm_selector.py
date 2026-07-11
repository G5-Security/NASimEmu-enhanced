"""
Design B evaluation driver: master plan Sec 15.11 -- "LLM as (evaluation-time)
subgoal selector over semantically labeled, RL-discovered abstractions."

Loads a --llm_distill-trained NASimNetDHRL checkpoint (its goal_bank slots
already mean GOAL_NAMES[k] by construction, since that's what the distillation
loss trained subgoal_head to predict -- no separate post-hoc slot-labeling
pass is needed here, unlike the master plan's original, more general Design B
spec). Runs each episode twice, from the identical seeded starting scenario:

  - 'live':      every switch decision is made by a live local-LLM teacher
                 call (llm_teacher.teacher_client), forced into the manager
                 via the existing force_action=(None, None, subgoal_idx)
                 mechanism (nasim_net_base_hrl.py forward()) -- this REPLACES
                 subgoal_head/subgoal_switch's role for goal choice AND
                 duration (the teacher's own `horizon` field controls how long
                 a goal is held before the teacher is asked again), while the
                 worker and goal_bank vectors are the exact same trained
                 weights used in 'distilled' mode. This is what makes the
                 causal chain state -> stated rationale -> goal -> worker
                 action real and auditable, not a parallel narrative.
  - 'distilled': the trained subgoal_head/subgoal_switch choose for
                 themselves, deterministically (net.eval() -> argmax), no
                 teacher calls at all -- ordinary deployed inference.

Output: one JSONL transcript per mode (state summary + teacher rationale +
the actions taken until the next switch, for 'live') plus a summary JSON
comparing episode return/length/captured-host-count between the two modes.

This is an evaluation-only script (Sec 15.10: the teacher must never sit
inside a path PPO's replay/target computation depends on) -- no training, no
gradient steps, no SubprocVecEnv. Sized per Sec 16.2: "a few hundred episodes,
not full training."
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from config import config as net_config  # noqa: E402
from nasim_problem import NASimRRL as Problem  # noqa: E402
from nasimemu import env_utils  # noqa: E402
from nasimemu.env import NASimEmuEnv  # noqa: E402
from llm_teacher.goal_ontology import GOAL_NAMES, GOAL_INDEX  # noqa: E402
from llm_teacher.state_summarizer import summarize_state  # noqa: E402
from llm_teacher.teacher_client import label_one_state, DEFAULT_MODEL  # noqa: E402

DEFAULT_SCENARIO = (
    "../scenarios/corp_100hosts_dynamic.v2.yaml:"
    "../scenarios/corp_100hosts_dynamic_varA.v2.yaml:"
    "../scenarios/corp_100hosts_dynamic_varB.v2.yaml"
)

FALLBACK_HORIZON = 4  # used only if a teacher call is invalid and there's no prior goal to hold


def _build_net(scenario, step_limit, checkpoint_path):
    """Same construction pattern as llm_teacher.label_states._build_dhrl_net,
    but checkpoint_path is required (this script only makes sense against an
    already --llm_distill-trained checkpoint) and the net is left in eval()
    mode by default -- callers switch to train() only if they specifically
    want stochastic sampling, which 'distilled' mode here does not."""
    from types import SimpleNamespace

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
    net.load(checkpoint_path)  # enforces the ontology-version check (net.py)
    net.eval()
    return net


def _graph_obs(env):
    return env_utils.convert_to_graph(env.s_raw, env.subnet_graph, version=2)


def run_episode_distilled(net, env, step_limit):
    """Trained subgoal_head/subgoal_switch choose for themselves -- ordinary
    deployed inference, no teacher calls, no forcing."""
    net.reset_state()
    env.reset()
    ep_t, ep_return, done = 0, 0.0, False

    captured = 0
    while ep_t < step_limit and not done:
        s_batch = [_graph_obs(env)]
        with torch.no_grad():
            a, _v, _pi, _raw_a = net(s_batch)
        _s, r, done, info = env.step(a[0])
        ep_return += float(r)
        ep_t += 1
        # NASimEmuEnv.step() calls self.reset() internally whenever d=True
        # (env.py, "if d: s = self.reset()"), which zeroes self.captured --
        # info['captured'] is snapshotted *before* that internal reset, so it
        # must be read from info on every step, not from env.captured after
        # the loop (which would read back 0 on the common case where the
        # episode actually ran to completion).
        captured = info["captured"]

    return {"mode": "distilled", "episode_len": ep_t, "episode_return": ep_return, "captured": captured}


def run_episode_live(net, env, step_limit, model, transcript_records, episode_idx):
    """Every switch decision comes from a live teacher call; the returned
    goal is forced into the manager via force_action's forced_subgoal slot
    every step (holding it constant across the teacher's own stated horizon),
    so subgoal_head/subgoal_switch's outputs are computed but never acted on
    -- see module docstring."""
    net.reset_state()
    env.reset()
    ep_t, ep_return, done, captured = 0, 0.0, False, 0

    recent_actions = []
    current_goal_idx = None
    current_goal_name = None
    steps_until_recall = 0
    switch_actions = []  # actions taken since the current goal was chosen

    def _flush_switch_record(record_meta):
        if record_meta is not None:
            record_meta["actions_until_next_switch"] = list(switch_actions)
            transcript_records.append(record_meta)
        switch_actions.clear()

    pending_record = None

    while ep_t < step_limit and not done:
        if steps_until_recall <= 0:
            _flush_switch_record(pending_record)

            summary = summarize_state(
                env.s_raw, ep_t, step_limit,
                active_goal_name=current_goal_name,
                active_goal_remaining_horizon=None,
                recent_actions=recent_actions,
            )
            teacher_record = label_one_state(summary, model=model)
            parsed = teacher_record["parsed_output"]

            if teacher_record["valid"]:
                current_goal_name = parsed["goal"]
                current_goal_idx = GOAL_INDEX[current_goal_name]
                steps_until_recall = int(parsed["horizon"])
                pending_record = {
                    "episode": episode_idx, "step": ep_t,
                    "goal": current_goal_name, "target_host": parsed.get("target_host"),
                    "target_subnet": parsed.get("target_subnet"), "confidence": parsed.get("confidence"),
                    "reason_code": parsed.get("reason_code"), "rationale": parsed.get("rationale"),
                    "state_summary": summary,
                }
            else:
                # Teacher call failed validation after retries (Sec 15.7:
                # rejected, not discarded -- but nothing to force here). Hold
                # the previous goal one more step and retry next step; on the
                # very first call with no previous goal, fall back to index 0
                # (DISCOVER_SUBNET) rather than crash the eval run.
                if current_goal_idx is None:
                    current_goal_idx = 0
                    current_goal_name = GOAL_NAMES[0]
                steps_until_recall = 1
                pending_record = {
                    "episode": episode_idx, "step": ep_t,
                    "goal": current_goal_name, "reject_reason": teacher_record["reject_reason"],
                    "state_summary": summary,
                }

        s_batch = [_graph_obs(env)]
        forced_subgoal = torch.tensor([current_goal_idx], dtype=torch.long)
        with torch.no_grad():
            a, _v, _pi, _raw_a = net(s_batch, force_action=(None, None, forced_subgoal))

        addr, action_id = a[0]
        action_name = env.action_list[action_id][0].__name__ if action_id != -1 else "TERMINATE"
        _s, r, done, info = env.step(a[0])
        success = bool(r > 0)
        target_str = f"{int(addr[0])}.{int(addr[1])}"
        recent_actions.append({"action": action_name, "target": target_str, "success": success})
        switch_actions.append({"step": ep_t, "action": action_name, "target": target_str, "success": success})

        ep_return += float(r)
        ep_t += 1
        steps_until_recall -= 1
        # see run_episode_distilled's comment: must read info['captured'], not env.captured
        captured = info["captured"]

    _flush_switch_record(pending_record)

    return {"mode": "live", "episode_len": ep_t, "episode_return": ep_return, "captured": captured}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True, help="A --llm_distill-trained NASimNetDHRL checkpoint")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--step_limit", type=int, default=400)
    ap.add_argument("--n_episodes", type=int, default=10, help="Paired (live, distilled) episodes -- small by design, see module docstring")
    ap.add_argument("--base_seed", type=int, default=1000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "training_data", "llm_selector_eval"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    transcript_path = os.path.join(args.out_dir, "live_transcript.jsonl")
    summary_path = os.path.join(args.out_dir, "episode_summary.jsonl")

    net = _build_net(args.scenario, args.step_limit, args.checkpoint_path)

    transcript_records = []
    episode_summaries = []
    t_start = time.time()

    for i in range(args.n_episodes):
        seed = args.base_seed + i

        env_distilled = NASimEmuEnv(scenario_name=args.scenario, step_limit=args.step_limit,
                                     observation_format="graph_v2", seed=seed)
        stats_distilled = run_episode_distilled(net, env_distilled, args.step_limit)
        stats_distilled["episode"] = i
        episode_summaries.append(stats_distilled)

        env_live = NASimEmuEnv(scenario_name=args.scenario, step_limit=args.step_limit,
                                observation_format="graph_v2", seed=seed)
        stats_live = run_episode_live(net, env_live, args.step_limit, args.model, transcript_records, i)
        stats_live["episode"] = i
        episode_summaries.append(stats_live)

        print(f"[evaluate_llm_selector] episode {i}: "
              f"distilled(return={stats_distilled['episode_return']:.2f}, captured={stats_distilled['captured']}, len={stats_distilled['episode_len']})  "
              f"live(return={stats_live['episode_return']:.2f}, captured={stats_live['captured']}, len={stats_live['episode_len']})")

    with open(transcript_path, "w") as f:
        for rec in transcript_records:
            f.write(json.dumps(rec) + "\n")
    with open(summary_path, "w") as f:
        for rec in episode_summaries:
            f.write(json.dumps(rec) + "\n")

    elapsed = time.time() - t_start
    live_returns = [s["episode_return"] for s in episode_summaries if s["mode"] == "live"]
    distilled_returns = [s["episode_return"] for s in episode_summaries if s["mode"] == "distilled"]
    print(f"[evaluate_llm_selector] done in {elapsed/60:.1f} min. "
          f"mean live return={sum(live_returns)/len(live_returns):.3f}  "
          f"mean distilled return={sum(distilled_returns)/len(distilled_returns):.3f}")
    print(f"[evaluate_llm_selector] transcript: {transcript_path}")
    print(f"[evaluate_llm_selector] summary: {summary_path}")


if __name__ == "__main__":
    main()
