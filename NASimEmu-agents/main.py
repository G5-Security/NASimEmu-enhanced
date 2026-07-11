import numpy as np
import gym, torch, logging

import argparse, itertools, os, random
import json, time

# Default to offline wandb mode so runs never block on the interactive
# "Create a W&B account / Use existing / Don't visualize" prompt -- explicit
# `WANDB_MODE=...` in the environment still overrides this.
os.environ.setdefault("WANDB_MODE", "offline")

import wandb

from vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm

from config import config
from nasim_problem import NASimRRL as Problem
from llm_teacher.reward_shaping import compute_shaping_terms, apply_shaping_to_trace, anneal_lambda
from llm_teacher.dataset_reader import load_dataset
from llm_teacher.distillation_loss import goal_distillation_loss
from llm_teacher.goal_ontology import GOAL_NAMES

# ----------------------------------------------------------------------------------------
def _print_curriculum_status(env, current_epoch):
	"""Print current curriculum stage based on epoch.
	
	Fetches actual curriculum settings from the environment and displays them.
	"""
	try:
		# Get curriculum info from first environment in the vectorized wrapper
		info = env.env_method('get_curriculum_info', indices=[0])[0]
		if not info:
			return  # No curriculum active
		
		params = info.get('realism_params', {})
		
		print(f"\n{'='*80}")
		print(f"[CURRICULUM] Epoch {current_epoch} | Stage: {info['name']}")
		print(f"  Epoch Range: {info['start_epoch']}-{info['end_epoch']}")
		print(f"{'='*80}")
		
		# IDS
		ids_config = params.get('ids_config', {})
		if ids_config.get('enabled'):
			print(f"  IDS: Enabled (decay={ids_config.get('detection_decay', 0):.2f}, "
			      f"threshold={ids_config.get('base_thresholds', [])})")
		else:
			print(f"  IDS: Disabled")
		
		# Scan Noise
		scan_noise = params.get('scan_noise', {})
		if 'service_scan' in scan_noise:
			ss = scan_noise['service_scan']
			print(f"  Scan Noise: service FP={ss.get('false_positive_rate', 0):.3f}, "
			      f"FN={ss.get('false_negative_rate', 0):.3f}")
		else:
			print(f"  Scan Noise: Disabled")
		
		# Network Reliability
		net_rel = params.get('network_reliability', {})
		if 'timeout_probability' in net_rel:
			print(f"  Network Reliability: timeout={net_rel['timeout_probability']:.3f}")
		else:
			print(f"  Network Reliability: Perfect (no timeouts)")
		
		# Service Dynamics
		svc_dyn = params.get('service_dynamics', {})
		if 'churn_probability' in svc_dyn:
			print(f"  Service Dynamics: churn={svc_dyn['churn_probability']:.3f}")
		else:
			print(f"  Service Dynamics: Disabled")
		
		print(f"{'='*80}\n")
	except Exception as e:
		# Silently skip if curriculum not available
		pass

def _print_llm_explainability_brief(current_epoch, goal_hist, args, shaping_ratio=None, shaping_lambda=None,
                                     distill_lambda=None, distill_loss=None, distill_n=None):
	"""Printed to stdout every epoch (not gated to an interval like the
	curriculum banner) -- the user asked for this explicitly so LLM-related
	training signals are visible live in the terminal, the same way the
	existing per-epoch `log` dict already is."""
	total = int(goal_hist.sum())
	print(f"\n{'-'*80}")
	print(f"[LLM EXPLAINABILITY BRIEF] Epoch {current_epoch}")
	if total > 0:
		labeled = bool(args.llm_distill)
		tag = "" if labeled else " (unlabeled positional index -- see NOTE at startup)"
		print(f"  Manager subgoal selections this epoch{tag}:")
		for idx, name in enumerate(GOAL_NAMES):
			pct = 100.0 * goal_hist[idx] / total
			label = name if labeled else f"idx_{idx}"
			print(f"    {label:<24} {pct:5.1f}%  ({int(goal_hist[idx])})")
	else:
		print("  Manager subgoal selections this epoch: n/a (net_class != NASimNetDHRL)")

	if args.llm_shaping and shaping_ratio is not None:
		print(f"  llm_shaping   (reward-level):     lambda={shaping_lambda:.4f}  |F|/|r_env|={shaping_ratio:.3f}")
	if args.llm_distill:
		loss_str = f"{distill_loss:.4f}" if distill_loss is not None else "n/a (lambda_goal ~ 0 this window)"
		print(f"  llm_distill   (manager-level):     lambda_goal={distill_lambda:.4f}  mean L_goal={loss_str}  dataset={distill_n} records")
	print(f"{'-'*80}")


# ----------------------------------------------------------------------------------------
def decay_time(step, start, min, factor, rate):
	exp = step / rate * factor
	value = (start - min) / (1 + exp) + min

	return value

def decay_exp(step, start, min, factor, rate):
	exp = step / rate
	value = (start - min) * (factor ** exp) + min

	return value

def llm_distill_lambda_goal(step, args, config):
	"""Master plan Sec 15.9's four-stage schedule: stage 2/3 is the
	fixed-then-decaying joint weight (anneal_lambda alone), stage 4 is
	RL-only with the teacher weight forced to EXACTLY 0 -- anneal_lambda's
	plain exponential decay only asymptotically approaches 0 and never
	guarantees a true RL-only phase actually occurs within a bounded run.
	This forces it exactly once the configured final fraction of training
	is reached. Only takes effect when -max_epochs is set (an unbounded run
	has no "final fraction" to compute against, so it falls back to the
	plain decay for the whole run)."""
	if config.max_epochs:
		total_steps = config.max_epochs * config.log_rate
		rl_only_start_step = total_steps * (1.0 - args.llm_distill_rl_only_frac)
		if step >= rl_only_start_step:
			return 0.0
	return anneal_lambda(step, lambda_start=args.llm_distill_lambda_start, lambda_min=0.0, rate=args.llm_distill_lambda_rate)

def init_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def get_args(problem_config):
	cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'] + cuda_devices, default='cpu', help="Which device to use")
	parser.add_argument('-cpus', type=str, default='2', help="How many CPUs to use")
	parser.add_argument('-batch', type=int, default=128, help="Number of parallel environments")
	parser.add_argument('-seed', type=int, default=None, help="Random seed (each of the -batch parallel envs gets seed+i; see NASimEmuEnv.__init__)")
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('-save_best_split', choices=['eval_trn', 'eval_tst'], default='eval_tst', help="Which eval split's metric to track for the best-checkpoint save (falls back to eval_trn if no -test_scenario is configured)")
	parser.add_argument('-save_best_metric', choices=['reward_avg', 'reward_avg_episodes', 'eplen_avg', 'captured_avg'], default='captured_avg', help="Metric to track for the best-checkpoint save; higher is always better for all four (eplen_avg is included for completeness, not recommended as the optimization target)")
	parser.add_argument('-epoch', type=int, default=1000, help="Epoch length")
	parser.add_argument('-max_epochs', type=int, default=None, help="Terminate after this many epochs")

	parser.add_argument('-mp_iterations', type=int, default=3, help="Number of message passes")
	parser.add_argument('-emb_dim', type=int, default=64, help="Embedding size")

	parser.add_argument('-force_continue_epochs', type=int, default=0, help="Disable force continue after this epochs (0=disable immediately; -1=never disable)")

	parser.add_argument('-lr', type=float, default=3e-3, help="Initial learning rate")
	parser.add_argument('-alpha_h', type=float, default=0.3, help="Initial entropy regularization constant")
	parser.add_argument('-max_norm', type=float, default=3., help="Maximal gradient norm")

	# Learning rate / entropy decay schedules
	parser.add_argument('--sched_lr_rate', type=int, default=None, help="Steps between LR decay updates")
	parser.add_argument('--sched_lr_factor', type=float, default=None, help="Exponential LR decay factor")
	parser.add_argument('--sched_lr_min', type=float, default=None, help="Minimum learning rate")
	parser.add_argument('--sched_alpha_h_rate', type=int, default=None, help="Steps between entropy coeff. decay updates")
	parser.add_argument('--sched_alpha_h_factor', type=float, default=None, help="Time-decay factor for entropy coeff.")
	parser.add_argument('--sched_alpha_h_min', type=float, default=None, help="Minimum entropy coefficient")

	parser.add_argument('--trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('--eval', action='store_const', const=True, help="Evaluate the agent")
	parser.add_argument('--debug', action='store_const', const=True, help="Debug the agent")
	parser.add_argument('--calc_baseline', action='store_const', const=True, help="Calculate required steps of a baseline agent")
	
	parser.add_argument('--no_debug', action='store_const', const=True, help="Do not debug the agent")

	# LLM-augmented Pen-DHRL (docs/llm_integration_plan.pdf, condition C2)
	parser.add_argument('--llm_shaping', action='store_const', const=True, help="Enable LLM-authored potential-based reward shaping over the active semantic subgoal (condition C2). Requires -net_class NASimNetDHRL.")
	parser.add_argument('--llm_shaping_lambda_rate', type=float, default=8000.0, help="Exponential anneal rate (in global training steps) for the shaping weight lambda_t")

	# LLM teacher-distillation (master plan Ch.15 Lgoal term; see llm_teacher/teacher_client.py
	# for the local free-model teacher and llm_teacher/label_states.py to build the dataset first)
	parser.add_argument('--llm_distill', action='store_const', const=True, help="Enable confidence-weighted cross-entropy distillation of subgoal_head against a precomputed local-LLM-teacher dataset (master plan Sec 15.9, Lgoal term only -- no LKD). Requires -net_class NASimNetDHRL and a dataset already built via 'python -m llm_teacher.label_states'.")
	parser.add_argument('--llm_distill_dataset', type=str, default=os.path.join('training_data', 'llm_teacher_dataset'), help="Directory containing dataset.jsonl + dataset_states.pkl written by llm_teacher/label_states.py")
	parser.add_argument('--llm_distill_split', choices=['train', 'test'], default=None, help="Train only against this split (see llm_teacher/split_dataset.py, run beforehand). Default: use every valid record, unsplit -- pass 'train' once a real held-out evaluation is wanted.")
	parser.add_argument('--llm_distill_shuffle_labels', action='store_const', const=True, help="Random-label control (master plan Ch.16 condition 5): train against the SAME dataset states but with teacher_goal_idx randomly permuted, isolating whether any observed gain comes from real teacher knowledge or just the regularizing effect of an extra auxiliary loss.")
	parser.add_argument('--llm_distill_warmup_epochs', type=int, default=3, help="Stage-1 supervised warm-start epochs over the full teacher dataset before PPO training starts (lambda_goal=1, L_RL=0)")
	parser.add_argument('--llm_distill_lambda_start', type=float, default=1.0, help="Stage-2/3 initial joint-training weight on L_goal, annealed toward 0 by --llm_distill_lambda_rate (stage 4 = RL-only once it reaches ~0)")
	parser.add_argument('--llm_distill_lambda_rate', type=float, default=8000.0, help="Exponential anneal rate (in global training steps) for the joint-training L_goal weight")
	parser.add_argument('--llm_distill_rl_only_frac', type=float, default=0.1, help="Final fraction of -max_epochs run with the teacher weight forced to exactly 0 (stage 4 of master plan Sec 15.9's schedule -- see llm_distill_lambda_goal()). No effect if -max_epochs is unset.")
	parser.add_argument('--llm_distill_batch', type=int, default=16, help="Minibatch size sampled from the teacher dataset for each joint-training distillation step")

	# delegate argparse to problem config
	problem_config.update_argparse(parser)

	cmd_args = parser.parse_args()
	return cmd_args

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
	# logging.basicConfig(level=logging.INFO)
	logging.basicConfig(level=logging.DEBUG)
	logging.getLogger('urllib3').setLevel(logging.INFO)
	logging.getLogger('numba').setLevel(logging.INFO)

	problem = Problem()
	problem_config = problem.make_config()

	np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

	args = get_args(problem_config)

	# --llm_shaping reads raw_a[2] (the active semantic subgoal) out of the
	# forward-pass's raw_actions tuple -- this field only exists on
	# NASimNetDHRL (nasim_net_base_hrl.py). Every other architecture's
	# raw_actions is shaped differently (e.g. NASimNetGNN: 2-tuple with no
	# subgoal; NASimNetMLP: a single tensor, not a tuple at all) and would
	# fail deep inside reward_shaping.py with a confusing IndexError rather
	# than a clear message -- so fail fast, here, before any training starts.
	if args.llm_shaping and args.net_class != 'NASimNetDHRL':
		raise SystemExit(
			f"--llm_shaping requires -net_class NASimNetDHRL (got '{args.net_class}'). "
			f"Other architectures' raw_actions have no active-subgoal field for the "
			f"potential-based shaping in llm_teacher/reward_shaping.py to read."
		)

	# --llm_distill reads subgoal_logits via only_subgoal_logits=True, which only
	# NASimNetDHRL.forward() implements -- and it needs a dataset built ahead of
	# time by llm_teacher/label_states.py (Sec 15.10: the teacher must never be
	# queried live from inside the training loop).
	llm_distill_dataset_dir = args.llm_distill_dataset
	if not os.path.isabs(llm_distill_dataset_dir):
		llm_distill_dataset_dir = os.path.join(os.path.dirname(__file__), llm_distill_dataset_dir)
	if args.llm_distill:
		if args.net_class != 'NASimNetDHRL':
			raise SystemExit(
				f"--llm_distill requires -net_class NASimNetDHRL (got '{args.net_class}'). "
				f"Only NASimNetDHRL exposes subgoal_logits via only_subgoal_logits=True."
			)
		if not os.path.exists(os.path.join(llm_distill_dataset_dir, 'dataset.jsonl')):
			raise SystemExit(
				f"--llm_distill requires a dataset at {llm_distill_dataset_dir} "
				f"(dataset.jsonl + dataset_states.pkl). Build one first, e.g.:\n"
				f"  python -m llm_teacher.label_states --target_records 300 --out_dir {llm_distill_dataset_dir}"
			)

	config.init(args)
	problem_config.update_config(config, args) # update config with problem specific settings

	print(f"Config: {config}")

	if config.seed:
		init_seed(config.seed)

	torch.set_num_threads(config.cpus)	

	problem.register_gym()
	problem_debug = problem.make_debug()
	
	if args.calc_baseline:
		problem_debug.calc_baseline()
		exit(0)

	net = problem.make_net()
	target_net = problem.make_net()
	print(net)
	print(f"Number of parameters: {net.get_param_count()}")

	if config.load_model:
		net.load(config.load_model)
		target_net.load(config.load_model)

		print(f"Model loaded: {config.load_model}")

	if args.trace:
		problem_debug.trace(net, config.load_model)
		exit(0)

	if args.eval:
		import pprint
		eval_res = problem_debug.evaluate(net)
		# print(f"Avg. reward: {r_avg}, Avg. solved per step: {s_ps_avg}, Avg. solved: {s_avg}, Tot. finished: {s_tot}")
		pprint.pp(eval_res)
		exit(0)

	if args.debug:
		problem_debug.debug(net, show=True)
		exit(0)

	# LLM-distillation Stage 1/4 (master plan Sec 15.9): supervised warm-start,
	# lambda_goal=1, L_RL=0 -- pure classification training of subgoal_head
	# against the precomputed teacher dataset, entirely before PPO starts.
	# Stages 2-4 (joint PPO at fixed then decaying lambda_goal, then RL-only)
	# run per training step inside the main loop below.
	llm_distill_records = None
	if args.llm_distill:
		llm_distill_records = load_dataset(llm_distill_dataset_dir, split=args.llm_distill_split)
		split_desc = f"split={args.llm_distill_split}" if args.llm_distill_split else "unsplit"
		print(f"[llm_distill] loaded {len(llm_distill_records)} valid teacher-labeled records ({split_desc}) from {llm_distill_dataset_dir}")
		if len(llm_distill_records) == 0:
			raise SystemExit(f"--llm_distill dataset at {llm_distill_dataset_dir} has 0 valid records ({split_desc}).")

		if args.llm_distill_shuffle_labels:
			# Ch.16 condition 5: same states, same confidences, but goal_idx
			# permuted across records -- shuffling in place rather than
			# resampling uniformly at random preserves the real dataset's
			# class-frequency distribution, isolating label *correctness* as
			# the only thing that changed.
			goal_idx_pool = [r["goal_idx"] for r in llm_distill_records]
			goal_name_pool = [r["goal_name"] for r in llm_distill_records]
			perm = np.random.permutation(len(llm_distill_records))
			for rec, p in zip(llm_distill_records, perm):
				rec["goal_idx"] = goal_idx_pool[p]
				rec["goal_name"] = goal_name_pool[p]
			print(f"[llm_distill] --llm_distill_shuffle_labels ENABLED (condition C5, random-label control) -- "
			      f"goal labels permuted across all {len(llm_distill_records)} records")

		print(f"[llm_distill] Stage 1/4: supervised warm-start, {args.llm_distill_warmup_epochs} epoch(s) over {len(llm_distill_records)} records")
		net.train()
		for warmup_epoch in range(args.llm_distill_warmup_epochs):
			perm = np.random.permutation(len(llm_distill_records))
			epoch_losses = []
			for i in range(0, len(perm), args.llm_distill_batch):
				batch_idx = perm[i:i + args.llm_distill_batch]
				batch_records = [llm_distill_records[j] for j in batch_idx]
				s_batch = [r['s_true'] for r in batch_records]
				goal_idx_t = torch.tensor([r['goal_idx'] for r in batch_records], dtype=torch.long, device=config.device)
				conf_t = torch.tensor([r['confidence'] for r in batch_records], dtype=torch.float32, device=config.device)

				subgoal_logits = net(s_batch, only_subgoal_logits=True, reset_hidden=True)
				loss_goal = goal_distillation_loss(subgoal_logits, goal_idx_t, conf_t)

				net.opt.zero_grad()
				loss_goal.backward()
				torch.nn.utils.clip_grad_norm_(net.parameters(), config.opt_max_norm)
				net.opt.step()
				epoch_losses.append(loss_goal.item())
			print(f"[llm_distill] warm-start epoch {warmup_epoch + 1}/{args.llm_distill_warmup_epochs}: mean L_goal={np.mean(epoch_losses):.4f}")

		# hard-sync target_net to the warm-started weights (rho=1.0 -> full copy,
		# see Net.copy_weights: val_new = rho*other + (1-rho)*self) before PPO begins
		target_net.copy_weights(net, rho=1.0)
		print("[llm_distill] Stage 1/4 complete -- entering Stage 2/4 (joint PPO + fixed-then-decaying lambda_goal) inside the main training loop")

	# Each parallel env gets its own deterministic-but-distinct seed derived from
	# config.seed (only when the user actually requested one) -- `i=i` binds the
	# loop variable at lambda-creation time, since these lambdas aren't called
	# until later inside the forked worker processes (late-binding closure).
	# See NASimEmuEnv.__init__ (env.py) for why this is required for -seed to
	# actually reproduce the same per-env scenario sequence across runs, instead
	# of every worker silently reseeding itself from fresh OS entropy.
	env = SubprocVecEnv([
		lambda i=i: gym.make(problem.get_gym_name(), seed=(config.seed + i) if config.seed else None)
		for i in range(config.batch)
	], in_series=(config.batch // config.cpus), context='fork')

	wandb.init(project=problem.get_project_name(), name=problem.get_run_name(), config=config)
	wandb.watch(net, log='all')

	# ---------------------------
	# Local per-episode JSON logger
	# ---------------------------
	log_dir = os.path.join(os.path.dirname(__file__), 'training_data', 'runs')
	os.makedirs(log_dir, exist_ok=True)
	jsonl_path = os.path.join(log_dir, f'{wandb.run.id}.json')

	# "latest" convenience mirror: same records, always at a fixed path, so you
	# don't need to look up the wandb run id to tail the current run. Truncated
	# fresh at the start of THIS run -- the per-run file above (keyed by run id)
	# remains the collision-safe source of truth if multiple runs ever overlap.
	latest_dir = os.path.join(os.path.dirname(__file__), 'training_data', 'latest')
	os.makedirs(latest_dir, exist_ok=True)
	latest_path = os.path.join(latest_dir, 'latest.json')
	open(latest_path, 'w').close()

	def _append_jsonl(path, obj):
		try:
			with open(path, 'a') as f:
				f.write(json.dumps(obj) + "\n")
		except Exception as e:
			logging.getLogger(__name__).warning(f"Failed to write JSON log: {e}")

	tot_env_steps = 0
	best_val = float('-inf')
	norm_log = []
	entropy_log = []
	shaping_log = []
	env_r_log = []
	goal_hist_log = np.zeros(len(GOAL_NAMES), dtype=np.int64)
	llm_distill_loss_log = []

	if args.llm_shaping:
		print(f"[llm_shaping] ENABLED (condition C2) -- lambda anneal rate = {args.llm_shaping_lambda_rate} steps. Requires -net_class NASimNetDHRL.")

	if args.llm_distill:
		print(f"[llm_distill] ENABLED -- subgoal_head distilled against {len(llm_distill_records)} teacher records "
		      f"(lambda_start={args.llm_distill_lambda_start}, anneal_rate={args.llm_distill_lambda_rate}).")
		print("[llm_distill] NOTE: the per-epoch goal-selection histogram below is only a valid GOAL_NAMES "
		      "labeling because --llm_distill is active and training subgoal_head toward that binding. "
		      "Without --llm_distill, the same histogram would just be positional index counts (master plan Sec 15.3).")

	if config.force_continue_steps == 0:
		print("Disabling force_continue")
		net.set_force_continue(False)
	else:
		print("Enabling force_continue")
		net.set_force_continue(True)

	tqdm_main = tqdm(desc='Training', unit=' steps')
	s = env.reset()
	
	# Track total episodes completed across all parallel envs for curriculum
	total_episodes_completed = 0
	
	# Get curriculum stage transition epochs dynamically from scenario
	try:
		curriculum_transition_epochs = env.env_method('get_stage_transition_epochs', indices=[0])[0]
	except:
		curriculum_transition_epochs = []

	for step in itertools.count(start=1):
		trace = []

		hidden_s0 = problem.make_net()		# save internal (recurrent) network state at s_0 and s_last
		hidden_s0.clone_state(net)

		for step_trace in range(config.ppo_t):
			s_orig = s
			
			a, v, pi, raw_a = net(s)
			a = np.array(a, dtype=object)
			s, r, d, i = env.step(a)
			net.reset_state(d)

			# Per-epoch explainability brief data (raw_a[2] = selected_subgoals,
			# only present on NASimNetDHRL's 3-tuple raw_actions -- see forward()).
			if args.net_class == 'NASimNetDHRL' and isinstance(raw_a, tuple) and len(raw_a) >= 3:
				subgoal_counts = torch.bincount(raw_a[2].flatten().cpu(), minlength=len(GOAL_NAMES)).numpy()
				goal_hist_log[:len(subgoal_counts)] += subgoal_counts[:len(GOAL_NAMES)]

			a_cnt = [0 if a_action == -1 else 1 for (a_node, a_action) in a] # action_q - 0 = terminate / 1 = continue

			s_true = [x['s_true'] for x in i]
			d_true = [x['d_true'] for x in i] # note: currently d == d_true (dependency in v_target, q_target computations and reccurency in ppo

			# Track episode completions for monitoring
			episodes_done_this_step = sum(d_true)
			total_episodes_completed += episodes_done_this_step

			trace.append( (s_orig, raw_a, a_cnt, r, s_true, d_true) )

		# LLM-augmented reward shaping (condition C2, docs/llm_integration_plan.pdf Sec 4/5.4)
		if args.llm_shaping:
			lambda_t = anneal_lambda(step, rate=args.llm_shaping_lambda_rate)
			shaping = compute_shaping_terms(trace, net, config.gamma, lambda_t, config.device)
			env_r_mean_abs = float(np.mean([np.mean(np.abs(np.asarray(t[3]))) for t in trace]))
			shaping_mean_abs = float(np.mean([np.mean(np.abs(f)) for f in shaping]))
			shaping_log.append(shaping_mean_abs)
			env_r_log.append(env_r_mean_abs)
			trace = apply_shaping_to_trace(trace, shaping)

		# update network
		# loss, entropy, norm, pi_deviations = net.update(s_orig, raw_a, r, s_true, d_true, target_net)
		target_net.clone_state(net)
		loss, entropy, norm, pi_deviations = net.update(trace, target_net, hidden_s0)
		target_net.copy_weights(net, rho=config.target_rho)

		# LLM-distillation Stage 2/3/4 (master plan Sec 15.9): a joint-training
		# step at fixed-then-decaying lambda_goal, sampled each outer PPO step
		# from the precomputed teacher dataset. Implemented as a SEPARATE
		# gradient step after the PPO update above rather than fused into a
		# single Ltotal backward -- fusing would require threading an auxiliary
		# loss through rl.py's shared ppo() update function used by every net
		# class, which this one-run scope doesn't extend to. This alternating
		# scheme optimizes the same two objectives, just not in one backward
		# pass; documented here rather than silently simplified.
		lambda_goal_t = 0.0
		if args.llm_distill:
			lambda_goal_t = llm_distill_lambda_goal(step, args, config)
			if lambda_goal_t > 1e-6:
				batch_n = min(args.llm_distill_batch, len(llm_distill_records))
				batch_idx = np.random.choice(len(llm_distill_records), size=batch_n, replace=False)
				batch_records = [llm_distill_records[j] for j in batch_idx]
				s_batch = [r['s_true'] for r in batch_records]
				goal_idx_t = torch.tensor([r['goal_idx'] for r in batch_records], dtype=torch.long, device=config.device)
				conf_t = torch.tensor([r['confidence'] for r in batch_records], dtype=torch.float32, device=config.device)

				subgoal_logits = net(s_batch, only_subgoal_logits=True, reset_hidden=True)
				loss_goal = lambda_goal_t * goal_distillation_loss(subgoal_logits, goal_idx_t, conf_t)

				net.opt.zero_grad()
				loss_goal.backward()
				torch.nn.utils.clip_grad_norm_(net.parameters(), config.opt_max_norm)
				net.opt.step()
				llm_distill_loss_log.append(loss_goal.item() / max(lambda_goal_t, 1e-6))

		# print([x.item() for x in pi_deviations])

		# save step stats
		tot_env_steps += config.batch
		tqdm_main.update()

		norm_log.append(norm)
		entropy_log.append(entropy)

		if step % config.sched_lr_rate == 0:
			lr = decay_exp(step, config.opt_lr, config.sched_lr_min, config.sched_lr_factor, config.sched_lr_rate)
			net.set_lr(lr)

		if step % config.sched_alpha_h_rate == 0:
			alpha_h = decay_time(step, config.alpha_h, config.sched_alpha_h_min, config.sched_alpha_h_factor, config.sched_alpha_h_rate)
			net.set_alpha_h(alpha_h)

		if step % config.log_rate == 0:
			log_step = step // config.log_rate
			current_epoch = log_step
			
			# Update curriculum based on current epoch
			env.env_method('set_epoch', current_epoch)

			# r_avg, s_ps_avg, s_avg, _ = problem_debug.evaluate(net)
			# r_avg_trn, s_ps_avg_trn, s_avg_trn, _ = problem_debug.evaluate(net, split='train', subset=config.subset)

			eval_perf = problem_debug.evaluate(net)
			# log_trn_eval = problem_debug.evaluate(net, split='train', subset=config.subset)

			if args.no_debug:
				log_debug = None
			else:
				log_debug = problem_debug.debug(net)
				# print(log_debug['value'], log_debug['q_val'])
		
			log = {
				'env_steps': tot_env_steps,
				'episodes_completed': total_episodes_completed,  # Track curriculum progress
				# 'el_env_steps': tot_el_env_steps,
				'rate': tqdm_main.format_dict['rate'],

				'loss': loss,
				# 'loss_pi': loss_pi,
				# 'loss_v': loss_v,
				# 'loss_h': loss_h,

				'pi_deviations': wandb.Histogram(pi_deviations),

				'grad_mean': np.mean(norm_log),
				'grad_min': np.min(norm_log),
				'grad_max': np.max(norm_log),

				'entropy_mean': np.mean(entropy_log),
				'entropy_min': np.min(entropy_log),
				'entropy_max': np.max(entropy_log),

				'lr': net.lr,
				'alpha_h': net.alpha_h,

				'eval_perf': eval_perf,

				'debug': log_debug,
			}

			# LLM-shaping diagnostics (risk register, docs/llm_integration_plan.pdf):
			# log |F_t| vs |r_env_t| so a "no measurable gradient change" wash outcome
			# is directly checkable rather than inferred from final performance alone.
			if args.llm_shaping and shaping_log:
				log['llm_shaping_mean_abs'] = float(np.mean(shaping_log))
				log['llm_shaping_env_r_mean_abs'] = float(np.mean(env_r_log))
				log['llm_shaping_lambda'] = anneal_lambda(step, rate=args.llm_shaping_lambda_rate)
				shaping_log = []
				env_r_log = []

			distill_lambda_cur = llm_distill_lambda_goal(step, args, config) if args.llm_distill else None
			distill_loss_cur = float(np.mean(llm_distill_loss_log)) if (args.llm_distill and llm_distill_loss_log) else None
			if args.llm_distill:
				log['llm_distill_lambda'] = distill_lambda_cur
				if distill_loss_cur is not None:
					log['llm_distill_loss'] = distill_loss_cur

			shaping_ratio_cur = None
			shaping_lambda_cur = log.get('llm_shaping_lambda')
			if args.llm_shaping and 'llm_shaping_mean_abs' in log:
				shaping_ratio_cur = log['llm_shaping_mean_abs'] / max(log['llm_shaping_env_r_mean_abs'], 1e-9)

			_print_llm_explainability_brief(
				current_epoch, goal_hist_log, args,
				shaping_ratio=shaping_ratio_cur, shaping_lambda=shaping_lambda_cur,
				distill_lambda=distill_lambda_cur, distill_loss=distill_loss_cur,
				distill_n=(len(llm_distill_records) if args.llm_distill else None),
			)
			# Snapshot before clearing -- log_json (built further below, after
			# this reset) needs the per-epoch counts too; reading goal_hist_log
			# there directly used to always see the zeroed array, so goal_hist
			# was silently never persisted to the JSONL log.
			goal_hist_snapshot = goal_hist_log.copy()
			goal_hist_log = np.zeros(len(GOAL_NAMES), dtype=np.int64)
			llm_distill_loss_log = []

			# Print curriculum status at meaningful intervals
			# Print at: early epochs (0-5), every 10 epochs, and at stage transitions from scenario
			should_print = (
				current_epoch <= 5 or  # Early training
				current_epoch % 10 == 0 or  # Every 10 epochs
				current_epoch in curriculum_transition_epochs  # Stage transitions from scenario
			)
			if should_print:
				_print_curriculum_status(env, current_epoch)

			norm_log = []
			entropy_log = []

			print(log)
			wandb.log(log)

			# Write one JSON record per logging interval (epoch-like)
			def _to_serializable(x):
				try:
					import numpy as _np
					if isinstance(x, (_np.floating, _np.integer)):
						return x.item()
				except Exception:
					pass
				return x

			log_json = {
				'timestamp': time.time(),
				'train_step': int(step),
				'env_steps_total': int(tot_env_steps),
				'loss': float(_to_serializable(log['loss'])),
				'grad_mean': float(_to_serializable(log['grad_mean'])),
				'grad_min': float(_to_serializable(log['grad_min'])),
				'grad_max': float(_to_serializable(log['grad_max'])),
				'entropy_mean': float(_to_serializable(log['entropy_mean'])),
				'entropy_min': float(_to_serializable(log['entropy_min'])),
				'entropy_max': float(_to_serializable(log['entropy_max'])),
				'lr': float(_to_serializable(log['lr'])),
				'alpha_h': float(_to_serializable(log['alpha_h'])),
				'eval_trn': {k: _to_serializable(v) for k, v in (log['eval_perf'].get('eval_trn') or {}).items()},
				'eval_tst': {k: _to_serializable(v) for k, v in (log['eval_perf'].get('eval_tst') or {}).items()},
			}
			if args.llm_shaping and 'llm_shaping_mean_abs' in log:
				log_json['llm_shaping_mean_abs'] = float(log['llm_shaping_mean_abs'])
				log_json['llm_shaping_env_r_mean_abs'] = float(log['llm_shaping_env_r_mean_abs'])
				log_json['llm_shaping_lambda'] = float(log['llm_shaping_lambda'])
			if args.llm_distill:
				log_json['llm_distill_lambda'] = float(distill_lambda_cur)
				if distill_loss_cur is not None:
					log_json['llm_distill_loss'] = float(distill_loss_cur)
			if args.net_class == 'NASimNetDHRL' and int(goal_hist_snapshot.sum()) > 0:
				log_json['goal_hist'] = {name: int(c) for name, c in zip(GOAL_NAMES, goal_hist_snapshot.tolist())}
				log_json['goal_hist_labeled'] = bool(args.llm_distill)
			_append_jsonl(jsonl_path, log_json)
			_append_jsonl(latest_path, log_json)
			
			# save model to wandb
			model_file = os.path.join(wandb.run.dir, "model.pt")
			os.makedirs(os.path.dirname(model_file), exist_ok=True)
			net.save(model_file)
			wandb.save(model_file)
		

			# if per-epoch auto mode, roll scenario for next epoch
			try:
				if getattr(config, 'auto_mode', 'off') == 'per_epoch':
					# broadcast to all workers
					env.env_method('set_roll_on_next_reset', True)
			except Exception as _:
				pass
 
			# save best checkpoint if improved -- tracks config.save_best_split/
			# save_best_metric (-save_best_split/-save_best_metric, default
			# eval_tst/captured_avg) across the whole run and, when it
			# improves, saves a SEPARATE model_best.pt alongside the regular
			# every-epoch model.pt above (which always reflects the latest
			# epoch, not the best one -- a run that degrades late, e.g. after
			# the teacher weight has fully decayed to 0 under --llm_distill,
			# would otherwise end on a worse checkpoint than one seen mid-run).
			split = config.save_best_split
			metric_name = config.save_best_metric
			split_perf = eval_perf.get(split) or eval_perf.get('eval_trn')
			cur_val = split_perf.get(metric_name) if split_perf else None

			if cur_val is not None and cur_val > best_val:
				best_val = cur_val
				best_model_file = os.path.join(wandb.run.dir, "model_best.pt")
				net.save(best_model_file)
				wandb.save(best_model_file)
				print(f"[save_best] new best {split}/{metric_name}={cur_val:.4f} at epoch {current_epoch} -> {best_model_file}")

		# finish if max_epochs exceeded
		if config.max_epochs and (step // config.log_rate >= config.max_epochs):
			break

		if step == config.force_continue_steps:
			print("Disabling force_continue")
			net.set_force_continue(False)

	env.close()
	tqdm_main.close()
