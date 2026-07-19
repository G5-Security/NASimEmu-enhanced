# Verified Run Commands: Baseline Pen-DHRL and LLM-Augmented Pen-DHRL

This document is for the current workstation checkout at
`/home/user5/NASimEmu-enhanced`. Every Python command below uses the project
virtual environment, CPU-only training, one BLAS thread per worker, and the
workstation's verified-stable E-core set.

For the diagnosis, repository changes, and remaining hardware caveats behind
this launch profile, see [runtime_fixes.md](runtime_fixes.md).

> Current status (2026-07-15): the baseline environment is certified by the
> complete 10-epoch run and the short-run matrix in this document. Real Ollama
> teacher calls are **not** certified on this workstation because `ollama` is
> not installed. The heuristic teacher, distillation, selector fallback, and
> mocked LLM logic are tested.

## 1. Required shell setup

Run this once in every new shell before using any later command block. Later
commands call `taskset` and the tested virtual-environment Python directly, so
they do not depend on a shell helper function.

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/nasimemu-matplotlib
export WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache
export WANDB_MODE=offline
export WANDB_DIR="$PWD"

mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR"

PYTHON=../venv/bin/python
CPUSET=16-31

test -x "$PYTHON"
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" --version
```

Expected interpreter: Python 3.10.20.

Why this launcher is mandatory on this workstation:

- Bare `python` is unavailable; `/usr/bin/python3` is Python 3.12 and does not
  contain the repository dependencies.
- `-device cpu` is intentional. CUDA wheels are installed, but
  `torch.cuda.is_available()` is false and this workload is CPU-bound.
- BLAS thread counts must be set explicitly before NumPy/PyTorch imports.
  Training already parallelizes with worker processes, so nested BLAS pools
  only oversubscribe the machine.
- `taskset -c 16-31` keeps this process and all child workers on the E-cores
  that passed the local stress tests. Isolated Python 3.10 and 3.12 workloads
  reproduced memory corruption on this workstation's P-core group. This CPU
  set is host-specific; do not copy it to the old i7-1255U laptop.
- `WANDB_MODE=offline` creates persistent, unique
  `wandb/offline-run-*/files/` checkpoint directories without uploading.
  Do not use `WANDB_MODE=disabled` with commands that discover a W&B run or
  with `dagger_loop.py`; disabled mode saves checkpoints in the temporary
  directory instead.

### Fast dependency and test gate

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -c "import torch, torch_geometric, torch_scatter, gym, yaml, wandb, nasimemu; print('dependencies: OK; torch', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'training device: cpu')"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m pytest -q tests
```

Run only `NASimEmu-agents/tests`. A repository-root pytest collection is not
the health gate: legacy/manual tests outside this directory require optional
MPI, Atari, and `baselines` packages that are not part of this environment.

## 2. Shared runtime rules

- Run every block from `NASimEmu-agents/` after Section 1.
- Use `-cpus 8` on this workstation. It is the worker count used by the
  successful 10-epoch job.
- `-batch` must be divisible by `-cpus`. The documented `16` and `128` values
  are valid.
- `-cpus` must also divide the fixed evaluation batch of 64. Safe values are
  1, 2, 4, 8, 16, 32, and 64; use 8 here.
- `-epoch` is the number of PPO update steps per logging/evaluation interval.
  `-max_epochs` is the number of those intervals. Total PPO updates are
  `epoch * max_epochs`.
- Never use `-max_epochs 0` as a smoke run. Zero is treated as unbounded.
- Only one training-mode `main.py` invocation is allowed at a time. The
  trainer enforces this with `training_data/.training.lock`; `--eval` remains
  available while training is active. Do not delete the lock file to bypass
  the guard: the file may remain after exit, but only a live kernel lock can
  block a new trainer.
- A meaningful baseline smoke uses `-epoch 1 -max_epochs 1`. A meaningful
  distillation smoke uses `-epoch 2 -max_epochs 1` so step 1 exercises the
  joint teacher loss before the final RL-only step.

Immediately after any offline training run, capture and validate its exact
checkpoint instead of searching for the newest directory with `ls | grep`:

```bash
TRAIN_RUN_DIR=$(readlink -f wandb/latest-run)
CHECKPOINT="$TRAIN_RUN_DIR/files/model_best.pt"
test -s "$CHECKPOINT"
printf 'checkpoint: %s\n' "$CHECKPOINT"
```

Do this before starting another run, because `wandb/latest-run` will then
point to the newer run.

## 3. Verified environment record

The following was observed on this workstation on 2026-07-15:

| Check | Result |
| --- | --- |
| Dependency import preflight | PASS |
| Python command-line entry points | 7/7 PASS |
| Supported pytest suite | 88 passed, 1 skipped |
| W&B offline initialization/checkpoint directory | PASS |
| One-epoch single-scenario baseline smoke | PASS |
| One-epoch three-scenario/scheduler smoke | PASS |
| Baseline checkpoint evaluations | 2/2 PASS |
| Heuristic dataset collect/audit/split | PASS |
| Random, untrained-net, and checkpoint collection policies | 3/3 PASS |
| C3 distillation warm-up, joint loss, RL-only step, eval, save | PASS |
| C5 shuffled-label control | PASS |
| Selector one-episode wiring/fallback without Ollama | PASS |
| DAgger two-round dry run | PASS |
| `compare_runs.py` against the long run | PASS |
| Real Ollama teacher and valid live-selector responses | BLOCKED: Ollama absent |

The completed long baseline run used the Section 1 CPU/thread restrictions
with the A1 model configuration:

- Exit code 0 after all 10 epochs, approximately 6 minutes 11 seconds.
- 100 PPO training steps, 1,600 logged environment steps, 64 completed
  episodes.
- Epoch-10 `eval_tst/captured_avg`: 27.328125.
- Best `eval_tst/captured_avg`: 28.109375 at epoch 9.
- No NaN, worker failure, heap error, segmentation fault, or new crash file.
- Persistent metrics: `training_data/runs/3afwaqbw.json`.
- That validation used W&B disabled because the Codex sandbox blocks local
  service sockets; its checkpoints are `/tmp/model.pt` and
  `/tmp/model_best.pt`. They are temporary and disappear after cleanup or a
  reboot. Normal shell runs should keep the offline setup from Section 1.

The repeatable artifact assertion is in Section 9.

---

## 4. Part A — Baseline Pen-DHRL (no LLM)

### A1. Verified 10-epoch baseline

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 10 -max_epochs 10 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02

A1_RUN_DIR=$(readlink -f wandb/latest-run)
A1_CHECKPOINT="$A1_RUN_DIR/files/model_best.pt"
test -s "$A1_CHECKPOINT"
```

This performs 100 PPO updates and matches the completed exploratory run,
which did not use a fixed seed. Add `-seed 1` when a deterministic rerun is
more important than matching that exact invocation.

### A2. Large three-scenario baseline (condition C0)

This is the final-scale baseline. It has not been run to 200 epochs during
this validation session; run the A2 smoke in Section 8 before committing the
full budget.

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 128 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetDHRL \
  -force_continue_epochs 0 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1

A2_RUN_DIR=$(readlink -f wandb/latest-run)
A2_CHECKPOINT="$A2_RUN_DIR/files/model_best.pt"
test -s "$A2_CHECKPOINT"
```

This is C0 because it contains no `--llm_*` flag. It performs 20,000 PPO
updates. No workstation runtime estimate is claimed until a same-shape
calibration run has been measured.

### A3. Evaluation

Evaluation must use the scenario, network, observation format, action
termination flag, and episode limit associated with its checkpoint.

Evaluate A1:

```bash
test -s "$A1_CHECKPOINT"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval \
  -load_model "$A1_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2
```

Evaluate A2:

```bash
test -s "$A2_CHECKPOINT"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval \
  -load_model "$A2_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2
```

Evaluate the already-proven 10-epoch checkpoint while it still exists:

```bash
test -s /tmp/model_best.pt

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval \
  -load_model /tmp/model_best.pt \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2
```

---

## 5. Part B — LLM-Augmented Pen-DHRL prerequisites

### B0. Ollama gate

Ollama is currently absent on this workstation. Install it and pull the
teacher model before treating any `--teacher_backend llm` or valid live
selector result as tested.

After installation, verify the executable and model service:

```bash
command -v ollama
ollama --version

# Run only when no Ollama service is already listening.
ollama serve > /tmp/nasimemu-ollama.log 2>&1 &
OLLAMA_PID=$!

ollama pull qwen2.5:3b-instruct
curl -fsS http://127.0.0.1:11434/api/tags
ollama list
```

The health request must succeed and `ollama list` must include
`qwen2.5:3b-instruct`. Pulling a model requires network access and local disk
space. Training itself never calls Ollama; only offline dataset collection
and the live selector do.

### B1. Small real-LLM dataset and C3 training

The old claim that a 300-record dataset already exists is no longer true.
Start by creating it explicitly. `--target_records` is an absolute total,
not "records to add".

```bash
LLM_DATASET=training_data/llm_teacher_dataset

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --policy random \
  --teacher_backend llm \
  --target_records 300 \
  --out_dir "$LLM_DATASET"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --policy untrained_net \
  --teacher_backend llm \
  --target_records 360 \
  --out_dir "$LLM_DATASET"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.audit_dataset --out_dir "$LLM_DATASET"
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.split_dataset --out_dir "$LLM_DATASET" --test_frac 0.2 --seed 0
```

The second collection raises the dataset's absolute target from 300 to 360,
so it adds approximately 60 records only when the first command reached
exactly 300.

Train C3 at the A1 scale:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 10 -max_epochs 10 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  -seed 1 \
  --llm_distill \
  --llm_distill_dataset "$LLM_DATASET" \
  --llm_distill_split train \
  --llm_distill_warmup_epochs 2

C3_SMALL_RUN_DIR=$(readlink -f wandb/latest-run)
C3_SMALL_CHECKPOINT="$C3_SMALL_RUN_DIR/files/model_best.pt"
test -s "$C3_SMALL_CHECKPOINT"
```

Standard checkpoint evaluation:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval \
  -load_model "$C3_SMALL_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2
```

Live-versus-distilled selector evaluation requires the Ollama gate to pass:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" experiments/evaluate_llm_selector.py \
  --checkpoint_path "$C3_SMALL_CHECKPOINT" \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --n_episodes 5 \
  --step_limit 200
```

### B1 controls

C1 is the A1/A2 baseline with no teacher-loss flags. Semantic goal names do
not change its architecture or training invocation.

C5 random-label control, shown as a complete command:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 10 -max_epochs 10 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  -seed 1 \
  --llm_distill \
  --llm_distill_dataset "$LLM_DATASET" \
  --llm_distill_split train \
  --llm_distill_warmup_epochs 2 \
  --llm_distill_shuffle_labels
```

C6 heuristic-teacher control uses a separate dataset so it cannot mix with
the LLM teacher records:

```bash
HEURISTIC_DATASET=training_data/llm_teacher_dataset_heuristic

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --teacher_backend heuristic \
  --policy random \
  --target_records 300 \
  --out_dir "$HEURISTIC_DATASET"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.audit_dataset --out_dir "$HEURISTIC_DATASET"
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.split_dataset --out_dir "$HEURISTIC_DATASET" --test_frac 0.2 --seed 0

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 10 -max_epochs 10 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  -seed 1 \
  --llm_distill \
  --llm_distill_dataset "$HEURISTIC_DATASET" \
  --llm_distill_split train \
  --llm_distill_warmup_epochs 2
```

---

## 6. Workstation-scale LLM pipeline

### B2. Diverse dataset collection

These real-LLM commands require Section B0 to pass:

```bash
LLM_DATASET=training_data/llm_teacher_dataset

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --policy random \
  --teacher_backend llm \
  --target_records 700 \
  --out_dir "$LLM_DATASET"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --policy untrained_net \
  --teacher_backend llm \
  --target_records 1400 \
  --out_dir "$LLM_DATASET"

BOOTSTRAP_CHECKPOINT="$A1_CHECKPOINT"
test -s "$BOOTSTRAP_CHECKPOINT"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --policy checkpoint \
  --checkpoint_path "$BOOTSTRAP_CHECKPOINT" \
  --teacher_backend llm \
  --target_records 2000 \
  --out_dir "$LLM_DATASET"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.audit_dataset --out_dir "$LLM_DATASET"
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.split_dataset --out_dir "$LLM_DATASET" --test_frac 0.2 --seed 0
```

### B2. DAgger aggregation

Always run the complete dry-run command first. Two dry-run rounds are enough
to show both the cold-start random policy and the next-round checkpoint
policy:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" experiments/dagger_loop.py \
  --dry_run \
  --rounds 2 \
  --records_per_round 1 \
  --scenario "../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" \
  --teacher_backend heuristic \
  --out_dir /tmp/nasimemu-dagger-smoke-dataset \
  --checkpoint_dir /tmp/nasimemu-dagger-smoke-checkpoints \
  --train_args "../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml -device cpu -cpus 8 -batch 16 -epoch 2 -max_epochs 1 --no_debug -net_class NASimNetDHRL -use_a_t -episode_step_limit 20 -observation_format graph_v2 -lr 0.0007 -alpha_h 0.02 --llm_distill --llm_distill_warmup_epochs 1"
```

Full DAgger plan after Ollama and the dry run pass:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" experiments/dagger_loop.py \
  --rounds 3 \
  --records_per_round 700 \
  --scenario "../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" \
  --teacher_backend llm \
  --out_dir training_data/llm_teacher_dataset \
  --checkpoint_dir training_data/dagger_checkpoints \
  --train_args "../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml -device cpu -cpus 8 -batch 128 -epoch 100 -max_epochs 200 --no_debug -net_class NASimNetDHRL -use_a_t -episode_step_limit 400 -observation_format graph_v2 -lr 0.0007 -alpha_h 0.02 --llm_distill --llm_distill_warmup_epochs 3"
```

`dagger_loop.py` launches child Python processes using its own interpreter;
the Section 1 affinity and thread environment are inherited. It locates
checkpoints under `NASimEmu-agents/wandb`, which is why offline W&B and
`WANDB_DIR="$PWD"` are required for a real run.

### B2. Full C3 training

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 128 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetDHRL \
  -force_continue_epochs 0 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1 \
  --llm_distill \
  --llm_distill_dataset training_data/llm_teacher_dataset \
  --llm_distill_split train \
  --llm_distill_warmup_epochs 3 \
  --llm_distill_rl_only_frac 0.1

C3_LARGE_RUN_DIR=$(readlink -f wandb/latest-run)
C3_LARGE_CHECKPOINT="$C3_LARGE_RUN_DIR/files/model_best.pt"
test -s "$C3_LARGE_CHECKPOINT"
```

Run C0 with A2 using the same seed. C1 is the same invocation without every
`--llm_distill*` flag. Keep scenario, seed, batch, epoch, and schedule flags
identical when comparing conditions.

Full standard and selector evaluations:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval \
  -load_model "$C3_LARGE_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" experiments/evaluate_llm_selector.py \
  --checkpoint_path "$C3_LARGE_CHECKPOINT" \
  --scenario "../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" \
  --n_episodes 50 \
  --step_limit 400
```

The selector's live mode makes real Ollama calls. Start with 1–5 episodes to
measure latency before committing to 50.

---

## 7. Run comparison

Compare a run with the published reference values:

```bash
RUN_JSON=training_data/runs/3afwaqbw.json
test -s "$RUN_JSON"
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" compare_runs.py "$RUN_JSON"
```

Compare two local runs:

```bash
RUN_JSON=training_data/runs/REPLACE_WITH_RUN_ID.json
BASELINE_JSON=training_data/runs/REPLACE_WITH_BASELINE_RUN_ID.json
test -s "$RUN_JSON"
test -s "$BASELINE_JSON"
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" compare_runs.py "$RUN_JSON" --baseline "$BASELINE_JSON"
```

---

## 8. Complete short-run certification suite

These are safe-size versions of every locally runnable pathway. They use
short episodes for wiring and lifecycle validation, not performance
measurement. Offline W&B still creates unique checkpoints. Each training
smoke replaces only the convenience mirror
`training_data/latest/latest.json`; collision-safe per-run JSON files remain
under `training_data/runs/`.

### S1. Baseline lifecycle

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 1 -max_epochs 1 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 20 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  -seed 1

S1_RUN_DIR=$(readlink -f wandb/latest-run)
S1_CHECKPOINT="$S1_RUN_DIR/files/model_best.pt"
test -s "$S1_CHECKPOINT"
```

Pass criteria: exit 0, one JSON record with `train_step == 1` and
`env_steps_total == 16`, finite train/test metrics, and nonempty latest/best
checkpoints.

### S2. Three-scenario and scheduler lifecycle

Rates are set to 1 in this smoke so both scheduler update branches execute.
The factors and minima match A2.

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 1 -max_epochs 1 \
  --no_debug \
  -net_class NASimNetDHRL \
  -force_continue_epochs 0 \
  -use_a_t \
  -episode_step_limit 20 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 1 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 1 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1

S2_RUN_DIR=$(readlink -f wandb/latest-run)
S2_CHECKPOINT="$S2_RUN_DIR/files/model_best.pt"
test -s "$S2_CHECKPOINT"
```

Evaluate both smoke checkpoints using the same 20-step behavior setting:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval -load_model "$S1_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL -use_a_t \
  -episode_step_limit 20 -observation_format graph_v2

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval -load_model "$S2_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL -use_a_t \
  -episode_step_limit 20 -observation_format graph_v2
```

### S3. Local teacher data and policy paths

This uses the heuristic teacher so it is fast and independent of Ollama.
`--test_frac 0` ensures the tiny smoke dataset has a nonempty train split;
the pytest suite covers real episode-level held-out splitting.

```bash
SMOKE_DATASET=$(mktemp -d /tmp/nasimemu-teacher-smoke.XXXXXX)

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --teacher_backend heuristic \
  --policy random \
  --step_limit 4 \
  --target_records 8 \
  --max_steps 32 \
  --periodic_every 1 \
  --out_dir "$SMOKE_DATASET"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.audit_dataset \
  --out_dir "$SMOKE_DATASET" \
  --min_class_count 0

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.split_dataset \
  --out_dir "$SMOKE_DATASET" \
  --test_frac 0 \
  --seed 0

UNTRAINED_DATASET=$(mktemp -d /tmp/nasimemu-untrained-smoke.XXXXXX)
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --teacher_backend heuristic \
  --policy untrained_net \
  --step_limit 4 --target_records 4 --max_steps 16 --periodic_every 1 \
  --out_dir "$UNTRAINED_DATASET"

CHECKPOINT_DATASET=$(mktemp -d /tmp/nasimemu-checkpoint-smoke.XXXXXX)
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m llm_teacher.label_states \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --teacher_backend heuristic \
  --policy checkpoint \
  --checkpoint_path "$S1_CHECKPOINT" \
  --step_limit 4 --target_records 4 --max_steps 16 --periodic_every 1 \
  --out_dir "$CHECKPOINT_DATASET"
```

### S4. C3 and C5 distillation paths

Two PPO updates are intentional: update 1 exercises joint PPO plus teacher
loss; update 2 is the final RL-only phase. Only one evaluation/checkpoint
interval runs.

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 2 -max_epochs 1 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 20 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  -seed 1 \
  --llm_distill \
  --llm_distill_dataset "$SMOKE_DATASET" \
  --llm_distill_split train \
  --llm_distill_warmup_epochs 1 \
  --llm_distill_batch 4

S4_RUN_DIR=$(readlink -f wandb/latest-run)
S4_CHECKPOINT="$S4_RUN_DIR/files/model_best.pt"
test -s "$S4_CHECKPOINT"

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 2 -max_epochs 1 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 20 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  -seed 1 \
  --llm_distill \
  --llm_distill_dataset "$SMOKE_DATASET" \
  --llm_distill_split train \
  --llm_distill_warmup_epochs 1 \
  --llm_distill_batch 4 \
  --llm_distill_shuffle_labels
```

Evaluate the C3 smoke checkpoint:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --eval -load_model "$S4_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL -use_a_t \
  -episode_step_limit 20 -observation_format graph_v2
```

### S5. DAgger planning and selector logic

Use the complete DAgger dry run from Section B2. It must print round 0 with
`--policy random`, round 1 with `--policy checkpoint`, and both training
commands, then exit 0 without creating checkpoints.

The supported automated suite tests selector live/distilled behavior with a
mock teacher:

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" -m pytest -q tests/test_evaluate_llm_selector.py tests/test_teacher_validator.py tests/test_teacher_schema.py
```

Without Ollama, a one-episode selector command exercises checkpoint loading,
distilled inference, API-error fallback, and output writing, but **does not**
certify valid teacher responses:

```bash
SELECTOR_SMOKE_DIR=$(mktemp -d /tmp/nasimemu-selector-smoke.XXXXXX)

taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" experiments/evaluate_llm_selector.py \
  --checkpoint_path "$S4_CHECKPOINT" \
  --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --n_episodes 1 \
  --step_limit 20 \
  --out_dir "$SELECTOR_SMOKE_DIR"

test "$(wc -l < "$SELECTOR_SMOKE_DIR/episode_summary.jsonl")" -eq 2
```

To certify real live behavior, install Ollama, pass B0, rerun this selector
smoke, and confirm the transcript records are valid rather than
`api_error:*` fallbacks.

---

## 9. Long-run artifact assertion

This read-only check turns the completed long run into a repeatable gate. It
validates all 10 JSON records, the expected step sequences, finite evaluation
metrics, the best/final values, and both checkpoint files.

```bash
taskset -c "${CPUSET:-16-31}" "${PYTHON:-../venv/bin/python}" - <<'PY'
import json
import math
from pathlib import Path

metrics_path = Path("training_data/runs/3afwaqbw.json")
rows = [
    json.loads(line)
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

assert len(rows) == 10
assert [row["train_step"] for row in rows] == list(range(10, 101, 10))
assert [row["env_steps_total"] for row in rows] == list(range(160, 1601, 160))
assert all(
    math.isfinite(float(value))
    for row in rows
    for split in ("eval_trn", "eval_tst")
    for value in row[split].values()
)

for checkpoint in (Path("/tmp/model.pt"), Path("/tmp/model_best.pt")):
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0

best_epoch, best_row = max(
    enumerate(rows, 1),
    key=lambda pair: pair[1]["eval_tst"]["captured_avg"],
)
assert best_epoch == 9
assert best_row["eval_tst"]["captured_avg"] == 28.109375
assert rows[-1]["eval_tst"]["captured_avg"] == 27.328125

print(
    "LONG RUN VERIFIED: 10/10 epochs, 100 train steps, "
    "1600 environment steps, finite metrics, checkpoints present"
)
PY
```

If `/tmp` was cleared after a reboot, the checkpoint assertions will fail
even though the persistent metrics still prove the completed run. In that
case, rerun A1 in offline mode to create persistent W&B checkpoints.

## 10. Where results land

| Result | Location |
| --- | --- |
| Offline run metadata and console logs | `wandb/offline-run-*/` |
| Latest checkpoint for an offline run | `wandb/offline-run-*/files/model.pt` |
| Best checkpoint for an offline run | `wandb/offline-run-*/files/model_best.pt` |
| Collision-safe per-run metrics | `training_data/runs/RUN_ID.json` |
| Convenience mirror for the most recent training run | `training_data/latest/latest.json` |
| Teacher dataset and state tensors | `DATASET_DIR/dataset.jsonl`, `DATASET_DIR/dataset_states.pkl` |
| Live-selector transcript | `training_data/llm_selector_eval/live_transcript.jsonl` |
| Selector episode summary | `training_data/llm_selector_eval/episode_summary.jsonl` |
| DAgger round checkpoints | `training_data/dagger_checkpoints/round_N.pt` |
| Active-trainer guard and holder metadata | `training_data/.training.lock` |

`training_data/latest/latest.json` is newline-delimited JSON: one complete
JSON object per training interval. It is not a single JSON array. Use the
per-run file under `training_data/runs/` for comparisons and durable audit.

## 11. Interrupted-run recovery

### 11.1 Recover run `4f6yqq9x` from step 11,600

The 2026-07-17 interruption left a valid weights-only `model.pt` at global
step 11,600. Its byte-verified recovery copy is:

```text
training_data/recovery/4f6yqq9x/model_step_11600.pt
```

The original best checkpoint (step 9,700, `eval_tst.captured_avg` =
30.613496932515336) is preserved separately as:

```text
training_data/recovery/4f6yqq9x/model_best_step_9700.pt
```

This command keeps the original 200-epoch/global-20,000-step endpoint. The
resume begins at step 11,601, restores curriculum epoch 116, derives the
active learning rate (`0.00062`) and entropy coefficient (`0.02`) from the
original schedules, preserves the old best-metric threshold, and therefore
runs only the remaining 8,400 updates:

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents
source ../venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/nasimemu-matplotlib
export WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache
export WANDB_MODE=offline
export WANDB_DIR="$PWD"

mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR"

RESUME_CHECKPOINT=training_data/recovery/4f6yqq9x/model_step_11600.pt
test -s "$RESUME_CHECKPOINT"

taskset -c 16-31 ../venv/bin/python main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -load_model "$RESUME_CHECKPOINT" \
  --resume \
  --resume_step 11600 \
  --resume_best_value 30.613496932515336 \
  -device cpu -cpus 8 -batch 128 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetDHRL \
  -force_continue_epochs 0 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1
```

Do not change `-max_epochs` to 84. The trainer now uses the restored global
step and stops at the original `200 * 100 = 20,000` endpoint. Setting 84
would put the configured endpoint behind step 11,600 and is rejected.

The legacy checkpoint predates full trainer-state persistence. Its neural
weights, global counters, schedule phase, curriculum position, and prior best
value are recovered. Its AdamW moments, target-network EMA, RNG state, and
active subprocess environments did not exist in the old file and cannot be
recovered retroactively; fresh episodes begin at the correct curriculum
difficulty.

### 11.2 Resume future checkpoints

New `model.pt` and `model_best.pt` files contain format-versioned training
state: global/environment steps, completed episodes, best metric, current
learning rate and entropy coefficient, optimizer state, target network, and
Python/NumPy/PyTorch RNG states. Checkpoint replacement is atomic, so an
interruption while writing leaves the prior complete checkpoint intact.

Repeat the original command and add only:

```bash
-load_model /absolute/path/to/model.pt --resume
```

Keep the original scenario, architecture, batch, epoch length, run endpoint,
and schedule arguments. Subprocess environment episodes still restart because
live simulator processes are not serializable.

## 12. Duplicate-training guard

Every training-mode `main.py` command now acquires one repository-wide lock
before it creates subprocess environments or touches `latest.json`. If a
trainer is already active, a second command exits with a message like:

```text
Another NASimEmu training run is already active.
Lock: .../training_data/.training.lock
Holder: {"pid": ..., "command": ...}
```

This prevents two independent PPO trainers from competing for CPUs and
interleaving records in the fixed convenience mirror. The eight
`SubprocVecEnv` children created by one `-cpus 8` trainer are expected worker
processes and are not duplicate trainers.

Do not remove `.training.lock` while that message is shown. The lock is held
by the kernel and releases automatically when its owner exits or crashes; the
small metadata file may remain safely afterward. Evaluation commands using
`--eval` intentionally bypass this guard and can still run when resources
permit.

New `latest.json` records include `run_id` and `trainer_pid`. For a clean,
run-specific history, continue to use:

```text
training_data/runs/RUN_ID.json
```
