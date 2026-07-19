# IDS Training and Final `model.pt` Evaluation Commands

Each section below is a complete, self-contained command. Use the copy button
on the `bash` block to copy the entire setup, training, final-checkpoint
selection, and full-IDS evaluation chain.

> Run only one training command at a time. Wait for the current recovery run
> to finish first. The repository training lock rejects overlapping trainers.
> Every standard evaluation below loads only the final `files/model.pt`; none
> selects `model_best.pt`.

For every 200-epoch training command, the scenario curriculum is:

- Epochs 0–59: baseline with IDS disabled
- Epochs 60–119: medium IDS
- Epochs 120–199: full IDS
- Final `main.py --eval`: full IDS difficulty

## 1. GNN-LSTM: train and evaluate final `model.pt`

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents && \
source ../venv/bin/activate && \
PYTHON=../venv/bin/python && \
CPUSET=16-31 && \
SCENARIOS="../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" && \
TEST_SCENARIO="../scenarios/corp_100hosts_dynamic.v2.yaml" && \
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nasimemu-matplotlib WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache WANDB_MODE=offline WANDB_DIR="$PWD" && \
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  -device cpu -cpus 8 -batch 128 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetGNN_LSTM \
  -mp_iterations 2 \
  -augment_with_action \
  -force_continue_epochs 150 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1 && \
GNN_RUN_DIR="$(readlink -f wandb/latest-run)" && \
GNN_CHECKPOINT="$GNN_RUN_DIR/files/model.pt" && \
test -s "$GNN_CHECKPOINT" && \
echo "=== GNN-LSTM COMPLETE: full-IDS evaluation of $GNN_CHECKPOINT ===" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  --eval \
  -load_model "$GNN_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetGNN_LSTM \
  -mp_iterations 2 \
  -augment_with_action \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -seed 1
```

## 2. Invariant: train and evaluate final `model.pt`

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents && \
source ../venv/bin/activate && \
PYTHON=../venv/bin/python && \
CPUSET=16-31 && \
SCENARIOS="../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" && \
TEST_SCENARIO="../scenarios/corp_100hosts_dynamic.v2.yaml" && \
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nasimemu-matplotlib WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache WANDB_MODE=offline WANDB_DIR="$PWD" && \
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  -device cpu -cpus 8 -batch 128 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetInvMAct \
  -augment_with_action \
  -force_continue_epochs 150 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format list \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1 && \
INV_RUN_DIR="$(readlink -f wandb/latest-run)" && \
INV_CHECKPOINT="$INV_RUN_DIR/files/model.pt" && \
test -s "$INV_CHECKPOINT" && \
echo "=== INVARIANT COMPLETE: full-IDS evaluation of $INV_CHECKPOINT ===" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  --eval \
  -load_model "$INV_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetInvMAct \
  -augment_with_action \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format list \
  -seed 1
```

## 3. Attention: train and evaluate final `model.pt`

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents && \
source ../venv/bin/activate && \
PYTHON=../venv/bin/python && \
CPUSET=16-31 && \
SCENARIOS="../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" && \
TEST_SCENARIO="../scenarios/corp_100hosts_dynamic.v2.yaml" && \
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nasimemu-matplotlib WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache WANDB_MODE=offline WANDB_DIR="$PWD" && \
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  -device cpu -cpus 8 -batch 128 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetXAttMAct \
  -augment_with_action \
  -force_continue_epochs 150 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format list \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1 && \
ATT_RUN_DIR="$(readlink -f wandb/latest-run)" && \
ATT_CHECKPOINT="$ATT_RUN_DIR/files/model.pt" && \
test -s "$ATT_CHECKPOINT" && \
echo "=== ATTENTION COMPLETE: full-IDS evaluation of $ATT_CHECKPOINT ===" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  --eval \
  -load_model "$ATT_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetXAttMAct \
  -augment_with_action \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format list \
  -seed 1
```

## 4. Distilled LLM: train and evaluate final `model.pt`

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents && \
source ../venv/bin/activate && \
PYTHON=../venv/bin/python && \
CPUSET=16-31 && \
SCENARIOS="../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" && \
TEST_SCENARIO="../scenarios/corp_100hosts_dynamic.v2.yaml" && \
LLM_DATASET=training_data/llm_teacher_dataset && \
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nasimemu-matplotlib WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache WANDB_MODE=offline WANDB_DIR="$PWD" && \
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" && \
test -s "$LLM_DATASET/dataset.jsonl" && \
test -s "$LLM_DATASET/dataset_states.pkl" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
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
  --llm_distill_dataset "$LLM_DATASET" \
  --llm_distill_split train \
  --llm_distill_warmup_epochs 3 \
  --llm_distill_rl_only_frac 0.1 && \
DIST_RUN_DIR="$(readlink -f wandb/latest-run)" && \
DIST_CHECKPOINT="$DIST_RUN_DIR/files/model.pt" && \
test -s "$DIST_CHECKPOINT" && \
echo "=== DISTILLED TRAINING COMPLETE: full-IDS evaluation of $DIST_CHECKPOINT ===" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  --eval \
  -load_model "$DIST_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -seed 1
```

## 5. Live LLM DAgger: train and evaluate final-round `model.pt`

Prerequisites:

- Ollama is installed and already running at `http://127.0.0.1:11434`.
- `qwen2.5:3b-instruct` is already pulled.
- No second DAgger command is using the same dataset/checkpoint directories.

This command performs three live-LLM DAgger rounds, selects the final W&B
run's `files/model.pt`, and performs standard full-IDS evaluation. It does not
use `round_2.pt` for final evaluation because that stable DAgger file may not
represent the final epoch.

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents && \
source ../venv/bin/activate && \
PYTHON=../venv/bin/python && \
CPUSET=16-31 && \
SCENARIOS="../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" && \
TEST_SCENARIO="../scenarios/corp_100hosts_dynamic.v2.yaml" && \
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nasimemu-matplotlib WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache WANDB_MODE=offline WANDB_DIR="$PWD" && \
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" && \
command -v ollama >/dev/null && \
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null && \
ollama list | grep -Fq 'qwen2.5:3b-instruct' && \
taskset -c "$CPUSET" "$PYTHON" experiments/dagger_loop.py \
  --rounds 3 \
  --records_per_round 700 \
  --scenario "$SCENARIOS" \
  --teacher_backend llm \
  --model qwen2.5:3b-instruct \
  --out_dir training_data/llm_teacher_dataset \
  --checkpoint_dir training_data/dagger_checkpoints \
  --train_args "$SCENARIOS --test_scenario $TEST_SCENARIO -device cpu -cpus 8 -batch 128 -epoch 100 -max_epochs 200 --no_debug -net_class NASimNetDHRL -force_continue_epochs 0 -use_a_t -episode_step_limit 400 -observation_format graph_v2 -lr 0.0007 -alpha_h 0.02 --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 -seed 1 --llm_distill --llm_distill_warmup_epochs 3 --llm_distill_rl_only_frac 0.1" && \
LIVE_RUN_DIR="$(readlink -f wandb/latest-run)" && \
LIVE_CHECKPOINT="$LIVE_RUN_DIR/files/model.pt" && \
test -s "$LIVE_CHECKPOINT" && \
echo "=== LIVE LLM DAGGER COMPLETE: full-IDS evaluation of $LIVE_CHECKPOINT ===" && \
taskset -c "$CPUSET" "$PYTHON" main.py \
  "$SCENARIOS" \
  --test_scenario "$TEST_SCENARIO" \
  --eval \
  -load_model "$LIVE_CHECKPOINT" \
  -device cpu -cpus 8 -batch 16 \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -seed 1
```

## 6. Optional live-versus-distilled selector comparison

Run this immediately after command 5 if you also want five paired episodes
with real Ollama calls. This is separate from the standard full-IDS
evaluation already performed by command 5.

```bash
cd /home/user5/NASimEmu-enhanced/NASimEmu-agents && \
source ../venv/bin/activate && \
PYTHON=../venv/bin/python && \
CPUSET=16-31 && \
SCENARIOS="../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml" && \
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nasimemu-matplotlib WANDB_CACHE_DIR=/tmp/nasimemu-wandb-cache WANDB_MODE=offline WANDB_DIR="$PWD" && \
LIVE_RUN_DIR="$(readlink -f wandb/latest-run)" && \
LIVE_CHECKPOINT="$LIVE_RUN_DIR/files/model.pt" && \
test -s "$LIVE_CHECKPOINT" && \
command -v ollama >/dev/null && \
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null && \
taskset -c "$CPUSET" "$PYTHON" experiments/evaluate_llm_selector.py \
  --checkpoint_path "$LIVE_CHECKPOINT" \
  --scenario "$SCENARIOS" \
  --n_episodes 5 \
  --step_limit 400 \
  --model qwen2.5:3b-instruct
```
