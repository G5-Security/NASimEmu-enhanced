# Current Workstation Runtime Fixes and Validation

This document records the fixes and diagnosis that made the NASimEmu
training environment run reliably on the current workstation. It is specific
to the checkout at `/home/user5/NASimEmu-enhanced` and the i9-14900K host
validated on 2026-07-15.

> **Accurate status:** the documented CPU launch profile completed the full
> 10-epoch job without an observed software or worker failure. The machine's
> unrestricted P-core path is contained by CPU affinity; it has **not** been
> proven permanently repaired. Do not describe this as a general hardware
> fix or remove the affinity restriction without repeating the stress and
> long-run gates.

Use [run_commands.md](run_commands.md) for the paste-ready commands. This
document explains why those commands differ from the original invocation and
what changed in the repository.

The older [environment_setup_and_fixes.md](environment_setup_and_fixes.md)
contains historical setup and IDS notes for a different machine state. Its
`/home/rakibul`, 12-thread, bare-`python`, and `-cpus 16` instructions are not
the current workstation launch profile.

## 1. Outcome

The reliable profile combines operational corrections with several
repository fixes:

| Area | Actual problem | Resolution | Status |
| --- | --- | --- | --- |
| Interpreter | Bare `python` does not exist; system Python 3.12 lacks project packages | Use `../venv/bin/python` (Python 3.10.20) | Verified |
| Working directory | `../scenarios/...` is relative to `NASimEmu-agents/` | Launch from `/home/user5/NASimEmu-enhanced/NASimEmu-agents` | Verified |
| CPU stability | P-core execution reproduced heap/GC corruption outside the repository | Pin the parent and all children to logical CPUs `16-31` | Contained, not repaired |
| Process parallelism | Old `-cpus 32` guidance oversubscribed the 16 verified E-cores | Use `-cpus 8`; keep training/eval batches divisible by 8 | Verified |
| BLAS parallelism | Every worker could create another large BLAS thread pool | Export BLAS/OpenMP thread counts as 1 before Python starts | Verified |
| Scenario loading | Dynamic YAML was parsed repeatedly and returned mutable loader data | Cache a pristine per-process parse and return a deep copy | Verified |
| W&B | Interactive/online behavior and disabled-mode checkpoint paths were mixed | Use offline mode for normal runs; disabled only for sandbox runs | Verified |
| CLI help | Literal `%` in argparse help strings broke help rendering | Escape it as `%%` | Verified |
| Tensor devices | PPO value-target tensors were implicitly created on CPU | Create them on `v_.device` | CPU verified; CUDA not certified |
| Interrupted runs | Checkpoints stored weights only, so scheduler/curriculum progress reset on `-load_model` | Add step-aware legacy recovery plus full trainer-state checkpoints and atomic replacement | Verified |
| Duplicate trainers | Independent `main.py` launches could train concurrently on the same CPUs and interleave the shared `latest.json` | Hold a kernel-backed exclusive training lock; evaluation commands bypass it | Verified |

The failure chain was:

`P-core instability → heap corruption → later, misleading exceptions in YAML/GC/worker code`

High worker count, nested BLAS pools, and repeated YAML parsing increased the
allocation/load pressure and made the fault easier to hit. They were useful
software fixes, but isolated pure-Python failures showed that neither
OpenBLAS nor PyYAML was the sole underlying cause.

## 2. Verified host

| Property | Observed value |
| --- | --- |
| CPU | Intel Core i9-14900K, 24 cores / 32 logical CPUs |
| P-core logical CPUs | `0-15` (8 cores with two threads each) |
| E-core logical CPUs | `16-31` (16 single-threaded cores) |
| Memory | 62 GiB total; memory exhaustion was not observed |
| Motherboard | Gigabyte Z790 AORUS PRO X |
| BIOS | F7, dated 2025-06-19 |
| Runtime microcode | `0x133` |
| Project Python | 3.10.20 from `venv/` |
| PyTorch | 2.7.1+cu126 |
| CUDA availability | `torch.cuda.is_available() == False` |
| Training device | CPU |

The installed CUDA-flavoured PyTorch wheels do not require a GPU when the
program is launched with `-device cpu`. The missing/unavailable GPU was not
the reason the original command crashed.

## 3. What the diagnosis established

### 3.1 The literal original launcher was wrong for this checkout

The initial command used `python main.py ... ../scenarios/...`.

- `python` is unavailable in the current shell.
- `/usr/bin/python3` is Python 3.12 and does not contain the pinned project
  environment.
- The repository virtual environment is `../venv/bin/python` when the
  working directory is `NASimEmu-agents/`.
- The relative scenario paths only resolve from that directory.

This was fixed in every command in [run_commands.md](run_commands.md) by
using a common `runpy` launcher.

### 3.2 The random parser errors were symptoms, not deterministic YAML bugs

The failing training command generated an apport record at
`/var/crash/_usr_bin_python3.10.1003.crash` with the exact command line and
signal 11 (`SIGSEGV`). Other attempts produced inconsistent exceptions deep
inside YAML scanner/composer code instead of one repeatable validation error.

The host-level isolation tests were more important than the surface stack
trace:

- Minimal pure-Python allocation/parse stress failed with isolated Python
  3.10 and Python 3.12 processes, without importing this repository.
- `PYTHONMALLOC=debug` detected a damaged trailing allocation guard.
- P-core group tests reproduced corruption; E-core-only tests did not.
- A pinned E-core reset stress completed 16,000 dynamic environment resets.
- General CPU/RAM capacity was sufficient, and short broad stress tests did
  not reproduce the more specific bursty Python allocation failure.

That evidence ruled out a repository-only or Python-version-only root cause.
The known-good training launcher therefore pins the entire process tree:

```bash
taskset -c 16-31 ../venv/bin/python ...
```

`-cpus 8` is still required. It controls the program's worker count; it does
not replace OS CPU affinity.

### 3.3 Why the old laptop could work

The earlier i7-1255U and the current i9 workstation are different hardware,
firmware, scheduler, and memory-controller environments. A workload running
on the old CPU does not prove the new CPU's unrestricted core group is
stable. The current mitigation is intentionally specific to the observed
`16-31` E-core mapping and must not be copied back to the 12-thread laptop.

## 4. Mandatory process settings

The complete setup is in
[run_commands.md, Section 1](run_commands.md#1-required-shell-setup). The
essential part is:

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

PYTHON=../venv/bin/python
CPUSET=16-31

runpy() {
  taskset -c "$CPUSET" "$PYTHON" "$@"
}
```

`MPLCONFIGDIR` and `WANDB_CACHE_DIR` point tools at writable temporary cache
locations. This avoids failures when a sandboxed or Snap-launched process
cannot write its default home-cache directory.

### 4.1 Why the shell exports are explicit

[main.py](../NASimEmu-agents/main.py) sets the four thread variables with
`os.environ.setdefault(...)` before importing NumPy. That provides safe
defaults only when the parent shell did not already define a value. If the
shell inherited `OPENBLAS_NUM_THREADS=32`, `setdefault` deliberately leaves
32 unchanged. The explicit exports guarantee the tested value of 1.

Training already parallelizes across subprocess workers. One BLAS pool per
worker would multiply thread counts, increase context switching, and add
unnecessary load to the unstable host.

### 4.2 W&B modes and checkpoint locations

Normal user runs use `WANDB_MODE=offline`:

- No cloud login or network upload is required.
- Each run gets a unique `wandb/offline-run-*/files/` directory.
- `model.pt` and `model_best.pt` are persistent and discoverable through
  `wandb/latest-run`.
- `experiments/dagger_loop.py` can find the checkpoint layout it expects.

The successful Codex-controlled 10-epoch run used
`WANDB_MODE=disabled` because the tool sandbox prohibits the local sockets
used by the W&B offline service. Disabled mode changed telemetry and the
checkpoint directory only; it did not change training or evaluation logic.
Its checkpoints are `/tmp/model.pt` and `/tmp/model_best.pt`, so they are
ephemeral.

## 5. Repository fixes

### 5.1 Early thread defaults and offline W&B default

File: [NASimEmu-agents/main.py](../NASimEmu-agents/main.py)

- Imports `os` before NumPy.
- Sets OpenBLAS, OpenMP, MKL, and NumExpr defaults to one thread.
- Defaults W&B to offline mode so training does not stop at an account prompt.
- Explicit shell variables still override these defaults.

The thread settings prevent oversubscription. They are not evidence that
OpenBLAS itself caused the isolated pure-Python P-core corruption.

### 5.2 Dynamic scenario YAML cache with copy isolation

File:
[src/nasimemu/nasim/scenarios/utils.py](../src/nasimemu/nasim/scenarios/utils.py)

Before the change, every environment reset reparsed the complete YAML file.
With multiple workers and two evaluation splits, the same dynamic scenario
was parsed repeatedly during every epoch.

The loader now:

1. Canonicalizes the scenario path.
2. Uses `(st_mtime_ns, st_size)` as the per-file cache signature.
3. Parses only on a cache miss or file change.
4. Stores one pristine template per Python process.
5. Returns `copy.deepcopy(...)` on every call.

The deep copy is necessary because later scenario normalization expands
ranged subnet values and mutates dictionaries. Returning the cached object
directly would leak one episode's mutations into later episodes.

This is a per-process cache, not shared memory across all workers. Each
worker parses once and then reuses its own pristine template.

Regression coverage:
[NASimEmu-agents/tests/test_scenario_yaml_cache.py](../NASimEmu-agents/tests/test_scenario_yaml_cache.py)

- Confirms repeated loads parse once.
- Confirms mutation of one returned value cannot affect the next load.
- Confirms a changed file invalidates and refreshes the cache.

### 5.3 Defensive scenario-load retry

File: [src/nasimemu/env.py](../src/nasimemu/env.py)

Static YAML scenario construction now gets at most three attempts. A
successful attempt continues normally; if all three fail, the final
exception is re-raised.

This is a narrow resilience layer for transient failures. It does not repair
hardware corruption, and it does not permanently hide a malformed scenario.
The cache reduces the number of parser invocations; the affinity restriction
is the primary host containment.

### 5.4 Argparse percent escaping

File:
[NASimEmu-agents/nasim_problem/nasim_config.py](../NASimEmu-agents/nasim_problem/nasim_config.py)

Argparse applies percent-style formatting to help text. Literal `10%` text
was changed to `10%%` in the domain-randomization help strings. This prevents
`--help` from raising a formatting error. All seven documented Python CLI
entry points subsequently rendered help and exited 0.

### 5.5 PPO value-target device consistency

File:
[NASimEmu-agents/nasim_problem/net_utils.py](../NASimEmu-agents/nasim_problem/net_utils.py)

Reward tensors, done tensors, and the `v_target` buffer are now constructed
on `v_.device`. This prevents mixing CPU tensors with model values on another
device.

The current CPU run validates the CPU path and confirms no regression. CUDA
is unavailable on this workstation, so this change's CUDA path is not
locally certified.

### 5.6 Duplicate-training prevention

Files:
[NASimEmu-agents/training_lock.py](../NASimEmu-agents/training_lock.py) and
[NASimEmu-agents/main.py](../NASimEmu-agents/main.py)

A training invocation now acquires an exclusive, non-blocking POSIX record
lock at `NASimEmu-agents/training_data/.training.lock` before it creates
worker environments, initializes W&B, or truncates the convenience log. A
second training command exits immediately and prints the holder metadata
instead of silently consuming the same CPU set.

The kernel lock, not the existence of the file, determines whether a trainer
is active. It is released automatically after a normal exit, exception,
crash, or reboot, so a stale metadata file is harmless and must not be
manually deleted. Read-only `--eval`, trace, debug, and baseline commands do
not acquire the training lock.

New `latest.json` records also contain `run_id` and `trainer_pid`, making the
source unambiguous during diagnostics. Collision-safe per-run records remain
the durable source of truth under `training_data/runs/RUN_ID.json`.

The recovery process for `4f6yqq9x` was already running when this guard was
added, so a lightweight lock holder adopted its PID without restarting it.
That process keeps its pre-change JSON record shape; runs launched from the
updated source include the two new identity fields automatically.

### 5.7 Change provenance

The YAML cache, restart persistence, duplicate-training guard, and their
regression tests were added during this runtime diagnosis. The BLAS-thread
addition in `main.py`, the argparse fix in
`nasim_config.py`, the device fix in `net_utils.py`, and the retry in `env.py`
were already present as uncommitted working-tree fixes when this diagnosis
began; they were inspected and validated as part of the successful test and
training matrix. The W&B offline default in `main.py` was already in `HEAD`.
The remaining working-tree fixes should not be mistaken for committed
upstream state until they are intentionally reviewed and committed.

## 6. Validation evidence

### 6.1 Automated and short-run gates

The complete commands and pass criteria are in
[run_commands.md, Section 8](run_commands.md#8-complete-short-run-certification-suite).

| Gate | Result |
| --- | --- |
| Dependency imports | PASS |
| Python CLI parsers | 7/7 PASS |
| Supported pytest suite | 95 passed, 1 skipped |
| Single-scenario train/eval/checkpoint lifecycle | PASS |
| Three-scenario loader and both scheduler branches | PASS |
| Baseline checkpoint evaluations | 2/2 PASS |
| Heuristic collect, audit, split | PASS |
| Random, untrained-net, checkpoint policies | 3/3 PASS |
| C3 warm-up, joint loss, RL-only transition, eval, save | PASS |
| C5 shuffled-label control | PASS |
| Selector mocked logic and no-Ollama fallback | PASS |
| Two-round DAgger dry run | PASS |
| Long-run comparison script | PASS |
| Two-stage checkpoint/resume lifecycle | PASS (step 1 resumed as step 2 with optimizer, target, and RNG state) |

The supported test suite is specifically `NASimEmu-agents/tests`. A
repository-root pytest run is not the environment gate because legacy/manual
tests outside that directory require optional MPI, Atari, and `baselines`
dependencies that are not installed for this project environment.

### 6.2 Full 10-epoch acceptance run

The complete command is
[A1 in run_commands.md](run_commands.md#a1-verified-10-epoch-baseline). The
actual acceptance run used the same model and CPU flags without a fixed seed
and with W&B disabled for sandbox compatibility.

| Evidence | Observed value |
| --- | --- |
| Process exit | 0 |
| Epochs | 10/10 |
| PPO training steps | 100 |
| Logged environment steps | 1,600 |
| Completed episodes | 64 |
| Runtime | Approximately 6 minutes 11 seconds |
| Epoch-10 test captured average | 27.328125 |
| Best test captured average | 28.109375 at epoch 9 |
| Final checkpoint | `/tmp/model.pt`, nonempty and loadable |
| Best checkpoint | `/tmp/model_best.pt`, nonempty and loadable |
| Persistent metrics | `NASimEmu-agents/training_data/runs/3afwaqbw.json` |
| New crash report during the run | None |

The metrics file contains exactly 10 newline-delimited JSON objects, training
steps 10 through 100, environment steps 160 through 1,600, and finite values
for both evaluation splits. A separate evaluation loaded the best checkpoint
and exited 0.

The repeatable read-only verifier is
[run_commands.md, Section 9](run_commands.md#9-long-run-artifact-assertion).

### 6.3 Acceptance statement

The environment is accepted for this workload when all of the following are
true:

- The project venv and `NASimEmu-agents/` working directory are used.
- BLAS/OpenMP thread counts are explicitly one.
- The process tree is restricted to logical CPUs `16-31`.
- Training uses `-device cpu -cpus 8` and a compatible batch.
- The supported tests pass.
- The short lifecycle gate passes before a new long configuration.
- Checkpoints and metrics are validated after the run.

Within those boundaries, the supplied 10-epoch command is demonstrated to
run to completion. This does not guarantee every future scenario, model, or
unbounded duration.

## 7. Remaining limitations and permanent host action

### 7.1 P-core stability remains unresolved

`taskset` is a software containment measure. For unrestricted use of the
workstation:

1. Confirm the exact motherboard revision and install its latest stable BIOS.
2. Load Intel Default Settings after the update.
3. Disable XMP, Gigabyte PerfDrive, manual overclocking, and undervolting
   while diagnosing.
4. Test memory at JEDEC settings with a multi-pass or overnight memory test.
5. Repeat the isolated Python allocator and training tests on P cores.
6. If corruption remains at defaults, service or RMA the CPU/system rather
   than adding more application retries.

Intel's public guidance for affected 13th/14th-generation desktop systems
also recommends current BIOS/microcode and Intel Default Settings:

- [Intel root-cause update](https://community.intel.com/t5/Mobile-and-Desktop-Processors/Intel-Core-13th-and-14th-Gen-Desktop-Instability-Root-Cause/td-p/1633442)
- [Intel Vmin-shift/microcode update](https://community.intel.com/t5/Mobile-and-Desktop-Processors/Intel-Core-13th-and-14th-Gen-Vmin-Shift-Instabilty-Update-New/m-p/1686948)
- [Intel warranty update](https://community.intel.com/t5/Mobile-and-Desktop-Processors/Additional-Warranty-Updates-on-Intel-Core-13th-14th-Gen-Desktop/m-p/1620853)

These links describe the affected product family and remediation guidance;
the local tests establish this host's instability, but do not by themselves
prove one specific silicon degradation mechanism.

### 7.2 Features not certified in this session

- Ollama is not installed. Real `--teacher_backend llm` collection and valid
  live-selector responses remain blocked. Heuristic, distilled, mocked, and
  API-error fallback paths passed.
- CUDA is unavailable. CPU operation is certified; CUDA execution is not.
- The 200-epoch, three-scenario A2/B2 configurations received short lifecycle
  tests but were not run to completion.
- `/tmp/model*.pt` is temporary. A normal W&B-offline rerun should create
  persistent checkpoints before relying on them for later experiments.

## 8. Troubleshooting after these fixes

| Symptom | First check |
| --- | --- |
| `python: command not found` | Use the `runpy` setup and `../venv/bin/python` |
| Scenario file not found | Confirm current directory is `NASimEmu-agents/` |
| Random YAML/GC errors, SIGSEGV, broken workers | Confirm the command starts through `taskset -c 16-31`; inspect `/var/crash` |
| Very high CPU/thread count | Re-export all four thread variables as 1 before Python starts |
| Evaluation assertion after training | Use `-cpus 8`; keep both training batch and fixed eval batch 64 divisible by it |
| No persistent W&B checkpoint | Use offline rather than disabled mode; inspect `wandb/latest-run/files/` |
| DAgger cannot locate a checkpoint | Keep `WANDB_MODE=offline` and `WANDB_DIR="$PWD"` |
| Distillation dataset missing | Run the collect/audit/split pipeline in `run_commands.md` |
| Ollama connection errors | Install/start Ollama, pull the configured model, and pass the `/api/tags` health check |

Do not treat repeated transient retries as an acceptable response to renewed
heap corruption or SIGSEGV. Recheck hardware defaults and affinity first.
