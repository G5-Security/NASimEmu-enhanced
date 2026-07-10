# Environment Setup & Runtime Error Fixes

> Reconstructed from the current state of this working tree: the installed
> venv (`venv/`), `requirements.txt` / `pyproject.toml`, and the three files
> still showing as uncommitted (`git status`) at the time this doc was
> written. The fixes below are not-yet-committed working-tree changes — they
> already exist in the repo, this doc explains *why* each one is there.

## 1. Environment setup

**Base interpreter**: Python 3.10.20, from a miniconda env (`venv/pyvenv.cfg`
points `home` at `~/miniconda3/envs/py310/bin`).

```bash
# 1. Create/activate a Python 3.10 base (miniconda used here)
conda create -n py310 python=3.10
conda activate py310

# 2. Create the project venv on top of it, from the repo root
python -m venv venv
source venv/bin/activate

# 3. Install pinned third-party dependencies
pip install -r requirements.txt

# 4. Install this repo's own package (nasimemu) in editable mode
#    (pyproject.toml -> src/nasimemu), so `import nasimemu` resolves
#    to the local source tree instead of a built wheel
pip install -e .
```

Confirmed from the live venv:

```
$ venv/bin/pip show nasimemu
Name: nasimemu
Version: 0.9.0
Location: venv/lib/python3.10/site-packages
Editable project location: /home/rakibul/NASimEmu-enhanced
```

Key pinned packages (`requirements.txt`, full CUDA wheel set included even
though this machine trains on CPU — see note below): `torch==2.7.1`,
`torch-geometric==2.6.1`, `torch-scatter==2.1.2`, `gym==0.21.0`,
`numba==0.61.2`, `wandb==0.21.0`, `numpy==2.2.6`.

**Watch-out**: `torch-scatter` is not a pure-Python package — it ships a
compiled extension that must match your exact `torch` build (CPU vs a
specific CUDA version). Installing it from plain PyPI can silently pull a
mismatched build and fail at *import time* rather than install time
(`OSError: ... undefined symbol ...` or `ImportError: libtorch_cpu.so ...`).
If that happens, install it from PyG's own wheel index instead, matching the
installed torch version and device:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cpu.html
```

This machine has no GPU (`nvidia-smi` isn't installed), so training is
launched with `-device cpu` explicitly.

**Training is invoked as** (this is the actual command line running against
this checkout — 16 requested CPU workers on the DHRL net):

```bash
python main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 16 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetDHRL \
  -force_continue_epochs 0 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005
```

(`nproc` reports 12 logical cores on this machine, so `-cpus 16` here
over-subscribes by 4 workers — not a crash, just wasted context-switch
overhead; drop it to `-cpus 10` or `11`.)

---

## 2. Runtime errors fixed and why

Three files carry uncommitted fixes. Each was needed because IDS
(Intrusion Detection System) fields were added to `Host`/`HostVector` in a
prior commit, and a couple of call sites hadn't caught up.

### 2.1 `AttributeError: 'Host' object has no attribute 'detection_threshold'`

**File**: [`src/nasimemu/nasim/scenarios/host.py`](../src/nasimemu/nasim/scenarios/host.py)

`HostVector.from_host()` (in `host_vector.py`) reads
`host.detection_level`, `host.detection_threshold`, and
`host.detection_multiplier` to build the observation vector for every host.
But `Host.__init__` never defined those attributes — anything that
constructs a `Host(...)` directly (e.g. the auto-scenario generator in
`env.py::_generate_auto_from_template`) produced an object missing them,
so the very first `env.reset()` crashed with an `AttributeError` as soon as
IDS observation encoding ran.

```diff
 class Host:
     def __init__(self, ...,
-                 access=0):
+                 access=0,
+                 detection_level=0.0,
+                 detection_threshold=None,
+                 detection_multiplier=1.0):
         ...
         self.access = access
+        self.detection_level = detection_level
+        self.detection_threshold = detection_threshold
+        self.detection_multiplier = detection_multiplier
```

**Fix**: give `Host` the three IDS attributes with safe defaults, so every
`Host` — however it's constructed — has them.

### 2.2 `TypeError: float() argument must be a string or a real number, not 'NoneType'`

**File**: [`src/nasimemu/nasim/envs/host_vector.py`](../src/nasimemu/nasim/envs/host_vector.py) (`from_host`, ~line 168)

Fixing 2.1 introduced a new problem: `detection_threshold` now defaults to
`None` (there's no single "correct" default threshold — it's meant to be
randomized per host). But `from_host` assigned it straight into a `float32`
numpy vector:

```python
vector[cls._detection_threshold_idx] = host.detection_threshold  # None -> crash
```

Assigning `None` into a numeric numpy array slot raises a `TypeError` at
scenario-generation time — every episode reset would fail immediately.

```diff
+        detection_threshold = host.detection_threshold
+        if detection_threshold is None:
+            threshold_range = cls.ids_config.get('base_thresholds', [0.7, 0.8])
+            detection_threshold = np.random.uniform(threshold_range[0], threshold_range[1])
         vector[cls._detection_level_idx] = host.detection_level
-        vector[cls._detection_threshold_idx] = host.detection_threshold
+        vector[cls._detection_threshold_idx] = detection_threshold
```

**Fix**: when `detection_threshold` is unset, sample one uniformly from the
scenario's configured `base_thresholds` range instead of storing `None`.
This also gives each host a per-episode randomized detection threshold,
which is presumably the intended behavior (a fixed global threshold would
make IDS trivially predictable).

### 2.3 Stale IDS features in the observation (silent correctness bug, not a crash)

**File**: `src/nasimemu/nasim/envs/host_vector.py` (`update` / detection-event
handling, ~lines 433–500)

`detection_level` decays every step (`self.detection_level *= decay_rate`)
and `detection_multiplier` changes on a "minor" detection event
(`self.detection_multiplier = 2.0`) — but both only updated the Python
attribute, not `self.vector[...]`, which is the array actually serialized
into the agent's observation (`HostVector.numpy()` reads from `self.vector`,
not from the Python attributes). Net effect: the agent's *observed* IDS
state silently lagged behind the *real* IDS state — no exception, just a
wrong signal the policy trains on.

```diff
         self.detection_level *= decay_rate
-        
+        self.vector[self._detection_level_idx] = self.detection_level
+
         if self.detection_level > self.detection_threshold:
             return 'DETECTED', self._handle_detection()
@@
         else:
             self.detection_multiplier = 2.0
+            self.vector[self._detection_multiplier_idx] = self.detection_multiplier
             return { 'type': 'monitor', 'penalty': -5, ... }
```

**Fix**: write both attributes back into `self.vector` immediately after
they change, so the observation the agent sees always matches the
underlying IDS state.

### 2.4 `AttributeError: 'NoneType' object has no attribute 'items'`

**File**: [`NASimEmu-agents/main.py`](../NASimEmu-agents/main.py) (~line 363, JSON logging block)

`nasim_debug.py::evaluate()` returns:

```python
return {'eval_trn': log_trn, 'eval_tst': log_tst}
```

where `log_tst` is explicitly set to **`None`** whenever no
`--test_scenario` is configured. The old logging code did:

```python
log['eval_perf'].get('eval_tst', {}).items()
```

`dict.get(key, default)` only falls back to `default` when the key is
**missing** — here the key `'eval_tst'` exists, its *value* is `None`, so
`.get(...)` returns `None`, and `.items()` on `None` crashes. This hit on
literally the first logging interval (`step % config.log_rate == 0`) of any
run started without a test scenario.

```diff
-'eval_trn': {k: _to_serializable(v) for k, v in log['eval_perf'].get('eval_trn', {}).items()},
-'eval_tst': {k: _to_serializable(v) for k, v in log['eval_perf'].get('eval_tst', {}).items()},
+'eval_trn': {k: _to_serializable(v) for k, v in (log['eval_perf'].get('eval_trn') or {}).items()},
+'eval_tst': {k: _to_serializable(v) for k, v in (log['eval_perf'].get('eval_tst') or {}).items()},
```

**Fix**: `(... or {})` coerces a `None` value to `{}` before `.items()` is
called, regardless of whether the key was missing or present-but-`None`.

### 2.5 `FileNotFoundError: [Errno 2] No such file or directory: '.../model.pt'`

**File**: `NASimEmu-agents/main.py` (~line 372, checkpoint save)

```python
model_file = os.path.join(wandb.run.dir, "model.pt")
net.save(model_file)   # torch.save under the hood
```

`wandb.run.dir` is guaranteed to exist as *wandb's* run directory, but
`torch.save` still needs every path component to exist — in offline mode
(as used here: `wandb/offline-run-.../files/`) the exact `files` subpath
isn't always pre-created before the first save call, so `torch.save` raised
`FileNotFoundError` on the very first checkpoint.

```diff
 model_file = os.path.join(wandb.run.dir, "model.pt")
+os.makedirs(os.path.dirname(model_file), exist_ok=True)
 net.save(model_file)
```

**Fix**: `os.makedirs(..., exist_ok=True)` right before saving, which is the
standard guard for "save to a path whose directory may not exist yet."

### 2.6 Log file collisions across runs (not a crash, but silent data loss)

**File**: `NASimEmu-agents/main.py` (~line 194, JSON logger path)

```diff
-log_dir = os.path.join(os.path.dirname(__file__), 'training_data', 'latest')
+log_dir = os.path.join(os.path.dirname(__file__), 'training_data', 'runs')
 os.makedirs(log_dir, exist_ok=True)
-jsonl_path = os.path.join(log_dir, 'latest.json')
+jsonl_path = os.path.join(log_dir, f'{wandb.run.id}.json')
```

**Reason**: every run used to append to the same
`training_data/latest/latest.json`. Two runs (or a restarted run) overwrite
or interleave into the same file with no way to tell which lines came from
which run — this is exactly why `training_data/latest/latest.json` is
tracked in git yet keeps showing local diffs (each run mutates it further).
Keying the filename by `wandb.run.id` under `training_data/runs/` gives
each run its own append-only log.

---

## Summary table

| # | File | Symptom | Fix |
|---|------|---------|-----|
| 2.1 | `host.py` | `AttributeError` on missing IDS attrs | add `detection_*` fields to `Host.__init__` |
| 2.2 | `host_vector.py` | `TypeError` storing `None` into float vector | sample random threshold when unset |
| 2.3 | `host_vector.py` | stale IDS features in observation (no crash) | sync `self.vector[...]` after attribute updates |
| 2.4 | `main.py` | `AttributeError: NoneType.items` when no test scenario | `.get(k) or {}` null coalescing |
| 2.5 | `main.py` | `FileNotFoundError` saving checkpoint | `os.makedirs` before `net.save` |
| 2.6 | `main.py` | silent log overwrite/interleave across runs | per-run log filename (`wandb.run.id`) |

None of these are committed yet (`git status` shows all three files as
modified, not staged) — worth a commit once verified against the currently
running training job.
