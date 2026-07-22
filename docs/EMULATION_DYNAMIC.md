# Emulating Dynamic Scenarios — Design & Roadmap

This document plans how to run the **dynamic** scenarios (IDS, service churn,
scan noise, network reliability) on the **real-VM emulation** backend, not just
in simulation. It covers all three goals the project asked for and all four
dynamic features, in feasibility order, and marks exactly which steps must be
validated on a machine with Vagrant/VirtualBox/Metasploit.

> **Hard constraint.** The emulation backend runs real VMs (Vagrant +
> VirtualBox) and drives them through Metasploit (`msfrpcd`). That software is
> **not present in the dev/CI sandbox**, so every VM-behavior claim below must
> be validated on the workstation that hosts the emulation. Steps marked
> **[VM]** need that box; steps marked **[sandbox]** are validated here (unit
> tests, YAML load, Vagrantfile generation, `bash -n`, dry-run).

---

## 1. Why this is not a one-liner

The dynamic realism is **simulation-model-only**. In `src/nasimemu/env.py`:

```python
if self.emulate:
    self.env = EmulatedNASimEnv(scenario=scenario)   # no curriculum, no realism model
else:
    self.env = NASimEnv(..., training_mode=..., curriculum_total_epochs=...)
```

`EmulatedNASimEnv` has no `curriculum_manager`, and the IDS `detection_level`,
scan-noise FP/FN, and churn math all live in `host_vector.py`'s **simulation**
code path, which emulation never runs (observations come from real hosts via
Metasploit). So running a dynamic scenario with `emulate=True` today emulates
only its **static topology**; the dynamics silently vanish.

To emulate the dynamics, each feature needs a real mechanism on the VMs:

| Feature | Emulation feasibility | Mechanism |
|---|---|---|
| `service_dynamics` (churn) | ✅ clean | Existing `vagrant/target/linux-up-down-script/{up,down}-*.sh` toggled by a churn daemon |
| `network_reliability` (timeouts) | ✅ clean | `tc`/`netem` loss+latency on the router (or per host) |
| `ids` (detection + response) | ⚠️ hard but doable | Real sensor (Suricata) mirroring router traffic + firewall quarantine; re-derive detection accounting from alerts |
| `scan_noise` (FP/FN) | ❌ does not map | Real scans have real behavior; injecting synthetic FP/FN contradicts emulating. **Stays simulation-only by design.** |

And a practical wall: the shipped dynamic scenario is `corp_100hosts_dynamic`
(~88 VMs). Emulating that on one machine is infeasible, so we start from a
**small dynamic scenario**.

---

## 2. Foundation (DONE, [sandbox]-verified)

`scenarios/sm_dynamic_one_subnet.v2.yaml` — the same procedural format as the
corp dynamic scenario, but a fixed **3 targets** (1 DMZ + 2 user) carrying the
full 3-stage dynamic-realism curriculum. Verified:

- loads in simulation; in eval mode (`training_mode=False`) the final stage
  pins **IDS on** (matches `--hardest_stage`);
- `python -m nasimemu.vagrant_gen scenarios/sm_dynamic_one_subnet.v2.yaml ...`
  generates a **5-VM** Vagrantfile (3 targets + attacker + router) — feasible.

This is the substrate every emulation goal below builds on.

---

## 3. Goals → phases

The three requested goals are layers on the same substrate:

- **Goal B — run dynamic scenarios in emulation without crashing** = Phase 1.
  (Largely already true: `vagrant_gen` + `load_scenario` ignore the dynamic
  keys and emulate the topology. Phase 1 makes that explicit and small-scale.)
- **Goal A — faithful dynamics on real VMs** = Phases 2–4 (churn, netem, IDS).
- **Goal C — sim-to-emulation eval bridge** = Phase 5 (train in sim, evaluate
  the checkpoint against the emulated network, measure transfer).

### Phase 1 — topology emulation of the small dynamic scenario  [VM]
1. `./setup_vagrant.sh scenarios/sm_dynamic_one_subnet.v2.yaml`  [sandbox: generation] / **[VM]: `vagrant up`**
2. `env = gym.make('NASimEmu-v0', emulate=True, scenario_name='scenarios/sm_dynamic_one_subnet.v2.yaml')`
3. Confirm the agent can scan/exploit/capture loot on the 3 real hosts.
   *Dynamics are ignored here — this is the working baseline to add them onto.*

### Phase 2 — service churn on the VMs  [mostly sandbox, behavior VM]
- A `churn-daemon` (systemd timer / cron) per target reads the final stage's
  `service_dynamics` (churn_probability, affected_services, churn_types,
  down_duration) and calls the existing `up-*.sh` / `down-*.sh` to stop/start a
  service for a sampled duration.
- `vagrant_gen` emits the daemon + config into each target's provisioning.
- **[sandbox]** validate: generated daemon script passes `bash -n`; the config
  JSON it reads matches the scenario's `service_dynamics`.
- **[VM]** validate: `systemctl` shows the service actually flapping; a msf
  `service_scan` sees it appear/disappear.

### Phase 3 — network reliability via netem  [mostly sandbox, behavior VM]
- Provision `tc qdisc ... netem loss <p>% delay <d>ms` on the router's internal
  interface, parameterised from `network_reliability` (timeout_probability →
  loss, timeout_types.duration → delay). Applied at `vagrant up`.
- **[sandbox]** validate: generated `tc` script passes `bash -n`; loss/latency
  numbers derive correctly from the YAML.
- **[VM]** validate: `ping`/exploit retries show the injected loss.

### Phase 4 — IDS: real detection + response  [design here, build+VM later]
- **Sensor:** Suricata on the router, sniffing the internal segment; alert
  volume per source IP ≈ the sim's `detection_increase` per action class.
- **Accounting bridge:** `EmulatedNetwork` polls Suricata's `eve.json`, maps
  alert counts → a `detection_level` per host, and writes the same
  `detection_level`/`detection_multiplier` observation columns the sim exposes
  (so the agent's `_aggregate_ids_features` path is unchanged; the **threshold
  stays hidden**, exactly like sim).
- **Response:** on threshold crossing, apply a router firewall rule to
  quarantine the host (drop its traffic) ≈ the sim's `response_types.quarantine`.
- **[sandbox]** validate: the alert→detection_level mapping is unit-tested
  against a canned `eve.json`.
- **[VM]** validate: noisy scanning raises the level and eventually quarantines.

### Phase 5 — sim-to-emulation eval bridge  [mostly sandbox, run VM]
- A script `experiments/eval_emulation.py` that loads a trained `NASimNetDHRL`
  checkpoint and runs `evaluate()` against `emulate=True` on the small dynamic
  scenario, reporting captured/reward/eplen — the transfer metric.
- **[sandbox]** validate: arg parsing + a `--dry_run` that builds the net and
  prints the intended emulation command without needing VMs.
- **[VM]** validate: real transfer numbers vs. the sim eval.

### scan_noise — intentionally sim-only
Documented as not emulated: real scans carry real (not modelled) noise.
Keeping the synthetic FP/FN model out of emulation is a correctness choice, not
an omission. The agent trained with sim scan-noise still runs against
emulation; it just meets the network's real scan behavior instead.

---

## 4. What to run on the VM box (quickstart)

```bash
# 1. generate the emulation for the small dynamic scenario  [sandbox-safe]
./setup_vagrant.sh scenarios/sm_dynamic_one_subnet.v2.yaml

# 2. bring up the 5 VMs (attacker + router + 3 targets)      [VM]
cd vagrant && vagrant up && cd ..

# 3. (after Phase 5 lands) evaluate a trained checkpoint against emulation
python experiments/eval_emulation.py \
    --scenario scenarios/sm_dynamic_one_subnet.v2.yaml \
    --checkpoint <trained_dhrl_checkpoint.pt>
```

## 5. Status

- [x] Phase 0 foundation: small dynamic scenario + verified small Vagrantfile.
- [ ] Phase 1 topology emulation (needs VM box).
- [ ] Phase 2 churn provisioning.
- [ ] Phase 3 netem provisioning.
- [ ] Phase 4 IDS sensor + response.
- [ ] Phase 5 eval bridge.
