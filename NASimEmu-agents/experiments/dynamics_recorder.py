"""Step 7: simulator validation event recorder (docs/eval_revision_plan.tex).

An opt-in event log. `HostVector.set_event_recorder(recorder)` and
`Network.set_event_recorder(recorder)` (src/nasimemu/nasim/envs/host_vector.py,
network.py) are the two hook points; both default to `None` and every call
site checks `is not None` before recording, so leaving this unset -- the
default, unaffecting every existing training/evaluation run -- costs one
attribute check per event and changes no existing behavior. Both are
class-level attributes (the same pattern `HostVector.ids_config`/
`scan_noise`/`churn_config` already use), not per-instance, because `Network`
and `HostVector` objects are recreated every episode
(`NASimEmuEnv._generate_env()`) -- a class-level recorder keeps accumulating
across episodes without needing to be re-wired into every fresh construction.

Recorded event kinds and fields
--------------------------------
  ids_increase     host, step, increase, level_before, level_after_decay, multiplier
                   -- one per update_detection() call, unconditionally (the
                   detection level rises and decays on every action against
                   a host, whether or not it crosses the threshold this step).
  ids_detected     host, step, response_type ('quarantine'/'patch'/'monitor')
                   -- one per threshold crossing; response_type is exactly
                   what _handle_detection() chose.
  scan_noise_trial scan_type, host, flipped (None / 'fp' / 'fn')
                   -- one per (host, feature-bit) pair evaluated in
                   _apply_scan_noise(), regardless of whether the bit flipped.
  churn_trial      service, host, step, fired (bool)
                   -- one per (service, step) churn-probability draw in
                   update_service_churn(), for a service that was actually
                   up and therefore eligible to go down this step.
  churn_recovery   service, host, step
                   -- one per service coming back up.
  timeout_trial    src, dest, action_type, step, fired (bool)
                   -- one per fresh timeout_probability draw in
                   Network.check_network_timeout() (only reached when no
                   timeout is already active for that key).
  timeout_blocked  src, dest, action_type, step, reason ('new'/'already_active')
                   -- one per action actually blocked by a timeout this step,
                   whichever reason. Not the same measurement as
                   timeout_trial: comparing *this* count's rate against the
                   configured timeout_probability would be biased upward,
                   since it also includes steps where no fresh draw happened
                   at all (an already-active timeout still blocks).
"""
from collections import Counter


class DynamicsRecorder:
    def __init__(self):
        self.events = []

    def record(self, kind, **fields):
        self.events.append({"kind": kind, **fields})

    def count(self, kind, **filters):
        return sum(
            1 for e in self.events
            if e["kind"] == kind and all(e.get(k) == v for k, v in filters.items())
        )

    def filtered(self, kind, **filters):
        return [
            e for e in self.events
            if e["kind"] == kind and all(e.get(k) == v for k, v in filters.items())
        ]

    def response_type_counts(self):
        return Counter(e["response_type"] for e in self.events if e["kind"] == "ids_detected")

    def level_trace(self, host):
        """(step, level_after_decay) pairs for one host, in recorded order --
        the one-host detection-level trace the plan calls for."""
        return [
            (e["step"], e["level_after_decay"])
            for e in self.events
            if e["kind"] == "ids_increase" and e["host"] == host
        ]

    def measured_rate(self, trial_kind, **filters):
        """(successes, trials, rate) for any *_trial event kind, whose
        `fired` field marks a Bernoulli-trial outcome -- the only event
        kinds directly comparable against a configured probability."""
        trials = self.filtered(trial_kind, **filters)
        n = len(trials)
        k = sum(1 for e in trials if e["fired"])
        return k, n, (k / n if n else float("nan"))

    def measured_flip_rate(self, direction):
        """(count, trials, rate) for scan_noise_trial's fp/fn flips.
        `direction` is 'fp' or 'fn'. The denominator is every trial where a
        flip in that direction was *possible*: an 'fp' trial only fires on a
        bit whose true_value was False, an 'fn' trial only on a bit whose
        true_value was True -- a bit that was already True can never
        register as a false positive and vice versa, so including those
        trials in the denominator would bias the measured rate downward
        relative to the configured false_positive_rate/false_negative_rate,
        which is itself conditioned on the bit's true value. Filtering by
        true_value directly (not by "flipped is None or <direction>") is
        required: flipped=None alone is ambiguous between "started False,
        stayed False" and "started True, stayed True"."""
        assert direction in ("fp", "fn")
        wants_true_value = False if direction == "fp" else True
        eligible = [e for e in self.events if e["kind"] == "scan_noise_trial"
                    and e["true_value"] == wants_true_value]
        n = len(eligible)
        k = sum(1 for e in eligible if e["flipped"] == direction)
        return k, n, (k / n if n else float("nan"))


def wilson_interval(k, n, z=1.96):
    """95% (default z) Wilson score confidence interval for a binomial
    proportion -- more reliable than the normal approximation at the small-n,
    extreme-p combinations these dynamics rates often fall into (e.g. a 1.5%
    configured churn probability over a few hundred trials)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half_width = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


class IdsCounter:
    """Recorder that keeps only per-episode IDS summaries and stores no events.

    A full DynamicsRecorder keeps every event, and a single policy episode
    produces roughly 90,000 churn/scan-noise trials; attaching that to every
    evaluation episode is wasteful when only IDS detections are wanted. This
    implements the same `record(kind, **fields)` interface but ignores every
    kind except the two IDS ones. Used by eval_harness.py's --record_ids.
    """

    RESPONSE_TYPES = ("quarantine", "patch", "monitor")

    def __init__(self):
        self.reset()

    def reset(self):
        self.detections = 0
        self.by_type = {t: 0 for t in self.RESPONSE_TYPES}
        self.first_step = -1
        self.max_level = 0.0
        self.increases = 0

    def record(self, kind, **fields):
        if kind == "ids_detected":
            self.detections += 1
            rt = fields.get("response_type")
            if rt in self.by_type:
                self.by_type[rt] += 1
            step = fields.get("step")
            if step is not None and (self.first_step < 0 or step < self.first_step):
                self.first_step = int(step)
        elif kind == "ids_increase":
            self.increases += 1
            level = fields.get("level_after_decay")
            if level is not None and level > self.max_level:
                self.max_level = float(level)

    def summary(self):
        return {"ids_detections": self.detections, "ids_quarantine": self.by_type["quarantine"],
                "ids_patch": self.by_type["patch"], "ids_monitor": self.by_type["monitor"],
                "ids_first_step": self.first_step, "ids_max_level": self.max_level,
                "ids_increases": self.increases}
