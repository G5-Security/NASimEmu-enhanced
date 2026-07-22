"""
DAgger-style aggregation loop -- master plan Sec 15.5/15.8: "state coverage
beyond what a single fixed teacher-labeling pass would see is addressed with
a DAgger-style aggregation loop (label states visited by the trained
student, not only by the original labeling policies, and retrain)."

Each round:
  1. Collect --records_per_round new teacher-labeled states via
     llm_teacher.label_states, using the PREVIOUS round's trained checkpoint
     as the rollout policy (round 0 has no checkpoint yet, so it falls back
     to --policy random) -- this is the actual DAgger move: label states the
     current student visits, not just the original labeling policies'.
     Collection runs with --hardest_stage by default (Phase 2/3): the labeling
     env is pinned to the final, IDS-on curriculum stage so the visited states
     carry detection pressure and the teacher can emit REDUCE_DETECTION -- the
     stealth labels the curriculum-aligned distillation (main.py
     --llm_distill_rewarm_scale) is built to revive when IDS turns on.
  2. Append those records into the same --out_dir (true aggregation, per
     "DAgger" = Dataset Aggregation -- prior rounds' data is never discarded)
     and re-split at the episode level (llm_teacher/split_dataset.py) so the
     newly aggregated records get a train/test assignment too.
  3. Train (via main.py --llm_distill, as a subprocess -- main.py is a
     top-level training script, not designed to be imported and called).
  4. Locate the checkpoint main.py just saved and copy it to a stable path
     for bookkeeping and as next round's --policy checkpoint source.

This is an orchestration script for real, multi-hour training runs -- it is
NOT something to smoke-test end-to-end on a laptop; --dry_run prints every
command it would execute without running any of them, which is what this
project's own development session validated instead of a live run.

Run from NASimEmu-agents/ (on a workstation with real training budget):
    python experiments/dagger_loop.py --rounds 3 --records_per_round 500 \\
        --scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \\
        --train_args "-device cuda -cpus 32 -epoch 100 -max_epochs 200 --no_debug \\
            -net_class NASimNetDHRL -use_a_t -episode_step_limit 400 \\
            -observation_format graph_v2 -lr 0.0007 -alpha_h 0.02 --llm_distill"
"""
import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_teacher.dataset_writer import DatasetWriter  # noqa: E402
from llm_teacher.split_dataset import split_dataset  # noqa: E402

AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _latest_wandb_checkpoint(after_ts):
    """main.py saves os.path.join(wandb.run.dir, 'model.pt') every epoch
    (always the latest, even if that epoch regressed) and, since the
    best-checkpoint fix, 'model_best.pt' whenever config.save_best_metric
    actually improved -- wandb.run.dir's exact name is only known inside
    that process, so both are located here by scanning for the most recently
    modified wandb/*/files/{model_best.pt,model.pt} created after this
    round's training subprocess started. Fragile by nature (depends on
    wandb's own directory layout, unchanged since at least the version
    pinned in this repo) -- documented, not hidden.

    Prefers model_best.pt: the next round's collection policy (and the
    round after that's warm-start via -load_model) should come from the
    best checkpoint this round produced, not whatever the last epoch
    happened to land on -- otherwise a late-round regression (e.g. from
    the teacher weight decaying to 0 and pure RL drifting) propagates
    straight into the next round's data collection."""
    def _candidates(name):
        found = glob.glob(os.path.join(AGENTS_DIR, "wandb", "*", "files", name))
        return [c for c in found if os.path.getmtime(c) >= after_ts]

    best_candidates = _candidates("model_best.pt")
    if best_candidates:
        return max(best_candidates, key=os.path.getmtime)

    fallback_candidates = _candidates("model.pt")
    if not fallback_candidates:
        raise RuntimeError(
            "No wandb/*/files/model.pt or model_best.pt found with mtime after this round's "
            "training started -- either training didn't reach its first checkpoint save, or "
            "wandb's directory layout changed. Check the training subprocess's own output above."
        )
    print("[dagger_loop] WARNING: no model_best.pt found for this round (eval metric may have "
          "never improved past -inf, e.g. if -save_best_split's eval never ran) -- falling back "
          "to the latest per-epoch model.pt instead.")
    return max(fallback_candidates, key=os.path.getmtime)


def _compute_target_records(out_dir_abs, records_per_round):
    """label_states.py's --target_records is an ABSOLUTE floor on the
    dataset's total valid-record count (it stops once writer.count_valid()
    reaches it), not "how many new records this call should add". Passing
    records_per_round directly here would work for round 0 (dataset starts
    empty) but silently collect ZERO new records in every later round once
    the dataset already holds >= records_per_round from prior rounds --
    compute the absolute target explicitly instead. Split out as its own
    function so the accumulation math is unit-testable without running a
    real collection subprocess."""
    existing_valid = DatasetWriter(out_dir_abs).count_valid() if os.path.isdir(out_dir_abs) else 0
    return existing_valid, existing_valid + records_per_round


def _build_collect_cmd(args, prev_checkpoint, target_records):
    """Assemble the llm_teacher.label_states command for one round's
    collection. Split out (like _compute_target_records) so the command shape
    -- especially that IDS-on collection (--hardest_stage) and the on-policy
    checkpoint rollout are actually wired through -- is unit-testable without
    launching a real collection subprocess.

    --hardest_stage is the Phase 2/3 alignment move: it pins the labeling env
    to the final, IDS-on curriculum stage so the states the student visits (and
    the teacher labels) actually contain detection pressure. Without it, DAgger
    relabels IDS-off baseline states and the aggregated dataset never gains the
    REDUCE_DETECTION examples the curriculum-aligned distillation needs."""
    policy = "checkpoint" if prev_checkpoint is not None else "random"
    collect_cmd = [
        sys.executable, "-m", "llm_teacher.label_states",
        "--scenario", args.scenario,
        "--target_records", str(target_records),
        "--out_dir", args.out_dir,
        "--policy", policy,
        "--teacher_backend", args.teacher_backend,
    ]
    if args.hardest_stage:
        collect_cmd += ["--hardest_stage"]
    if prev_checkpoint is not None:
        collect_cmd += ["--checkpoint_path", prev_checkpoint]
    if args.model:
        collect_cmd += ["--model", args.model]
    return collect_cmd


def run_round(round_idx, args, prev_checkpoint):
    print(f"\n{'='*80}\n[dagger_loop] Round {round_idx}/{args.rounds - 1}\n{'='*80}")

    # Step 1-2: collect + aggregate.
    out_dir_abs = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(AGENTS_DIR, args.out_dir)
    existing_valid, target_records = _compute_target_records(out_dir_abs, args.records_per_round)

    collect_cmd = _build_collect_cmd(args, prev_checkpoint, target_records)
    print(f"[dagger_loop] round {round_idx} collection: {existing_valid} existing -> {target_records} target: "
          f"{' '.join(shlex.quote(c) for c in collect_cmd)}")
    if not args.dry_run:
        subprocess.run(collect_cmd, cwd=AGENTS_DIR, check=True)

        # Re-split after every round's aggregation, not just once at the
        # start -- otherwise this round's newly-appended records have no
        # "split" field at all (load_dataset(split=...) would then raise,
        # per dataset_reader.py's own guard) or, if an earlier ad-hoc split
        # is reused, they'd train unsplit while looking split. Same seed
        # every round keeps prior episodes' train/test assignment stable
        # (split_dataset.py hashes by episode_id, not position -- see its
        # own docstring) while newly aggregated episodes get assigned too.
        split_dataset(out_dir_abs, test_frac=args.test_frac, seed=args.split_seed)

    # Step 3: train. Always against the train split, now that every round
    # re-splits right before this runs -- training on the unsplit set would
    # mean "held-out" evaluation later is measuring states the distillation
    # loss was itself trained against.
    train_cmd = [sys.executable, "main.py"] + shlex.split(args.train_args)
    train_cmd += ["--llm_distill_dataset", args.out_dir, "--llm_distill_split", "train"]
    if prev_checkpoint is not None:
        train_cmd += ["-load_model", prev_checkpoint]
    print(f"[dagger_loop] round {round_idx} training: {' '.join(shlex.quote(c) for c in train_cmd)}")
    train_start = time.time()
    if not args.dry_run:
        subprocess.run(train_cmd, cwd=AGENTS_DIR, check=True)

    # Step 4: locate + stash the checkpoint this round produced
    if args.dry_run:
        return os.path.join(args.checkpoint_dir, f"round_{round_idx}.pt")

    latest = _latest_wandb_checkpoint(after_ts=train_start)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    round_ckpt = os.path.join(args.checkpoint_dir, f"round_{round_idx}.pt")
    shutil.copy2(latest, round_ckpt)
    print(f"[dagger_loop] round {round_idx} checkpoint: {latest} -> {round_ckpt}")
    return round_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--records_per_round", type=int, default=500)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--train_args", required=True,
                     help="Everything main.py needs besides --llm_distill_dataset/-load_model "
                          "(this script adds those). Must include --llm_distill and -net_class NASimNetDHRL.")
    ap.add_argument("--out_dir", default=os.path.join("training_data", "llm_teacher_dataset"))
    ap.add_argument("--checkpoint_dir", default=os.path.join("training_data", "dagger_checkpoints"))
    ap.add_argument("--test_frac", type=float, default=0.2, help="Re-applied every round after aggregation, see llm_teacher/split_dataset.py")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--hardest_stage", action=argparse.BooleanOptionalAction, default=True,
                     help="Collect each round with the labeling env pinned to the final, IDS-on "
                          "curriculum stage (label_states.py --hardest_stage). ON by default: the "
                          "aggregated dataset only gains REDUCE_DETECTION examples if the states the "
                          "student visits actually contain detection pressure. Use --no-hardest_stage "
                          "for the original IDS-off baseline collection.")
    ap.add_argument("--teacher_backend", choices=["llm", "heuristic", "llm_cascade"], default="llm", help="Passed through to every round's llm_teacher.label_states call")
    ap.add_argument("--model", default=None, help="Passed through to every round's llm_teacher.label_states call (ignored for --teacher_backend heuristic)")
    ap.add_argument("--initial_checkpoint", default=None,
                     help="Skip round 0's cold start -- collect round 0 with this checkpoint instead of --policy random")
    ap.add_argument("--dry_run", action="store_true", help="Print every command without running any of them")
    args = ap.parse_args()

    if "--llm_distill" not in shlex.split(args.train_args):
        raise SystemExit("--train_args must include --llm_distill (this loop only makes sense for that condition)")
    if "--llm_distill_dataset" in shlex.split(args.train_args) or "--llm_distill_split" in shlex.split(args.train_args):
        raise SystemExit("--train_args must not set --llm_distill_dataset/--llm_distill_split -- this script "
                          "always adds --llm_distill_dataset=--out_dir and --llm_distill_split=train itself")

    checkpoint = args.initial_checkpoint
    for round_idx in range(args.rounds):
        checkpoint = run_round(round_idx, args, checkpoint)

    print(f"\n[dagger_loop] done. Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
