"""Step 3: post-hoc Pen-DHRL component interventions (docs/eval_revision_plan.tex).

Each condition uses the *same* checkpoint, same weights, same tensor
shapes -- only which mechanism fires during inference changes. Implemented
as four instance flags directly on NASimNetDHRL (nasim_problem/nasim_net_base_hrl.py:
`disable_ids_branch`, `disable_recurrent_memory`, `disable_goal_persistence`,
`disable_goal_conditioning`), each read at the exact point in forward() where
the corresponding mechanism actually takes effect, plus the pre-existing
`force_continue` flag reused for the fifth. This module is a thin,
consistent interface over those flags -- a context manager per condition,
not new net-internal logic -- because the actual hooks had to live inside
forward() itself: they touch local tensors (the goal vector's exact position
in a concatenation, the GRU hidden state before the message-passing call
that consumes it) that a wrapper outside forward() cannot reach without
fragile assumptions about internal tensor layout.

All four flags default to False; an untouched checkpoint's forward pass is
unaffected unless a condition below is explicitly applied. Restored on
exit even if the episode raises, so one condition's evaluation can never
silently leak into the next.

Conditions
----------
  full_checkpoint       reference condition, no flags set
  no_ids_branch          disable_ids_branch=True: the IDS-derived bias term
                         (self.ids_projection(ids_summary)) is zeroed before
                         being added to the manager's global context --
                         contrast with Step 2's IDS conditions, which edit
                         the *raw observation* the whole network sees;
                         this instead disables one internal computation
                         path after the observation has already been read,
                         so the IDS features still reach the rest of the
                         network (embed_node, the manager GNN's node
                         features) even though this one bias term does not.
  no_recurrent_memory    disable_recurrent_memory=True: the manager GRU's
                         hidden state is cleared before every single
                         forward() call, not just at episode start -- the
                         manager message-passing effectively becomes
                         stateless every step.
  no_goal_persistence    disable_goal_persistence=True: the subgoal
                         persistence counter is forced to 0 before every
                         decision (equivalent to an inference-time horizon
                         of H=1), so the manager must resample a subgoal
                         every single step instead of holding one for
                         self.goal_horizon steps.
  no_goal_conditioning   disable_goal_conditioning=True: the goal vector
                         reaching the worker (not the value/termination
                         head, which keeps its own separate goal_vectors
                         term) is replaced with zeros.
  no_learned_stopping    force_continue=True (pre-existing flag, reused
                         as-is): the learned termination decision never
                         fires; episodes always run to the step limit.
                         Results from this condition should be read
                         alongside fixed-budget metrics (Step 8), since
                         forcing continuation changes the trajectory
                         distribution, not just whether/when it stops.

Verification (belongs in "tests before large runs" for this module, same
principle as Step 1/2): each flag must be confirmed to actually change the
intended internal value (a direct unit check against forward()'s own
tensors) and, for the branch/conditioning-disabling flags, the resulting
action probabilities on a fixed input, without changing the unrelated
outputs the plan says are untouched (e.g. no_goal_conditioning must not
change the value head's output, since that keeps its own copy of the goal
vector). A performance decrease under a condition supports only that the
evaluated checkpoint functionally depends on the disabled mechanism at
inference time -- not that a separately retrained architecture lacking that
mechanism would learn as well or as differently (docs/eval_revision_plan.tex,
"Scope limit").
"""
from contextlib import contextmanager


CONDITIONS = (
    "full_checkpoint",
    "no_ids_branch",
    "no_recurrent_memory",
    "no_goal_persistence",
    "no_goal_conditioning",
    "no_learned_stopping",
)

# Follow-up A: goal-mask conditions on the manager's eight named subgoals
# (llm_teacher.goal_ontology.GOAL_NAMES, index k <-> GOAL_NAMES[k]). `mask_*`
# blocks goals; `only_*` forces a single goal (all others masked).
from llm_teacher.goal_ontology import GOAL_NAMES  # noqa: E402

_GOAL_INDEX = {name: i for i, name in enumerate(GOAL_NAMES)}
_IDS_GOALS = (_GOAL_INDEX["REDUCE_DETECTION"], _GOAL_INDEX["RECOVER_OR_REPLAN"])


def _allowed_excluding(*blocked):
    return tuple(k for k in range(len(GOAL_NAMES)) if k not in blocked)


GOAL_CONDITION_ALLOWED = {
    "mask_reduce_detection": _allowed_excluding(_GOAL_INDEX["REDUCE_DETECTION"]),
    "mask_recover_or_replan": _allowed_excluding(_GOAL_INDEX["RECOVER_OR_REPLAN"]),
    "mask_both_ids_goals": _allowed_excluding(*_IDS_GOALS),
}
GOAL_CONDITION_ALLOWED.update({f"only_{name.lower()}": (i,) for name, i in _GOAL_INDEX.items()})
GOAL_CONDITIONS = tuple(GOAL_CONDITION_ALLOWED)
ALL_CONDITIONS = CONDITIONS + GOAL_CONDITIONS

_COMPONENT_FLAGS = {
    "no_ids_branch": "disable_ids_branch",
    "no_recurrent_memory": "disable_recurrent_memory",
    "no_goal_persistence": "disable_goal_persistence",
    "no_goal_conditioning": "disable_goal_conditioning",
}


@contextmanager
def apply_condition(net, condition):
    """Apply one Step 3 condition to `net` for the duration of the `with`
    block, restoring every flag this module touches to its prior value on
    exit (including on an exception), so conditions can be evaluated back to
    back on the same net instance without leaking into each other."""
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"unknown component-intervention condition: {condition!r}")

    prior = {
        "disable_ids_branch": net.disable_ids_branch,
        "disable_recurrent_memory": net.disable_recurrent_memory,
        "disable_goal_persistence": net.disable_goal_persistence,
        "disable_goal_conditioning": net.disable_goal_conditioning,
        "force_continue": net.force_continue,
        "allowed_goals": net.allowed_goals,
    }
    try:
        net.set_component_interventions()  # all four off, i.e. full_checkpoint baseline
        net.set_force_continue(False)
        net.set_allowed_goals(None)

        if condition in _COMPONENT_FLAGS:
            setattr(net, _COMPONENT_FLAGS[condition], True)
        elif condition == "no_learned_stopping":
            net.set_force_continue(True)
        elif condition in GOAL_CONDITION_ALLOWED:
            net.set_allowed_goals(GOAL_CONDITION_ALLOWED[condition])
        # full_checkpoint: nothing further to set, already the baseline above

        yield net
    finally:
        net.set_component_interventions(
            disable_ids_branch=prior["disable_ids_branch"],
            disable_recurrent_memory=prior["disable_recurrent_memory"],
            disable_goal_persistence=prior["disable_goal_persistence"],
            disable_goal_conditioning=prior["disable_goal_conditioning"],
        )
        net.set_force_continue(prior["force_continue"])
        net.allowed_goals = prior["allowed_goals"]
