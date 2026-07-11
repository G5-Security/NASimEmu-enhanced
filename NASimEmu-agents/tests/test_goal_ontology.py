"""Master plan Sec 15.13. Guards the one invariant everything else in
llm_teacher/ depends on: the fixed ontology stays a fixed 8-slot binding
matching NASimNetDHRL's default num_subgoals."""
from llm_teacher.goal_ontology import GOAL_NAMES, GOAL_INDEX, GOAL_ONTOLOGY_VERSION


def test_eight_names_matching_num_subgoals_default():
    assert len(GOAL_NAMES) == 8


def test_names_are_unique():
    assert len(set(GOAL_NAMES)) == len(GOAL_NAMES)


def test_index_is_exact_inverse_of_names():
    assert len(GOAL_INDEX) == len(GOAL_NAMES)
    for i, name in enumerate(GOAL_NAMES):
        assert GOAL_INDEX[name] == i
    for name, i in GOAL_INDEX.items():
        assert GOAL_NAMES[i] == name


def test_version_is_a_positive_int():
    assert isinstance(GOAL_ONTOLOGY_VERSION, int)
    assert GOAL_ONTOLOGY_VERSION >= 1


def test_expected_names_present():
    # Master plan Table 15.1 -- pinning the literal names guards against a
    # silent rename that would desync goal_bank slot k's *meaning* from
    # every prompt/summary/checkpoint that assumes the current ontology.
    expected = {
        "DISCOVER_SUBNET", "ENUMERATE_HOST", "GAIN_INITIAL_ACCESS",
        "ESCALATE_PRIVILEGE", "PIVOT", "CAPTURE_SENSITIVE_HOST",
        "REDUCE_DETECTION", "RECOVER_OR_REPLAN",
    }
    assert set(GOAL_NAMES) == expected
