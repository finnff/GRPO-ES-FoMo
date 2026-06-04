import pytest

from grpo_es.rewards.registry import get_rubric, make_trl_reward_funcs
from grpo_es.rewards.trl_bridge import format_reward_func, rubric_reward_func


# --- format reward -----------------------------------------------------------


def test_format_reward_exact_and_miss():
    good = "<think>ab</think><answer>x</answer>"
    bad = "no tags at all"
    assert format_reward_func([good, bad]) == [1.0, 0.0]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<think>only thinking</think>", 0.25),
        ("prose <think>a</think> prose <answer>b</answer> prose", 0.75),
        ("<answer>b</answer> then <think>a</think>", 0.5),
    ],
)
def test_format_reward_partial_credit(completion, expected):
    assert format_reward_func([completion]) == [expected]


# --- rubrics through the TRL bridge -----------------------------------------


def test_toy_rubric_exact_match():
    func = rubric_reward_func(get_rubric("toy"))
    prompts = ["Concatenate the last letters of: cat, dog"] * 2
    completions = ["<think>...</think><answer>tg</answer>", "<answer>xx</answer>"]
    assert func(prompts, completions, answer=["tg", "tg"]) == [1.0, 0.0]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2*3+4+1", 1.0),  # hits 11 with the right multiset
        ("2*3+4", 0.0),  # missing an operand
        ("2*3+4+2", 0.0),  # wrong operand
        ("2*/3", 0.0),  # syntax error
        ("__import__('os')", 0.0),  # disallowed node, not an eval()
    ],
)
def test_countdown_rubric(expr, expected):
    func = rubric_reward_func(get_rubric("countdown"))
    scores = func(
        ["reach 11"],
        [f"<think>...</think><answer>{expr}</answer>"],
        answer=["11"],
        numbers=[[2, 3, 4, 1]],
        target=[11],
    )
    assert scores == [expected]


def test_countdown_division_tolerance():
    func = rubric_reward_func(get_rubric("countdown"))
    # 7/7 + 9 + 1 = 11 exercises true division landing within 1e-6.
    scores = func(
        ["reach 11"],
        ["<answer>7/7 + 9 + 1</answer>"],
        answer=["11"],
        numbers=[[7, 7, 9, 1]],
        target=[11],
    )
    assert scores == [1.0]


def test_countdown_zero_division_scores_zero():
    func = rubric_reward_func(get_rubric("countdown"))
    scores = func(
        ["reach 11"],
        ["<answer>3/(2-2)+4</answer>"],
        answer=["11"],
        numbers=[[3, 2, 2, 4]],
        target=[11],
    )
    assert scores == [0.0]


# --- reward list assembly ----------------------------------------------------


def test_make_trl_reward_funcs_task_plus_format():
    funcs, weights = make_trl_reward_funcs("toy")
    assert len(funcs) == 2
    assert weights == [1.0, 0.5]


def test_make_trl_reward_funcs_single_func_uniform_weights():
    funcs, weights = make_trl_reward_funcs("toy", use_format_reward=False)
    assert len(funcs) == 1
    assert weights is None


def test_make_trl_reward_funcs_trims_default_pair_without_format():
    # The dataclass default [task, format] should be tolerated when the
    # format reward is dropped.
    funcs, weights = make_trl_reward_funcs(
        "toy", use_format_reward=False, reward_weights=[1.0, 0.5]
    )
    assert len(funcs) == 1
    assert weights is None


def test_make_trl_reward_funcs_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="reward weights"):
        make_trl_reward_funcs("toy", reward_weights=[1.0, 0.5, 0.25])


def test_unknown_rubric_raises():
    with pytest.raises(KeyError, match="unknown rubric"):
        get_rubric("nope")
