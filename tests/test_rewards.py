import pytest

from grpo_es.rewards.mcq_rubric import extract_choice, normalize_letter
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


def test_bridge_forwards_non_list_columns():
    # datasets>=4 hands the ES leg a lazy Column, not a list; the seam must
    # still forward per-sample columns or the rubric scores against missing
    # data (empty answer -> every completion wrong). A tuple stands in for any
    # non-list length-n sequence.
    func = rubric_reward_func(get_rubric("toy"))
    prompts = ["Concatenate the last letters of: cat, dog"] * 2
    completions = ["<think>...</think><answer>tg</answer>", "<answer>xx</answer>"]
    assert func(prompts, completions, answer=("tg", "tg")) == [1.0, 0.0]


def test_toy_rubric_rejects_bare_answer():
    # The bare-answer reward hack: a tag-less "tg" must NOT score — only an
    # answer delivered inside the <answer> scaffold earns task reward.
    func = rubric_reward_func(get_rubric("toy"))
    prompts = ["Concatenate the last letters of: cat, dog"] * 3
    completions = [
        "<think>...</think><answer>tg</answer>",  # tagged, correct  -> 1.0
        "tg",  # bare correct answer, no tags  -> 0.0 (used to be 1.0)
        "<answer>xx</answer>",  # tagged, wrong  -> 0.0
    ]
    assert func(prompts, completions, answer=["tg", "tg", "tg"]) == [1.0, 0.0, 0.0]


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


# --- gsm8k (math rubric) -----------------------------------------------------


def test_gsm8k_rubric_verifies_boxed_number():
    func = rubric_reward_func(get_rubric("gsm8k"))
    completions = [
        "<think>2+2</think><answer>\\boxed{4}</answer>",
        "<answer>\\boxed{4.0}</answer>",  # symbolic equivalence, not string match
        "<think>2+2</think><answer>\\boxed{5}</answer>",
    ]
    scores = func(["q"] * 3, completions, answer=["4"] * 3)
    assert scores == [1.0, 1.0, 0.0]


# --- mmlu_pro (MCQ letter rubric) --------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C", "C"),
        ("c", "C"),
        ("(C)", "C"),
        ("C.", "C"),
        ("**C**", "C"),
        ("C) Paris", "C"),
        ("\\text{C}", "C"),
        ("Paris", ""),  # no standalone letter — must NOT read the P
    ],
)
def test_normalize_letter(raw, expected):
    assert normalize_letter(raw) == expected


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<think>r</think><answer>\\boxed{C}</answer>", 1.0),
        ("<think>r</think><answer>\\boxed{C.}</answer>", 1.0),
        ("<think>r</think><answer>\\boxed{**C**}</answer>", 1.0),
        ("<think>r</think><answer>C) some option text</answer>", 1.0),
        ("<answer>C</answer>", 1.0),  # no box, answer-tag fallback
        ("<think>r</think><answer>\\boxed{D}</answer>", 0.0),
        ("<answer>none of these</answer>", 0.0),
    ],
)
def test_mcq_rubric_scores(completion, expected):
    func = rubric_reward_func(get_rubric("mmlu_pro"))
    assert func(["q"], [completion], answer=["C"]) == [expected]


def test_mcq_ignores_letters_inside_think():
    text = "<think>maybe \\boxed{A}, no...</think><answer>\\boxed{C}</answer>"
    assert extract_choice(text) == "C"


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


def test_mmlu_shadow_rubric_appended_at_zero_weight():
    # The math rubric rides along on mmlu_pro purely for logging: it
    # quantifies its own false-negative rate against the letter rubric.
    funcs, weights = make_trl_reward_funcs("mmlu_pro")
    assert [f.__name__ for f in funcs] == [
        "MCQLetterRubric",
        "format_reward_func",
        "MathRubric",
    ]
    assert weights == [1.0, 0.5, 0.0]


def test_gsm8k_has_no_shadow():
    funcs, weights = make_trl_reward_funcs("gsm8k")
    assert [f.__name__ for f in funcs] == ["MathRubric", "format_reward_func"]
    assert weights == [1.0, 0.5]


def test_unknown_rubric_raises():
    with pytest.raises(KeyError, match="unknown rubric"):
        get_rubric("nope")
