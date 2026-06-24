"""Rubric registry + assembly of the TRL reward-function list.

``get_rubric`` is the single place a task name turns into a scoring object;
``make_trl_reward_funcs`` is the single place the reward list (and its
weights) is assembled. Keep both here — duplicated assembly logic in the
method legs is exactly the kind of drift this layer exists to prevent.
"""

from __future__ import annotations

from typing import Callable

from verifiers.rubrics.math_rubric import MathRubric
from verifiers.rubrics.rubric import Rubric

from grpo_es.rewards.ascii_tree_glyphnorm import AsciiTreeGlyphNormRubric
from grpo_es.rewards.countdown_rubric import CountdownRubric
from grpo_es.rewards.mcq_rubric import MCQLetterRubric
from grpo_es.rewards.pydantic_graded import PydanticGradedRubric
from grpo_es.rewards.toy_rubric import ToyRubric
from grpo_es.rewards.trl_bridge import format_reward_func, rubric_reward_func

# Per-rollout cap on math_verify's symbolic check; runaway parses otherwise
# stall the whole reward batch.
_MATH_TIMEOUT_S = 5


def _math_rubric() -> Rubric:
    return MathRubric(timeout_seconds=_MATH_TIMEOUT_S)


_FACTORIES: dict[str, Callable[[], Rubric]] = {
    "toy": ToyRubric,
    "countdown": CountdownRubric,
    "gsm8k": _math_rubric,
    "mmlu_pro": MCQLetterRubric,
    # Local graded overrides for hub envs whose stock reward is near-binary;
    # wired in via ENV_CUSTOMIZATIONS (tasks/from_env.py). Static so they
    # resolve at both train and eval.
    "pydantic_graded": PydanticGradedRubric,
    "ascii_tree_glyphnorm": AsciiTreeGlyphNormRubric,
}

# Hub-env tasks register their rubric factory here at task-registration time
# (tasks/from_env.py); _FACTORIES above stays the closed set of built-ins.
DYNAMIC_RUBRICS: dict[str, Callable[[], Rubric]] = {}


def register_rubric(name: str, factory: Callable[[], Rubric]) -> None:
    DYNAMIC_RUBRICS[name] = factory

# Shadow rubrics: scored and logged every step but weighted 0, so they shape
# nothing and cost only compute. mmlu_pro shadows the math rubric to measure
# its false-negative rate on letter answers (the reason MCQLetterRubric
# exists — see its docstring); drop once the gap is quantified.
_SHADOW_FACTORIES: dict[str, list[Callable[[], Rubric]]] = {
    "mmlu_pro": [_math_rubric],
}


def get_rubric(name: str) -> Rubric:
    if name in _FACTORIES:
        return _FACTORIES[name]()
    if name in DYNAMIC_RUBRICS:
        return DYNAMIC_RUBRICS[name]()
    raise KeyError(
        f"unknown rubric {name!r}; known: {sorted(_FACTORIES) + sorted(DYNAMIC_RUBRICS)}"
    )


def make_trl_reward_funcs(
    rubric_name: str,
    *,
    use_format_reward: bool = True,
    reward_weights: list[float] | None = None,
) -> tuple[list[Callable], list[float] | None]:
    """Build ``(reward_funcs, reward_weights)`` for the TRL trainer.

    Order is [task, format?, shadows...]. With a single function the weights
    come back as None (TRL's uniform default); a default-shaped [task, format]
    weight pair is tolerated when the format reward is dropped, anything else
    mismatched raises. User-supplied weights never cover the shadows — those
    are always appended at 0.0.
    """
    funcs: list[Callable] = [rubric_reward_func(get_rubric(rubric_name))]
    weights = [1.0]
    if use_format_reward:
        funcs.append(format_reward_func)
        weights.append(0.5)

    if reward_weights is not None:
        wanted = list(reward_weights)
        if not use_format_reward and len(wanted) == len(funcs) + 1:
            wanted = wanted[: len(funcs)]
        if len(wanted) != len(funcs):
            raise ValueError(
                f"got {len(reward_weights)} reward weights for {len(funcs)} "
                f"reward functions (order is [task, format])"
            )
        weights = wanted

    for factory in _SHADOW_FACTORIES.get(rubric_name, []):
        funcs.append(rubric_reward_func(factory()))  # logs under the class name
        weights.append(0.0)

    return funcs, (weights if len(funcs) > 1 else None)
