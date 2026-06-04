"""Rubric registry + assembly of the TRL reward-function list.

``get_rubric`` is the single place a task name turns into a scoring object;
``make_trl_reward_funcs`` is the single place the reward list (and its
weights) is assembled. Keep both here — duplicated assembly logic in the
method legs is exactly the kind of drift this layer exists to prevent.
"""

from __future__ import annotations

from typing import Callable

from verifiers.rubrics.rubric import Rubric

from grpo_es.rewards.countdown_rubric import CountdownRubric
from grpo_es.rewards.toy_rubric import ToyRubric
from grpo_es.rewards.trl_bridge import format_reward_func, rubric_reward_func

_FACTORIES: dict[str, Callable[[], Rubric]] = {
    "toy": ToyRubric,
    "countdown": CountdownRubric,
}


def get_rubric(name: str) -> Rubric:
    try:
        return _FACTORIES[name]()
    except KeyError:
        raise KeyError(f"unknown rubric {name!r}; known: {sorted(_FACTORIES)}") from None


def make_trl_reward_funcs(
    rubric_name: str,
    *,
    use_format_reward: bool = True,
    reward_weights: list[float] | None = None,
) -> tuple[list[Callable], list[float] | None]:
    """Build ``(reward_funcs, reward_weights)`` for the TRL trainer.

    Order is [task, format?]. With a single function the weights come back as
    None (TRL's uniform default); a default-shaped [task, format] weight pair
    is tolerated when the format reward is dropped, anything else mismatched
    raises.
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

    return funcs, (weights if len(funcs) > 1 else None)
