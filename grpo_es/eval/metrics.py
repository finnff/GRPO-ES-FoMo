"""Scoring a batch of completions outside the training loop.

This is the single scoring path for anything eval-shaped: it goes through the
same ``VerifiersRubricAdapter`` the trainer's reward funcs use, so a number
produced here is directly comparable to the training reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verifiers.rubrics.rubric import Rubric

from grpo_es.rewards.trl_bridge import VerifiersRubricAdapter, format_reward_func


@dataclass
class EvalMetrics:
    mean_reward: float = 0.0
    accuracy: float = 0.0  # fraction of completions with reward >= 1.0
    format_pass: float = 0.0
    mean_length: float = 0.0  # chars
    n: int = 0
    per_sample: list[dict] = field(default_factory=list)


def score_completions(
    rubric: Rubric,
    prompts: list[str],
    completions: list[str],
    answers: list[str],
    **extra_columns: Any,
) -> EvalMetrics:
    """Score completions against a rubric; extra columns are per-sample lists
    threaded through to the rubric by name (numbers, target, ...)."""
    adapter = VerifiersRubricAdapter(rubric)
    fmt = format_reward_func(completions)

    rows: list[tuple[str, str, dict]] = []
    for i, (prompt, completion, answer) in enumerate(zip(prompts, completions, answers)):
        columns = {key: values[i] for key, values in extra_columns.items()}
        columns["answer"] = answer
        rows.append((prompt, completion, columns))
    rewards = adapter.score_batch(rows)

    per_sample = [
        {"reward": reward, "format": fmt[i], "length": len(completion)}
        for i, (reward, completion) in enumerate(zip(rewards, completions))
    ]

    n = len(prompts) or 1
    return EvalMetrics(
        mean_reward=sum(rewards) / n,
        accuracy=sum(1 for r in rewards if r >= 1.0) / n,
        format_pass=sum(fmt) / n,
        mean_length=sum(len(c) for c in completions) / n,
        n=len(prompts),
        per_sample=per_sample,
    )
