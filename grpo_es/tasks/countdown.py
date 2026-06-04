"""Countdown: reach a target number using each given operand exactly once.

Generated on the fly like the toy task, but with a real search space —
useful as a first non-trivial reward before the benchmark tasks land.
"""

from __future__ import annotations

import random

from datasets import Dataset

from grpo_es.tasks.base import TaskSpec
from grpo_es.tasks.prompts import R1_SYSTEM

_NUM_OPERANDS = 4
_OPERAND_RANGE = (1, 9)
_TARGET_RANGE = (10, 30)


def _build_prompt(row: dict) -> str:
    return (
        f"{R1_SYSTEM}\n\n"
        f"Use each of {row['numbers']} exactly once with + - * / to reach {row['target']}.\n"
        "Respond with <think>...</think><answer>expression</answer>."
    )


COUNTDOWN_SPEC = TaskSpec(
    name="countdown",
    rubric="countdown",
    system_prompt=R1_SYSTEM,
    build_prompt=_build_prompt,
)


def load_countdown(
    spec: TaskSpec,
    split: str = "train",
    seed: int = 0,
    max_samples: int | None = 64,
) -> Dataset:
    """Generate `max_samples` rows; `split` is ignored (seed-driven data).

    Targets are sampled independently of the operands, so some rows may be
    unsolvable — that's fine for a reward signal (they score 0 for everyone).
    """
    rng = random.Random(seed)
    n = 64 if max_samples is None else max_samples
    rows = []
    for _ in range(n):
        numbers = [rng.randint(*_OPERAND_RANGE) for _ in range(_NUM_OPERANDS)]
        target = rng.randint(*_TARGET_RANGE)
        row = {"numbers": numbers, "target": target}
        rows.append(
            {
                "prompt": spec.build_prompt(row),
                "answer": str(target),
                "numbers": numbers,
                "target": target,
            }
        )
    return Dataset.from_list(rows)
