"""Toy task: concatenate the last letters of three words.

Trivially verifiable and generated on the fly — exists purely as a smoke
target for the training loop, not as a benchmark.
"""

from __future__ import annotations

import random

from datasets import Dataset

from grpo_es.tasks.base import TaskSpec
from grpo_es.tasks.prompts import TOY_SYSTEM

_WORDS = ("cat", "dog", "fish", "bird")
_WORDS_PER_ROW = 3


def _build_prompt(row: dict) -> str:
    return (
        f"{TOY_SYSTEM}\n\n"
        f"Concatenate the last letters of: {row['words']}\n"
        "Example format: <think>...</think><answer>xy</answer>"
    )


TOY_SPEC = TaskSpec(
    name="toy",
    rubric="toy",
    system_prompt=TOY_SYSTEM,
    build_prompt=_build_prompt,
    # Generated data has no real splits: a fresh generator seed IS the
    # held-out set (the loader ignores `split`).
    eval_seed=999,
    eval_max_new=128,
)


def load_toy(
    spec: TaskSpec,
    split: str = "train",
    seed: int = 0,
    max_samples: int | None = 32,
) -> Dataset:
    """Generate `max_samples` rows; `split` is ignored (seed-driven data)."""
    rng = random.Random(seed)
    n = 32 if max_samples is None else max_samples
    rows = []
    for _ in range(n):
        words = [rng.choice(_WORDS) for _ in range(_WORDS_PER_ROW)]
        row = {"words": ", ".join(words)}
        rows.append(
            {
                "prompt": spec.build_prompt(row),
                "answer": "".join(w[-1] for w in words),
                "words": row["words"],
            }
        )
    return Dataset.from_list(rows)
