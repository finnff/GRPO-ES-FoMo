"""MMLU-Pro: 10-way multiple choice across academic domains.

The dataset ships no train split, so the spine's split names are remapped:
training rolls out on the big upstream ``test`` split and held-out eval uses
the small ``validation`` split — disjoint by construction.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

from grpo_es.tasks.base import TaskSpec, shuffle_take
from grpo_es.tasks.prompts import R1_SYSTEM

_DATASET_ID = "TIGER-Lab/MMLU-Pro"

# Spine split name -> upstream HF split.
_HF_SPLIT = {
    "train": "test",
    "test": "validation",
    "validation": "validation",
    "eval": "validation",
}


def _build_prompt(row: dict) -> str:
    options_block = "\n".join(
        f"{chr(65 + i)}. {option}" for i, option in enumerate(row["options"])
    )
    return (
        f"{R1_SYSTEM}\n\n"
        f"Question: {row['question']}\n\n"
        f"{options_block}\n\n"
        "Respond with <think>your reasoning</think> then "
        "<answer>\\boxed{X}</answer>, where X is the letter of the correct option."
    )


MMLU_PRO_SPEC = TaskSpec(
    name="mmlu_pro",
    rubric="mmlu_pro",
    system_prompt=R1_SYSTEM,
    build_prompt=_build_prompt,
    eval_split="validation",
    eval_size=None,  # the whole validation split — it's only 70 rows
    # Wider caps than gsm8k: 10 options inflate the prompt and the chain of
    # thought that walks through them.
    eval_max_prompt=1024,
    eval_max_new=1536,
)


def load_mmlu_pro(
    spec: TaskSpec,
    split: str = "train",
    seed: int = 0,
    max_samples: int | None = None,
) -> Dataset:
    raw = shuffle_take(load_dataset(_DATASET_ID, split=_HF_SPLIT[split]), seed, max_samples)

    def _to_row(ex: dict) -> dict:
        return {
            "prompt": spec.build_prompt(ex),
            "answer": ex["answer"],  # a single letter A-J
            "question": ex["question"],
            "category": ex["category"],  # kept for per-domain breakdowns
        }

    return raw.map(_to_row, remove_columns=raw.column_names)
