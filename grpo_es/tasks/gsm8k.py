"""GSM8K: grade-school math word problems, gold answer after ``####``.

First real benchmark task — small models are far from saturated on it, which
is exactly what an optimizer comparison needs.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset
from verifiers.utils.data_utils import extract_hash_answer

from grpo_es.tasks.base import TaskSpec, shuffle_take
from grpo_es.tasks.prompts import R1_SYSTEM

_DATASET_ID = "openai/gsm8k"
_DATASET_CONFIG = "main"


def _build_prompt(row: dict) -> str:
    return (
        f"{R1_SYSTEM}\n\n"
        f"Problem: {row['question']}\n\n"
        "Respond with <think>your reasoning</think> then "
        "<answer>\\boxed{final_number}</answer>."
    )


GSM8K_SPEC = TaskSpec(
    name="gsm8k",
    rubric="gsm8k",
    system_prompt=R1_SYSTEM,
    build_prompt=_build_prompt,
    eval_split="test",  # train/test are disjoint upstream
    eval_size=300,
    eval_max_prompt=512,
    eval_max_new=768,
)


def load_gsm8k(
    spec: TaskSpec,
    split: str = "train",
    seed: int = 0,
    max_samples: int | None = None,
) -> Dataset:
    raw = shuffle_take(load_dataset(_DATASET_ID, _DATASET_CONFIG, split=split), seed, max_samples)

    def _to_row(ex: dict) -> dict:
        return {
            "prompt": spec.build_prompt(ex),
            # The gold field is full worked reasoning ending in "#### N";
            # the rubric only ever compares against N.
            "answer": extract_hash_answer(ex["answer"]),
            "question": ex["question"],
        }

    return raw.map(_to_row, remove_columns=raw.column_names)
