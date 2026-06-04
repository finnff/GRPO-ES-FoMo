"""TaskSpec: the contract every task implements.

A task is a prompt builder plus a rubric name; the dataset loaders all share
one signature so both method legs (and later the eval runner) can stay
task-agnostic. Datasets always carry a ``prompt`` and an ``answer`` column;
anything else a rubric needs rides along as extra columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets import Dataset


@dataclass(frozen=True)
class TaskSpec:
    name: str
    rubric: str  # key into rewards.registry
    system_prompt: str
    build_prompt: Callable[[dict], str]


def build_dataset(
    spec: TaskSpec,
    split: str = "train",
    seed: int = 0,
    max_samples: int | None = None,
) -> Dataset:
    # Local import: the registry imports the task modules, which import this
    # module — a top-level import here would be circular.
    from grpo_es.tasks.registry import LOADERS

    loader = LOADERS[spec.name]
    return loader(spec, split=split, seed=seed, max_samples=max_samples)
