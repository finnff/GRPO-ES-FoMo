"""Task registry: name -> (spec, loader). Adding a task = one module plus two
dict entries here; nothing downstream changes."""

from __future__ import annotations

from grpo_es.tasks.base import TaskSpec
from grpo_es.tasks.countdown import COUNTDOWN_SPEC, load_countdown
from grpo_es.tasks.toy import TOY_SPEC, load_toy

SPECS: dict[str, TaskSpec] = {
    "toy": TOY_SPEC,
    "countdown": COUNTDOWN_SPEC,
}

LOADERS = {
    "toy": load_toy,
    "countdown": load_countdown,
}


def get_task_spec(name: str) -> TaskSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known tasks: {sorted(SPECS)}") from None
