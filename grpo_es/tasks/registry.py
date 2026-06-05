"""Task registry: name -> (spec, loader). Adding a task = one module plus two
dict entries here; nothing downstream changes."""

from __future__ import annotations

from grpo_es.tasks.base import TaskSpec
from grpo_es.tasks.countdown import COUNTDOWN_SPEC, load_countdown
from grpo_es.tasks.gsm8k import GSM8K_SPEC, load_gsm8k
from grpo_es.tasks.mmlu_pro import MMLU_PRO_SPEC, load_mmlu_pro
from grpo_es.tasks.toy import TOY_SPEC, load_toy

SPECS: dict[str, TaskSpec] = {
    "toy": TOY_SPEC,
    "countdown": COUNTDOWN_SPEC,
    "gsm8k": GSM8K_SPEC,
    "mmlu_pro": MMLU_PRO_SPEC,
}

LOADERS = {
    "toy": load_toy,
    "countdown": load_countdown,
    "gsm8k": load_gsm8k,
    "mmlu_pro": load_mmlu_pro,
}


def get_task_spec(name: str) -> TaskSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known tasks: {sorted(SPECS)}") from None
