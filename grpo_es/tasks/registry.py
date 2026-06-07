"""Task registry: name -> (spec, loader). Adding a task = one module plus two
dict entries here; nothing downstream changes. Hub-env tasks — the raw
``env:<owner>/<env>`` form and the named hub-backed ones (ifeval, ifbench) —
register themselves into the same dicts on first lookup."""

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


# Hub-backed named tasks; specs need the .venv-prime worker, so they only
# register on first lookup (grpo_es/tasks/if_tasks.py).
HUB_BACKED_TASKS = ("ifeval", "ifbench")


def register_task(spec: TaskSpec, loader) -> None:
    """Dynamic registration (hub-env tasks) into the same dicts as above."""
    SPECS[spec.name] = spec
    LOADERS[spec.name] = loader


def get_task_spec(name: str) -> TaskSpec:
    # Local imports: from_env starts the worker subprocess machinery, which
    # only hub-env tasks should ever pay for.
    if name not in SPECS and name.startswith("env:"):
        from grpo_es.tasks.from_env import register_environment_task

        register_environment_task(name)
    elif name not in SPECS and name in HUB_BACKED_TASKS:
        from grpo_es.tasks.if_tasks import register_if_task

        register_if_task(name)
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown task {name!r}; known tasks: "
            f"{sorted({*SPECS, *HUB_BACKED_TASKS})} "
            f"or 'env:<owner>/<env>' (PrimeIntellect hub)"
        ) from None
