"""IFEval / IFBench as named tasks, served by the PrimeIntellect hub.

Instruction-following with programmatic checkers ("no commas", "300+ words",
"three highlighted sections"). Nothing is vendored: the hub environments own
the checker code (``scripts/setup_prime_venv.sh primeintellect/ifeval
primeintellect/ifbench`` installs them); this module only pins the
spine-side choices the raw ``env:`` form can't make:

- Both envs publish a single eval-only split (IFEval 541 rows, IFBench 300),
  so each spec pins a holdout window and the env loader keeps the training
  draw disjoint from it.
- Both stock rubrics put all reward weight on the all-or-nothing
  ``followed_instructions``, which collapses GRPO's within-group advantage
  on hard prompts. Training rewards the graded
  ``followed_instructions_rate`` instead; the strict pass/fail stays
  available in the metrics.

Checker modes keep the upstream defaults: IFEval strict, IFBench loose.
"""

from __future__ import annotations

from grpo_es.tasks.base import TaskSpec
from grpo_es.tasks.from_env import task_from_environment

_ENV_IDS = {
    "ifeval": "primeintellect/ifeval",
    "ifbench": "primeintellect/ifbench",
}
_EVAL_SIZE = 100
# Instructions like "write 400+ words" need more room than the default cap.
_EVAL_MAX_NEW = 1024


def register_if_task(name: str) -> TaskSpec:
    """Lazy entry point used by the task registry on first lookup."""
    spec, _ = task_from_environment(
        name,
        _ENV_IDS[name],
        reward_metric="followed_instructions_rate",
        eval_size=_EVAL_SIZE,
        eval_max_new=_EVAL_MAX_NEW,
        metric_label="instruction_rate",
    )
    return spec
