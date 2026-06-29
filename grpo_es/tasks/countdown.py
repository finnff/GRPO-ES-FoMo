"""Countdown-plain: reach a target by combining the given numbers.

!!!!
https://app.primeintellect.ai/dashboard/environments/sivit/countdown-plain
!!!!

Ported from Prime Intellect's ``sivit/countdown-plain`` a sibling of the
classic Countdown game adapted for plain chat models: no system message, plain
instructions prepended to the question, no tag format requirement, and targets
*constructed* from the operands so every puzzle is solvable. Numbers may be used
at most once (a subset is allowed); the rubric grades the last arithmetic
expression in the response.

Generated on the fly like the toy task: a fresh generator seed IS the held-out
set (the loader ignores ``split``).
"""

from __future__ import annotations

import random

from datasets import Dataset

from grpo_es.tasks.base import TaskSpec

# Prime exposes num_numbers/num_large as environment args; the repo's loader
# signature is fixed, so we pin them to countdown-plain's defaults here.
_NUM_NUMBERS = 4
_NUM_LARGE = 1

USER_INSTRUCTION = (
    "Reason through the problem step by step. "
    "Use the given numbers at most once. "
    "Use only ordinary arithmetic with plus, minus, times, and divide. "
    "Write plainly, and put your final arithmetic expression in the last sentence."
)

LARGE_NUMBERS = [25, 50, 75, 100]
SMALL_NUMBERS = list(range(1, 11)) * 2


def generate_numbers(num_large: int, num_numbers: int) -> list[int]:
    num_large = max(0, min(num_large, len(LARGE_NUMBERS), num_numbers))
    large = random.sample(LARGE_NUMBERS, num_large)
    small = random.sample(SMALL_NUMBERS, num_numbers - num_large)
    numbers = large + small
    random.shuffle(numbers)
    return numbers


def make_solvable_target(numbers: list[int], min_steps: int = 2) -> tuple[int, str]:
    available = [(float(number), str(number)) for number in numbers]
    steps = random.randint(min_steps, max(min_steps, min(4, len(numbers) - 1)))
    for _ in range(steps):
        if len(available) < 2:
            break
        left = available.pop(random.randrange(len(available)))
        right = available.pop(random.randrange(len(available)))
        a_value, a_expr = left
        b_value, b_expr = right
        # Upstream formatted the subtraction with the operand *values*
        # (`{max:g} - {min:g}`), which bakes intermediate results into the gold
        # string and makes it use numbers not in the puzzle. Order the
        # sub-*expressions* by value instead: same value (abs difference), same
        # random sequence, but a faithful expression over the real operands.
        larger_expr, smaller_expr = (
            (a_expr, b_expr) if a_value >= b_value else (b_expr, a_expr)
        )
        choices = [
            (a_value + b_value, f"({a_expr} + {b_expr})"),
            (abs(a_value - b_value), f"({larger_expr} - {smaller_expr})"),
            (a_value * b_value, f"({a_expr} * {b_expr})"),
        ]
        if b_value != 0 and a_value / b_value > 0 and (a_value / b_value).is_integer():
            choices.append((a_value / b_value, f"({a_expr} / {b_expr})"))
        if a_value != 0 and b_value / a_value > 0 and (b_value / a_value).is_integer():
            choices.append((b_value / a_value, f"({b_expr} / {a_expr})"))
        value, expr = random.choice([choice for choice in choices if choice[0] > 0])
        available.append((value, expr))
    value, expr = available[-1]
    return int(value), expr


def _build_prompt(row: dict) -> str:
    return (
        f"{USER_INSTRUCTION}\n\n"
        f"Numbers: {', '.join(map(str, row['numbers']))}\n"
        f"Target: {row['target']}"
    )


COUNTDOWN_SPEC = TaskSpec(
    name="countdown",
    rubric="countdown",
    system_prompt="",  # countdown-plain uses no system message
    build_prompt=_build_prompt,
    # Plain prose answers — no <think>/<answer> scaffold, so the format reward
    # is dropped (registry gates it on spec.format_scaffold).
    format_scaffold=False,
    # Generated data has no real splits: a fresh generator seed IS the
    # held-out set (the loader ignores `split`).
    eval_seed=999,
    eval_max_new=512,
)


def load_countdown(
    spec: TaskSpec,
    split: str = "train",
    seed: int = 0,
    max_samples: int | None = 64,
) -> Dataset:
    """Generate `max_samples` solvable rows; `split` is ignored (seed-driven data).

    Targets are *constructed* by applying 2-4 random operations to the operands,
    so every puzzle has at least one valid solution.
    """
    random.seed(seed)
    n = 64 if max_samples is None else max_samples
    rows = []
    for _ in range(n):
        numbers = generate_numbers(num_large=_NUM_LARGE, num_numbers=_NUM_NUMBERS)
        target, solution = make_solvable_target(numbers)
        rows.append(
            {
                "prompt": _build_prompt({"numbers": numbers, "target": target}),
                "answer": solution,
                "numbers": numbers,
                "target": target,
            }
        )
    return Dataset.from_list(rows)
