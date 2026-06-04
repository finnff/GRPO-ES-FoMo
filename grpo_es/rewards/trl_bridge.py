"""The verifiers <-> TRL seam.

A verifiers ``Rubric`` scores one rollout at a time through an async
``score_rollout(state)`` interface; TRL wants a synchronous batch function
``(prompts, completions, **columns) -> list[float]``. This module owns that
translation in one place, so the very same rubric object can later be consumed
directly by a second optimizer leg without TRL in the loop.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State

_THINK_ANSWER_RE = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def as_messages(text: str, role: str) -> Messages:
    return [{"role": role, "content": text}]


class VerifiersRubricAdapter:
    """Synchronous scoring on top of an async Rubric.

    Besides ``answer``, every per-sample dataset column is threaded into the
    rollout state, so rubrics that need more than the gold string (operand
    lists, instruction metadata, ...) can pick their inputs up by name.

    Scoring is batched through a single event loop: thousands of completions
    are scored per training step, and a fresh ``asyncio.run`` per completion
    spends most of its time building and tearing the loop down.
    """

    def __init__(self, rubric: Rubric) -> None:
        self.rubric = rubric

    def _build_state(self, prompt: str, completion: str, columns: dict) -> State:
        state: State = {
            "prompt": as_messages(prompt, "user"),
            "completion": as_messages(completion, "assistant"),
            "answer": columns.pop("answer", ""),
        }
        if "info" in columns:
            state["info"] = columns.pop("info") or {}
        if columns:
            # verifiers exposes per-sample extras to reward funcs only via
            # state["input"] (Rubric.task_score_fields) — a top-level
            # state.update() here would silently never reach them.
            state["input"] = columns
        return state

    def score_batch(self, rows: list[tuple[str, str, dict]]) -> list[float]:
        """Score ``(prompt, completion, columns)`` rows in one event loop."""
        states = [self._build_state(p, c, dict(cols)) for p, c, cols in rows]

        async def _run() -> None:
            await asyncio.gather(*(self.rubric.score_rollout(s) for s in states))

        asyncio.run(_run())
        return [float(s.get("reward", 0.0)) for s in states]

    def score(self, prompt: str, completion: str, **columns: Any) -> float:
        return self.score_batch([(prompt, completion, columns)])[0]


def rubric_reward_func(rubric: Rubric, name: str | None = None) -> Callable:
    """Wrap a Rubric as a TRL ``reward_func``.

    With ``remove_unused_columns=False`` TRL hands every dataset column to the
    reward function as a list aligned with ``prompts``; we forward exactly
    those (scalars and wrong-length values are dropped) so the rubric sees the
    same per-sample fields the loader produced.
    """
    adapter = VerifiersRubricAdapter(rubric)

    def reward_func(
        prompts: list[str], completions: list[str], **kwargs: Any
    ) -> list[float]:
        n = len(prompts)
        columns = {
            k: v for k, v in kwargs.items() if isinstance(v, list) and len(v) == n
        }
        rows: list[tuple[str, str, dict]] = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            row = {k: v[i] for k, v in columns.items()}
            row.setdefault("answer", "")
            rows.append((prompt, completion, row))
        return adapter.score_batch(rows)

    # TRL logs per-reward means under the function's __name__.
    reward_func.__name__ = name or type(rubric).__name__
    return reward_func


def _format_score(text: str) -> float:
    """Graded score for the R1 scaffold ``<think>...</think><answer>...</answer>``.

    All-or-nothing format checks are ~always 0 on a fresh model, which makes
    every completion in a group identical → zero within-group advantage → no
    gradient. Partial credit keeps the signal alive while the tags are being
    learned: +0.25 per closed block, +0.25 for think-before-answer, 1.0 for an
    exact match.
    """
    if _THINK_ANSWER_RE.match(text):
        return 1.0
    think = _THINK_RE.search(text)
    answer = _ANSWER_RE.search(text)
    score = 0.0
    if think is not None:
        score += 0.25
    if answer is not None:
        score += 0.25
    if think is not None and answer is not None and think.start() < answer.start():
        score += 0.25
    return score


def format_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    return [_format_score(completion.strip()) for completion in completions]
