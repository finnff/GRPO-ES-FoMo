"""Rubric for the toy last-letter task: exact match inside the answer tag."""

from __future__ import annotations

import re

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer(text: str) -> str:
    match = _ANSWER_TAG.search(text)
    return (match.group(1) if match else text).strip()


class ToyRubric(Rubric):
    """1.0 iff the extracted answer equals the gold string (case-insensitive)."""

    def __init__(self) -> None:
        super().__init__(parser=Parser())
        self.add_reward_func(self.exact_match)

    async def exact_match(
        self,
        parser: Parser,
        completion: Messages,
        answer: str,
        **kwargs,
    ) -> float:
        text = completion[-1]["content"] if completion else ""
        return 1.0 if _extract_answer(text).lower() == str(answer).strip().lower() else 0.0
