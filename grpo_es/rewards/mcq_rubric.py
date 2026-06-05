"""Rubric for multiple-choice tasks: normalized single-letter match.

Why not score MCQ with the math rubric? Symbolic verification silently
false-negatives on the variant forms a small model actually emits — ``C.``,
``**C**``, ``C) Paris`` — scoring correct answers 0. Under GRPO that's worse
than noise: a correct-but-oddly-formatted completion gets negative
within-group advantage, actively pushing the policy away from right answers.
Normalizing to a bare letter before comparing removes that failure mode.
"""

from __future__ import annotations

import re

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\s*{")
# A standalone capital: matches the C in "C) Paris" but not the P in "Paris".
_LETTER_RE = re.compile(r"([A-Z])(?![A-Za-z])")


def _last_boxed_content(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` in `text`, brace-matched (regex
    alone can't handle nesting); None when absent or unbalanced."""
    match = None
    for match in _BOXED_RE.finditer(text):
        pass
    if match is None:
        return None
    depth, i = 1, match.end()
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth:
        return None
    return text[match.end() : i - 1]


def normalize_letter(raw: str) -> str:
    """Boil a model's choice down to one capital letter, or '' if there isn't
    one. Tolerates LaTeX wrappers, markdown bold, brackets, trailing option
    text — every variant observed in real rollouts gets a row in the tests."""
    s = raw.strip()
    s = re.sub(r"\\(?:text|mathrm|mathbf|rm)\b", "", s)
    s = re.sub(r"[*${}\\]", "", s)
    s = s.strip().strip("()[]").strip()
    s = s.upper()
    match = _LETTER_RE.match(s)
    return match.group(1) if match else ""


def extract_choice(completion_text: str) -> str:
    """The model's chosen letter: last ``\\boxed{}`` after the think block,
    falling back to the ``<answer>`` tag. Splitting at ``</think>`` first
    keeps letters mentioned mid-reasoning from being read as the choice."""
    tail = completion_text.split("</think>")[-1]
    boxed = _last_boxed_content(tail)
    if boxed is not None:
        return normalize_letter(boxed)
    match = _ANSWER_RE.search(tail)
    return normalize_letter(match.group(1)) if match else ""


class MCQLetterRubric(Rubric):
    """1.0 iff the extracted letter equals the gold letter; no partial credit."""

    def __init__(self) -> None:
        super().__init__(parser=Parser())
        self.add_reward_func(self.correct_choice)

    async def correct_choice(
        self,
        parser: Parser,
        completion: Messages,
        answer: str,
        **kwargs,
    ) -> float:
        text = completion[-1]["content"] if completion else ""
        pred = extract_choice(text)
        return 1.0 if pred and pred == normalize_letter(str(answer)) else 0.0
