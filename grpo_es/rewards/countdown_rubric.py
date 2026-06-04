"""Countdown rubric: evaluate the proposed expression, enforce the operand
multiset, compare against the target.

The expression is parsed and evaluated over a whitelisted AST — never
``eval()`` — since it is model-generated text.
"""

from __future__ import annotations

import ast
import operator
import re
from collections import Counter

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_TOLERANCE = 1e-6

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _operands(expr: str) -> list[int | float] | None:
    """All numeric literals in the expression, or None if it doesn't parse."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    ]


def _safe_eval(expr: str) -> float | None:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    try:
        return _eval(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return None


class CountdownRubric(Rubric):
    """1.0 iff the expression uses exactly the given numbers and hits the target."""

    def __init__(self) -> None:
        super().__init__(parser=Parser())
        self.add_reward_func(self.solves_target)

    async def solves_target(
        self,
        parser: Parser,
        completion: Messages,
        answer: str,
        numbers: list[int] | None = None,
        target: int | None = None,
        **kwargs,
    ) -> float:
        if numbers is None or target is None:
            return 0.0
        text = completion[-1]["content"] if completion else ""
        match = _ANSWER_TAG.search(text)
        expr = (match.group(1) if match else "").strip()
        if not expr:
            return 0.0
        used = _operands(expr)
        if used is None or Counter(used) != Counter(numbers):
            return 0.0
        value = _safe_eval(expr)
        if value is None:
            return 0.0
        return 1.0 if abs(value - float(target)) < _TOLERANCE else 0.0
