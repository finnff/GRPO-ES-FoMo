"""Countdown-plain rubric: extract the last arithmetic expression from free
prose, enforce the operand multiset (each number at most once), and score
exact-match plus a smooth closeness term.

Ported from Prime Intellect's ``sivit/countdown-plain`` verifier. Expressions
are parsed and evaluated over a whitelisted AST — never ``eval()`` — since the
text is model-generated.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

EXPRESSION_RE = re.compile(r"[\d(][0-9\s,()+\-*/xX÷×.]*[\d)]")


def normalize_operators(text: str) -> str:
    replacements = {
        "×": "*",
        "x": "*",
        "X": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _message_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(message.get("content", "")) for message in value)
    return str(value)


def _safe_eval_expr(expr: str) -> float | None:
    expr = normalize_operators(expr).replace(",", "")
    if not re.fullmatch(r"[0-9\s()+\-*/.]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                return None
            return left / right
        return None

    return eval_node(tree)


def extract_candidate_expressions(text) -> list[str]:
    text = normalize_operators(_message_text(text))
    candidates = []
    for line in text.splitlines():
        pieces = [line]
        if "=" in line:
            pieces.extend(part.strip() for part in line.split("="))
        for piece in pieces:
            for match in EXPRESSION_RE.finditer(piece):
                expr = match.group(0).strip(" .,:;")
                if any(op in expr for op in ["+", "-", "*", "/"]):
                    candidates.append(expr)
    return candidates


def extract_last_expression(text) -> str | None:
    candidates = extract_candidate_expressions(text)
    for expr in reversed(candidates):
        if _safe_eval_expr(expr) is not None:
            return expr
    return None


def validate_expression(
    expr: str | None, numbers: list[int], target: int
) -> tuple[bool, float | None]:
    if not expr:
        return False, None
    expr = normalize_operators(expr).replace(",", "")
    # Disallow decimal literals; the only non-integer values should come from operations.
    if "." in expr:
        return False, None
    used_numbers = [int(n) for n in re.findall(r"\d+", expr)]
    available = Counter(numbers)
    for number in used_numbers:
        if available[number] <= 0:
            return False, None
        available[number] -= 1

    result = _safe_eval_expr(expr)
    if result is None or result <= 0:
        return False, None
    return math.isclose(result, target, rel_tol=1e-9, abs_tol=1e-9), result


class CountdownRubric(Rubric):
    """Weighted exact-match + closeness over a prose-tolerant expression parser."""

    def __init__(
        self, exact_weight: float = 1.0, closeness_weight: float = 0.25
    ) -> None:
        super().__init__(parser=Parser(extract_fn=extract_last_expression))
        self.add_reward_func(self.exact_match_reward, weight=exact_weight)
        self.add_reward_func(self.closeness_reward, weight=closeness_weight)

    async def exact_match_reward(
        self,
        parser: Parser,
        completion: Messages,
        numbers: list[int] | None = None,
        target: int | None = None,
        **kwargs,
    ) -> float:
        if numbers is None or target is None:
            return 0.0
        expr = parser.parse_answer(completion)
        is_exact, _ = validate_expression(expr, numbers, target)
        return 1.0 if is_exact else 0.0

    async def closeness_reward(
        self,
        parser: Parser,
        completion: Messages,
        numbers: list[int] | None = None,
        target: int | None = None,
        **kwargs,
    ) -> float:
        if numbers is None or target is None:
            return 0.0
        expr = parser.parse_answer(completion)
        _, result = validate_expression(expr, numbers, target)
        if result is None:
            return 0.0
        distance = abs(result - target)
        return 1.0 if distance == 0 else 0.5 ** (distance / 10)
