"""Glyph-normalized graded rubric for ``primeintellect/ascii-tree``.

The hub env scores ``SequenceMatcher.ratio()`` over the *exact* tree text, so a
structurally correct tree drawn with different connector glyphs than gold
(base ``--`` / box-drawing ``├──``/``└──``/``│`` vs gold ``+--`` / backtick
``` `-- ``` / ``|``) scores ~0 — the nominally "dense" reward is near-binary in
practice. This rubric canonicalizes connector glyphs and indentation on *both*
sides before diffing, so drawing style cancels out and the score reflects
structure. The blend matches the env (0.3 * line-set ratio + 0.7 *
longest-matching-block / line-count); the env's ``--``/leading-space
``*= 0.5`` penalties are dropped — moot once glyphs are normalized.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from verifiers.rubrics.rubric import Rubric

_BLOCK_RE = re.compile(r"<ascii_formatted>(.*?)</ascii_formatted>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:\w+)?\s*([\s\S]*?)\s*```")

# Box-drawing connectors -> the ASCII alphabet gold uses, so both sides share
# one vocabulary before we measure structure.
_GLYPH_TRANS = str.maketrans(
    {
        "│": "|",  # │ vertical
        "┃": "|",  # ┃
        "├": "+",  # ├ tee
        "┣": "+",  # ┣
        "└": "`",  # └ last-child elbow
        "┗": "`",  # ┗
        "╰": "`",  # ╰
        "┌": "+",  # ┌
        "╭": "+",  # ╭
        "─": "-",  # ─ horizontal
        "━": "-",  # ━
        "╷": "|",  # ╷
        " ": " ",  # nbsp
    }
)
# Characters that encode tree structure (indent + connector), not node identity.
_TREE_LEAD = set(" |+-`")


def _extract_block(text: str) -> str:
    """Pull the tree out of a completion: ``<ascii_formatted>`` tag, else a
    fenced block, else the whole text."""
    m = _BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _canonicalize(line: str) -> str:
    """Reduce a tree line to ``"<depth>|<node-name>"``, glyph- and width-agnostic.

    Depth = length of the leading run of tree-drawing chars, in ~4-char units
    (round-half-up, so a bare ``+-- `` cell counts as one level and 2-space
    indents still register). Node name = the remainder, preserving any ``-`` in
    real names. Applied identically to candidate and gold, so style cancels.
    """
    norm = line.translate(_GLYPH_TRANS)
    lead = 0
    for ch in norm:
        if ch in _TREE_LEAD:
            lead += 1
        else:
            break
    depth = (lead + 2) // 4
    name = norm[lead:].strip()
    return f"{depth}|{name}"


def _canon_lines(text: str) -> list[str]:
    return [_canonicalize(ln) for ln in text.split("\n") if ln.strip()]


class AsciiTreeGlyphNormRubric(Rubric):
    """Glyph-normalized graded reward for ascii-tree (gold lives in ``answer``)."""

    async def score_rollout(self, state: dict, **_: Any) -> dict:
        score = self._score(state)
        state["reward"] = score
        state["metrics"] = {"ascii_tree_glyphnorm": score}
        return state

    @staticmethod
    def _score(state: dict) -> float:
        completion = state.get("completion") or []
        text = completion[-1].get("content", "") if completion else ""
        gold = state.get("answer") or ""
        truth_lines = _canon_lines(gold.strip())
        if not truth_lines:
            return 0.0
        cand_lines = _canon_lines(_extract_block(text))
        matcher = difflib.SequenceMatcher(None, cand_lines, truth_lines)
        ratio = matcher.ratio()
        longest = max(
            matcher.get_matching_blocks(),
            key=lambda b: b.size,
            default=difflib.Match(0, 0, 0),
        )
        continuous = longest.size / len(truth_lines)
        return 0.3 * ratio + 0.7 * continuous
