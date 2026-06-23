"""Graded local rubric + row-filter for ``primeintellect/pydantic-adherence``.

The hub env scores all-or-nothing (1.0 iff the emitted JSON validates against a
per-sample Pydantic model, else 0.0), which collapses GRPO within-group
advantage / ES fitness spread on a base that only occasionally lands a fully
valid object. Worse, its reward fn *throws* on a sizeable slice of rows: schemas
built with ``from __future__ import annotations`` leave forward refs unresolved,
so ``model_json_schema()`` / ``model_validate()`` raise unless ``model_rebuild()``
is called first — the env never calls it, so those rows auto-score 0 no matter
the output (unwinnable, poisons the gradient).

This module fixes both, locally (scored in the training venv, no worker
round-trip):

* ``load_pydantic_model`` rebuilds forward refs before the structural check and
  is the single source of truth for both the rubric and the filter.
* ``row_is_loadable`` is the dataset filter — drop rows whose schema can't be
  loaded *here* (forward-ref failures, plus schemas needing optional deps such
  as ``email-validator`` that aren't installed in the training venv); unwinnable
  either way, so they only flatten the gradient.
* ``PydanticGradedRubric`` grades by the *fraction of model fields that
  validate* rather than all-or-nothing, turning the binary signal dense.

Security note: ``load_pydantic_model`` ``exec``s dataset-provided config code.
This mirrors the env's own ``_load_model_from_code`` trust boundary, now running
in the training venv rather than the isolated hub venv.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ValidationError
from verifiers.rubrics.rubric import Rubric

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


@lru_cache(maxsize=512)
def load_pydantic_model(config_str: str, model_name: str) -> type[BaseModel] | None:
    """Exec a dataset schema config and return its model class, or ``None``.

    Mirrors the env's ``_load_model_from_code`` but adds the ``model_rebuild()``
    rescue the env omits (resolves forward refs left dangling by
    ``from __future__ import annotations``). Returns ``None`` on *any* failure so
    both the filter and the rubric treat unloadable schemas uniformly. Cached:
    the same schema is scored once per generation in a group and reused across
    steps/epochs, and re-``exec``ing it each time dominated step time.
    """
    if not config_str or not model_name:
        return None
    module = ModuleType("dyn_pydantic_cfg")
    try:
        # exec of dataset-provided config — same trust boundary as the env's
        # _load_model_from_code, now in the training venv (see module docstring).
        exec(config_str, module.__dict__)
        cls = getattr(module, model_name, None)
        if cls is None or not isinstance(cls, type) or not issubclass(cls, BaseModel):
            return None
        try:
            # resolve forward refs the env leaves dangling; the scratch module
            # isn't in sys.modules, so hand model_rebuild the exec namespace
            # explicitly (sibling classes live there).
            cls.model_rebuild(_types_namespace=module.__dict__)
        except Exception:
            pass  # best-effort; the schema build below is the real gate
        cls.model_json_schema()  # structural self-check (matches the env)
        return cls
    except Exception:
        return None


def row_is_loadable(row: dict) -> bool:
    """Dataset filter: ``True`` iff this row's schema loads in the training venv.

    Reads the same nested ``info`` structure the rubric reads. Drops the env's
    unwinnable rows (forward-ref / missing-optional-dep schemas) so they don't
    auto-score 0 and flatten the gradient.
    """
    info = (row.get("info") or {}).get("verification_info") or {}
    return load_pydantic_model(
        info.get("pydantic_config", ""), info.get("model_name", "")
    ) is not None


def _extract_last_json(text: str) -> dict | None:
    """Last JSON object in ``text`` — fenced ```json block else last balanced ``{...}``.

    Mirrors the env's ``_find_last_json_block`` / ``extract_last_json``.
    """
    matches = list(_FENCE_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip()
    else:
        end = text.rfind("}")
        if end == -1:
            return None
        depth = 0
        candidate = None
        i = end
        while i >= 0:
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[i : end + 1].strip()
                    break
            i -= 1
        if candidate is None:
            return None
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


class PydanticGradedRubric(Rubric):
    """Per-field-graded replacement for the env's binary reward.

    1.0 if the JSON fully validates; otherwise the fraction of top-level model
    fields *not* implicated in a ``ValidationError``. Dense where the env was
    binary, so a base that gets most fields right earns partial credit.
    """

    async def score_rollout(self, state: dict, **_: Any) -> dict:
        score = self._score(state)
        state["reward"] = score
        state["metrics"] = {"pydantic_graded": score}
        return state

    @staticmethod
    def _score(state: dict) -> float:
        completion = state.get("completion") or []
        text = completion[-1].get("content", "") if completion else ""
        parsed = _extract_last_json(text)
        if parsed is None:
            return 0.0
        info = (state.get("info") or {}).get("verification_info") or {}
        model = load_pydantic_model(
            info.get("pydantic_config", ""), info.get("model_name", "")
        )
        if model is None:
            return 0.0
        try:
            model.model_validate(parsed)
            return 1.0
        except ValidationError as e:
            n = len(model.model_fields)
            if n == 0:
                return 0.0
            bad = {err["loc"][0] for err in e.errors() if err.get("loc")}
            if not bad:
                # a ValidationError with no locatable field — charge one field
                return max(0.0, 1.0 - 1.0 / n)
            return max(0.0, 1.0 - len(bad) / n)
        except Exception:
            return 0.0
