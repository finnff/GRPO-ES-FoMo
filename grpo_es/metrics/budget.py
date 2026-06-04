"""Token accounting per run.

Tokens processed are the cost currency for comparing optimizers later, so
every run writes a ``token_budget.json`` next to its checkpoints from day one
— retrofitting cost accounting after the comparison runs exist is how budgets
end up estimated instead of measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenBudgetLog:
    num_tokens: int | None = None
    global_step: int | None = None
    train_runtime: float | None = None
    mean_step_time: float | None = None
    tokens_per_second: float | None = None
    peak_vram_bytes: int | None = None
    source: str = "trl_logs"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def _last_value(history: list[dict[str, Any]], key: str) -> Any:
    for entry in reversed(history):
        if key in entry:
            return entry[key]
    return None


def extract_trl_token_budget(history: list[dict[str, Any]]) -> TokenBudgetLog:
    """Pull token/throughput totals out of a TRL trainer's ``log_history``.

    The final history entry is a train summary (has ``train_runtime`` but no
    ``step``/``num_tokens``), so each key is taken from the last entry that
    actually carries it rather than from ``history[-1]``.
    """
    num_tokens = _last_value(history, "num_tokens")
    step_times = [e["step_time"] for e in history if "step_time" in e]
    train_runtime = _last_value(history, "train_runtime")

    elapsed = train_runtime if train_runtime is not None else sum(step_times) or None
    return TokenBudgetLog(
        num_tokens=num_tokens,
        global_step=_last_value(history, "step"),
        train_runtime=train_runtime,
        mean_step_time=sum(step_times) / len(step_times) if step_times else None,
        tokens_per_second=num_tokens / elapsed if num_tokens and elapsed else None,
    )
