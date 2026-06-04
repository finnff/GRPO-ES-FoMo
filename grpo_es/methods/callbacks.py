"""Trainer callbacks shared by the method legs."""

from __future__ import annotations

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class CompactMetricsCallback(TrainerCallback):
    """One short line per logging step instead of TRL's raw dict dump.

    Precision is per-key: 4 decimals for signal metrics, coarse for progress.
    Keys missing from a log entry are simply skipped, so the same callback
    works across tasks (each rubric logs under its own class name).
    """

    _FORMATS = {
        "reward": ".4f",
        "rewards/ToyRubric/mean": ".4f",
        "rewards/CountdownRubric/mean": ".4f",
        "rewards/format_reward_func/mean": ".4f",
        "completions/clipped_ratio": ".3f",
        "kl": ".4f",
        "loss": ".2f",
        "epoch": ".2f",
        "step_time": ".1f",
    }

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs or "reward" not in logs:
            return  # train-end summary entries etc.
        parts = [f"step {state.global_step}"]
        for key, fmt in self._FORMATS.items():
            if key in logs:
                short = key.removeprefix("rewards/").removesuffix("/mean")
                parts.append(f"{short}={logs[key]:{fmt}}")
        logger.info(" ".join(parts))
