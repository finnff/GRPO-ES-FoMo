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


class InspectStepCallback(TrainerCallback):
    """Feeds the live trainer step to the inspect observer reward func.

    Reward funcs don't receive the step; the observer reads it from a shared
    mutable box this callback updates at each step's start, so dumped records
    carry the real ``global_step``.
    """

    def __init__(self, step_box: dict) -> None:
        self._box = step_box

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        # global_step is 0 for the about-to-run first step; the CompactMetrics
        # log line calls that completed step "1", so +1 keeps the inspector's
        # step numbers aligned with the scalar log.
        self._box["step"] = state.global_step + 1
