"""Per-completion rollout dump for the live inspector.

Both training legs append one JSON line per rollout to
``<output_dir>/inspect.jsonl``: the prompt, the completion text, the rubric
reward, the format score, the token count and a clipped flag. The standalone
``inspect_run.py`` viewer tails that file and renders it green/yellow/red. This
is on by default; pass ``--no-inspect-dump`` to skip it and leave the training
path untouched.

Scoring goes through the same :func:`grpo_es.eval.metrics.score_completions`
path the trainer's reward funcs use, so a number shown by the inspector matches
the training reward for that completion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from transformers import PreTrainedTokenizerBase
from verifiers.rubrics.rubric import Rubric

from grpo_es.eval.metrics import score_completions


class InspectDumper:
    """Appends per-completion records to a JSONL file.

    Opens the file per ``write`` call (append mode): the cost is negligible next
    to generation, and it keeps the file consistent for a viewer tailing it
    concurrently. The file is truncated once at construction so a re-run of the
    same output dir starts clean.
    """

    def __init__(self, path: Path, method: str) -> None:
        self.path = Path(path)
        self.method = method
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")  # fresh file per run

    def write(self, records: list[dict]) -> None:
        if not records:
            return
        with self.path.open("a") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")


def build_records(
    rubric: Rubric,
    tok: PreTrainedTokenizerBase,
    max_tokens: int,
    step: int,
    method: str,
    items: list[dict],
) -> list[dict]:
    """Score ``items`` and turn them into inspector records.

    Each item is ``{prompt, completion, answer, columns, clipped, group,
    member, sign}``. ``columns`` holds the per-sample extra dataset fields the
    rubric may need (numbers, target, info, ...). Scoring is one batched call so
    the whole sampled population costs a single rubric pass.
    """
    if not items:
        return []
    prompts = [it["prompt"] for it in items]
    completions = [it["completion"] for it in items]
    answers = [it.get("answer", "") for it in items]
    keys = {k for it in items for k in it.get("columns", {})}
    extra = {k: [it.get("columns", {}).get(k) for it in items] for k in keys}

    metrics = score_completions(rubric, prompts, completions, answers, **extra)

    records: list[dict] = []
    for it, sample in zip(items, metrics.per_sample):
        completion = it["completion"]
        tokens = len(tok(completion)["input_ids"])
        records.append(
            {
                "step": step,
                "method": method,
                "group": it.get("group", 0),
                "member": it.get("member", 0),
                "sign": it.get("sign"),
                "prompt": it["prompt"],
                "completion": completion,
                "answer": it.get("answer", ""),
                "task_reward": sample["reward"],
                "format": sample["format"],
                "tokens": tokens,
                "max_tokens": max_tokens,
                "clipped": bool(it.get("clipped", False)),
            }
        )
    return records


def make_inspect_reward_func(
    dumper: InspectDumper,
    rubric: Rubric,
    tok: PreTrainedTokenizerBase,
    max_tokens: int,
    num_generations: int,
    max_prompts: int,
    step_box: dict,
    every: int,
) -> Callable:
    """A zero-weight TRL reward func that dumps rollouts and returns all zeros.

    TRL hands reward funcs ``(prompts, completions, **columns)`` with each prompt
    repeated ``num_generations`` times. We grade the first ``max_prompts`` groups
    on dump steps and write them out; the returned zeros (weight 0 in the
    trainer) keep it from shaping the policy. ``step_box["step"]`` is updated by
    :class:`grpo_es.methods.callbacks.InspectStepCallback` so records carry the
    real trainer step.
    """

    def inspect_observer(
        prompts: list[str], completions: list[str], **kwargs: Any
    ) -> list[float]:
        zeros = [0.0] * len(completions)
        step = step_box.get("step", 0)
        if every <= 0 or step % every != 0:
            return zeros

        n = len(prompts)
        columns = {
            k: v for k, v in kwargs.items() if isinstance(v, list) and len(v) == n
        }
        items: list[dict] = []
        groups = min(max_prompts, n // num_generations) if num_generations else 0
        for g in range(groups):
            for m in range(num_generations):
                i = g * num_generations + m
                completion = completions[i]
                tokens = len(tok(completion)["input_ids"])
                items.append(
                    {
                        "prompt": prompts[i],
                        "completion": completion,
                        "answer": columns.get("answer", [""] * n)[i]
                        if "answer" in columns
                        else "",
                        "columns": {k: v[i] for k, v in columns.items()},
                        # The reward layer can't see EOS; tokens at the cap is
                        # the best truncation signal available here.
                        "clipped": tokens >= max_tokens,
                        "group": g,
                        "member": m,
                        "sign": None,
                    }
                )
        dumper.write(build_records(rubric, tok, max_tokens, step, "grpo", items))
        return zeros

    inspect_observer.__name__ = "inspect_observer"
    return inspect_observer
