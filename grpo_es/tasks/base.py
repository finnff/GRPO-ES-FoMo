"""TaskSpec: the contract every task implements.

A task is a prompt builder plus a rubric name; the dataset loaders all share
one signature so both method legs (and later the eval runner) can stay
task-agnostic. Datasets always carry a ``prompt`` and an ``answer`` column;
anything else a rubric needs rides along as extra columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets import Dataset


@dataclass(frozen=True)
class TaskSpec:
    name: str
    rubric: str  # key into rewards.registry
    system_prompt: str
    build_prompt: Callable[[dict], str]

    # Whether completions are expected to carry the <think>/<answer>
    # scaffold. Hub-env rubrics grade the raw response, so env tasks turn
    # this off and the format reward is dropped with it.
    format_scaffold: bool = True

    # Held-out eval slice: shuffle `eval_split` with `eval_seed`, keep rows
    # [eval_offset : eval_offset + eval_size]. Each task picks these so the
    # slice is disjoint from the training draw; generated tasks get a fresh
    # seed instead of a split.
    eval_split: str = "test"
    eval_seed: int = 0
    eval_offset: int = 0
    eval_size: int | None = None  # None = the whole split

    # Decode caps for eval generation (prompt tokens / new tokens).
    eval_max_prompt: int = 512
    eval_max_new: int = 768

    # What the headline number means in eval tables.
    metric_label: str = "solve_rate"


def chat_template_enabled(mode: str, tok) -> bool:
    """Whether to wrap prompts in the model's chat template.

    ``auto`` turns it on iff the tokenizer ships a chat template — instruct
    models do, most base models don't — so the common case needs no per-model
    flag. ``on``/``off`` force it.
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    return getattr(tok, "chat_template", None) is not None


def apply_chat_template(dataset: Dataset, tok, mode: str = "auto") -> Dataset:
    """Rewrite the ``prompt`` column as a templated user turn (+ generation
    prompt), so an instruct model sees the turn structure it was tuned for.

    Called identically in both method legs and the eval runner — that keeps the
    train==eval invariant. The template only changes the model *input*; the raw
    response the rubric grades is untouched. A no-op when disabled (``off``, or
    ``auto`` on a base model without a template).
    """
    if not chat_template_enabled(mode, tok):
        return dataset

    # Hybrid-reasoning models (e.g. Qwen3.5) default their template to an OPEN
    # ``<think>`` block, so the model spends its whole completion budget on a
    # chain-of-thought preamble and never reaches the answer on these short,
    # structured tasks (observed: Qwen3.5-4B clipped at 99% on ascii-tree,
    # reward ~0). Force thinking OFF (the empty ``<think></think>`` prefill) for
    # any template that exposes the switch; this is a no-op for templates that
    # don't reference it (LFM2.5, SmolLM2), so it cannot perturb the existing
    # poster legs. Applied here means train AND eval get it identically — the
    # train==eval invariant holds.
    extra = {}
    if "enable_thinking" in (getattr(tok, "chat_template", "") or ""):
        extra["enable_thinking"] = False

    def _wrap(row: dict) -> dict:
        return {
            "prompt": tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                **extra,
            )
        }

    return dataset.map(_wrap, load_from_cache_file=False, desc="chat-template")


def shuffle_take(dataset: Dataset, seed: int, max_samples: int | None) -> Dataset:
    """Seeded shuffle keeping the first ``max_samples`` rows (all if ``None``).

    The one slicing path every hub-backed loader shares, so ``offset`` math in
    ``build_eval_dataset`` and the train draw stay defined by the same code.
    """
    if max_samples is None:
        return dataset
    return dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))


def build_dataset(
    spec: TaskSpec,
    split: str = "train",
    seed: int = 0,
    max_samples: int | None = None,
) -> Dataset:
    # Local import: the registry imports the task modules, which import this
    # module — a top-level import here would be circular.
    from grpo_es.tasks.registry import LOADERS

    loader = LOADERS[spec.name]
    return loader(spec, split=split, seed=seed, max_samples=max_samples)


def build_eval_dataset(
    spec: TaskSpec,
    split: str | None = None,
    seed: int | None = None,
    offset: int | None = None,
    size: int | None = None,
) -> Dataset:
    """Build the held-out slice; ``None`` arguments fall back to the spec.

    Loaders shuffle with ``seed`` and keep the first ``max_samples`` rows, so
    loading ``offset + size`` rows and slicing ``[offset:]`` is exactly
    ``shuffle(seed)[offset : offset + size]`` — no second shuffle code path
    to keep consistent.
    """
    from grpo_es.tasks.registry import LOADERS

    split = spec.eval_split if split is None else split
    seed = spec.eval_seed if seed is None else seed
    offset = spec.eval_offset if offset is None else offset
    size = spec.eval_size if size is None else size

    loader = LOADERS[spec.name]
    take = None if size is None else offset + size
    ds = loader(spec, split=split, seed=seed, max_samples=take)
    if offset:
        ds = ds.select(range(offset, len(ds)))
    return ds
