"""Generation + scoring for the held-out eval.

Reproducibility invariants, in one place so they can't drift per call site:
bf16 on cuda (never fp32/auto), left padding with the completion sliced off
at the prompt length, a fresh ``torch.manual_seed`` per adapter, raw
``build_prompt`` strings (no chat template — the trainer sees none either).
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import torch
import transformers
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from verifiers.rubrics.rubric import Rubric

from grpo_es.eval.kl import sequence_kl_to_base
from grpo_es.eval.metrics import EvalMetrics, coherence_gate, score_completions

logger = logging.getLogger(__name__)

_GEN_BATCH = 16


@dataclass
class DecodeParams:
    decode: str = "greedy"  # "greedy" | "sample"
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    max_new: int | None = None  # None -> TaskSpec.eval_max_new
    max_prompt: int | None = None  # None -> TaskSpec.eval_max_prompt

    @property
    def do_sample(self) -> bool:
        return self.decode == "sample"

    def label(self) -> str:
        return f"sample T={self.temperature}" if self.do_sample else "greedy"

    @classmethod
    def from_run_config(cls, run_dir: str | Path) -> "DecodeParams":
        """Decode settings matching a training run — for judging an adapter
        in the regime it was actually optimized under."""
        rc = json.loads((Path(run_dir) / "run_config.json").read_text())
        return cls(
            decode="sample",  # training rollouts sample
            temperature=rc["temperature"],
            repetition_penalty=rc["repetition_penalty"],
            max_new=rc["max_completion_length"],
            max_prompt=rc["max_prompt_length"],
        )

    def merged(self, **overrides) -> "DecodeParams":
        """A copy with the non-None overrides applied (CLI beats file)."""
        return replace(
            self, **{k: v for k, v in overrides.items() if v is not None}
        )


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _model_class(model_id: str) -> type:
    """The concrete class named in the checkpoint's config — NOT AutoModel*.

    This mirrors how the trainer loads models. The distinction bites on
    wrapper architectures (e.g. Qwen3.5 is a ForConditionalGeneration shell
    around a ``language_model``): AutoModelForCausalLM resolves to the inner
    text model, which has a different module tree than what training wrapped
    — and a tree mismatch makes PeftModel silently load zero adapter weights.
    """
    config = AutoConfig.from_pretrained(model_id)
    arch = (config.architectures or [None])[0]
    if arch and hasattr(transformers, arch):
        return getattr(transformers, arch)
    return AutoModelForCausalLM


def load_model(model_id: str, adapter: str = "base") -> PreTrainedModel:
    """Base model in bf16 on cuda, optionally with a LoRA adapter on top."""
    model = _model_class(model_id).from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter and adapter != "base":
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


class Generation(NamedTuple):
    """One ``generate`` call's output. ``sequences`` is the
    ``(batch_ids, prompt_len)`` list for KL teacher-forcing over the very same
    rollouts, populated only when ``want_sequences`` is set (it pins every
    batch's ids in memory), ``None`` otherwise."""

    completions: list[str]
    clipped: list[bool]  # clipped[i]: ran to max_new without an EOS
    tokens: int
    sequences: list[tuple[torch.Tensor, int]] | None = None


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    decode: DecodeParams,
    seed: int = 0,
    batch_size: int = _GEN_BATCH,
    want_sequences: bool = False,
) -> Generation:
    """Generate one completion per prompt."""
    torch.manual_seed(seed)  # per-adapter reseed: same draw order for everyone
    completions: list[str] = []
    clipped: list[bool] = []
    sequences: list[tuple[torch.Tensor, int]] = []
    total_tokens = 0

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=decode.max_prompt,
        ).to(model.device)
        input_len = enc["input_ids"].shape[1]

        kwargs: dict = {
            "max_new_tokens": decode.max_new,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
            "use_cache": True,
            "do_sample": decode.do_sample,
        }
        if decode.do_sample:
            kwargs["temperature"] = decode.temperature
        if decode.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = decode.repetition_penalty

        gen = model.generate(**enc, **kwargs)
        new = gen[:, input_len:]
        completions.extend(tok.batch_decode(new, skip_special_tokens=True))
        clipped.extend(tok.eos_token_id not in row.tolist() for row in new)

        prompt_tokens = int(enc["attention_mask"].sum().item())
        gen_tokens = int((new != tok.pad_token_id).sum().item())
        total_tokens += prompt_tokens + gen_tokens
        if want_sequences:
            sequences.append((gen.cpu(), input_len))

        logger.info("generated %d/%d", min(start + batch_size, len(prompts)), len(prompts))

    return Generation(completions, clipped, total_tokens, sequences if want_sequences else None)


@dataclass
class AdapterEval:
    label: str
    metrics: EvalMetrics
    mean_gate: float
    clip_frac: float
    completions: list[str]
    clipped: list[bool]
    tokens: int
    kl_to_base: float | None = None  # None for the bare base (trivially 0)


def evaluate_adapter(
    model_id: str,
    adapter: str,
    tok: PreTrainedTokenizerBase,
    dataset: Dataset,
    rubric: Rubric,
    decode: DecodeParams,
    seed: int = 0,
    want_kl: bool = False,
) -> AdapterEval:
    model = load_model(model_id, adapter)
    prompts = dataset["prompt"]
    is_base = adapter in ("", "base")

    gen = generate(
        model, tok, prompts, decode, seed=seed, want_sequences=want_kl and not is_base
    )
    kl = sequence_kl_to_base(model, tok, gen.sequences) if gen.sequences else None

    extra = {
        col: dataset[col]
        for col in dataset.column_names
        if col not in ("prompt", "answer")
    }
    metrics = score_completions(
        rubric, prompts, gen.completions, dataset["answer"], **extra
    )

    result = AdapterEval(
        label=adapter,
        metrics=metrics,
        mean_gate=sum(coherence_gate(c) for c in gen.completions) / max(len(gen.completions), 1),
        clip_frac=sum(gen.clipped) / max(len(gen.clipped), 1),
        completions=gen.completions,
        clipped=gen.clipped,
        tokens=gen.tokens,
        kl_to_base=kl,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
