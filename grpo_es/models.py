"""The model ladder: short aliases for the checkpoints this repo trains on.

Ascending size, deliberately small — the comparison between optimizers is
only informative while the base model has headroom on the task. Everything
model-specific (alias, LoRA target naming) lives here so the method legs and
the eval runner stay architecture-agnostic.
"""

from __future__ import annotations

MODEL_ALIASES: dict[str, str] = {
    # SmolLM2: plain Llama architecture, instruct checkpoints. Not trained
    # with think tags, so the graded format reward does the lifting early on.
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # Repo default (see config.run_config.DEFAULT_MODEL).
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    # LFM2.5: hybrid short-conv + GQA attention blocks, native in
    # transformers>=5 (no trust_remote_code).
    "lfm2.5-350m": "LiquidAI/LFM2.5-350M",
    "lfm2.5-1.2b": "LiquidAI/LFM2.5-1.2B-Instruct",
}


def resolve_model_alias(name: str) -> str:
    """Map a known alias (case-insensitive) to its HF repo id; pass anything
    else through untouched, so full repo ids and local paths keep working."""
    return MODEL_ALIASES.get(name.strip().lower(), name)


def lora_targets_for(model_id: str) -> list[str]:
    """LoRA target module names for the model's architecture.

    Llama-family (SmolLM2, Qwen) names its attention projections q/k/v/o_proj
    and every layer is attention, so that list covers the whole stack. LFM2 is
    a hybrid where only a minority of layers are self-attention and the output
    projection is named ``out_proj`` — Llama's ``o_proj`` matches nothing
    there. The ``out_proj`` suffix also matches the short-conv blocks'
    ``conv.out_proj``, which (deliberately) lets the adapter reach the conv
    pathway carrying most layers.
    """
    if "lfm2" in model_id.lower():
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]
