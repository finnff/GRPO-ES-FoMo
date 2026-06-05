from grpo_es.config.run_config import parse_args
from grpo_es.models import MODEL_ALIASES, lora_targets_for, resolve_model_alias


def test_aliases_resolve():
    assert resolve_model_alias("smollm2-360m") == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert resolve_model_alias("qwen3.5-0.8b") == "Qwen/Qwen3.5-0.8B"
    assert resolve_model_alias("lfm2.5-1.2b") == "LiquidAI/LFM2.5-1.2B-Instruct"


def test_alias_resolution_case_insensitive_and_stripped():
    assert resolve_model_alias("SmolLM2-135M") == MODEL_ALIASES["smollm2-135m"]
    assert resolve_model_alias("  Qwen3.5-0.8B  ") == "Qwen/Qwen3.5-0.8B"


def test_unknown_model_passes_through():
    # Full repo ids and local paths must survive untouched.
    assert resolve_model_alias("HuggingFaceTB/SmolLM2-135M-Instruct") == (
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert resolve_model_alias("/tmp/my-checkpoint") == "/tmp/my-checkpoint"


def test_parse_args_resolves_alias():
    cfg = parse_args(["--model", "smollm2-360m"])
    # run_config.json must record the canonical id, never the alias.
    assert cfg.model == "HuggingFaceTB/SmolLM2-360M-Instruct"


def test_lora_targets_per_architecture():
    # Llama-family names the attention output o_proj; LFM2 names it out_proj
    # (and the suffix also reaches the conv blocks' out_proj on purpose).
    assert lora_targets_for("Qwen/Qwen3.5-0.8B") == [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ]
    assert lora_targets_for("LiquidAI/LFM2.5-350M") == [
        "q_proj", "k_proj", "v_proj", "out_proj",
    ]
