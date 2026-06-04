"""GRPO leg: a thin wrapper around TRL's ``GRPOTrainer``.

TRL owns the rollout loop. This module only wires the spine into it — task
dataset, rubric-backed reward functions, LoRA — and writes the run artifacts
(``run_config.json``, ``checkpoint-final/``, ``token_budget.json``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import GRPOConfig, GRPOTrainer

from grpo_es.config.run_config import RunConfig
from grpo_es.methods.callbacks import CompactMetricsCallback
from grpo_es.metrics.budget import extract_trl_token_budget
from grpo_es.rewards.registry import make_trl_reward_funcs
from grpo_es.tasks.base import build_dataset
from grpo_es.tasks.registry import get_task_spec

logger = logging.getLogger(__name__)

# Llama-family attention projections (SmolLM2, Qwen, ...). Revisit when a
# non-Llama rung joins the model ladder — don't fork this list per model.
_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
_LORA_DROPOUT = 0.05


def _check_generation_batching(cfg: RunConfig) -> None:
    """Fail fast on TRL's group-divisibility rule.

    TRL builds its generation batch as pdtbs x num_processes x grad_accum and
    requires it to hold whole prompt groups; violating that surfaces as a
    cryptic mid-startup error, so check it before anything heavy loads.
    """
    local_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    if local_batch % cfg.num_generations != 0:
        raise ValueError(
            f"num_generations ({cfg.num_generations}) must divide "
            f"per_device_train_batch_size * gradient_accumulation_steps "
            f"({cfg.per_device_train_batch_size} * {cfg.gradient_accumulation_steps} "
            f"= {local_batch})"
        )


def _training_args(cfg: RunConfig) -> GRPOConfig:
    return GRPOConfig(
        output_dir=cfg.output_dir,
        save_steps=cfg.save_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        seed=cfg.seed,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        beta=cfg.beta,
        logging_steps=cfg.logging_steps,
        report_to="trackio" if cfg.use_trackio else "none",
        project=Path(cfg.output_dir).name,
        trackio_space_id=cfg.trackio_space_id or "trackio",
        log_level="info" if cfg.verbose else "warning",
        log_level_replica="error",
        # Keep dataset columns: reward funcs read answer/numbers/target/... .
        remove_unused_columns=False,
        bf16=True,
    )


def _peft_config(cfg: RunConfig) -> LoraConfig | None:
    if not cfg.use_peft:
        return None
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_LORA_TARGETS,
    )


class QuietGRPOTrainer(GRPOTrainer):
    """GRPOTrainer minus the Printer/Progress callback noise."""

    def __init__(self, *args, quiet: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if quiet:
            self.remove_callback(PrinterCallback)
            self.remove_callback(ProgressCallback)


def run_grpo(cfg: RunConfig) -> Path:
    if cfg.method != "grpo":
        raise ValueError(f"run_grpo called with method={cfg.method!r}")
    _check_generation_batching(cfg)

    spec = get_task_spec(cfg.task)
    train_ds = build_dataset(
        spec, split="train", seed=cfg.data_seed, max_samples=cfg.max_samples
    )
    reward_funcs, reward_weights = make_trl_reward_funcs(
        spec.rubric,
        use_format_reward=cfg.use_format_reward,
        reward_weights=cfg.reward_weights,
    )

    training_args = _training_args(cfg)
    if reward_weights is not None:
        training_args.reward_weights = reward_weights

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "run_config.json")

    trainer = QuietGRPOTrainer(
        model=cfg.model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_ds,
        peft_config=_peft_config(cfg),
        processing_class=tokenizer,
        callbacks=None if cfg.verbose else [CompactMetricsCallback()],
        quiet=not cfg.verbose,
    )

    logger.info(
        "GRPO: task=%s model=%s rows=%d generations/prompt=%d",
        cfg.task,
        cfg.model,
        len(train_ds),
        cfg.num_generations,
    )
    trainer.train()
    trainer.save_model(str(out / "checkpoint-final"))

    budget = extract_trl_token_budget(trainer.state.log_history)
    if torch.cuda.is_available():
        budget.peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    budget.save(out / "token_budget.json")

    logger.info("GRPO finished; artifacts in %s", out)
    return out
