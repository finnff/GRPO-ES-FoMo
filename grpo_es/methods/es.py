"""ES leg: antithetic LoRA-subspace evolution strategies, forward passes only.

A custom loop with no TRL in it: each step draws N antithetic perturbation
pairs in the adapter's parameter space, scores all 2N members on the same
prompt mini-batch with the same rubric-backed rewards GRPO trains on, and
moves the adapter along the rank-weighted noise sum (OpenAI-ES). Perturbing
only the LoRA tensors keeps the search dimension ~10^6 instead of the full
model's ~10^8-9.

First cut, deliberately rough: members generate sequentially (the batched
population forward is the planned throughput fix), and there is no warm-start
and no trust region yet — see GRPO_ES_ARCHITECTURE.md §6 for where this is
headed.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Callable

import torch
from peft import get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from grpo_es.config.run_config import RunConfig
from grpo_es.eval.runner import DecodeParams, generate, load_model, load_tokenizer
from grpo_es.metrics.budget import TokenBudgetLog
from grpo_es.models import lora_config
from grpo_es.rewards.registry import make_trl_reward_funcs
from grpo_es.tasks.base import TaskSpec, build_dataset
from grpo_es.tasks.registry import get_task_spec

logger = logging.getLogger(__name__)

# Per-step seed strides: distinct primes so the noise stream and the sampling
# stream never collide across (seed, step) pairs.
_NOISE_SEED_STRIDE = 1_000_003
_GEN_SEED_STRIDE = 7_919


def centered_ranks(values: list[float]) -> list[float]:
    """Map fitnesses to utilities in [-0.5, 0.5] by rank — the standard
    OpenAI-ES transform, invariant to the reward's scale and to outliers."""
    order = sorted(range(len(values)), key=values.__getitem__)
    utils = [0.0] * len(values)
    span = max(len(values) - 1, 1)
    for rank, i in enumerate(order):
        utils[i] = rank / span - 0.5
    return utils


class ESEngine:
    """The θ handle: an fp32 master copy of the LoRA tensors + the update rule.

    The live module is treated as a scratchpad — ``perturb`` writes
    master + scale·ε into it for one member's rollout, ``update`` folds the
    ranked noise back into the master and re-syncs the live weights. The
    master is fp32 because at σ ~1e-2 perturbations would drown in the bf16
    noise floor if accumulated in a half dtype (PEFT keeps LoRA tensors fp32
    even on a bf16 base, so today the copy is exact).
    """

    def __init__(
        self, model: PreTrainedModel, *, sigma: float, lr: float, seed: int
    ) -> None:
        self.sigma = sigma
        self.lr = lr
        self.seed = seed
        # Fixed capture order: every noise vector and update zips against it.
        self.params = [p for name, p in model.named_parameters() if "lora_" in name]
        if not self.params:
            raise ValueError("no LoRA parameters to perturb (is PEFT attached?)")
        self.master = [p.detach().clone().float() for p in self.params]

    def sample_noise(self, step: int, pairs: int) -> list[list[torch.Tensor]]:
        """``pairs`` ε-vectors shaped like θ, deterministic in (seed, step)."""
        gen = torch.Generator(device=self.params[0].device)
        gen.manual_seed(self.seed * _NOISE_SEED_STRIDE + step)
        return [
            [
                torch.randn(p.shape, generator=gen, device=p.device, dtype=torch.float32)
                for p in self.params
            ]
            for _ in range(pairs)
        ]

    @torch.no_grad()
    def perturb(self, eps: list[torch.Tensor], scale: float) -> None:
        """Write θ = master + scale·ε into the live module (fp32 add, then cast)."""
        for p, m, e in zip(self.params, self.master, eps):
            p.copy_((m + scale * e).to(p.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        for p, m in zip(self.params, self.master):
            p.copy_(m.to(p.dtype))

    @torch.no_grad()
    def update(self, noise: list[list[torch.Tensor]], fitnesses: list[float]) -> None:
        """One ES step: θ += lr/(2Nσ) · Σ_pairs (u⁺−u⁻)·ε, then sync the live
        module. ``fitnesses`` are ordered [+ε₀, −ε₀, +ε₁, −ε₁, ...]."""
        assert len(fitnesses) == 2 * len(noise), (
            f"expected 2 fitnesses per noise pair, got {len(fitnesses)} for "
            f"{len(noise)} pairs"
        )
        if max(fitnesses) == min(fitnesses):
            # A flat population carries no ranking signal; the arbitrary
            # tie-break order would otherwise become a random parameter walk.
            self.restore()
            return
        utils = centered_ranks(fitnesses)
        coef = self.lr / (len(fitnesses) * self.sigma)
        for pair, eps in enumerate(noise):
            weight = coef * (utils[2 * pair] - utils[2 * pair + 1])
            for m, e in zip(self.master, eps):
                m.add_(e, alpha=weight)
        self.restore()


def _make_fitness(
    cfg: RunConfig, spec: TaskSpec
) -> Callable[[list[str], list[str], dict], float]:
    """Per-member fitness = GRPO's weighted total reward, averaged over the
    mini-batch — both legs optimize the identical scalar, assembled by the
    same registry call the GRPO trainer makes."""
    funcs, weights = make_trl_reward_funcs(
        spec.rubric,
        use_format_reward=cfg.use_format_reward and spec.format_scaffold,
        reward_weights=cfg.reward_weights,
    )
    weights = weights or [1.0] * len(funcs)
    # Weight-0 shadows shape nothing, and at 2N rubric calls per step they
    # would only burn wall-clock — the fitness skips them.
    scored = [(func, w) for func, w in zip(funcs, weights) if w]

    def fitness(prompts: list[str], completions: list[str], columns: dict) -> float:
        # Keyword calls, like TRL makes them — the funcs' positional
        # signatures differ (format_reward_func takes no prompts).
        return sum(
            w
            * sum(func(prompts=prompts, completions=completions, **columns))
            / len(prompts)
            for func, w in scored
        )

    return fitness


def _score_population(
    engine: ESEngine,
    noise: list[list[torch.Tensor]],
    prompts: list[str],
    columns: dict,
    fitness: Callable[[list[str], list[str], dict], float],
    decode: DecodeParams,
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    gen_seed: int,
) -> tuple[list[float], list[int], int]:
    """Score all 2N antithetic members on one mini-batch.

    Returns ``(fitnesses, completion_lengths, tokens)`` with the fitnesses
    ordered [+ε₀, −ε₀, +ε₁, −ε₁, ...] — the layout ``ESEngine.update`` zips
    its noise against. ``gen_seed`` is reused for every member (common random
    numbers across the population), so decode noise partly cancels inside each
    antithetic pair.
    """
    fits: list[float] = []
    lengths: list[int] = []
    tokens = 0
    for eps in noise:
        for sign in (1.0, -1.0):
            engine.perturb(eps, sign * engine.sigma)
            gen = generate(
                model,
                tok,
                prompts,
                decode,
                seed=gen_seed,
                batch_size=len(prompts),
                progress=False,
            )
            fits.append(fitness(prompts, gen.completions, columns))
            lengths.extend(len(c) for c in gen.completions)
            tokens += gen.tokens
    return fits, lengths, tokens


def run_es(cfg: RunConfig) -> Path:
    if cfg.method != "es":
        raise ValueError(f"run_es called with method={cfg.method!r}")
    if not cfg.use_peft:
        raise ValueError(
            "the ES leg perturbs the LoRA subspace; full-weight ES is not "
            "implemented (drop --no-peft)"
        )

    spec = get_task_spec(cfg.task)
    train_ds = build_dataset(
        spec, split="train", seed=cfg.data_seed, max_samples=cfg.max_samples
    )
    fitness = _make_fitness(cfg, spec)

    tok = load_tokenizer(cfg.model)
    # Same loader as eval, so the saved adapter's module tree matches what
    # eval (and TRL) reload — wrapper architectures otherwise silently load
    # zero adapter weights (see eval.runner._model_class).
    model = load_model(cfg.model, "base")
    model = get_peft_model(model, lora_config(cfg.model, cfg.lora_r, cfg.lora_alpha))
    model.eval()  # dropout off: each member must score exactly the policy it perturbs

    engine = ESEngine(model, sigma=cfg.es_sigma, lr=cfg.es_lr, seed=cfg.seed)
    decode = DecodeParams(
        decode="greedy" if cfg.es_greedy_fitness or cfg.temperature <= 0 else "sample",
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        max_new=cfg.max_completion_length,
        max_prompt=cfg.max_prompt_length,
    )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "run_config.json")

    logger.info(
        "ES: task=%s model=%s rows=%d population=%dx2 sigma=%g lr=%g steps=%d",
        cfg.task,
        cfg.model,
        len(train_ds),
        cfg.es_population,
        cfg.es_sigma,
        cfg.es_lr,
        cfg.es_steps,
    )

    # The mini-batch picker rides the optimizer seed (like GRPO's data order),
    # not data_seed — the train slice itself is already pinned by data_seed.
    batch_rng = random.Random(cfg.seed)
    total_tokens = 0
    step_times: list[float] = []
    t_start = time.perf_counter()

    for step in range(cfg.es_steps):
        t_step = time.perf_counter()
        idx = batch_rng.sample(
            range(len(train_ds)), k=min(cfg.es_eval_batch, len(train_ds))
        )
        batch = train_ds.select(idx)
        prompts = batch["prompt"]
        columns = {c: batch[c] for c in batch.column_names if c != "prompt"}

        noise = engine.sample_noise(step, cfg.es_population)
        # One sampling seed per step (distinct prime stride keeps it off the
        # noise stream), reused for every member inside _score_population.
        gen_seed = cfg.seed * _GEN_SEED_STRIDE + step
        fits, lengths, tokens = _score_population(
            engine, noise, prompts, columns, fitness, decode, tok, model, gen_seed
        )
        total_tokens += tokens
        engine.update(noise, fits)

        step_times.append(time.perf_counter() - t_step)
        logger.info(
            "step %d/%d fitness mean=%.4f best=%.4f mean_len=%.0f tokens=%d step_time=%.1f",
            step + 1,
            cfg.es_steps,
            sum(fits) / len(fits),
            max(fits),
            sum(lengths) / len(lengths),
            total_tokens,
            step_times[-1],
        )
        if cfg.save_steps and (step + 1) % cfg.save_steps == 0 and step + 1 < cfg.es_steps:
            model.save_pretrained(str(out / f"checkpoint-{step + 1}"))

    model.save_pretrained(str(out / "checkpoint-final"))

    runtime = time.perf_counter() - t_start
    budget = TokenBudgetLog(
        num_tokens=total_tokens,
        global_step=cfg.es_steps,
        train_runtime=runtime,
        mean_step_time=sum(step_times) / len(step_times) if step_times else None,
        tokens_per_second=total_tokens / runtime if runtime else None,
        source="es_loop",
    )
    if torch.cuda.is_available():
        budget.peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    budget.save(out / "token_budget.json")

    logger.info("ES finished; artifacts in %s", out)
    return out
