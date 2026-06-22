"""ES leg: antithetic LoRA-subspace evolution strategies, forward passes only.

A custom loop with no TRL in it: each step draws N antithetic perturbation
pairs in the adapter's parameter space, scores all 2N members on the same
prompt mini-batch with the same rubric-backed rewards GRPO trains on, and
moves the adapter along the rank-weighted noise sum (OpenAI-ES). Perturbing
only the LoRA tensors keeps the search dimension ~10^6 instead of the full
model's ~10^8-9.

The population rides the batch dimension instead of looping: a forward hook
on each LoRA layer adds per-member deltas via grouped ``bmm`` while the
adapter itself is disabled, so one ``generate`` call scores a whole chunk of
members. Two stay-honest levers complete the leg: ``--es-init-adapter``
warm-starts from a trained adapter (charging that run's tokens to this
budget), and ``--es-trust-region`` caps the cumulative parameter-space step —
naive ES at large σ reward-hacks dense checkers into token-spam, which lives
at adapter norms an order of magnitude above any honest solution, and the
norm cap removes that basin geometrically (see README).
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import torch
from peft import get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from grpo_es.config.run_config import RunConfig
from grpo_es.eval.runner import DecodeParams, generate, load_model, load_tokenizer
from grpo_es.inspect_dump import InspectDumper, build_records
from grpo_es.metrics.budget import TokenBudgetLog
from grpo_es.models import lora_config
from grpo_es.rewards.registry import get_rubric, make_trl_reward_funcs
from grpo_es.tasks.base import TaskSpec, apply_chat_template, build_dataset
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

    Members never touch the live weights — ``population`` overlays per-member
    deltas through forward hooks while PEFT's adapter path is disabled, so the
    live module always holds θ. The master is fp32 because at σ ~1e-2
    perturbations would drown in the bf16 noise floor if accumulated in a half
    dtype (PEFT keeps LoRA tensors fp32 even on a bf16 base, so today the copy
    is exact).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        sigma: float,
        lr: float,
        seed: int,
        trust_region: float = 0.0,
    ) -> None:
        self.sigma = sigma
        self.lr = lr
        self.seed = seed
        self.trust_region = trust_region
        self._model = model
        # Fixed capture order — (A, B) per LoRA layer in module-walk order;
        # every noise vector and update zips against ``params``.
        self.layers: list[tuple[torch.nn.Module, float]] = []
        self.params: list[torch.nn.Parameter] = []
        for module in model.modules():
            lora_A = getattr(module, "lora_A", None)
            if not isinstance(lora_A, torch.nn.ModuleDict) or "default" not in lora_A:
                continue
            self.layers.append((module, module.scaling["default"]))
            self.params.append(module.lora_A["default"].weight)
            self.params.append(module.lora_B["default"].weight)
        if not self.layers:
            raise ValueError("no LoRA layers to perturb (is PEFT attached?)")
        self.master = [p.detach().clone().float() for p in self.params]
        # Anchor for the trust region and the theta_dev diagnostic. For a
        # warm start this is the loaded adapter, not zero.
        self.init = [m.clone() for m in self.master]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.params)

    def init_norm(self) -> float:
        return math.sqrt(sum(i.pow(2).sum().item() for i in self.init))

    def theta_dev(self) -> float:
        """Global L2 distance of the master from its start point."""
        return math.sqrt(
            sum((m - i).pow(2).sum().item() for m, i in zip(self.master, self.init))
        )

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

    @contextmanager
    def population(
        self, noise: list[list[torch.Tensor]], sign: float
    ) -> Iterator[None]:
        """Overlay members θ + sign·σ·ε_i for the duration of the context.

        Inputs must be member-major — row block j of the batch belongs to
        member j. Each LoRA layer's hook reshapes its input to
        ``[members, rows/members · seq, d]`` and adds the per-member
        ``x·Aᵢᵀ·Bᵢᵀ·scaling`` via two grouped ``bmm``s in fp32 (matching
        PEFT's cast-input-to-adapter-dtype semantics), while
        ``disable_adapter`` keeps the shared adapter path out of the way.
        """
        members = len(noise)
        handles = []
        for l, (module, scaling) in enumerate(self.layers):
            a_pop = torch.stack(
                [self.master[2 * l] + sign * self.sigma * eps[2 * l] for eps in noise]
            )
            b_pop = torch.stack(
                [
                    self.master[2 * l + 1] + sign * self.sigma * eps[2 * l + 1]
                    for eps in noise
                ]
            )
            handles.append(
                module.register_forward_hook(
                    _member_delta_hook(a_pop, b_pop, scaling, members)
                )
            )
        try:
            with self._model.disable_adapter():
                yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def sync_live(self) -> None:
        for p, m in zip(self.params, self.master):
            p.copy_(m.to(p.dtype))

    @torch.no_grad()
    def update(self, noise: list[list[torch.Tensor]], fitnesses: list[float]) -> None:
        """One ES step: θ += lr/(2Nσ) · Σ_pairs (u⁺−u⁻)·ε, project back onto
        the trust-region ball, then sync the live module. ``fitnesses`` are
        ordered [+ε₀, −ε₀, +ε₁, −ε₁, ...]."""
        assert len(fitnesses) == 2 * len(noise), (
            f"expected 2 fitnesses per noise pair, got {len(fitnesses)} for "
            f"{len(noise)} pairs"
        )
        if max(fitnesses) == min(fitnesses):
            # A flat population carries no ranking signal; the arbitrary
            # tie-break order would otherwise become a random parameter walk.
            return
        utils = centered_ranks(fitnesses)
        coef = self.lr / (len(fitnesses) * self.sigma)
        for pair, eps in enumerate(noise):
            weight = coef * (utils[2 * pair] - utils[2 * pair + 1])
            for m, e in zip(self.master, eps):
                m.add_(e, alpha=weight)
        if self.trust_region:
            dev = self.theta_dev()
            if dev > self.trust_region:
                shrink = self.trust_region / dev
                for m, i in zip(self.master, self.init):
                    m.copy_(i + (m - i) * shrink)
        self.sync_live()


def _member_delta_hook(
    a_pop: torch.Tensor, b_pop: torch.Tensor, scaling: float, members: int
) -> Callable:
    def hook(module, args, output):
        x = args[0]
        assert x.shape[0] % members == 0, (
            f"batch rows {x.shape[0]} not divisible by {members} members"
        )
        h = x.reshape(members, -1, x.shape[-1]).float()
        delta = torch.bmm(torch.bmm(h, a_pop.transpose(1, 2)), b_pop.transpose(1, 2))
        return output + (scaling * delta).reshape(output.shape).to(output.dtype)

    return hook


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
    member_batch: int,
    collect: list | None = None,
    collect_prompts: int = 0,
) -> tuple[list[float], list[int], int]:
    """Score all 2N antithetic members on one mini-batch.

    All +ε members run first in chunked batched generates, then all −ε
    members through identically laid-out chunks with the same seeds —
    ``generate`` reseeds per call, so aligned rows of a +/− chunk pair
    consume the same sampling stream and decode noise partly cancels inside
    each antithetic pair (common random numbers). Returns ``(fitnesses,
    completion_lengths, tokens)`` with the fitnesses interleaved back to
    [+ε₀, −ε₀, +ε₁, −ε₁, ...] — the layout ``ESEngine.update`` zips against.
    """
    by_sign: dict[float, list[float]] = {1.0: [], -1.0: []}
    lengths: list[int] = []
    tokens = 0
    k = len(prompts)
    for sign in (1.0, -1.0):
        for start in range(0, len(noise), member_batch):
            chunk = noise[start : start + member_batch]
            chunk_prompts = [p for _ in chunk for p in prompts]
            with engine.population(chunk, sign):
                gen = generate(
                    model,
                    tok,
                    chunk_prompts,
                    decode,
                    seed=gen_seed,
                    batch_size=len(chunk_prompts),
                    progress=False,
                )
            tokens += gen.tokens
            lengths.extend(len(c) for c in gen.completions)
            for j in range(len(chunk)):
                member = gen.completions[j * k : (j + 1) * k]
                by_sign[sign].append(fitness(prompts, member, columns))
                if collect is not None:
                    member_idx = start + j
                    for p in range(min(collect_prompts, k)):
                        collect.append(
                            (sign, member_idx, p, member[p], gen.clipped[j * k + p])
                        )
    fits = [f for pair in zip(by_sign[1.0], by_sign[-1.0]) for f in pair]
    return fits, lengths, tokens


def _warm_start_tokens(adapter: str) -> int | None:
    """The token bill of the run that produced ``adapter`` — checkpoints live
    inside run dirs, so the budget file sits next to or one level above."""
    for d in (Path(adapter), Path(adapter).parent):
        budget = d / "token_budget.json"
        if budget.exists():
            tokens = json.loads(budget.read_text()).get("num_tokens")
            # TRL reports token counts as floats; budgets sum as ints.
            return None if tokens is None else int(tokens)
    return None


def _load_es_model(cfg: RunConfig) -> tuple[PreTrainedModel, int | None]:
    """Load the policy the ES leg perturbs and report what it already cost.

    Returns ``(model, warm_tokens)``: a cold base + fresh LoRA (``warm_tokens``
    None), or a warm start from a trained adapter whose own run's token bill is
    charged to this run's budget. Same loader as eval, so the saved adapter's
    module tree matches what eval (and TRL) reload — wrapper architectures
    otherwise silently load zero adapter weights (see eval.runner._model_class);
    the warm-start path rides the same loader's adapter branch.
    """
    if cfg.es_init_adapter:
        model = load_model(cfg.model, cfg.es_init_adapter)
        warm_tokens = _warm_start_tokens(cfg.es_init_adapter)
        if warm_tokens is None:
            logger.warning(
                "no token_budget.json found next to %s — the warm-start "
                "tokens go uncharged and this budget understates the true cost",
                cfg.es_init_adapter,
            )
    else:
        model = load_model(cfg.model, "base")
        model = get_peft_model(model, lora_config(cfg.model, cfg.lora_r, cfg.lora_alpha))
        warm_tokens = None
    model.eval()  # dropout off: each member must score exactly the policy it perturbs
    return model, warm_tokens


def resolve_es_scale(cfg: RunConfig, init_norm: float, num_params: int) -> None:
    """Auto-scale σ and the trust region to the adapter geometry, in place.

    σ and R left at the ``-1`` sentinel are derived from ``init_norm`` so the
    defaults transfer across model sizes: σ makes the per-member noise norm
    ``σ·√P`` equal ``es_noise_ratio`` of ``init_norm``, and ``R =
    es_trust_ratio·init_norm``. A fixed σ does not transfer — its noise norm
    grows with √P, so a constant tuned on a small LoRA buries a larger one's
    init under the perturbations (gibberish from step 1). Explicit positive
    values pass through untouched; ``es_trust_region=0`` stays off.
    """
    if cfg.es_sigma < 0:
        cfg.es_sigma = cfg.es_noise_ratio * init_norm / math.sqrt(num_params)
    if cfg.es_trust_region < 0:
        cfg.es_trust_region = cfg.es_trust_ratio * init_norm


def _log_calibration(engine: ESEngine, cfg: RunConfig, num_rows: int) -> None:
    """Startup line on the σ/θ_init geometry, plus a warning when a warm-start's
    per-step noise norm (~σ·√P) dwarfs the init it should refine — at which
    point the first population erases the adapter it started from."""
    noise_norm = cfg.es_sigma * math.sqrt(engine.num_params)
    init_norm = engine.init_norm()
    logger.info(
        "ES: task=%s model=%s rows=%d population=%dx2 sigma=%g lr=%g steps=%d "
        "init_norm=%.2f step_noise_norm=%.2f trust_region=%s",
        cfg.task,
        cfg.model,
        num_rows,
        cfg.es_population,
        cfg.es_sigma,
        cfg.es_lr,
        cfg.es_steps,
        init_norm,
        noise_norm,
        cfg.es_trust_region or "off",
    )
    if noise_norm > init_norm:
        anchor = "warm-start adapter" if cfg.es_init_adapter else "cold LoRA init"
        logger.warning(
            "per-step noise norm %.1f exceeds the %s norm %.1f — σ this large "
            "buries the init under the perturbations (gibberish completions); "
            "lower --es-sigma or use the -1 auto default so σ·√P sits well "
            "below the init norm",
            noise_norm,
            anchor,
            init_norm,
        )


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
    # Same chat-template wrap as GRPO + eval (train==eval invariant).
    train_ds = apply_chat_template(train_ds, tok, cfg.chat_template)
    model, warm_tokens = _load_es_model(cfg)

    # Construct with safe placeholders for any -1 auto sentinels — a negative σ
    # or trust region is a fragile state to leave the engine in (e.g. a negative
    # trust region projects with a negative shrink). The real values are pushed
    # back below before any step runs.
    engine = ESEngine(
        model,
        sigma=cfg.es_sigma if cfg.es_sigma >= 0 else 0.0,
        lr=cfg.es_lr,
        seed=cfg.seed,
        trust_region=cfg.es_trust_region if cfg.es_trust_region >= 0 else 0.0,
    )
    # Resolve auto σ/R now that the adapter geometry is known, then push the
    # resolved values back onto the engine so the first step uses them (and
    # run_config.json records the real numbers, not the -1 sentinels).
    resolve_es_scale(cfg, engine.init_norm(), engine.num_params)
    engine.sigma = cfg.es_sigma
    engine.trust_region = cfg.es_trust_region
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
    history_path = out / "history.jsonl"
    history_path.unlink(missing_ok=True)

    dumper = InspectDumper(out / "inspect.jsonl", "es") if cfg.inspect_dump else None
    rubric = get_rubric(spec.rubric) if cfg.inspect_dump else None

    _log_calibration(engine, cfg, len(train_ds))

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
        # datasets>=4 returns a lazy Column from batch[c]; materialize to real
        # lists so the reward seam forwards them (see is_per_sample_column) and
        # nothing downstream sees a Column.
        columns = {c: list(batch[c]) for c in batch.column_names if c != "prompt"}

        noise = engine.sample_noise(step, cfg.es_population)
        # One sampling seed per step (distinct prime stride keeps it off the
        # noise stream), shared by every chunk inside _score_population.
        gen_seed = cfg.seed * _GEN_SEED_STRIDE + step
        dump_this_step = (
            dumper is not None
            and cfg.inspect_every > 0
            and (step + 1) % cfg.inspect_every == 0
        )
        collect: list | None = [] if dump_this_step else None
        fits, lengths, tokens = _score_population(
            engine,
            noise,
            prompts,
            columns,
            fitness,
            decode,
            tok,
            model,
            gen_seed,
            cfg.es_member_batch,
            collect=collect,
            collect_prompts=cfg.inspect_max_prompts if dump_this_step else 0,
        )
        total_tokens += tokens
        engine.update(noise, fits)

        if collect:
            items = [
                {
                    "prompt": prompts[p],
                    "completion": completion,
                    "answer": columns.get("answer", [""] * len(prompts))[p]
                    if "answer" in columns
                    else "",
                    "columns": {c: columns[c][p] for c in columns},
                    "clipped": clipped,
                    "group": p,
                    "member": member_idx,
                    "sign": "+" if sign > 0 else "-",
                }
                for sign, member_idx, p, completion, clipped in collect
            ]
            dumper.write(
                build_records(
                    rubric, tok, cfg.max_completion_length, step + 1, "es", items
                )
            )

        step_times.append(time.perf_counter() - t_step)
        # Fraction of the whole population's rollouts that came back empty
        # (immediate EOS). A spike here while fitness holds/climbs is the tell
        # for a length-collapse / format-dropping reward hack — the kind the
        # mean alone hides (it stays high while mean_len cratered).
        empty_frac = sum(1 for l in lengths if l == 0) / len(lengths)
        record = {
            "step": step + 1,
            "fitness_mean": sum(fits) / len(fits),
            "fitness_best": max(fits),
            "mean_len": sum(lengths) / len(lengths),
            "empty_frac": empty_frac,
            "tokens": total_tokens,
            "theta_dev": engine.theta_dev(),
            "step_time": step_times[-1],
        }
        with history_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info(
            "step %d/%d fitness mean=%.4f best=%.4f mean_len=%.0f empty=%.2f "
            "tokens=%d theta_dev=%.2f step_time=%.1f",
            step + 1,
            cfg.es_steps,
            record["fitness_mean"],
            record["fitness_best"],
            record["mean_len"],
            record["empty_frac"],
            total_tokens,
            record["theta_dev"],
            record["step_time"],
        )
        if cfg.save_steps and (step + 1) % cfg.save_steps == 0 and step + 1 < cfg.es_steps:
            model.save_pretrained(str(out / f"checkpoint-{step + 1}"))

    model.save_pretrained(str(out / "checkpoint-final"))

    runtime = time.perf_counter() - t_start
    budget = TokenBudgetLog(
        num_tokens=total_tokens + (warm_tokens or 0),
        global_step=cfg.es_steps,
        train_runtime=runtime,
        mean_step_time=sum(step_times) / len(step_times) if step_times else None,
        tokens_per_second=total_tokens / runtime if runtime else None,
        warm_start_tokens=warm_tokens,
        source="es_loop",
    )
    if torch.cuda.is_available():
        budget.peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    budget.save(out / "token_budget.json")

    logger.info("ES finished; artifacts in %s", out)
    return out
