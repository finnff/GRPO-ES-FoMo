"""RunConfig: one run = one config = one grid cell.

Every knob lives here and maps 1:1 onto a CLI flag — no module-level constants
to hand-edit between runs. ``save()`` stamps the git commit so an output
directory is always traceable to the code that produced it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from grpo_es.models import resolve_model_alias

KNOWN_METHODS = ("grpo", "es")
# ifeval/ifbench are hub-backed (need .venv-prime, see README).
KNOWN_TASKS = ("toy", "countdown", "gsm8k", "mmlu_pro", "ifeval", "ifbench")

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def task_arg(value: str) -> str:
    """``--task`` validator: a built-in name, or ``env:<owner>/<env>`` for a
    PrimeIntellect hub environment (registered lazily, needs .venv-prime)."""
    if value in KNOWN_TASKS or value.startswith("env:"):
        return value
    raise argparse.ArgumentTypeError(
        f"unknown task {value!r}; choose from {KNOWN_TASKS} or "
        f"'env:<owner>/<env>' (PrimeIntellect hub — see README)"
    )


@dataclass
class RunConfig:
    # What to run.
    method: str = "grpo"
    task: str = "toy"
    model: str = DEFAULT_MODEL

    # Data & reproducibility. `seed` drives the optimizer (trainer init,
    # sampling); `data_seed` drives the train-slice shuffle and stays pinned,
    # so a seed sweep trains every run on the same rows.
    seed: int = 0
    data_seed: int = 0
    max_samples: int | None = 100
    output_dir: str = "outputs/grpo"
    git_commit: str | None = None  # auto-filled by save()

    # Rollouts.
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 512
    temperature: float = 1.0
    repetition_penalty: float = 1.0

    # Optimization.
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    beta: float = 0.04  # KL-to-base coefficient in the GRPO objective

    # Rewards. Weights are [task, format]; format stays below the task reward
    # so the scaffold never dominates the signal.
    use_format_reward: bool = True
    reward_weights: list[float] = field(default_factory=lambda: [1.0, 0.5])

    # Adapters.
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # ES leg (--method es). Shares seed/max_samples/temperature/
    # max_completion_length/lora_* with the GRPO knobs above; `es_lr` stays
    # separate from `learning_rate` because a rank-utility step size is not
    # comparable to an Adam learning rate.
    es_population: int = 32  # antithetic pairs -> 2N generations per step
    es_sigma: float = 0.03  # perturbation scale in the fp32 LoRA subspace
    es_lr: float = 0.05
    es_steps: int = 200  # ES iterations (ES has no epoch notion)
    es_eval_batch: int = 8  # prompts scored per member per step
    es_greedy_fitness: bool = False  # greedy instead of temperature-matched sampling
    es_member_batch: int = 8  # members per batched generate call (VRAM lever)
    es_init_adapter: str | None = None  # warm-start adapter; its run's tokens get charged
    es_trust_region: float = 0.0  # cap on ||theta - theta_init||_2 (0 = off)

    # Logging.
    verbose: bool = False
    logging_steps: int = 1
    save_steps: int = 50
    use_trackio: bool = False
    trackio_space_id: str | None = None
    inspect_dump: bool = False  # write per-completion rollouts for inspect_run.py
    inspect_every: int = 1  # dump every Nth optimizer/ES step
    inspect_max_prompts: int = 1  # prompts sampled per dumped step

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        if self.git_commit is None:
            self.git_commit = _current_git_commit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _current_git_commit() -> str | None:
    """Best-effort short HEAD hash, marked ``+dirty`` on uncommitted edits."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()

    try:
        head = _git("rev-parse", "--short", "HEAD")
        return f"{head}+dirty" if _git("status", "--porcelain") else head
    except Exception:
        return None


# Flags that don't map straight onto a field: the store_true inversions and
# one rename. Every other flag uses its field name as the argparse dest, so a
# new plain knob is one dataclass field + one add_argument — no third site to
# keep in sync (and no way to silently drop a flag by forgetting one).
_INVERTED_FLAGS = {
    "no_gradient_checkpointing": "gradient_checkpointing",
    "no_format_reward": "use_format_reward",
    "no_peft": "use_peft",
}
_RENAMED_FLAGS = {"trackio": "use_trackio"}


def _build_parser() -> argparse.ArgumentParser:
    d = RunConfig()
    p = argparse.ArgumentParser(
        description="Fine-tune a small LM on a verifiable task (GRPO or ES)."
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        help="TOML file of defaults (e.g. configs/smoke_grpo.toml); keys are "
        "config field names. Explicit flags override it.",
    )

    # SUPPRESS defaults so an unset flag stays absent from the namespace —
    # that's what lets parse_args layer file config under explicit flags
    # without a per-field "was it passed?" sentinel.
    def opt(*names: str, default, **kwargs) -> None:
        help_text = kwargs.pop("help", "")
        kwargs["help"] = f"{help_text} (default: {default})".strip()
        p.add_argument(*names, default=argparse.SUPPRESS, **kwargs)

    def flag(*names: str, **kwargs) -> None:
        p.add_argument(*names, action="store_true", default=argparse.SUPPRESS, **kwargs)

    opt("--method", default=d.method, choices=KNOWN_METHODS)
    opt(
        "--task",
        default=d.task,
        type=task_arg,
        help=f"one of {KNOWN_TASKS} or env:<owner>/<env>",
    )
    opt(
        "--model",
        default=d.model,
        help="HF repo id, local path, or a short alias from grpo_es.models "
        "(e.g. smollm2-360m, qwen3.5-0.8b, lfm2.5-1.2b)",
    )

    opt("--seed", default=d.seed, type=int, help="optimizer seed (sweep this)")
    opt(
        "--data-seed",
        default=d.data_seed,
        type=int,
        help="train-slice shuffle seed (keep pinned across a seed sweep)",
    )
    opt("--max-samples", default=d.max_samples, type=int)
    opt("--output-dir", default=d.output_dir)

    opt("--num-generations", default=d.num_generations, type=int)
    opt("--max-prompt-length", default=d.max_prompt_length, type=int)
    opt("--max-completion-length", default=d.max_completion_length, type=int)
    opt("--temperature", default=d.temperature, type=float)
    opt("--repetition-penalty", default=d.repetition_penalty, type=float)

    opt("--learning-rate", default=d.learning_rate, type=float)
    opt("--num-train-epochs", default=d.num_train_epochs, type=float)
    opt("--per-device-train-batch-size", default=d.per_device_train_batch_size, type=int)
    opt("--gradient-accumulation-steps", default=d.gradient_accumulation_steps, type=int)
    flag("--no-gradient-checkpointing")
    opt("--beta", default=d.beta, type=float)

    flag("--no-format-reward")
    opt(
        "--reward-weights",
        default=d.reward_weights,
        type=float,
        nargs="+",
        metavar="W",
        help="per-reward weights, order [task, format]",
    )

    flag("--no-peft", help="full fine-tune instead of LoRA")
    opt("--lora-r", default=d.lora_r, type=int)
    opt("--lora-alpha", default=d.lora_alpha, type=int)

    opt(
        "--es-population",
        default=d.es_population,
        type=int,
        help="antithetic pairs per ES step (2N generations)",
    )
    opt("--es-sigma", default=d.es_sigma, type=float)
    opt("--es-lr", default=d.es_lr, type=float)
    opt("--es-steps", default=d.es_steps, type=int)
    opt(
        "--es-eval-batch",
        default=d.es_eval_batch,
        type=int,
        help="prompts scored per member per ES step",
    )
    flag("--es-greedy-fitness", help="greedy fitness decoding (low-variance ablation)")
    opt(
        "--es-member-batch",
        default=d.es_member_batch,
        type=int,
        help="population members per batched generate call (VRAM lever)",
    )
    opt(
        "--es-init-adapter",
        default=d.es_init_adapter,
        metavar="CKPT",
        help="warm-start the ES master from a trained adapter; that run's "
        "tokens are charged to this run's budget",
    )
    opt(
        "--es-trust-region",
        default=d.es_trust_region,
        type=float,
        metavar="R",
        help="project ||theta - theta_init||_2 back to R after each step (0 = off)",
    )

    flag("-v", "--verbose")
    opt("--logging-steps", default=d.logging_steps, type=int)
    opt("--save-steps", default=d.save_steps, type=int)
    flag("--trackio", help="log metrics to trackio")
    opt("--trackio-space-id", default=d.trackio_space_id)
    flag(
        "--inspect-dump",
        help="dump per-completion rollouts to <output_dir>/inspect.jsonl "
        "for the inspect_run.py live viewer",
    )
    opt("--inspect-every", default=d.inspect_every, type=int)
    opt("--inspect-max-prompts", default=d.inspect_max_prompts, type=int)
    return p


def _load_config_file(path: Path) -> dict:
    """Read a TOML defaults file; its keys are ``RunConfig`` field names."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(RunConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown config keys in {path}: {sorted(unknown)}; "
            f"known: {sorted(known)}"
        )
    return data


def _cli_overrides(a: argparse.Namespace) -> dict:
    """The flags actually passed, mapped onto field names (inversions handled)."""
    out: dict = {}
    for key, value in vars(a).items():
        if key == "config":
            continue
        if key in _INVERTED_FLAGS:
            out[_INVERTED_FLAGS[key]] = not value  # store_true → field is its negation
        elif key in _RENAMED_FLAGS:
            out[_RENAMED_FLAGS[key]] = value
        else:
            out[key] = value
    return out


def parse_args(argv: list[str] | None = None) -> RunConfig:
    a = _build_parser().parse_args(argv)
    # Precedence: dataclass defaults < TOML config file < explicit CLI flags.
    values: dict = {}
    if getattr(a, "config", None):
        values.update(_load_config_file(Path(a.config)))
    values.update(_cli_overrides(a))
    cfg = RunConfig(**values)
    # Resolve here, after the TOML/CLI merge, so run_config.json always
    # records the canonical repo id — never an alias.
    cfg.model = resolve_model_alias(cfg.model)
    return cfg
