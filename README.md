# GRPO × ES

One training spine, two optimizers: **GRPO** (gradient RL via TRL) and **Evolution
Strategies** (gradient-free, perturb-and-rank), selected with `--method`. Same prompts, same
verifiable reward, same config schema — only the inner optimizer loop differs.

Design: [GRPO_ES_ARCHITECTURE.md](GRPO_ES_ARCHITECTURE.md). Current state: the shared spine
(config / tasks / rewards / token budget) plus the **GRPO leg** on two generated smoke tasks
(`toy`, `countdown`). The benchmark tasks, held-out eval runner, hub-env adapter, and the ES
leg land on top of this skeleton.

Hardware target: a single RTX 5090 (CUDA, bf16). Default model: `Qwen/Qwen3.5-0.8B`
(instruct — follows the `<think>`/`<answer>` template far more reliably than the `-Base`
checkpoint).

## Install

Linux, Python 3.11, CUDA for real training.

```bash
conda create -n FoMo-RL python=3.11
conda activate FoMo-RL
./scripts/setup_env.sh
```

The setup script runs **two** pip invocations on purpose: `trl` and `verifiers` declare
incompatible `datasets` bounds, so a single resolve refuses the combination while the
two-step install lands on the known-good pairing (`datasets` 4.6.x). Details in
`requirements.txt` — the single source of truth for dependency pins.

Optional: `pip install -r requirements-trackio.txt` for the `--trackio` metrics dashboard.

## Run

```bash
# Smallest end-to-end check (seconds on GPU):
python run.py --config configs/smoke_grpo.toml

# A real run:
python run.py --method grpo --task countdown --model Qwen/Qwen3.5-0.8B \
  --max-samples 100 --seed 0 --output-dir outputs/grpo-countdown
```

Defaults live in TOML files under `configs/` (keys are config field names);
pass one with `--config` and override any of it with explicit flags. Every knob
is also a CLI flag (see `python run.py --help`); a run writes
`run_config.json` (all hyperparameters + git commit), `checkpoint-final/` (the LoRA
adapter), and `token_budget.json` (tokens processed — the cost currency for comparing
optimizers) into `--output-dir`.

Two seeds, deliberately separate: `--seed` drives the optimizer, `--data-seed` (pinned by
default) drives the train-slice shuffle — so a seed sweep trains every run on the same rows.

## Tasks

| `--task` | Reward | Purpose |
|---|---|---|
| `toy` | last-letter concatenation, exact match | wiring smoke test |
| `countdown` | reach a target using each operand exactly once (safe AST eval) | first non-trivial reward |

Rewards are [PrimeIntellect `verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
`Rubric` objects wrapped as TRL reward functions (`grpo_es/rewards/trl_bridge.py`), plus a
**graded** format reward for the `<think>/<answer>` scaffold — partial credit keeps the
within-group advantage alive on fresh models where an all-or-nothing check scores 0
everywhere.

## Tests

```bash
pytest            # unit tests, no GPU needed
pytest -m slow    # (none yet at this stage)
```
