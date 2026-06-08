# GRPO × ES

One training spine, two optimizers: **GRPO** (gradient RL via TRL) and **Evolution
Strategies** (gradient-free, perturb-and-rank), selected with `--method`. Same prompts, same
verifiable reward, same config schema — only the inner optimizer loop differs.

Design: [GRPO_ES_ARCHITECTURE.md](GRPO_ES_ARCHITECTURE.md). Current state: the shared spine
(config / tasks / rewards / token budget), the **GRPO leg**, a first cut of the **ES leg**
(antithetic LoRA-subspace ES — population batching, warm-start and the trust region still
to come), two generated smoke tasks (`toy`, `countdown`), two benchmark tasks (`gsm8k`,
`mmlu_pro`), two instruction-following tasks (`ifeval`, `ifbench`), the held-out
**eval runner** (KL-to-base, paired significance tests), a small model alias ladder, and
the **hub-env adapter** (`--task env:<owner>/<env>`).

Hardware target: a single RTX 5090 (CUDA, bf16).

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
python run.py --config configs/grpo_gsm8k.toml --seed 0
```

Defaults live in TOML files under `configs/` (keys are config field names);
pass one with `--config` and override any of it with explicit flags. Every knob
is also a CLI flag (see `python run.py --help`); a run writes
`run_config.json` (all hyperparameters + git commit), `checkpoint-final/` (the LoRA
adapter), and `token_budget.json` (tokens processed — the cost currency for comparing
optimizers) into `--output-dir`.

Two seeds, deliberately separate: `--seed` drives the optimizer, `--data-seed` (pinned by
default) drives the train-slice shuffle — so a seed sweep trains every run on the same rows.

`--method es` runs the same task / reward / budget spine through evolution strategies
instead of TRL (`configs/smoke_es.toml` is the seconds-scale wiring check): each step
draws `--es-population` antithetic perturbation pairs in the LoRA parameter space, scores
all 2N members on an `--es-eval-batch`-prompt mini-batch with the exact weighted reward
GRPO trains on, and moves the adapter along the rank-weighted noise sum (`--es-sigma`,
`--es-lr`, `--es-steps`). Fitness decoding matches the training temperature with one
shared sampling seed per step, so decode noise partly cancels inside each antithetic
pair (`--es-greedy-fitness` for the low-variance ablation). First cut: members generate
sequentially — the batched population forward, warm-start and the trust region are the
planned follow-ups (architecture §6).

## Models

`--model` takes an HF repo id, a local path, or one of these aliases (resolved to the
canonical id before `run_config.json` is written):

| Alias | Repo |
|---|---|
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` (default) |
| `smollm2-135m` / `smollm2-360m` / `smollm2-1.7b` | `HuggingFaceTB/SmolLM2-*-Instruct` |
| `lfm2.5-350m` | `LiquidAI/LFM2.5-350M` |
| `lfm2.5-1.2b` | `LiquidAI/LFM2.5-1.2B-Instruct` |

Deliberately small: the optimizer comparison is only informative while the base model has
headroom on the task. LoRA targets are architecture-aware (`grpo_es/models.py`) — LFM2's
hybrid conv+attention stack names its projections differently from Llama-family models.

## Tasks

| `--task` | Reward | Purpose |
|---|---|---|
| `toy` | last-letter concatenation, exact match | wiring smoke test |
| `countdown` | reach a target using each operand exactly once (safe AST eval) | first non-trivial reward |
| `gsm8k` | boxed number, symbolic equivalence via `math_verify` | math benchmark |
| `mmlu_pro` | normalized letter match (10-way MCQ) | knowledge benchmark |
| `ifeval` | fraction of verifiable instructions followed | instruction following |
| `ifbench` | same checkers, newer and harder instruction set | harder IF benchmark |

`ifeval` and `ifbench` are served by their PrimeIntellect hub environments
(`primeintellect/ifeval`, `primeintellect/ifbench` — nothing vendored), so they need the
one-time `.venv-prime` setup from the **Hub environments** section below. Their stock
reward is the all-or-nothing "every instruction followed", which collapses GRPO's
within-group advantage on hard prompts — these tasks train on the env's graded
per-instruction rate instead, and the strict pass/fail stays in the metrics. Both
benchmarks publish a single eval-only split (541 / 300 rows); each spec pins a 100-row
holdout and the training draw excludes exactly that window.

`mmlu_pro` scores with a letter-match rubric rather than the math rubric: symbolic
verification false-negatives on the variants a small model emits (`C.`, `**C**`,
`C) Paris`), which under GRPO actively pushes the policy *away* from correct answers. The
math rubric still rides along at weight 0 to log the size of that gap.

Rewards are [PrimeIntellect `verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
`Rubric` objects wrapped as TRL reward functions (`grpo_es/rewards/trl_bridge.py`), plus a
**graded** format reward for the `<think>/<answer>` scaffold — partial credit keeps the
within-group advantage alive on fresh models where an all-or-nothing check scores 0
everywhere.

## Hub environments

Single-turn tasks from the
[PrimeIntellect Environments Hub](https://app.primeintellect.ai/dashboard/environments)
plug in as `--task env:<owner>/<env>` — their dataset and rubric are consumed directly,
nothing gets vendored. Hub environments are pip wheels with their own dependency trees
(a fresh `verifiers`, a newer `openai`); installing one into FoMo-RL would re-resolve the
pinned trl/verifiers/datasets combination, so hub deps live in a dedicated venv and the
trainer talks to it through a small worker subprocess (JSON lines over stdin/stdout,
`scripts/prime_env_worker.py`):

```bash
# one-time: build .venv-prime and install environment(s) into it (needs uv);
# the ifeval/ifbench tasks install the same way:
#   scripts/setup_prime_venv.sh primeintellect/ifeval primeintellect/ifbench
scripts/setup_prime_venv.sh primeintellect/gsm8k

# then train / eval from the normal FoMo-RL env:
python run.py --task env:primeintellect/gsm8k --model smollm2-360m \
  --output-dir outputs/grpo-env-gsm8k
python -m grpo_es.eval --task env:primeintellect/gsm8k --model smollm2-360m \
  --adapter base outputs/grpo-env-gsm8k/checkpoint-final
```

`$PRIME_ENV_PYTHON` points the worker at a different venv's python if needed. Env tasks
score the **raw response** (no `<think>/<answer>` scaffold), so the format reward is
dropped for them automatically.

Two boundaries to know about. Multi-turn / tool / sandbox environments are out of scope:
both optimizer legs generate locally and bypass the env's rollout harness, so only
dataset + rubric are reachable (the adapter warns on non-`SingleTurnEnv` envs). And when
an env publishes only one split, training and the holdout share a pool — the loader warns
unless the spec pins an eval window (`task_from_environment(..., eval_offset=, eval_size=)`
in `grpo_es/tasks/from_env.py`), in which case the training draw excludes exactly that
window. The same function takes `reward_metric=` to promote one of the env's graded
metrics to the reward, or `rubric_override=` to swap in a spine rubric entirely — the
escape hatches for envs whose all-or-nothing stock reward would collapse the within-group
advantage (the named `ifeval`/`ifbench` tasks use `reward_metric`).

## Eval

Held-out evaluation lives behind one entry point and never touches training code paths:

```bash
python -m grpo_es.eval --task gsm8k --model qwen3.5-0.8b \
  --adapter base outputs/grpo-gsm8k/checkpoint-final --kl --json out.json
```

Each task pins its held-out slice in the `TaskSpec` (split, shuffle seed, offset, size) so
it is disjoint from the training draw; generated tasks hold out a fresh generator seed
instead. Per adapter the runner reports the task metric, exact-solve fraction, format
score, a zlib-based **coherence gate** (degenerate-repetition detector), clip fraction,
and — with `--kl` — a k3 estimate of KL(adapter‖base) teacher-forced over the very
completions being scored. `--decode-from RUNDIR` replays a training run's sampling regime;
the default is greedy.

The `--json` payload keeps the full per-sample reward vectors: since every method is
scored on the same prompts, comparisons go through *paired* bootstrap CIs and permutation
tests (`grpo_es/eval/stats.py`), not pooled means.

## Tests

```bash
pytest            # unit tests, no GPU needed (two download a few HF rows;
                  # the hub-env end-to-end tests skip without .venv-prime)
pytest -m slow    # (none yet at this stage)
```
