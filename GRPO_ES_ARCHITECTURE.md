# GRPO × ES — Architecture

One codebase that fine-tunes small language models on verifiable tasks with either **GRPO**
(gradient RL via TRL) or **Evolution Strategies** (gradient-free, perturb-and-rank), selected by
`--method`, on a shared reward + eval + data spine. **Hardware target: a single RTX 5090 (CUDA,
bf16).**

This document describes the **target architecture** — the shape the codebase is being built
toward and the design decisions behind it. See [README.md](README.md) for install + usage, and
**§0 Status** below for what exists today versus what is still planned.

---

## 0. Status

**Built:** the shared spine, the GRPO leg, the held-out eval runner, the benchmark and
instruction-following tasks, the hub-env adapter, and a first cut of the ES leg.

- `config/` — `RunConfig` (one run = one config), CLI + TOML defaults, `run_config.json` stamped
  with the git commit.
- `tasks/` — `TaskSpec` contract + the generated smoke tasks `toy`/`countdown`, the benchmark
  tasks `gsm8k`/`mmlu_pro`, the instruction-following tasks `ifeval`/`ifbench`, and the
  PrimeIntellect hub-env adapter (`env:<owner>/<env>`, behind an isolated `.venv-prime` worker).
- `rewards/` — verifiers `Rubric` → TRL `reward_func` bridge, graded `<think>/<answer>` format
  reward, rubric registry (built-in + dynamic hub rubrics).
- `metrics/budget.py` — forward-token budget pulled from TRL's logs (and the ES loop).
- `eval/` — rubric scoring outside the training loop (`metrics.py`, same scoring path as training)
  plus the held-out **runner** (`runner.py`): disjoint-slice generation, KL-to-base (`kl.py`), and
  paired significance tests (`stats.py`).
- `methods/grpo.py` — the GRPO leg (thin TRL `GRPOTrainer` wrapper).
- `methods/es.py` — the ES leg, **first cut**: antithetic LoRA-subspace OpenAI-ES, forward passes
  only. The batched population forward, warm-start, and the trust region are still to come.

**Planned (designed here, not yet implemented):** the `simpleqa` judge task and the ES
follow-ups — the grouped-bmm batched population forward, warm-start, and trust-region projection
(§6). The sections below describe how the remaining pieces are **intended** to slot onto the
existing spine; planned components are marked *(planned)* where the distinction matters.

---

## 1. Design goal

One repo where the *only* thing that differs between a GRPO run and an ES run is the inner
optimizer loop. Same prompts, same verifiable reward, same eval, same config schema. The spine
boundary is the whole game: if it leaks into the method legs, "ES vs GRPO" stops being a
controlled comparison.

This is what the spine is built to enable: a question like *"does ES reward-hack a dense checker
where GRPO doesn't?"* is only answerable if both legs score through the identical rubric object
and the identical held-out eval — so that the optimizer is provably the only difference. Holding
that invariant is the reason the spine exists, and the reason scoring/eval/data live above the
method split rather than inside either leg.

**The spine's scope is also its boundary:** text-in/text-out, **verifiable, single-turn** tasks.
Multi-turn, tool-use, sandboxed, and multimodal rewards are out of scope by construction (both
legs generate locally; there is no rollout-harness seam — see §4.2).

## 2. The spine (shared, method-agnostic)

```
   config/   RunConfig: one grid cell (method × model × task × seed × knobs) = one config,
       │     saved as run_config.json with the git commit
       ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │ tasks/    TaskSpec + loaders: toy · countdown · gsm8k ·            │
 │           mmlu_pro · ifeval · ifbench · env:<owner>/<env>          │
 │           (hub)  (built);   simpleqa  (planned)                    │
 │ rewards/  verifiers Rubric contract: task rubric + graded format;  │
 │           trl_bridge wraps a rubric as a TRL reward_func  (built). │
 │           weight-0 shadows (built) + soft-overlong  (planned)      │
 │ eval/     rubric scoring outside loop · held-out runner ·          │
 │           KL-to-base · stats  (built);  aggregate  (planned)       │
 │ metrics/  forward-token budget from TRL logs  (built);            │
 │           FLOPs estimate · warm-start charge  (planned)            │
 └────────────────────────────────────────────────────────────────────┘
       │                                      │
       ▼                                      ▼
 ┌──────────────────────────┐    ┌──────────────────────────────────┐
 │ methods/grpo.py  (built) │    │ methods/es.py  (planned)         │
 │ TRL GRPOTrainer wrapper  │    │ ESEngine: batched antithetic     │
 │ KL-to-base β=0.04        │    │ population via grouped-bmm LoRA   │
 │ arch-aware LoRA targets  │    │ hook · rank-norm · warm-start +   │
 │ bf16 load                │    │ trust region + coherence gate     │
 └──────────────────────────┘    └──────────────────────────────────┘
```

**The two contracts both legs agree on:**
- `tasks.base.TaskSpec` — prompt builder + dataset split + rubric name (and, once the eval runner
  lands, the held-out slice spec: `eval_split/seed/offset/size`, decode caps, `metric_label`).
  Putting the eval slice spec *on the task* is what is meant to keep train/holdout disjointness a
  property of the spine, not of individual eval scripts.
- `verifiers` `Rubric.score_rollout(state) -> float` — the single scoring interface. The same
  rubric object is wrapped as a TRL `reward_func` (`rewards/trl_bridge.py`) today, and is intended
  to be consumed directly by the ES fitness loop when that leg lands.

Everything above the method split is written once; below it live two genuinely different loops.

**The TRL × verifiers seam:** TRL's `GRPOTrainer` owns rollout and just calls `reward_funcs`.
PrimeIntellect `verifiers`, in native use, owns the whole Environment (dataset + rollout + rubric)
for its own trainer. Combining them means `verifiers` is used as a **rubric/parser library only** —
its `Rubric`/`Parser` get wrapped as a TRL `reward_func`, and the same rubric object is meant to be
imported by the ES leg. A `verifiers` Environment never drives rollout. That keeps the reward
identical across both legs without two rollout drivers fighting — and it is the same boundary that
shapes what the hub-env adapter can and cannot consume (§4.2).

## 3. Module map

Built today:

| Module | Responsibility |
|---|---|
| `config/run_config.py` | `RunConfig` dataclass + argparse + `--config` TOML defaults; saves `run_config.json` (with git commit) per run |
| `tasks/` | one file per task; `base.py` (`TaskSpec` + `build_dataset`), `registry.py` (`SPECS`/`LOADERS`), the generated `toy`/`countdown` loaders, shared `prompts.py` |
| `rewards/` | `registry.py` (`get_rubric` + `make_trl_reward_funcs`), `trl_bridge.py` (rubric→`reward_func`, graded format reward), `toy_rubric.py`, `countdown_rubric.py` |
| `methods/grpo.py` | thin TRL `GRPOTrainer` wrapper; arch-aware LoRA targets; bf16 load |
| `methods/callbacks.py` | compact one-line-per-step metrics logging |
| `eval/metrics.py` | score a batch of completions through the same adapter the trainer uses |
| `metrics/budget.py` | forward-token budget + throughput, extracted from TRL's `log_history` |
| `run.py` | entrypoint: `--method grpo --task ... --model ...` |
| `scripts/` | `setup_env.sh` (install); `configs/*.toml` hold run presets |

Planned modules (described in §4–§8):

| Module | Intended responsibility |
|---|---|
| `methods/es.py` | `ESEngine`: antithetic population, rank-normalized update, grouped-bmm forward hook, trust-region projection, coherence gate, warm-start token charge |
| `tasks/from_env.py` | hub-env adapter (§4.2): `task_from_environment` / `register_environment_task`, `HubEnvClient`, `RemoteEnvRubric` |
| `eval/runner.py`, `eval/kl.py`, `eval/stats.py`, `eval/aggregate.py` | held-out generation (no chat template), KL-to-base, significance stats, aggregation |
| `models.py` | model aliases + per-arch LoRA-target notes |
| `scripts/prime_env_worker.py`, `setup_prime_venv.sh` | the isolated `.venv-prime` worker for hub envs |

## 4. Task layer — builtins + the PrimeIntellect hub adapter

### 4.1 Builtin tasks

Built today, both generated on the fly:

| `--task` | Reward | Purpose |
|---|---|---|
| `toy` | exact match on last-letter concatenation | wiring smoke test |
| `countdown` | reach a target using each operand exactly once (safe AST eval) | first non-trivial reward |

Planned benchmark lineup, with the reward shaping each is meant to use:

| `--task` | Reward (planned) | Why it's shaped this way |
|---|---|---|
| `gsm8k` | `verifiers` MathRubric (boxed) | R1 `<think>/<answer>` scaffold + graded format reward |
| `mmlu_pro` | MCQ letter rubric (exact letter) | a MathRubric false-negatives on MCQ forms (`C.`, `**C**`) → bad advantage; it would ride as a weight-0 shadow |
| `ifeval` / `ifbench` | **dense** instruction-following *rate* (vendored Google/AllenAI checkers) | an all-or-nothing prompt-level pass collapses within-group variance → no GRPO gradient, flat ES fitness. Strict drives the gradient; loose rides as a weight-0 shadow |
| `simpleqa` | LLM-judge (OpenRouter) | no local gold-match exists; every rollout is a paid judge call → eval-oriented |

The intended rule for tasks graded on the **raw response** (`ifeval`, `ifbench`, `simpleqa`, hub
envs): set `r1_template=False` — no `<think>/<answer>` scaffold, format reward auto-dropped — so the
template can't corrupt what the checker/judge sees.

### 4.2 The hub-env adapter (`--task env:<owner>/<env>`) — *(built)*

The spine speaks the `verifiers` data model natively, so any **single-turn** env
from the [PrimeIntellect Environments Hub](https://app.primeintellect.ai/dashboard/environments)
is consumed plug-and-play — *dataset + rubric only*.

```
 grpo-es venv (core pins: trl, transformers 5.x, verifiers==0.1.14, openai pin)
 ─────────────────────────────────────────────────────────────────────────────
  tasks/from_env.py
    task_from_environment / register_environment_task   ← lazy, on first 'env:' lookup
    HubEnvClient ──── JSON-lines over stdin/stdout ────┐
    RemoteEnvRubric (verifiers-Rubric facade:           │   subprocess
      score_rollout → remote 'score' op)                ▼
 ─────────────────────────────────────────────────  .venv-prime  ──────────────
  scripts/prime_env_worker.py: owns load_environment(env_id); ops:
    ping / load / dataset / score        (scripts/setup_prime_venv.sh builds it)
```

**Why a separate venv, not an install extra:** hub envs are pip wheels with their own deps, and
`verifiers.v1`'s loader drags in `openai-agents`, which conflicts with the core `openai` pin. pip
resolves extras against the same site-packages, so an extra can't isolate — a dedicated
`.venv-prime` bridged by a worker subprocess can. The core pins are never perturbed;
`$PRIME_ENV_PYTHON` overrides the worker python.

**The hard ceiling — rollout harnesses are unreachable.** A verifiers `Environment`'s generation
surface drives an OpenAI-compatible inference *server*. Both legs generate **locally** (TRL owns
GRPO rollouts; ES needs `generate` under its grouped-bmm hook), so only `env.get_dataset()` +
`env.rubric` are consumable. Multi-turn / tool-use / sandbox / browser envs are out of scope by
construction — the same line as the spine's verifiable-reward boundary.

The deliberate spine adaptations are meant to be preserved as hooks, not overridden by env
defaults: chat-message prompts flattened to raw strings (so eval's "no chat template" invariant and
the R1 scaffold survive); a `rubric_override=` to keep a spine rubric (dense IF rate, MCQ letter)
where an env's stock rubric would kill the gradient; `eval_offset/eval_size` to carve a disjoint
holdout when an env ships a single split; and the `_ifeval`/`_ifbench` checkers stay vendored even
though hub envs for them exist, to escape the dependency pull the vendoring avoids.

Net intent: onboarding a new single-turn hub task is ~10 lines (vs vendor a checker + write a loader
+ write a rubric), with existing builtin tasks untouched.

## 5. Generation: plain PyTorch everywhere, no inference engine

Both legs generate with HF/PyTorch `generate` in bf16. There is deliberately **no vLLM (or other
inference engine) anywhere in the codebase**:

- The head-to-head lives at **135M–1.2B**, where rollout generation is not the bottleneck — the
  main GRPO wall-clock lever is the *train micro-batch* (`pdtbs × ga`), not the rollout engine.
- A second inference stack on one GPU costs VRAM (reserved cache starving the trainer), startup
  time, and failure modes (weight-sync shape asserts, sleep-mode silently generating from the base
  model) — none of which is the science.
- Fairness is not meant to depend on a shared engine: it is intended to rest on the **forward-token
  budget** logged in `metrics/`, with bf16 HF `generate` keeping both legs' numerics identical
  (which matters for KL-to-base and the Goodhart gap).

ES generation throughput is intended to be solved *inside* PyTorch via the **grouped-bmm population
hook** (§6): the whole population batched into one `generate` call rather than a sequential
population loop. The same reasoning excludes Unsloth from the head-to-head (its fused attention
replaces the PEFT LoRA `Linear` the hook attaches to, and 4-bit confounds KL/Goodhart). A GRPO-only
7–9B run that ever needs a fast engine would put it in an isolated venv, off the comparison path.

## 6. The two legs

GRPO is built; the ES column describes the intended design.

| | GRPO (`methods/grpo.py`, built) | ES (`methods/es.py`, planned) |
|---|---|---|
| Engine | TRL `GRPOTrainer` (rollouts owned by TRL) | custom `ESEngine` loop |
| Pass | forward + backward | forward-only |
| Params | LoRA (arch-aware targets: `o_proj` vs `out_proj` …; `--no-peft` for full FT) | same LoRA subspace — perturbs the adapter's `(A,B)` only |
| Population | G=`--num-generations` rollouts/prompt | 2N antithetic members in one batched `generate`: a forward hook computes per-member `x·Aᵢᵀ·Bᵢᵀ` via grouped `bmm` with the adapter disabled |
| Update | group-relative advantage + clip | rank-normalized utilities → weighted noise sum |
| Stay-near-base | KL-to-base in the objective (β=0.04) | param-space trust region: cap ‖cumulative LoRA delta‖₂ from the start point and project back each step — the weight-space analogue of the KL anchor |
| Init | base + zero-delta LoRA | cold, or warm-start from a GRPO checkpoint (its training tokens charged to the ES budget) |
| Anti-Goodhart extras | (KL does the work) | optional coherence gate: zlib-ratio degeneracy gate on each member's greedy pass, multiplying fitness ∈ [0,1] |

The ES extras anticipate a known failure mode rather than being speculative: naïve ES tends to
reward-hack dense checkers into token-spam. The planned mitigation is **warm-start + trust region +
small σ** (σ is only meaningful relative to the current delta norm, so warm-starting without
rescaling σ is expected to reproduce the collapse), and the intended honest hack-detector is **mean
completion length**, not (ES − base). Cold-start defaults are expected to be roughly
`--es-population 32 --es-sigma 0.03 --es-lr 0.05`; these will be validated as the leg is built.

`--seed` is intended to drive the optimizer (ES noise / GRPO trainer) while `--data-seed`
(default 0, pinned) drives the train-slice shuffle, so a seed sweep keeps the split fixed and the
holdout disjoint. (This split is already in place on the GRPO leg.)

## 7. Models (small-model ladder, arch-aware)

The contest is only meaningful where the base model hasn't already saturated the task (a saturated
base ties every optimizer), so the model axis is intended to be a **small-model ladder with
headroom**, not a race to the biggest rung. The default model (`Qwen/Qwen3.5-0.8B`) is wired today;
the wider ladder and its arch-aware LoRA targeting are the planned `models.py` lineup:

| Alias | Model | Arch / notes |
|---|---|---|
| `qwen3.5-0.8b` (default) | Qwen/Qwen3.5-0.8B | hybrid linear-attention; instruct, follows the R1 scaffold |
| `smollm2-135m/360m/1.7b` | HuggingFaceTB SmolLM2-Instruct | plain Llama arch — the clean ladder for head-to-heads |
| `lfm2.5-350m` / `lfm2.5-1.2b` | LiquidAI LFM2.5 | hybrid conv+attn → LoRA targets `out_proj`, not `o_proj` |

`--model` already takes any HF repo id or local path; the intent is that LoRA targeting resolves
per-architecture so both legs run on every rung with no per-model code.

## 8. Eval & fairness (the part that catches lies)

The fairness design (the eval runner is built; the ES-side accounting lands with the ES follow-ups):

- **Held-out is the only number that counts.** Train on one slice, score on a disjoint slice — this
  is what exposes reward-hacking that training fitness hides. The slice spec lives on
  `TaskSpec`. *(built: held-out runner in `eval/runner.py`, rubric scoring in `eval/metrics.py`.)*
- **KL-to-base** lives in `eval/kl.py`, computed identically for both legs and never
  inside a method — an inconsistent estimator would bias the Goodhart comparison at every rung.
  *(built.)*
- **No chat template in the eval runner** — a reproducibility invariant, so historical numbers
  reproduce up to the greedy-bf16 nondeterminism floor. *(built.)*
- **Forward-token budget** (`metrics/budget.py`, built) is the intended cost currency (ES = many
  cheap forwards; GRPO = fewer forwards + backward): TRL's rollout counts on the GRPO side, the
  population×tokens count on the ES side, and a warm-started ES run charged the GRPO tokens it
  consumed.
- Coherence/degeneracy metrics are intended to live in one place (`eval/metrics.py`) and be shared
  by both the held-out eval and the ES gate, so the two never drift.

## 9. Design decisions

| Decision | Rationale |
|---|---|
| **GRPO via TRL `GRPOTrainer`; ES as a separate custom loop** | Share reward/eval/data only; the rollout engines are two genuinely different beasts and pretending otherwise would leak the spine. GRPO cost accounting derives from TRL's reported rollout counts (we don't drive its loop). |
| **`verifiers` as scoring/parsing library only** (pinned `0.1.14`) | One rubric object serves both legs; no Environment ever drives rollout. Prompt builders live in `tasks/`, guaranteeing identical prompts across legs and tasks. |
| **No inference engine; bf16 HF `generate` for both legs** | See §5 — fairness via token budget + identical numerics; throughput via the micro-batch lever (GRPO) and the grouped-bmm population hook (ES). |
| **ES population batched in the forward** (grouped-bmm hook on PEFT LoRA layers) | Intended to remove ES's multiplicative population penalty without custom kernels or an engine; plain batched matmuls. |
| **Hub envs through an isolated `.venv-prime` + worker subprocess** | Install extras can't isolate dependency pins; a venv + JSON-lines bridge can. Core pins never move. |
| **Stock env rubrics replaceable via `rubric_override`** | Dense/MCQ rubric choices are gradient-survival decisions; silent plug-and-play would regress them. |
| **Vendored `_ifeval`/`_ifbench` checkers** | Escapes the verifiers-version dependency pull; the checkers are the reward ground truth and must not drift. |
| **KL estimator in `eval/`, single implementation** | Both legs must measure KL-to-base identically or the Goodhart headline is biased. |
| **Single shared `--seed` for optimizer, separate pinned `--data-seed` for the split** | A naive seed sweep would otherwise train each seed on different rows and leak the holdout. |
| **Unsloth excluded from the head-to-head** | Fused attention replaces the PEFT `Linear` the ES hook needs; 4-bit confounds KL/Goodhart. GRPO-only big-model runs may use it in an isolated venv. |

## 10. Standing risks / boundaries

These are the boundaries the design accepts; several apply to components still being built.

- **Single-turn ceiling:** anything needing an env's rollout harness (multi-turn, tools, sandboxes)
  cannot enter the spine without giving up local generation — which the ES hook and TRL's rollout
  ownership both forbid. Treat that as a project boundary, not a TODO.
- **Hub-env worker is a process boundary:** rubric scoring would cross a subprocess per batch; fine
  at current scales, but a hot rubric in a big ES population would feel it.
- **Base saturation, not optimizer quality, caps the contest** at the upper rungs — picking
  task/model pairs with headroom is the experimental bottleneck.
- **Judge tasks cost money per rollout** (simpleqa and judge-based hub envs): eval-oriented;
  reasoning judges need `max_tokens` headroom or they silently zero every grade.
- **Rubric hot-loop hazard:** sympy-backed math scoring can hang on adversarial completions, so
  rubric calls should carry hard per-call timeouts (thousands of generations are scored per step).
- **ES precision:** perturbations should be drawn against fp32 master weights even though generation
  is bf16 — at small σ the bf16 noise floor would otherwise eat the signal.
