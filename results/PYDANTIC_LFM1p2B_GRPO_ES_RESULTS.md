# Pydantic-adherence (LFM2.5-1.2B): GRPO vs ES — matched-protocol results

Appendix table for the report. All numbers are held-out, difficulty-0, same data and budget for both
arms. Full debugging trail in `PYDANTIC_ES_DEBUGGING.md` (§7) and `PYDANTIC_ES_REVELATIONS.md`.

## Setup

| | |
|---|---|
| Model | `LiquidAI/LFM2.5-1.2B-Instruct` (LoRA: r=16, α=32, dropout=0.05, targets q/k/v/out_proj; adapter ‖θ₀‖=13.46) |
| Task | `env:primeintellect/pydantic-adherence` (emit JSON validating against a per-sample Pydantic schema) |
| Reward | graded rubric = fraction of model fields that validate (`grpo_es/rewards/pydantic_graded.py`); dense, not the env's binary |
| Difficulty filter | difficulty-0 rows only (`PYDANTIC_DIFFICULTY_KEEP_FILE`, ~1.3k of 1971), AND schema-loadable |
| Train prompts | 256 unique per seed = `shuffle(seed)[0:256]` |
| Held-out | `[500:600]` = 100 prompts, disjoint from train (500 ≥ 256) |
| Base anchor | reward **0.4421**, exact-match **0.070** (same slice) |
| Decode (eval) | greedy, temp 0.8, max_new 1024, max_prompt 2048; n=100 paired |
| Seeds | 0, 1, 2 |
| Hardware | 1× RTX 5090 (32 GB), env `FoMo-RL` |

## Hyperparameters

| | **GRPO** | **ES** |
|---|---|---|
| LR | 3e-5 | es_lr 1e-4 |
| KL / trust | β = 0.02 | trust_ratio 1.0 → cap = ‖θ₀‖ = 13.46 |
| Noise | — | noise_ratio 0.2 → σ = 1.94e-3, step-noise-norm 2.69 |
| Batch | num_gen 8, per-device 1, grad-accum 8 (1 prompt-group/step) | population 16×2 (antithetic), eval_batch 8, member_batch 16 |
| Steps | 1 epoch → **256** optimizer steps (save 25) | **48** ES steps (matched endpoint; s0 also ran to 96) (save 24) |
| Lengths | max_prompt 2048, max_completion 1024, chat_template on | (same) |
| Rollouts / seed | 256 × 8 = **2,048** | 48 × 32 × 8 = **12,288** (~14.5M generated tokens) |

## Compute & wall-time (per seed)

| arm | train steps | train wall | eval wall | peak VRAM |
|---|---|---|---|---|
| GRPO | 256 | ~12.7 min | base+final ~1 min (s0 full curve, 13 ckpts: ~5 min) | ~14.3 GB (measured) |
| ES | 48 | ~25 min (~31 s/step) | ~1 min | < GRPO (forward-only: no backprop / optimizer state) |

ES does **~6× the generations** and **~2× the wall-clock** of GRPO to reach its endpoint, yet lands at 36%
of GRPO's gain (below). (ES s0 was extended to 96 steps ≈ 47 min — it eroded, see controls.)

## Held-out results

Δ = vs base reward 0.4421; exact = absolute exact-match (base 0.070).

| seed | GRPO Δreward | GRPO exact | ES Δreward | ES exact |
|---|---|---|---|---|
| 0 | +0.1551 | 0.230 | +0.0986 | 0.140 |
| 1 | +0.2119 | 0.240 | +0.0433 | 0.070 |
| 2 | +0.2216 | 0.270 | +0.0672 | 0.110 |
| **mean** | **+0.1962 ± 0.0359** | **0.247** | **+0.0697 ± 0.0277** | **0.107** |
| seed-t (2 df) | **9.46** | | **4.35** (p < 0.05) | |

**Matched ratio: ES recovers 36% of GRPO's held-out gain (+0.0697 / +0.1962) on identical data and budget.**
GRPO also moves exact-match decisively (0.070 → ~0.247); ES barely touches it (0.070 → ~0.107, and seed 1
returns to base). The GRPO seed-0 curve climbs then plateaus without erosion (peak +0.175 @ step 75, settles
+0.155 through step 256); ES peaks then decays (s0: +0.0986 @ step 48 → +0.0550 @ step 96).

## Controls (why the ES +0.070 is real, and bounded)

| control | config | step-48 Δreward | reading |
|---|---|---|---|
| **Shuffle** (s0) | same cap/lr, per-step advantages permuted → same step-norm, random direction | **−0.0280** | selection edge = +0.0986 − (−0.0280) = **+0.1266** (paired t=3.58). The gain is direction (selection), not same-norm movement. |
| **Uncapped** (s0) | trust_ratio 4.0 (cap ~54) | **−0.3279** @ step 96 (KL 0.835, len 1010, exact 0.010) | removing the cap → monotonic collapse into the token-spam reward-hacking basin. **The cap is load-bearing.** |

## Verdict

ES on this cell is **regime-dependent**: null at low KL (movement floor), a real selection-driven
**+0.070 (3-seed, all positive, p < 0.05 = 36% of matched GRPO)** in the capped forced-KL window, and
destructive uncapped. GRPO is the clear winner here (+0.196, t=9.46, decisive exact-match lift) at lower
compute, but the "ES is a dead end on this task" claim is false: it has a usable, cap-gated window.

---
*Repro: `scratchpad/grpo_diff0.sh` (GRPO), `es_diff0_big_rep.sh` (ES s1/s2), `es_diff0_shuf_big.sh`
(shuffle), `es_diff0_grow.sh` (uncapped). Evals: `outputs/prime_sweep/pydantic/eval_{grpo_diff0,diff0_big,
diff0_shuf_big,diff0_grow}_s*.json`. Configs: `configs/grpo_pydantic_adherence.toml` (max_samples→256),
`configs/es_pydantic_adherence.toml` (+ CLI overrides above). The `--es-shuffle-fitness` flag and the
difficulty-filter plumbing are uncommitted working-tree changes.*
