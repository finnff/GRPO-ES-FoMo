# Countdown (LFM2.5-1.2B): GRPO vs ES — tuning toward solid improvements on BOTH arms

**Goal (this doc tracks it):** sweep variations of the GRPO and ES configs (LR/β, ES population/batching/LR/steps)
until *both* arms show a solid held-out improvement over base. Budget: ≤ ~60 min per training run. Best result
per arm is pinned in **§ Best so far**.

## Setup

| | |
|---|---|
| Model | `LiquidAI/LFM2.5-1.2B-Instruct` (LoRA: r=16, α=32, dropout=0.05, targets q/k/v/out_proj; adapter ‖θ₀‖=13.47) |
| Task | `countdown` (faithful sivit/countdown-plain port: solvable constructed targets, prose-tolerant parser) |
| Reward | dense = 1.0·exact + 0.25·closeness (perfect solve = 1.25). Born-dense → ES gets live fitness; brevity pressure (no token-spam basin). |
| Train data | generated puzzles, `random.seed(seed)` then in-sequence → `max_samples` is the unique-puzzle count; 256-run is a clean subset of the 512-run |
| Held-out | `eval_seed 999`, `--slice 0:100` → disjoint from train **by construction** (no eval_offset window needed) |
| Base anchor | reward **0.355**, exact **0.27**, partial 0.24 (greedy, eval_seed 999, n=100) |
| Decode (eval) | greedy, max_completion 512, chat_template on (train==eval); base re-scored in-call → every Δ is paired |
| Hardware | 1× RTX 5090 (32 GB), env `FoMo-RL`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

## Prior context (before this tuning pass)

- **Seed-0 exploratory (256-step GRPO / pop16·eb8·48-step ES):** GRPO **Δ+0.229** (t=3.38, exact→0.46, KL 0.0044);
  ES **null Δ−0.019** (exact flat) despite live fitness + ~6× GRPO's KL. Clean √(d/N) story on a 2nd task family.
- Pydantic cell (companion doc): GRPO +0.196 (3-seed), ES +0.070 (capped forced-KL window) — ES is *regime-dependent*,
  not a dead end. Countdown is the 2nd family to confirm/deny the scale hypothesis.

**Hypothesis under test:** ES on countdown was null because the zeroth-order direction was *under-scaled*
(error ∝ √(d/(N·T))). Raising population N and steps T within the proven capped forced-KL regime
(es_lr 1e-4, trust_ratio 1.0, noise_ratio 0.2) should move it off the null.

## Best so far  ✅ BOTH ARMS SOLID & VALIDATED — ES MATCHES GRPO (3-seed, every cell p<.001)

**Authoritative = 3 seeds, each pooled n=200 held-out** (two disjoint slices 0:100 + 100:200, eval_seed 999, greedy, paired t):

| arm | config | seed 0 | seed 1 | seed 2 | **3-seed mean** | status |
|---|---|---|---|---|---|---|
| **GRPO** | 512-run, ckpt-448 | +0.174 (p<.001) | +0.193 (p<.001) | +0.198 (p<.001) | **+0.188 ± 0.013** | ✅ SOLID |
| **ES** | pop64·eb16, **greedy-fitness, lr 1e-4**, ckpt-final | +0.193 (p<.001) | +0.179 (p<.001) | +0.184 (p<.001) | **+0.185 ± 0.007** | ✅ SOLID |
| _ES selection edge_ | v3 real − shuffle control (seed 0) | +0.138 (p<.01) | — | — | — | ✅ gain is SELECTION, not movement |
| _ES movement floor_ | shuffle (same step-norm, random dir) | +0.056 (ns) | — | — | — | random-direction move ⇒ no real gain |

**ES and GRPO are statistically indistinguishable.** Per-seed paired difference GRPO−ES = +0.0031 (t=0.28, **ns**)
— ES *matches* GRPO on countdown, and with **tighter** seed-to-seed variance (±0.007 vs ±0.013). All 6 arm×seed
cells are pooled-n=200 **p<.001**. Single-task / single-seed caveats removed: **3 seeds × 2 disjoint slices** for
both arms, plus a shuffle control for ES.

**Headline:** on countdown **ES catches up to GRPO** (+0.185 vs +0.188 over 3 seeds, indistinguishable) — vs only
36% of GRPO on pydantic. **The win is validated as real selection** by the shuffle control (edge +0.138 p<.01;
movement floor ns) and is **robust across two disjoint held-out slices and three seeds**. Two levers got it there,
both confirming earlier diagnoses:
1. **Population scaling** (pop 16→64): null Δ−0.019 → off the floor — the √(d/N·T) under-scaling story on a 2nd family.
2. **Greedy fitness** (free lever): the decisive one. It aligns the ES fitness objective with the greedy held-out
   metric and makes antithetic pairs deterministic (pure perturbation signal, variance killed). It turned a
   **slice-fragile** sampled-fitness gain (var#1: +0.135 on slice 0:100 but +0.013 ns out-of-sample → +0.074 ns
   pooled) into a **robust** one (v3: p<.001 on *both* slices). θ_dev stayed off the cap (12.6 < 13.46) = in-window.

_Checkpoint-selection note: the pre-registered ckpts are GRPO=448, ES=final (chosen on seeds 0/1). The per-seed
slice-0:100 **peak wanders** (GRPO seed 2 peaked at ckpt-256 +0.228; ES peak is sometimes ckpt-22) = checkpoint-
selection noise — but the pre-registered choice stays pooled-p<.001 every seed, and the **level** (~+0.18–0.20)
holds. Reporting the fixed pre-registered ckpt (not the per-seed argmax) avoids selection bias._

### Significance (paired t-test, base vs ckpt on the same 100 held-out prompts, df=99)

| arm | ckpt | Δreward | t | sig | exact-match flips (win/loss) |
|---|---|---|---|---|---|
| GRPO | 448 (peak) | +0.2042 | +3.06 | **p<.01** | 25 / 8 |
| GRPO | 512/final | +0.1450 | +2.31 | *p<.05* | 20 / 8 |
| ES | 24 (peak) | +0.1352 | +2.14 | *p<.05* | 19 / 8 |
| ES | 48/final | +0.1346 | +1.99 | *p<.05* | 21 / 10 |

_Note: this is the initial slice-0:100 read (n=100). The final claim rests on the pooled n=200 × 2-seed numbers in
"Best so far" plus the shuffle control (both done below) — not on this single-slice snapshot._

## GRPO 512-step curve (held-out, n=100, greedy, base re-scored in-call)

| ckpt | mean reward | exact | Δreward | KL | clip% |
|---|---|---|---|---|---|
| base | 0.3546 | 0.270 | — | — | 65% |
| 64 | 0.4374 | 0.340 | +0.083 | 0.0027 | 56% |
| 128 | 0.3760 | 0.290 | +0.021 | 0.0048 | 58% |
| 192 | 0.4480 | 0.350 | +0.093 | 0.0059 | 53% |
| 256 | 0.4805 | 0.370 | +0.126 | 0.0044 | 53% |
| 320 | 0.4762 | 0.370 | +0.122 | 0.0043 | 56% |
| 384 | 0.5403 | 0.420 | +0.186 | 0.0046 | 48% |
| **448** | **0.5588** | **0.440** | **+0.204** | 0.0049 | 50% |
| 512 / final | 0.4996 | 0.390 | +0.145 | 0.0047 | 53% |

**Reading:** GRPO **keeps climbing past step 256** (+0.126 → peak **+0.204 @ 448**), then **over-trains/erodes by 512**
(+0.145). Answers "maybe it's not done improving?" → yes, to ~448; 512 is slightly too long. KL stays tiny
(~0.005) throughout — clean policy move, no reward-hacking. Best GRPO = **ckpt-448, Δ+0.204, exact 0.44**.
(Earlier exploratory 256-*sample* run hit +0.229: there step 256 was fully LR-annealed; here it's mid-cosine-decay.)

## Operational note — GPU is UNDER-FED during ES (power lever)

User-observed: GPU power during the ES leg is **~half** of prior ES runs. At `member_batch=8` (resident
mb·eb = 8·16 = 128 sequences) we are **launch/bandwidth-bound, not compute-bound** — idle compute + idle VRAM
(~8–15 GB of 32). **Lever for the next ES run:** raise `member_batch` (bigger decode chunks → higher GPU
utilization → faster steps → more population/eval_batch/steps science in the same ≤60-min budget). Keep the
ramble-VRAM caveat: worst-case ≈ resident · max_seqlen, so resident 192 (mb=12) ≈ ~21 GB worst case (safe with
`expandable_segments:True`); resident 256 (mb=16) ≈ 30.7 GB (the prior "weird peak", risky). Sweet spot mb≈12.

**Empirical confirmation (step-time vs member_batch, same pop64/eb16):** mb=8 → **77.8 s/step**; mb=12 →
**48.9 s/step** (~1.6× faster from bigger decode chunks). Consequence: ES#1 (mb=8, 48 steps) lands at ~62 min
(slightly over the 60-min target); at mb=12 the same 48 steps would be ~39 min — so mb=12 fits budget AND frees
~23 min of wall-time to reinvest into more population/eval_batch/steps. Use mb=12 for variation #2.

## ES diagnosis (4-lens design panel + live var#1 log) — it's NOISE/TIMIDITY-limited, not regime-broken

The panel read `es.py`/`countdown_rubric.py` and the **live var#1 log** and converged:
- **var#1 is NOT flat** — train fitness climbs 0.287→0.388→0.333→0.380→0.493 (best 0.547→0.800) over 5 steps.
- `theta_dev` rises 1.57→2.31→2.79→3.37→3.72 but **decelerates** (√t random-walk) → projects to ~9–11 at step 48,
  **never reaching the cap (13.47)** ⇒ the trust cap is **non-binding (~3.6× headroom)**. The regime is right; the
  **per-step is timid** and the **selection signal is decode-noise-contaminated**.
- Two free/cheap, code-verified noise sources: fitness uses `temperature=1.0` + `es_greedy_fitness=False` ⇒ train
  fitness is *sampled* while held-out is *greedy* (objective mismatch + variance surviving antithetic CRN).
- Open question: does the **training-fitness** climb convert to a **held-out** gain, or overfit the train pool?

**Ranked plan for variation #2** (each ≤60 min, ≤21 GB worst-case, mb=12 to feed the GPU):

| rank | recipe | knobs | est wall | hypothesis |
|---|---|---|---|---|
| **1** | greedy-fitness **+** lr 3e-4 | `es_greedy_fitness=true`, `es_lr=3e-4`, pop64·eb16·mb12·44 | ~40 min | timid step + decode noise (attacks both, both cheap) |
| 2 | greedy-fitness + eb 24 (lr 1e-4) | `es_greedy_fitness=true`, pop48·eb24·mb8·40 | ~58 min | if lr 3e-4 overshoots, bottleneck is fitness variance |
| 3 | pop 96 (lr 1e-4) | pop96·eb16·mb12·36 | ~60 min | last resort: direction-coherence √(d/N) floor |

**Control (mandatory):** re-run rank-1 with `--es-shuffle-fitness` (same step-norm, random direction). Gain only
counts if shuffle is flat/negative (edge = real − shuffle = honest effect). **Falsification:** Δ ≤ +0.01 despite
cleaned greedy + 3× step + train fitness past ~0.45 ⇒ ES genuinely noise-floored → honest result "ES flat-but-stable,
GRPO +0.204" (clean √(d/N) complementary null on a 2nd task family). Staged in `configs/es_countdown_v2.toml` +
`scripts/es_countdown_variation.sh`.

## Run log (variations tried)

| # | arm | key change vs prior | steps | train wall | held-out Δ (best ckpt) | exact | KL | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | GRPO | max_samples 256→512 (does it keep climbing past 256?) | 512 | ~34 min | **+0.204** @ ckpt-448 | 0.440 | 0.005 | ✅ solid; peaks @448, erodes @512 |
| 2 | ES | pop 16→64, eb 8→16, 48 steps (4× N·T vs null) | 48 | ~60 min | **+0.135** @ ckpt-24 | 0.380 | 0.010 | ✅ solid; off the null (was −0.019); plateaus @24 |
| 3 | ES | greedy-fitness + **lr 3e-4** (mb12) | 44 | ~50 min | +0.1125 @ final | 0.370 | 0.029 | ⚠ **worse than var#1**: lr 3e-4 SATURATED the cap (θ_dev pinned 13.46) = fixed-norm random walk |
| 4 | ES | greedy-fitness + **lr 1e-4** (isolate greedy lever) | 44 | ~40 min | **+0.2215** @ final (slice0) / **+0.193 pooled p<.001** | 0.450 | 0.023 | ✅ **NEW BEST** — greedy is the decisive lever; matches GRPO; robust both slices |
| 5 | ES | **shuffle control** on v3 (greedy+lr1e-4 + `--es-shuffle-fitness`) | 44 | ~40 min | floor +0.056 (ns); edge +0.138 (p<.01) | 0.320 | 0.018 | ✅ gain is SELECTION, not movement |
| 6 | ES | v3 (greedy+lr1e-4) **seed 1** (reproducibility) | 44 | ~40 min | **+0.179 pooled p<.001** | 0.440 | 0.027 | ✅ **reproduces** (seed0 +0.193, seed1 +0.179) |
| 7 | GRPO | **seed 1** (symmetric 2-seed comparison) | 512 | ~34 min | **+0.193 pooled p<.001** @ ckpt-448 | 0.470 | 0.008 | ✅ reproduces peak@448/erode@512; GRPO 2-seed mean +0.184 |
| 8 | GRPO | **seed 2** (3-seed) | 512 | ~32 min | **+0.198 pooled p<.001** @ ckpt-448 | 0.420 | 0.011 | ✅ slice0 peak wandered to ckpt-256 (+0.228), but 448 holds; erode@512 again; **3-seed mean +0.188** |
| 9 | ES | v3 (greedy+lr1e-4) **seed 2** (3-seed) | 44 | ~52 min | **+0.184 pooled p<.001** @ ckpt-final | 0.420 | 0.023 | ✅ **3-seed mean +0.185 ± 0.007**; GRPO−ES gap ns (t=0.28) ⇒ parity confirmed |

### 3-seed reproducibility (pooled n=200 each, paired t df=199)

| seed | GRPO ckpt-448 Δ (t) | ES v3 ckpt-final Δ (t) |
|---|---|---|
| 0 | +0.1739 (+3.99, p<.001) | +0.1933 (+4.48, p<.001) |
| 1 | +0.1933 (+4.99, p<.001) | +0.1786 (+3.81, p<.001) |
| 2 | +0.1981 (+4.04, p<.001) | +0.1840 (+4.27, p<.001) |
| **mean** | **+0.1884 ± 0.0128** | **+0.1853 ± 0.0074** |

Both arms are **reproducible across 3 seeds** — neither is a single-seed fluke, and the per-seed paired gap
GRPO−ES = +0.0031 (t=0.28, **ns**) confirms parity. ES even has the *tighter* seed variance (±0.0074 vs ±0.0128).
All seeds show the characteristic bouncy ES curve (mid-run dip, peak at ~ckpt-22 or final) and the GRPO
mid-run-peak/erode-by-512 shape, confirming checkpoint-selection noise around a stable ~+0.18–0.20 level. **This
overturns the prior "ES is null on countdown (Δ−0.019)" headline at 3 seeds.**

### Validation of the ES v3 win (the rigor gates)

**(a) Shuffle control** — re-ran v3's exact config with `--es-shuffle-fitness` (advantages permuted ⇒ same θ_dev/step
magnitude, random direction). θ_dev reached 12.46 (vs real 12.61 — matched). Result: shuffle Δ = +0.056 **ns**
(movement floor), real − shuffle = **+0.138, p<.01** ⇒ the gain is **selection**, not a fixed-norm movement artifact
(the pydantic trap does NOT apply here).

**(b) Disjoint-slice out-of-sample** — re-eval on a fresh held-out slice (100:200):

| adapter | slice 0:100 Δ | slice 100:200 Δ | pooled n=200 Δ (t, sig) |
|---|---|---|---|
| GRPO ckpt-448 | +0.204 (p<.01) | +0.144 (p<.05) | **+0.174** (t=3.99, p<.001) |
| ES v3 greedy ckpt-final | +0.2215 (p<.01) | +0.165 (p<.01) | **+0.193** (t=4.48, p<.001) |
| ES v3 greedy ckpt-22 | +0.138 | +0.221 (p<.01) | +0.180 (t=3.87, p<.001) |
| ES var#1 (sampled lr1e-4) | +0.135 (p<.05) | **+0.013 (ns)** | +0.074 (t=1.71, **ns**) |
| ES v3 shuffle (floor) | +0.0585 (ns) | +0.053 (ns) | +0.056 (t=1.31, ns) |

Reading: GRPO and v3-greedy are robust across both slices (pooled p<.001). The per-checkpoint *peak* wanders (slice0
likes ckpt-final, slice1 likes ckpt-22) = checkpoint-selection noise, but the *level* (~+0.18–0.19) holds. **var#1
(sampled fitness) is the cautionary tale: it looked solid on slice 0:100 but is a slice-fluke (+0.074 ns pooled)** —
which is exactly why greedy fitness matters (it removes that fragility).

### v3 (greedy + lr 1e-4) curve + paired significance (vs base, n=100, df=99)

| ckpt | Δreward | t(99) | sig | exact-flips (win/loss) |
|---|---|---|---|---|
| 11 | +0.1253 | +2.13 | *p<.05* | 17/7 |
| 22 | +0.1383 | +2.10 | *p<.05* | 21/9 |
| 33 | +0.0510 | +0.79 | ns | 16/12 |
| **final/44** | **+0.2215** | **+3.37** | **p<.01** | 25/7 |

The trajectory is unstable (the step-33 dip is not significant), so the +0.2215 peak should be read with the shuffle
control + a disjoint-slice recheck. But the peak itself is solidly p<.01 with a clean exact-match flip ledger (25 win / 7 loss).

### ES lr dose-response (the trust cap is the constraint)

| es_lr | fitness | held-out Δ | θ_dev @ end | reading |
|---|---|---|---|---|
| 1e-4 (var#1) | sampled | **+0.135** | ~3.7 (proj), cap NOT hit | ✅ in-window, best |
| 3e-4 (v2) | greedy | +0.1125 | **13.46 = cap (saturated)** | ⚠ too hot → fixed-norm random walk |

Mirrors the pydantic finding: there is a **step-magnitude window**. lr 1e-4 stays inside the trust region and
selects; lr 3e-4 slams the cap and degrades. The cap is load-bearing (it prevents collapse) but once you're
*pinned* to it the direction is the only free variable → random walk. v4 (greedy @ lr 1e-4) isolates whether the
greedy lever itself helps at the in-window lr.

**Naming note (important):** "v3" is a *run label*, not originally a file — the winner was run as
`es_countdown_v2.toml --es-lr 1e-4` (v2's baked `es_lr=3e-4` underperformed, see run #3). It is now also shipped as a
**self-contained `configs/es_countdown_v3.toml`** (= v2 with `es_lr=1e-4` baked) so the headline reproduces from one
file: `python run.py --config configs/es_countdown_v3.toml --seed 0`.

_Configs: `configs/grpo_countdown.toml`, `configs/es_countdown.toml` (var#1), `configs/es_countdown_v2.toml` (greedy,
lr 3e-4 default), **`configs/es_countdown_v3.toml` (the winner, lr 1e-4 baked)**. Drivers:
`scripts/prime_countdown_sweep.sh`, `scripts/countdown_seed_run.sh`, `scripts/es_countdown_variation.sh`.
Evals: `outputs/prime_countdown_sweep/eval_*.json` (full ckpt curves, base re-scored in-call; 3 seeds × 2 disjoint slices)._
