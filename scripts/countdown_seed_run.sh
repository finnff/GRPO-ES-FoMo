#!/usr/bin/env bash
# Full per-seed countdown cell: GRPO 512-run + ES v3 (greedy+lr1e-4), then the
# disjoint-slice (100:200) evals needed to pool to n=200. Reproduces the EXACT
# seed-0/seed-1 protocol so a 3rd (or 4th) seed is apples-to-apples:
#   GRPO : configs/grpo_countdown.toml  (512 samples, lr 3e-5, beta 0.02)  -> ckpt curve 64..512
#   ES   : configs/es_countdown_v2.toml --es-lr 1e-4  (greedy fitness, pop64/eb16/mb12, 44 steps)
# mb=12 keeps the GPU fed (power lever; ~21 GB worst-case, safe w/ expandable_segments).
#
#   bash scripts/countdown_seed_run.sh 2          # one seed, both arms + disjoint evals
#   ONLY=grpo bash scripts/countdown_seed_run.sh 2
#   ONLY=es   bash scripts/countdown_seed_run.sh 2
set -u
set -o pipefail
PY="${PY:-/home/sga/miniconda3/envs/FoMo-RL/bin/python}"
cd /home/sga/MASTER/Foundation/Project/GRPO-ES-FoMo
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED="${1:?usage: countdown_seed_run.sh SEED}"
ONLY="${ONLY:-both}"
OUT="${OUT_DIR:-outputs/prime_countdown_sweep}"
mkdir -p "$OUT"
DJ=(--task countdown --model lfm2.5-1.2b --chat-template on --slice 100:200 --decode greedy --kl)

echo "########## COUNTDOWN SEED $SEED (ONLY=$ONLY) START $(date +%H:%M:%S) ##########"

# ---- GRPO (train + eval slice 0:100 full curve via the canonical sweep) ----
if [ "$ONLY" = both ] || [ "$ONLY" = grpo ]; then
  ONLY=grpo bash scripts/prime_countdown_sweep.sh "$SEED" || { echo "GRPO arm FAILED"; exit 1; }
  # disjoint slice 100:200 at the protocol checkpoint (ckpt-448, the peak)
  GD="$OUT/grpo_countdown_s${SEED}/checkpoint-448"
  if [ -d "$GD" ]; then
    echo "===== EVAL grpo s${SEED} disjoint 100:200 [ckpt-448] ($(date +%H:%M:%S)) ====="
    "$PY" -m grpo_es.eval "${DJ[@]}" --adapter base "$GD" \
      --json "$OUT/eval_grpo_s${SEED}_disjoint_100_200.json" \
      --per-sample "$OUT/persample_grpo_s${SEED}_disjoint_100_200.jsonl" \
      2>&1 | tee "$OUT/eval_grpo_s${SEED}_disjoint_100_200.log"
  else
    echo "WARN: $GD missing -> no GRPO disjoint eval"
  fi
fi

# ---- ES v3 = greedy fitness + lr 1e-4 (train + eval slice 0:100 full curve) ----
if [ "$ONLY" = both ] || [ "$ONLY" = es ]; then
  SEED="$SEED" bash scripts/es_countdown_variation.sh es_v3_greedy_lr1e4 es_countdown_v2 --es-lr 1e-4 \
    || { echo "ES arm FAILED"; exit 1; }
  # disjoint slice 100:200 at ckpt-22 + ckpt-final (matches seed-0/1 disjoint set)
  E22="$OUT/es_v3_greedy_lr1e4_s${SEED}/checkpoint-22"
  EFN="$OUT/es_v3_greedy_lr1e4_s${SEED}/checkpoint-final"
  ADJ=(base); [ -d "$E22" ] && ADJ+=("$E22"); [ -d "$EFN" ] && ADJ+=("$EFN")
  if [ "${#ADJ[@]}" -ge 2 ]; then
    echo "===== EVAL es_v3 s${SEED} disjoint 100:200 [${ADJ[*]:1}] ($(date +%H:%M:%S)) ====="
    "$PY" -m grpo_es.eval "${DJ[@]}" --adapter "${ADJ[@]}" \
      --json "$OUT/eval_es_v3_s${SEED}_disjoint_100_200.json" \
      --per-sample "$OUT/persample_es_v3_s${SEED}_disjoint_100_200.jsonl" \
      2>&1 | tee "$OUT/eval_es_v3_s${SEED}_disjoint_100_200.log"
  else
    echo "WARN: no ES checkpoints -> no ES disjoint eval"
  fi
fi

echo "########## COUNTDOWN SEED $SEED DONE $(date +%H:%M:%S) ##########"
