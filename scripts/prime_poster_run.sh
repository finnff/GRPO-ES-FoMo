#!/usr/bin/env bash
# Full seed-0 poster cells for the two PrimeIntellect hub envs (ascii-tree,
# pydantic-adherence), GRPO + stabilized-ES, on LFM2.5-1.2B. Hyperparameters
# match the IFEval/IFBench poster cells (es_population=16 / es_eval_batch=8 /
# es_steps=40; GRPO 8x8 rollout; chat_template ON) — see
# configs/{grpo,es}_{ascii_tree,pydantic_adherence}.toml. NOTE: the pydantic GRPO
# cell now uses the tuned 500-sample config (lr 3e-5, beta 0.02), so its disjoint
# holdout is [500:600]; the other cells stay 100/64 -> [100:200].
#
# Pipeline: baseline eval (run separately, already done) -> 4 training runs ->
# paired greedy held-out eval (base + GRPO + ES on each env's disjoint holdout
# from ENV_CUSTOMIZATIONS, identical decode/lengths to the baseline, paired).
#
#   bash scripts/prime_poster_run.sh
set -u
OUT="outputs/prime"
mkdir -p "$OUT"
SEED=0

train() {  # name  config
  local name="$1" cfg="$2"
  echo "===== TRAIN $name ($(date +%H:%M:%S)) ====="
  python run.py --config "configs/$cfg.toml" --seed "$SEED" \
    --output-dir "$OUT/${name}_s${SEED}" > "$OUT/${name}_s${SEED}.log" 2>&1 \
    && echo "OK  TRAIN $name ($(date +%H:%M:%S))" \
    || echo "FAIL($?) TRAIN $name ($(date +%H:%M:%S))"
}

# GRPO first (cheap per step -> fast end-to-end validation), then ES (long pole).
train grpo_ascii    grpo_ascii_tree
train grpo_pydantic grpo_pydantic_adherence
train es_ascii      es_ascii_tree
train es_pydantic   es_pydantic_adherence

# --- Paired held-out eval: base vs GRPO vs ES on one disjoint holdout. ---
# Greedy (poster headline, deterministic), spec lengths == training lengths,
# each env's spec window (ENV_CUSTOMIZATIONS) -> directly paired with baseline.
eval_env() {  # short  env-id  grpo-name  es-name
  local short="$1" env="$2" g="$3" e="$4"
  echo "===== EVAL $short ($(date +%H:%M:%S)) ====="
  python -m grpo_es.eval --task "env:$env" --model lfm2.5-1.2b --chat-template on --kl \
    --adapter base "$OUT/${g}_s${SEED}/checkpoint-final" "$OUT/${e}_s${SEED}/checkpoint-final" \
    --json "$OUT/eval_${short}_s${SEED}.json" \
    --per-sample "$OUT/persample_${short}_s${SEED}.jsonl" \
    > "$OUT/eval_${short}_s${SEED}.log" 2>&1 \
    && echo "OK  EVAL $short ($(date +%H:%M:%S))" \
    || echo "FAIL($?) EVAL $short ($(date +%H:%M:%S))"
}

eval_env ascii    primeintellect/ascii-tree         grpo_ascii    es_ascii
eval_env pydantic primeintellect/pydantic-adherence grpo_pydantic es_pydantic

echo "ALL DONE ($(date +%H:%M:%S))"
