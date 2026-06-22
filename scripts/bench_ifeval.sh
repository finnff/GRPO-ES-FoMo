#!/usr/bin/env bash
# Controlled step-time benchmark: ifeval, identical config, only --model varies,
# across the three candidate models, both optimizers. Apples-to-apples timing /
# VRAM / reward-trajectory so we can pick the model to iterate fast on.
#
#   bash scripts/bench_ifeval.sh
set -u

OUT="outputs/bench-ifeval"
mkdir -p "$OUT"
PROG="$OUT/PROGRESS.log"
: > "$PROG"

MODELS=("lfm2.5-350m" "lfm2.5-1.2b" "qwen3.5-0.8b")

# Shared, fixed knobs (every cell identical except --model / --method).
COMMON=(--task ifeval --max-samples 64 --max-completion-length 512 \
        --max-prompt-length 512 --seed 0 --save-steps 100000 --no-inspect-dump)
ES_KNOBS=(--method es --es-population 8 --es-eval-batch 8 --es-steps 12)
GRPO_KNOBS=(--method grpo --num-train-epochs 1)   # max-samples drives ~16 steps

log() { echo "$@" | tee -a "$PROG"; }

run_cell() {
  local name="$1"; shift
  local dir="$OUT/$name"
  log "===== START $name ====="
  local t0; t0=$(date +%s)
  if timeout 1200 python run.py "$@" --output-dir "$dir" > "$OUT/$name.log" 2>&1; then
    local st="OK"
  else
    local st="FAIL(rc=$?)"
  fi
  local t1; t1=$(date +%s)
  log "===== END   $name :: $st ($((t1 - t0))s) ====="
}

for model in "${MODELS[@]}"; do
  tag="${model//./}"; tag="${tag//-/}"
  run_cell "es_${tag}"   "${ES_KNOBS[@]}"   "${COMMON[@]}" --model "$model"
  run_cell "grpo_${tag}" "${GRPO_KNOBS[@]}" "${COMMON[@]}" --model "$model"
done

log "ALL DONE"
python scripts/bench_summary.py "$OUT" | tee -a "$PROG"
