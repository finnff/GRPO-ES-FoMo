#!/usr/bin/env bash
# Run ONE ES variation on countdown: train to a NAMED output dir, then eval the full
# checkpoint curve (base re-scored in-call). Unlike the both-arms sweep, the run name
# is a parameter, so variations + their shuffle controls never collide on disk.
#
#   bash scripts/es_countdown_variation.sh NAME CONFIG [extra run.py flags...]
# e.g.
#   bash scripts/es_countdown_variation.sh es_v2_greedy_lr3e4 es_countdown_v2
#   bash scripts/es_countdown_variation.sh es_v2_shuffle     es_countdown_v2 --es-shuffle-fitness
set -u
set -o pipefail
PY="${PY:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEED="${SEED:-0}"
OUT="${OUT_DIR:-outputs/prime_countdown_sweep}"
MODEL="lfm2.5-1.2b"
HELD=(--task countdown --model "$MODEL" --chat-template on --slice 0:100 --decode greedy --kl)
mkdir -p "$OUT"

NAME="$1"; CFG="$2"; shift 2
RUNDIR="$OUT/${NAME}_s${SEED}"

echo "===== TRAIN $NAME  [cfg=$CFG]  [$*]  ($(date +%H:%M:%S)) ====="
"$PY" run.py --config "configs/$CFG.toml" --seed "$SEED" \
  --output-dir "$RUNDIR" "$@" 2>&1 \
  | tee "$OUT/${NAME}_s${SEED}.log" \
  && echo "OK  TRAIN $NAME ($(date +%H:%M:%S))" \
  || { echo "FAIL($?) TRAIN $NAME ($(date +%H:%M:%S)) -- see $OUT/${NAME}_s${SEED}.log"; exit 1; }

# Build adapter list: base + interior numeric ckpts ascending + final.
ADAPTERS=(base)
while IFS= read -r c; do [ -n "$c" ] && ADAPTERS+=("$c"); done < <(
  ls -d "$RUNDIR"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sort -t- -k2 -n)
[ -d "$RUNDIR/checkpoint-final" ] && ADAPTERS+=("$RUNDIR/checkpoint-final")
if [ "${#ADAPTERS[@]}" -lt 2 ]; then
  echo "SKIP EVAL $NAME: no checkpoints under $RUNDIR"; exit 1
fi

echo "===== EVAL $NAME  [${ADAPTERS[*]:1}]  ($(date +%H:%M:%S)) ====="
"$PY" -m grpo_es.eval "${HELD[@]}" \
  --adapter "${ADAPTERS[@]}" \
  --json "$OUT/eval_${NAME}_s${SEED}.json" \
  --per-sample "$OUT/persample_${NAME}_s${SEED}.jsonl" 2>&1 \
  | tee "$OUT/eval_${NAME}_s${SEED}.log" \
  && echo "OK  EVAL $NAME ($(date +%H:%M:%S))" \
  || echo "FAIL($?) EVAL $NAME ($(date +%H:%M:%S)) -- see $OUT/eval_${NAME}_s${SEED}.log"

echo "########## DONE $NAME ($(date +%H:%M:%S)) ##########"
