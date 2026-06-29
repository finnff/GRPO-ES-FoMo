#!/usr/bin/env bash
# Countdown GRPO-vs-ES sweep on LFM2.5-1.2B, parameterized by seed. Mirrors
# scripts/prime_pydantic_sweep.sh (tee file-logging, per-seed output dirs,
# use_trackio via the configs) but adapted to countdown's GENERATED held-out:
# eval_seed 999 -> the held-out is disjoint from train BY CONSTRUCTION, so we
# just eval --slice 0:100 (no eval_offset window dance like pydantic). Base is
# re-scored in the SAME eval call so every delta is paired.
#
#   bash scripts/prime_countdown_sweep.sh 0          # one seed
#   bash scripts/prime_countdown_sweep.sh 0 1 2      # seeds 0,1,2 (sequential)
#   ONLY=grpo bash scripts/prime_countdown_sweep.sh 0
#   ONLY=es   bash scripts/prime_countdown_sweep.sh 0
#
# This evals the FULL checkpoint curve (base + every interior ckpt + final) for
# each trained arm, so the held-out JSON shows climb-vs-plateau (GRPO) and the
# peak-then-decay trajectory (ES). The configs carry the recipe:
#   grpo_countdown.toml  max_samples=512 (was 256), save_steps=64  -> ckpts @64..512
#   es_countdown.toml    pop=64 eb=16 mb=16 steps=96 (scaled up),  save_steps=24
set -u
set -o pipefail                         # `... | tee` reports python's exit, not tee's

PY="${PY:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Tame fragmentation-driven transient VRAM spikes (the "weird peak"): grow
# segments instead of grabbing huge contiguous blocks. Cheap insurance on a
# long run where a rambling ES step can momentarily balloon the KV cache.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=("${@:-0}")                       # positional args = seeds
ONLY="${ONLY:-both}"                    # both | grpo | es
CFG_GRPO="${CFG_GRPO:-grpo_countdown}"
CFG_ES="${CFG_ES:-es_countdown}"
MODEL="lfm2.5-1.2b"
HELD=(--task countdown --model "$MODEL" --chat-template on --slice 0:100 --decode greedy --kl)

case "$ONLY" in both|grpo|es) ;; *) echo "ONLY must be both|grpo|es (got '$ONLY')"; exit 2 ;; esac

OUT="${OUT_DIR:-outputs/prime_countdown_sweep}"
mkdir -p "$OUT"

train() {  # name  config  extra-flags...
  local name="$1" cfg="$2"; shift 2
  echo "===== TRAIN $name  [$*]  ($(date +%H:%M:%S)) ====="
  # tee -> live step output on the terminal AND a full log on disk.
  "$PY" run.py --config "configs/$cfg.toml" --seed "$SEED" \
    --output-dir "$OUT/${name}_s${SEED}" "$@" 2>&1 \
    | tee "$OUT/${name}_s${SEED}.log" \
    && echo "OK  TRAIN $name ($(date +%H:%M:%S))" \
    || { echo "FAIL($?) TRAIN $name ($(date +%H:%M:%S)) -- see $OUT/${name}_s${SEED}.log"; return 1; }
}

# Echo "base ckpt-N1 ckpt-N2 ... ckpt-final": numeric ckpts ascending, final last.
ckpt_list() {  # run-dir
  local d="$1"
  echo -n "base"
  while IFS= read -r c; do [ -n "$c" ] && echo -n " $c"; done < <(
    ls -d "$d"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sort -t- -k2 -n)
  [ -d "$d/checkpoint-final" ] && echo -n " $d/checkpoint-final"
  echo
}

# Paired held-out eval over the full checkpoint curve (base re-scored in-call).
eval_curve() {  # name  run-dir
  local name="$1" d="$2"
  read -r -a ADAPTERS <<<"$(ckpt_list "$d")"
  if [ "${#ADAPTERS[@]}" -lt 2 ]; then
    echo "SKIP EVAL $name s${SEED}: no checkpoints under $d"; return 1
  fi
  echo "===== EVAL $name s${SEED}  [${ADAPTERS[*]:1}]  ($(date +%H:%M:%S)) ====="
  "$PY" -m grpo_es.eval "${HELD[@]}" \
    --adapter "${ADAPTERS[@]}" \
    --json "$OUT/eval_${name}_s${SEED}.json" \
    --per-sample "$OUT/persample_${name}_s${SEED}.jsonl" 2>&1 \
    | tee "$OUT/eval_${name}_s${SEED}.log" \
    && echo "OK  EVAL $name s${SEED} ($(date +%H:%M:%S))" \
    || echo "FAIL($?) EVAL $name s${SEED} ($(date +%H:%M:%S)) -- see $OUT/eval_${name}_s${SEED}.log"
}

for SEED in "${SEEDS[@]}"; do
  echo "########## SEED $SEED  (ONLY=$ONLY)  ($(date +%H:%M:%S)) ##########"
  if [ "$ONLY" = both ] || [ "$ONLY" = grpo ]; then
    train grpo_countdown "$CFG_GRPO" && eval_curve grpo_countdown "$OUT/grpo_countdown_s${SEED}"
  fi
  if [ "$ONLY" = both ] || [ "$ONLY" = es ]; then
    train es_countdown "$CFG_ES" && eval_curve es_countdown "$OUT/es_countdown_s${SEED}"
  fi
done

echo "########## ALL SEEDS DONE (${SEEDS[*]}) ONLY=$ONLY ($(date +%H:%M:%S)) ##########"
# Compact summary of every eval JSON produced.
"$PY" - "$OUT" <<'PY' 2>/dev/null || true
import json, sys, glob, os
out = sys.argv[1]
for f in sorted(glob.glob(os.path.join(out, "eval_*_s*.json"))):
    try:
        rows = json.load(open(f))
    except Exception:
        continue
    print(f"\n##### {os.path.basename(f)} #####")
    for r in (rows if isinstance(rows, list) else [rows]):
        name = str(r.get("adapter", "?"))
        print(f"  {name[-46:]:46s} mean={r.get('mean_reward',0):.4f} "
              f"exact={r.get('accuracy',0):.3f} d={r.get('delta_mean_reward',0):+.4f} "
              f"kl={r.get('kl_to_base',0):.4f} len={r.get('mean_length',0):.0f} "
              f"clip={r.get('clip_frac',0):.2f}")
PY
