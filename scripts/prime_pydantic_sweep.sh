#!/usr/bin/env bash
# Pydantic-adherence ES-vs-GRPO sweep on LFM2.5-1.2B, parameterized by seed.
# Compares base -> GRPO -> ES on the SAME disjoint held-out window, paired per
# seed (base is re-scored on that window in the same eval call). The window is
# derived from max_samples (below); the 500-sample GRPO default -> [500:600].
#
#   bash scripts/prime_pydantic_sweep.sh 0        # one seed
#   bash scripts/prime_pydantic_sweep.sh 1 2      # seeds 1 and 2 (sequential)
#   bash scripts/prime_pydantic_sweep.sh          # defaults to seed 0
#
# Then aggregate across whatever seeds / methods exist:
#   python scripts/agg_pydantic_sweep.py
#
# --- Pick which method(s) to run:  ONLY={both,grpo,es}  (default both) ---
#   ONLY=grpo bash scripts/prime_pydantic_sweep.sh 1 2   # GRPO only, skip ES
#   ONLY=es   bash scripts/prime_pydantic_sweep.sh 1 2   # ES only, skip GRPO
# both -> eval writes eval_pydantic_s{N}.json       (base, GRPO, ES)
# grpo -> eval writes eval_pydantic_s{N}_grpo.json  (base, GRPO)   [no clobber]
# es   -> eval writes eval_pydantic_s{N}_es.json    (base, ES)     [no clobber]
# The aggregator merges per-seed across all three filename forms.
#
# --- Pick the config(s):  CFG_GRPO / CFG_ES  (config basename, no .toml) ---
#   CFG_GRPO=<other> ONLY=grpo bash scripts/prime_pydantic_sweep.sh 1 2
# The held-out eval window is placed AUTOMATICALLY disjoint from training:
# train = shuffle(seed)[0:max_samples], so the window is [start:start+100] with
# start = max(100, largest max_samples across BOTH configs) — regardless of ONLY.
# The 500-sample GRPO default fixes this at [500:600], and an ONLY=es run uses the
# SAME [500:600] (still disjoint from ES train[0:64]) so the ES bar is paired with
# the GRPO bar on identical prompts. Base is re-scored on that window in the SAME
# eval call, so every delta stays paired. Override with EVAL_START=<n> if needed.
#
# --- ES is TUNED, not the default poster ES. ---
# The default es_lr=0.05 saturates the trust region every step (theta_dev pinned
# at the cap R~3.37 from step 1 -> the master random-walks the R-sphere instead of
# integrating a coherent direction; documented in outputs/prime/ANALYSIS_es_vs_grpo.md
# and memory es-trust-region-saturation-seed0). We lower es_lr so the raw pre-clamp
# step is ~0.17x R (the "best window" leg from scripts/es_trustcap_sweep.sh): the
# cap unbinds and a weak-but-real signal accumulates across steps. Read theta_dev in
# each ES run's history.jsonl: if it climbs from <R instead of pinning, the cap is
# unbound. Overridable:
#   ES_LR=2e-5   (default; 0.05 reproduces the known-inert poster ES)
#   ES_EVAL_BATCH=8  (raise to 16 for lower fitness variance / fewer zero-gradient
#                     antithetic pairs -- the "best-shot" leg -- at ~2x ES step cost)
set -u
set -o pipefail                         # so `... | tee` reports python's exit, not tee's

SEEDS=("${@:-0}")                       # all positional args are seeds
ONLY="${ONLY:-both}"                    # both | grpo | es
CFG_GRPO="${CFG_GRPO:-grpo_pydantic_adherence}"
CFG_ES="${CFG_ES:-es_pydantic_adherence}"
ES_LR="${ES_LR:-2e-5}"
ES_EVAL_BATCH="${ES_EVAL_BATCH:-8}"
EVAL_SIZE=100                           # held-out window size (== ENV_CUSTOMIZATIONS eval_size)

case "$ONLY" in
  both|grpo|es) ;;
  *) echo "ONLY must be one of: both grpo es  (got '$ONLY')"; exit 2 ;;
esac

# OUT_DIR keeps separate runs from clobbering each other (the output subdir is
# named per-method, NOT per-config). Use a distinct dir to compare an alt config:
#   OUT_DIR=outputs/prime_pydantic_altcfg CFG_GRPO=<other_config> ONLY=grpo \
#     bash scripts/prime_pydantic_sweep.sh 1 2
OUT="${OUT_DIR:-outputs/prime_pydantic_sweep}"
mkdir -p "$OUT"
ENV="env:primeintellect/pydantic-adherence"
MODEL="lfm2.5-1.2b"

# max_samples from a config (default 100 if unset/unreadable) -> used to place
# the eval window disjoint from the train draw shuffle[0:max_samples].
cfg_max_samples() {  # config-basename
  python3 - "$1" <<'PY'
import sys, pathlib
try:
    import tomllib
    d = tomllib.load(open(pathlib.Path("configs") / (sys.argv[1] + ".toml"), "rb"))
    v = d.get("max_samples")
    print(int(v) if isinstance(v, int) else 100)
except Exception:
    print(100)
PY
}

# Window start clears the LARGEST training draw across BOTH legs (not just the
# selected one), so an ONLY=es run evals on the SAME held-out window as the
# already-done GRPO arm -> the ES/GRPO bars are paired on identical prompts.
# [500:600] is trivially disjoint from ES train[0:64] too (500 >= 64). Floored at
# 100. Override with EVAL_START=<n> only if you deliberately want another window.
_g=$(cfg_max_samples "$CFG_GRPO")
_e=$(cfg_max_samples "$CFG_ES")
_ms=$(( _g > _e ? _g : _e ))
SLICE_START=$(( _ms > 100 ? _ms : 100 ))
SLICE_START="${EVAL_START:-$SLICE_START}"
SLICE_END=$(( SLICE_START + EVAL_SIZE ))
echo "configs: GRPO=$CFG_GRPO ES=$CFG_ES | held-out window [${SLICE_START}:${SLICE_END}] (clears max train draw ${_ms} across both legs -> ES & GRPO paired)"

train() {  # name  config  extra-flags...
  local name="$1" cfg="$2"; shift 2
  echo "===== TRAIN $name  [$*]  ($(date +%H:%M:%S)) ====="
  # tee -> live step output on the terminal AND a full log on disk.
  python run.py --config "configs/$cfg.toml" --seed "$SEED" \
    --output-dir "$OUT/${name}_s${SEED}" "$@" 2>&1 \
    | tee "$OUT/${name}_s${SEED}.log" \
    && echo "OK  TRAIN $name ($(date +%H:%M:%S))" \
    || { echo "FAIL($?) TRAIN $name ($(date +%H:%M:%S)) -- see $OUT/${name}_s${SEED}.log"; return 1; }
}

for SEED in "${SEEDS[@]}"; do
  echo "########## SEED $SEED  (ONLY=$ONLY)  ($(date +%H:%M:%S)) ##########"

  # Build the eval adapter list from whichever legs are selected + train OK.
  ADAPTERS=(base)
  if [ "$ONLY" = both ] || [ "$ONLY" = grpo ]; then
    train grpo_pydantic "$CFG_GRPO" \
      && ADAPTERS+=("$OUT/grpo_pydantic_s${SEED}/checkpoint-final")
  fi
  if [ "$ONLY" = both ] || [ "$ONLY" = es ]; then
    # Tuned ES (low es_lr unbinds the saturated trust cap; see header).
    train es_pydantic "$CFG_ES" --es-lr "$ES_LR" --es-eval-batch "$ES_EVAL_BATCH" \
      && ADAPTERS+=("$OUT/es_pydantic_s${SEED}/checkpoint-final")
  fi

  if [ "${#ADAPTERS[@]}" -lt 2 ]; then
    echo "SKIP EVAL s${SEED}: no adapter trained (all selected legs failed)"; continue
  fi

  # Method-tagged eval filename so grpo-only and es-only runs of the SAME seed
  # don't clobber each other; both -> the canonical unsuffixed name.
  case "$ONLY" in both) tag="" ;; *) tag="_$ONLY" ;; esac

  # --- Paired held-out eval on the auto-placed [SLICE_START:SLICE_END] window.
  # Greedy (deterministic poster headline), spec lengths (2048/1024) == training
  # lengths; --slice keeps it disjoint from train[0:max_samples] and re-scores
  # base on the same window so every delta is paired. ---
  echo "===== EVAL pydantic s${SEED}${tag} [legs: ${ADAPTERS[*]:1}] window [${SLICE_START}:${SLICE_END}] ($(date +%H:%M:%S)) ====="
  python -m grpo_es.eval --task "$ENV" --model "$MODEL" --chat-template on --kl \
    --slice "${SLICE_START}:${SLICE_END}" \
    --adapter "${ADAPTERS[@]}" \
    --json "$OUT/eval_pydantic_s${SEED}${tag}.json" \
    --per-sample "$OUT/persample_pydantic_s${SEED}${tag}.jsonl" 2>&1 \
    | tee "$OUT/eval_pydantic_s${SEED}${tag}.log" \
    && echo "OK  EVAL s${SEED}${tag} ($(date +%H:%M:%S))" \
    || echo "FAIL($?) EVAL s${SEED}${tag} ($(date +%H:%M:%S)) -- see $OUT/eval_pydantic_s${SEED}${tag}.log"
done

echo "ALL SEEDS DONE (${SEEDS[*]}) ONLY=$ONLY ($(date +%H:%M:%S))"
python scripts/agg_pydantic_sweep.py "$OUT" 2>/dev/null || true
