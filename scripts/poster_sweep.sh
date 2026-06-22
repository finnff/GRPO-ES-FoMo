#!/usr/bin/env bash
# POSTER sweep driver: LFM2.5-350M. Trains base-vs-GRPO-vs-stabilized-ES across
# seeds (plus one naive-ES hacker demo) on TRAIN_TASK, then runs the paired
# held-out evals: the train task (within-task generalization) and the other IF
# task (transfer). Re-runnable: existing checkpoint-final dirs are skipped.
#
#   bash scripts/poster_sweep.sh                      # ifeval-train (default), seeds 0 1 2
#   SEEDS="0" bash scripts/poster_sweep.sh            # one-seed dry run first
#   SEEDS="0" TRAIN_TASK=ifbench bash scripts/poster_sweep.sh   # the headroom cell
#
# TRAIN_TASK=ifeval  -> transfer=ifbench, out=outputs/poster        (saturated base)
# TRAIN_TASK=ifbench -> transfer=ifeval,  out=outputs/poster_ifbench (headroom cell)
#
# Ctrl-C stops the whole sweep (a trap kills the current child and exits).
# Per-step reward/fitness lines stream to the terminal; full output -> <dir>.log.
set -u

MODEL="${MODEL:-lfm2.5-350m}" # env-overridable (e.g. MODEL=lfm2.5-1.2b for a capacity probe)
SEEDS="${SEEDS:-0 1 2}"
TRAIN_TASK="${TRAIN_TASK:-ifeval}"

case "$TRAIN_TASK" in
ifeval)
    TRANSFER="ifbench"
    OUT="outputs/poster"
    ;;
ifbench)
    TRANSFER="ifeval"
    OUT="outputs/poster_ifbench"
    ;;
*)
    echo "unknown TRAIN_TASK=$TRAIN_TASK (want ifeval|ifbench)" >&2
    exit 2
    ;;
esac
OUT="${OUT_OVERRIDE:-$OUT}"
mkdir -p "$OUT"
PROG="$OUT/PROGRESS.log"

# Config paths default to the cell's poster configs; override any via env to
# probe a variant (e.g. ES_CFG=configs/es_ifbench_big.toml for the higher-power
# ES test). SKIP_NAIVE=1 drops the hacker-demo leg (already proven) for speed.
GRPO_CFG="${GRPO_CFG:-configs/poster/grpo_${TRAIN_TASK}_poster.toml}"
ES_CFG="${ES_CFG:-configs/poster/es_${TRAIN_TASK}_poster.toml}"
NAIVE_CFG="${NAIVE_CFG:-configs/poster/es_${TRAIN_TASK}_naive.toml}"
SKIP_NAIVE="${SKIP_NAIVE:-0}"

# --- Ctrl-C handling -------------------------------------------------------
# Without a trap, bash runs the next loop iteration after a child is killed by
# SIGINT — which is why the old script ignored Ctrl-C. This kills the current
# child's process group and exits the whole sweep.
child_pgid=""
on_int() {
    echo
    echo "[sweep] interrupted — stopping."
    [ -n "$child_pgid" ] && kill -TERM -- "-$child_pgid" 2>/dev/null
    exit 130
}
trap on_int INT TERM

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }

# Stream the interesting per-step lines (reward / fitness / loss) to the
# terminal while the full output goes to <dir>.log. setsid puts the child in its
# own process group so the trap can signal the whole tree; process substitution
# keeps $? equal to python's exit code (a pipe would give tee's).
STEP_RE='step [0-9]|fitness|reward=|loss=|epoch=|theta_dev|mean_len|finished'

run_logged() { # run_logged <logfile> <cmd...>
    local logf="$1"
    shift
    setsid "$@" > >(tee "$logf" | stdbuf -oL grep -E --line-buffered "$STEP_RE") 2>&1 &
    child_pgid=$!
    wait "$child_pgid"
    local rc=$?
    child_pgid=""
    return $rc
}

train() { # train <config> <run_dir> [extra flags...]
    local cfg="$1" dir="$2"
    shift 2
    if [ -f "$dir/checkpoint-final/adapter_config.json" ]; then
        log "SKIP (already trained): $dir"
        return
    fi
    log "TRAIN $dir"
    local t0
    t0=$(date +%s)
    run_logged "$dir.log" timeout 3600 python -u run.py --config "$cfg" --model "$MODEL" --output-dir "$dir" "$@"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        log "  ok ($(($(date +%s) - t0))s)"
    elif [ "$rc" -ge 130 ]; then
        log "  INTERRUPTED (rc=$rc)"
        exit "$rc"
    else log "  FAIL rc=$rc (see $dir.log)"; fi
}

log "SWEEP train=$TRAIN_TASK transfer=$TRANSFER out=$OUT seeds='$SEEDS'"

# ---- 1. Train -------------------------------------------------------------
for s in $SEEDS; do
    train "$GRPO_CFG" "$OUT/grpo_${TRAIN_TASK}_s$s" --seed "$s"
    train "$ES_CFG" "$OUT/es_${TRAIN_TASK}_s$s" --seed "$s"
done
# Hacker demo: one seed is enough to make the contrast (skippable for fast probes).
if [ "$SKIP_NAIVE" != "1" ]; then
    train "$NAIVE_CFG" "$OUT/es_${TRAIN_TASK}_naive_s0" --seed 0
fi

# ---- 2. Held-out eval (chat_template MUST match training = on) ------------
# Greedy decode (deterministic, the locked headline estimator) + k3 KL-to-base.
eval_seed() { # eval_seed <eval_task> <seed> [extra adapters...]
    local task="$1" s="$2"
    shift 2
    local json="$OUT/eval_${task}_s$s.json"
    log "EVAL $task seed $s -> $json"
    run_logged "$OUT/eval_${task}_s$s.log" \
        timeout 3600 python -u -m grpo_es.eval --task "$task" --model "$MODEL" --kl \
        --chat-template on \
        --adapter base \
        "$OUT/grpo_${TRAIN_TASK}_s$s/checkpoint-final" \
        "$OUT/es_${TRAIN_TASK}_s$s/checkpoint-final" "$@" \
        --json "$json" --per-sample "$OUT/persample_${task}_s$s.jsonl"
    local rc=$?
    [ "$rc" -ge 130 ] && {
        log "  EVAL INTERRUPTED"
        exit "$rc"
    }
    [ -f "$json" ] && python scripts/paired_stats.py "$json" | tee -a "$PROG"
}

for s in $SEEDS; do
    # Within-task held-out. Seed 0 also scores the naive hacker leg (unless skipped).
    if [ "$s" = "0" ] && [ "$SKIP_NAIVE" != "1" ]; then
        eval_seed "$TRAIN_TASK" 0 "$OUT/es_${TRAIN_TASK}_naive_s0/checkpoint-final"
    else
        eval_seed "$TRAIN_TASK" "$s"
    fi
    # Transfer to the other IF task (generalization check).
    eval_seed "$TRANSFER" "$s"
done

# ---- 3. Pooled significance across seeds ----------------------------------
log "POOLED $TRAIN_TASK (within-task) across seeds:"
python scripts/paired_stats.py "$OUT"/eval_${TRAIN_TASK}_s*.json | tee -a "$PROG"
log "POOLED $TRANSFER (transfer) across seeds:"
python scripts/paired_stats.py "$OUT"/eval_${TRANSFER}_s*.json | tee -a "$PROG"
log "ALL DONE"
