#!/usr/bin/env bash
# Base-model-only held-out eval grid: 4 models x 2 hub envs (ascii-tree,
# pydantic-adherence). No GRPO, no ES — just where each bare base model sits on
# the graded rubric, to map task headroom for the poster.
#
# Each env auto-wires its local graded rubric + eval_offset=100/eval_size=100
# disjoint holdout via ENV_CUSTOMIZATIONS, so the slice and rubric match the
# training legs exactly. Greedy decode (eval default), chat-template ON (all
# four ship one; enable_thinking=False auto-applies for the two Qwen).
#
#   bash scripts/base_eval_grid.sh
set -u
OUT="outputs/base_eval_grid"
mkdir -p "$OUT/ascii" "$OUT/pydantic"

# name|model_id  (name = output slug; model_id = alias or repo id)
MODELS=(
  "lfm350m|lfm2.5-350m"
  "qwen0.8b|qwen3.5-0.8b"
  "lfm1.2b|lfm2.5-1.2b"
  "qwen2b|Qwen/Qwen3.5-2B"
)
# slug|env id
TASKS=(
  "ascii|primeintellect/ascii-tree"
  "pydantic|primeintellect/pydantic-adherence"
)

run_cell () {
  local tslug="$1" envid="$2" mslug="$3" model="$4"
  local dir="$OUT/$tslug"
  echo "===== [$tslug / $mslug] base eval ($model) ($(date +%H:%M:%S)) ====="
  timeout 1800 python -m grpo_es.eval \
    --task "env:$envid" --model "$model" --chat-template on \
    --adapter base \
    --json "$dir/$mslug.json" --per-sample "$dir/$mslug.persample.jsonl" \
    > "$dir/$mslug.log" 2>&1 \
    && echo "OK   $tslug/$mslug ($(date +%H:%M:%S))" \
    || echo "FAIL($?) $tslug/$mslug ($(date +%H:%M:%S)) — see $dir/$mslug.log"
}

for t in "${TASKS[@]}"; do
  IFS='|' read -r tslug envid <<< "$t"
  for m in "${MODELS[@]}"; do
    IFS='|' read -r mslug model <<< "$m"
    run_cell "$tslug" "$envid" "$mslug" "$model"
  done
done

echo "===== GRID SUMMARY ($(date +%H:%M:%S)) ====="
python3 - "$OUT" <<'PY' 2>/dev/null || true
import json, glob, os, sys
root = sys.argv[1]
rows = []
for tslug in ("ascii", "pydantic"):
    for path in sorted(glob.glob(os.path.join(root, tslug, "*.json"))):
        mslug = os.path.basename(path)[:-5]
        try:
            d = json.load(open(path))
            a = d["adapters"][0]
            rows.append((tslug, mslug, a["mean_reward"], a["accuracy"],
                         a.get("format_pass", 0.0), a["mean_length"],
                         a.get("clip_frac", 0.0), d.get("holdout")))
        except Exception as e:
            rows.append((tslug, mslug, None, None, None, None, None, None))
hdr = f"{'task':9} {'model':10} {'mean':>7} {'exact':>6} {'fmt':>5} {'len':>7} {'clip%':>6} {'n':>4}"
print(hdr); print("-"*len(hdr))
for r in rows:
    t, m, mean, acc, fmt, ln, clip, n = r
    if mean is None:
        print(f"{t:9} {m:10} {'ERR':>7}"); continue
    print(f"{t:9} {m:10} {mean:>7.4f} {acc:>6.3f} {fmt:>5.2f} {ln:>7.0f} {clip:>5.0%} {n:>4}")
PY

echo "ALL DONE ($(date +%H:%M:%S))"
