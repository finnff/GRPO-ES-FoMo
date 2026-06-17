#!/usr/bin/env bash
# Quick cross-model smoke: one GRPO + one ES config (the cheapest, *_toy) run
# unchanged except --model, over qwen3.5-0.8b (the config default), lfm2.5-350m,
# lfm2.5-1.2b. Confirms a single config transfers across model sizes (the ES
# sigma/trust-region auto-scale in particular). Only --model and --output-dir
# move; every other knob comes from the committed config.
#
#   bash scripts/model_test.sh
set -u

OUT_DIR="outputs/model-test"
mkdir -p "$OUT_DIR"
PROGRESS="$OUT_DIR/PROGRESS.log"
: > "$PROGRESS"

MODELS=("qwen3.5-0.8b" "lfm2.5-350m" "lfm2.5-1.2b")
CONFIGS=("grpo_toy" "es_toy")

for cfg in "${CONFIGS[@]}"; do
  for model in "${MODELS[@]}"; do
    tag="${model//./}"; tag="${tag//-/}"
    name="${cfg}__${tag}"
    log="$OUT_DIR/$name.log"
    echo "===== START $name (--model $model) =====" | tee -a "$PROGRESS"
    t0=$(date +%s)
    if python run.py --config "configs/$cfg.toml" --model "$model" --output-dir "$OUT_DIR/$name" > "$log" 2>&1; then
      status="OK"
    else
      status="FAIL"
    fi
    t1=$(date +%s)
    echo "===== END   $name :: $status ($((t1 - t0))s) =====" | tee -a "$PROGRESS"
  done
done

echo "ALL DONE" | tee -a "$PROGRESS"

python - "$OUT_DIR" <<'PY'
import json, sys, glob, os, re
out_dir = sys.argv[1]
rows = []
for cell in sorted(glob.glob(os.path.join(out_dir, "*", ""))):
    name = os.path.basename(cell.rstrip("/"))
    cfg_p = os.path.join(cell, "run_config.json")
    if not os.path.exists(cfg_p):
        rows.append((name, "FAIL", "", "", "", "")); continue
    method = json.load(open(cfg_p)).get("method", "?")
    first = last = mx = extra = ""
    if method == "grpo":
        states = glob.glob(os.path.join(cell, "checkpoint-*", "trainer_state.json"))
        best = None
        for s in states:
            try:
                d = json.load(open(s))
                if best is None or d.get("global_step", 0) > best.get("global_step", 0): best = d
            except Exception: pass
        if best:
            r = [e["reward"] for e in best["log_history"] if "reward" in e]
            if r: first, last, mx = round(r[0],3), round(r[-1],3), round(max(r),3)
    else:
        hp = os.path.join(cell, "history.jsonl")
        f = [json.loads(l)["fitness_mean"] for l in open(hp)] if os.path.exists(hp) else []
        if f: first, last, mx = round(f[0],3), round(f[-1],3), round(max(f),3)
        lp = os.path.join(out_dir, name + ".log")
        if os.path.exists(lp):
            m = re.search(r"init_norm=([\d.]+) step_noise_norm=([\d.]+) trust_region=([\d.]+)", open(lp).read())
            ms = re.search(r"sigma=(\S+)", open(lp).read())
            if m and ms: extra = "sigma=%s init=%s noise=%s R=%s" % (ms.group(1), m.group(1), m.group(2), m.group(3))
    rows.append((name, method, first, last, mx, extra))
summ = os.path.join(out_dir, "SUMMARY.tsv")
with open(summ, "w") as fh:
    fh.write("cell\tmethod\tfirst\tlast\tmax\tnotes\n")
    for r in rows: fh.write("\t".join(str(v) for v in r) + "\n")
print(open(summ).read())
PY
