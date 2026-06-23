#!/usr/bin/env bash
# ES trust-cap / sigma sweep — diagnose whether the seed-0 ES inertness is a
# tuning artifact (trust region SATURATED every step: theta_dev pinned at the
# cap R from step 1) or a genuine signal failure.
#
# Mechanism (verified, see outputs/prime/ANALYSIS_es_vs_grpo.md): the raw
# pre-clamp update is ~130-400x R because raw_step ∝ es_lr/es_sigma and the
# auto-sigma is tiny. So the clamp fires every step and the master random-walks
# the R-sphere. Shrinking es_lr is the clean way to unbind the cap WITHOUT
# moving theta far from base (raising es_trust_ratio to unbind needs R≈450 ≫
# init_norm 13.5 → destroys the model; raising es_noise_ratio enough corrupts the
# perturbations). We bracket es_lr down from 0.05 and add a trust-only leg as a
# negative control.
#
# Read-out is `theta_dev` in each run's history.jsonl (and trackio): if it climbs
# from 0 instead of pinning at R, the cap unbound; then check whether fitness_mean
# develops an upward trend. Trackio is ON (logs fitness/theta_dev per step) — ES
# trackio is wired in grpo_es/methods/es.py; it falls back to history.jsonl if
# trackio init fails, so the diagnostic survives either way.
#
#   bash scripts/es_trustcap_sweep.sh ascii      # phase 1 (cheap, ~70min)
#   bash scripts/es_trustcap_sweep.sh pydantic    # phase 2 (run winners)
set -u

TASK="${1:-ascii}"
case "$TASK" in
  ascii)    CFG="configs/es_ascii_tree.toml" ;;
  pydantic) CFG="configs/es_pydantic_adherence.toml" ;;
  *) echo "usage: $0 {ascii|pydantic}"; exit 2 ;;
esac

OUT="outputs/prime_sweep/$TASK"
mkdir -p "$OUT"
SEED=0

# CALIBRATED (cap-off 1-step measurement, ascii): raw_step ≈ 29000*es_lr, R≈3.37.
#   es_lr 0.05 -> raw 1642 = 487x R (PINNED, control)
#   es_lr 1e-4 -> raw  2.9 = 0.86x R (just sub-cap; theta_dev climbs to ~R fast)
#   es_lr 5e-5 -> raw  1.5 = 0.43x R (sub-cap; climbs over ~5-10 steps)
#   es_lr 2e-5 -> raw  0.6 = 0.17x R (best window: climbs over ~10-30 steps)
# Unbinding the cap lets a weak-but-real coherent signal ACCUMULATE across steps
# (the every-step clamp was renormalizing it away, 99.6% direction overwrite).
# lr2e5_eb16 is the best-shot leg: unbound cap AND es_eval_batch 8->16 (more
# prompts/member -> lower fitness variance -> fewer zero-signal antithetic pairs,
# attacking the 65-76% zero-gradient problem directly).
run_cfg() {  # name  flags...
  local name="$1"; shift
  local dir="$OUT/${name}_s${SEED}"
  echo "===== TRAIN $TASK/$name  [$*]  ($(date +%H:%M:%S)) ====="
  python run.py --config "$CFG" --seed "$SEED" --trackio \
    --output-dir "$dir" "$@" > "$OUT/${name}_s${SEED}.log" 2>&1 \
    && echo "OK  $TASK/$name ($(date +%H:%M:%S))" \
    || echo "FAIL($?) $TASK/$name ($(date +%H:%M:%S))"
}

run_cfg base        --es-lr 0.05
run_cfg lr1e4       --es-lr 0.0001
run_cfg lr5e5       --es-lr 0.00005
run_cfg lr2e5       --es-lr 0.00002
run_cfg lr2e5_eb16  --es-lr 0.00002 --es-eval-batch 16

echo "SWEEP DONE $TASK ($(date +%H:%M:%S))"

# --- self-summary: the diagnostic read-out (theta_dev unbind + fitness trend) ---
echo; echo "===== SWEEP SUMMARY $TASK ====="
python3 - "$OUT" <<'PY'
import json, sys, glob, os
root = sys.argv[1]
def trend(xs):
    n=len(xs)
    if n<2: return 0.0
    mx=(n-1)/2; mean=sum(xs)/n
    num=sum((i-mx)*(x-mean) for i,x in enumerate(xs)); den=sum((i-mx)**2 for i in range(n))
    return num/den if den else 0.0
print(f"{'config':14} {'R':>7} {'dev1':>7} {'dev_last':>8} {'dev_max':>8} {'pinned?':>8} "
      f"{'fit1':>7} {'fit_last':>8} {'fit_slope/step':>14}")
for hp in sorted(glob.glob(os.path.join(root,'*_s0','history.jsonl'))):
    name=os.path.basename(os.path.dirname(hp)).replace('_s0','')
    rows=[json.loads(l) for l in open(hp)]
    if not rows: continue
    dev=[r['theta_dev'] for r in rows]; fit=[r['fitness_mean'] for r in rows]
    # R: read from the run's log line if present, else infer ~3.37
    R=3.37
    cfgp=os.path.join(os.path.dirname(hp),'run_config.json')
    if os.path.exists(cfgp):
        R=json.load(open(cfgp)).get('es_trust_region',R)
    pinned = (max(dev)-min(dev))/R < 0.02  # within 2% of flat at the cap
    print(f"{name:14} {R:7.2f} {dev[0]:7.2f} {dev[-1]:8.2f} {max(dev):8.2f} {str(pinned):>8} "
          f"{fit[0]:7.3f} {fit[-1]:8.3f} {trend(fit):14.5f}")
print("\nREAD: pinned?=True => cap still saturated (no unbinding). If a config UNBINDS")
print("(dev climbs from <R) AND fit_slope turns clearly positive => ES was under-tuned;")
print("if it unbinds but slope stays ~0 => weak antithetic signal, not the cap.")
PY
