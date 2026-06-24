#!/usr/bin/env python3
"""Aggregate the pydantic ES-vs-GRPO sweep across whatever seeds / methods exist.

Reads outputs/prime_pydantic_sweep/eval_pydantic_s*.json -- including the
method-tagged forms eval_pydantic_s{N}_grpo.json / _es.json written by
ONLY=grpo / ONLY=es runs -- and merges them per seed. Each adapter is
classified into a leg (base / GRPO / ES) by its checkpoint path, so the
adapter order does not matter and partial (single-method) seeds are fine.

Prints per-seed and across-seed mean+-std for the two headline metrics:
mean graded reward and exact accuracy (P[reward>=1]). Base is re-scored on the
same disjoint held-out window as the GRPO/ES legs ([500:600] for the 500-sample
default), so every delta is paired.

  python scripts/agg_pydantic_sweep.py [OUT_DIR]
"""
import glob
import json
import math
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/prime_pydantic_sweep"
LEGS = ["base", "GRPO", "ES"]


def leg_of(adapter_label):
    """Map an eval adapter label to a leg, or None."""
    s = adapter_label.lower()
    if s == "base":
        return "base"
    if "grpo_pydantic" in s:
        return "GRPO"
    if "es_pydantic" in s:
        return "ES"
    return None


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main():
    files = sorted(glob.glob(os.path.join(OUT, "eval_pydantic_s*.json")))
    if not files:
        print(f"no eval_pydantic_s*.json under {OUT} yet")
        return

    # seed -> leg -> adapter dict (merged across both/grpo/es files for that seed)
    by_seed = {}
    for f in files:
        m = re.search(r"eval_pydantic_s(\d+)", os.path.basename(f))
        if not m:
            continue
        seed = m.group(1)
        legs = by_seed.setdefault(seed, {})
        for a in json.load(open(f))["adapters"]:
            leg = leg_of(a["adapter"])
            if leg and leg not in legs:  # first writer wins; bases agree anyway
                legs[leg] = a

    seeds = sorted(by_seed)
    print(f"\n=== pydantic ES-vs-GRPO sweep ({len(seeds)} seed(s): "
          f"{', '.join(seeds)}) ===")
    print(f"{'seed':>5} | {'leg':>5} | {'mean_reward':>11} {'Δ_vs_base':>10} | "
          f"{'exact_acc':>9} {'Δ_acc':>7} | {'gate':>5} {'clip%':>6} {'KL':>7}")
    print("-" * 78)
    per_leg = {leg: {"mean": [], "acc": []} for leg in LEGS}
    for seed in seeds:
        legs = by_seed[seed]
        base = legs.get("base")
        base_mean = base["mean_reward"] if base else float("nan")
        base_acc = base["accuracy"] if base else float("nan")
        for leg in LEGS:
            a = legs.get(leg)
            if not a:
                continue
            mr, acc = a["mean_reward"], a["accuracy"]
            dmr = mr - base_mean
            dacc = acc - base_acc
            kl = a.get("kl_to_base")
            kl_s = f"{kl:.4f}" if isinstance(kl, (int, float)) else "  -  "
            print(f"{seed:>5} | {leg:>5} | {mr:11.4f} {dmr:+10.4f} | "
                  f"{acc:9.3f} {dacc:+7.3f} | {a.get('mean_gate', float('nan')):5.2f} "
                  f"{a.get('clip_frac', float('nan'))*100:5.1f}% {kl_s:>7}")
            if leg != "base" or not math.isnan(mr):
                per_leg[leg]["mean"].append(mr)
                per_leg[leg]["acc"].append(acc)
        print("-" * 78)

    if len(seeds) >= 2:
        print(f"\n=== across-seed mean +- std ===")
        print(f"{'leg':>5} | {'n':>2} | {'mean_reward':>18} | {'exact_acc':>16}")
        print("-" * 52)
        for leg in LEGS:
            n = len(per_leg[leg]["mean"])
            if n == 0:
                continue
            mm, ms = mean_std(per_leg[leg]["mean"])
            am, as_ = mean_std(per_leg[leg]["acc"])
            print(f"{leg:>5} | {n:>2} | {mm:8.4f} +- {ms:6.4f} | {am:6.3f} +- {as_:6.3f}")
        if per_leg["base"]["mean"]:
            bm, _ = mean_std(per_leg["base"]["mean"])
            for leg in ("GRPO", "ES"):
                if per_leg[leg]["mean"]:
                    lm, _ = mean_std(per_leg[leg]["mean"])
                    print(f"  Δ {leg} vs base (mean of per-seed means): {lm - bm:+.4f}")
    print()


if __name__ == "__main__":
    main()
