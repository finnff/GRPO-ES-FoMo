#!/usr/bin/env python3
"""Paired CIs + p-values from an eval --json payload.

The eval runner scores `base` + every adapter on the SAME held-out prompts and
stores each one's `per_sample_reward` vector. This turns those vectors into the
paired comparisons the poster table needs (every adapter vs base, and the two
trained legs against each other), using grpo_es.eval.stats.

    python scripts/paired_stats.py outputs/poster/eval_ifeval_s0.json

Pass several JSONs (e.g. one per seed) to pool per-prompt scores across seeds:

    python scripts/paired_stats.py outputs/poster/eval_ifeval_s*.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_es.eval.stats import compare_paired


def norm(label):
    """Drop the per-seed suffix (``grpo_ifbench_s0`` -> ``grpo_ifbench``) so the
    same leg's vectors from seed 0/1/2 pool into one paired series."""
    return re.sub(r"_s\d+(?=/|$)", "", label)


def load(paths):
    """Merge adapters across payloads by label -> concatenated per-sample vector.

    Labels are seed-normalized first, so passing one JSON per seed pools the
    per-prompt scores of each leg across seeds (base[i] and a trained leg[i]
    stay aligned because every file is read in the same order for every leg).
    """
    label_to_scores, task, metric = {}, None, None
    order = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        task, metric = d["task"], d.get("metric_label", "reward")
        for a in d["adapters"]:
            lab = norm(a["adapter"])
            if lab not in label_to_scores:
                label_to_scores[lab] = []
                order.append(lab)
            label_to_scores[lab].extend(a["per_sample_reward"])
    return task, metric, order, label_to_scores


def short(label):
    return "base" if label in ("", "base") else "/".join(Path(label).parts[-2:])


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    task, metric, order, scores = load(paths)
    base_label = order[0]
    base = scores[base_label]
    n = len(base)
    print(f"task={task}  metric={metric}  n={n} prompts  ({len(paths)} payload(s))\n")
    print(f"{'comparison':<46}{'Δmean':>9}{'95% CI':>20}{'p':>9}  sig")
    print("-" * 92)

    def row(name, b, t):
        r = compare_paired(b, t)
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else \
              "*" if r.p_value < 0.05 else "ns"
        ci = f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        print(f"{name:<46}{r.mean_diff:>+9.3f}{ci:>20}{r.p_value:>9.4f}  {sig}")

    # Only legs present in every payload pool to the full n; a leg scored in a
    # subset of seeds (e.g. the naive hacker, seed 0 only) is reported separately
    # rather than crashing the paired comparison on a length mismatch.
    trained = [l for l in order if l != base_label]
    full = [l for l in trained if len(scores[l]) == n]
    partial = [l for l in trained if len(scores[l]) != n]
    for lab in full:
        row(f"{short(lab)}  vs  base", base, scores[lab])
    # Pairwise between the fully-pooled trained legs (parity: is GRPO != ES?).
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            a, b = full[i], full[j]
            row(f"{short(a)}  vs  {short(b)}", scores[b], scores[a])
    # Legs not in every seed: compare on their own n against the matching base slice.
    for lab in partial:
        m = len(scores[lab])
        row(f"{short(lab)}  vs  base [n={m}]", base[:m], scores[lab])


if __name__ == "__main__":
    main()
