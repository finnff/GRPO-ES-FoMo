#!/usr/bin/env python3
"""Rigorous PAIRED statistical analysis: GRPO vs ES vs base on held-out eval slices.

Per-sample rewards are paired by index (same disjoint held-out window, greedy decode).
scipy is unavailable in .venv-prime, so Wilcoxon signed-rank (normal approx with tie &
continuity correction), exact sign test (binomial), and bootstrap CIs are implemented
by hand with numpy + stdlib math.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIME = ROOT / "outputs" / "prime"
TOL = 1e-9
EXACT = 1.0 - 1e-9  # threshold for "reward >= 1.0" full credit

TASKS = [
    ("ascii-tree", PRIME / "eval_ascii_s0.json"),
    ("pydantic-adherence", PRIME / "eval_pydantic_s0.json"),
]


# ---------------------------------------------------------------------------
# statistics helpers (no scipy)
# ---------------------------------------------------------------------------
def _norm_sf(z):
    """Upper-tail of standard normal via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(delta, tol=TOL):
    """Two-sided Wilcoxon signed-rank test, normal approximation.

    Zeros (|d|<=tol) dropped (Wilcoxon's standard handling). Average ranks for ties,
    variance corrected for ties, continuity correction applied. Returns dict.
    """
    d = np.asarray(delta, dtype=float)
    nz = d[np.abs(d) > tol]
    n = nz.size
    if n == 0:
        return dict(n_effective=0, W_plus=0.0, W_minus=0.0, z=0.0, p_two_sided=1.0,
                    note="all paired deltas are zero")
    absd = np.abs(nz)
    order = np.argsort(absd, kind="mergesort")
    sorted_abs = absd[order]
    # average ranks for ties
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # ranks are 1-based
        ranks[i:j + 1] = avg
        i = j + 1
    rank_of = np.empty(n, dtype=float)
    rank_of[order] = ranks
    signs = np.sign(nz)
    W_plus = float(np.sum(rank_of[signs > 0]))
    W_minus = float(np.sum(rank_of[signs < 0]))
    W = min(W_plus, W_minus)
    mu = n * (n + 1) / 4.0
    # tie correction term
    _, counts = np.unique(sorted_abs, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    var = (n * (n + 1) * (2 * n + 1)) / 24.0 - tie_term / 48.0
    sigma = math.sqrt(var) if var > 0 else 0.0
    if sigma == 0.0:
        z = 0.0
        p = 1.0
    else:
        # continuity correction toward the mean
        cc = 0.5
        z = (W - mu + cc) / sigma  # W is the smaller sum -> below mean
        p = 2.0 * _norm_sf(abs(z))
        p = min(1.0, p)
    return dict(n_effective=n, W_plus=W_plus, W_minus=W_minus, W=W, mu=mu,
                sigma=sigma, z=z, p_two_sided=p, tie_corrected=tie_term > 0)


def _binom_cdf(k, n, p=0.5):
    """P(X<=k) for Binomial(n,p) via exact summation of log-comb terms."""
    total = 0.0
    for i in range(0, k + 1):
        logc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        total += math.exp(logc + i * math.log(p) + (n - i) * math.log(1 - p))
    return total


def sign_test(delta, tol=TOL):
    """Exact two-sided sign test on non-zero paired deltas (binomial, p=0.5)."""
    d = np.asarray(delta, dtype=float)
    pos = int(np.sum(d > tol))
    neg = int(np.sum(d < -tol))
    n = pos + neg
    if n == 0:
        return dict(n_effective=0, pos=0, neg=0, p_two_sided=1.0)
    k = min(pos, neg)
    cdf = _binom_cdf(k, n, 0.5)
    p = min(1.0, 2.0 * cdf)
    return dict(n_effective=n, pos=pos, neg=neg, k=k, p_two_sided=p)


def bootstrap_ci_mean(delta, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap 95% CI on the paired mean delta. Deterministic (seed=0)."""
    d = np.asarray(delta, dtype=float)
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return dict(point=float(d.mean()), lo=lo, hi=hi, n_boot=n_boot, seed=seed,
                boot_mean=float(means.mean()), boot_se=float(means.std(ddof=1)))


def bucket_counts(rewards):
    """Count exactly-0, open-(0,1), exactly-1 (>=1-eps)."""
    r = np.asarray(rewards, dtype=float)
    at0 = int(np.sum(np.abs(r) <= TOL))
    at1 = int(np.sum(r >= EXACT))
    mid = int(r.size - at0 - at1)
    return dict(at0=at0, mid=mid, at1=at1, n=int(r.size),
                mean=float(r.mean()),
                accuracy=float(np.mean(r >= EXACT)))


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------
def load_task(path):
    data = json.loads(Path(path).read_text())
    ads = {a["adapter"]: a for a in data["adapters"]}
    # canonical order base, GRPO, ES by position
    by_role = {}
    for a in data["adapters"]:
        name = a["adapter"]
        if name == "base":
            by_role["base"] = a
        elif "grpo" in name.lower():
            by_role["GRPO"] = a
        elif "es_" in name.lower() or "/es" in name.lower():
            by_role["ES"] = a
    return data, by_role


def fmt(x, p=6):
    return f"{x:.{p}f}"


def analyze():
    lines = []
    W = lines.append
    W("# Paired statistical analysis — GRPO vs ES vs base (seed 0, held-out [100:200])\n")
    W("All adapters evaluated on the SAME disjoint held-out window, greedy decode, so")
    W("per-sample rewards are paired by index (n=100). Wilcoxon signed-rank (normal")
    W("approx w/ tie + continuity correction), exact sign test (binomial p=0.5), and")
    W("percentile bootstrap (10k resamples, RNG seed=0) implemented by hand (scipy absent).\n")
    W("metric semantics: mean_reward = graded PARTIAL credit; accuracy = fraction with")
    W("reward >= 1.0 (FULL validation pydantic / exact normalized tree ascii).\n")

    struct = {}

    for task_name, path in TASKS:
        data, roles = load_task(path)
        base = np.asarray(roles["base"]["per_sample_reward"], dtype=float)
        struct[task_name] = {}
        W("\n" + "=" * 78)
        W(f"## TASK: {task_name}   (n={base.size})")
        W("=" * 78 + "\n")

        # distribution buckets for all three
        W("### 3. Reward-distribution buckets (mass at extremes vs middle)\n")
        W(f"{'adapter':>8} | {'=0.0':>5} | {'(0,1)':>6} | {'=1.0':>5} | "
          f"{'mean':>9} | {'acc':>6}")
        W("-" * 56)
        buckets = {}
        for role in ["base", "GRPO", "ES"]:
            b = bucket_counts(roles[role]["per_sample_reward"])
            buckets[role] = b
            W(f"{role:>8} | {b['at0']:>5} | {b['mid']:>6} | {b['at1']:>5} | "
              f"{fmt(b['mean']):>9} | {fmt(b['accuracy'],4):>6}")
        struct[task_name]["buckets"] = buckets
        W("")

        for role in ["GRPO", "ES"]:
            cur = np.asarray(roles[role]["per_sample_reward"], dtype=float)
            delta = cur - base
            stated_dmr = roles[role].get("delta_mean_reward")
            stated_dacc = roles[role].get("delta_accuracy")

            W("\n" + "-" * 78)
            W(f"### {role} vs base — {task_name}")
            W("-" * 78 + "\n")

            # 1. recompute deltas
            recomputed_dmr = float(cur.mean() - base.mean())
            base_acc = float(np.mean(base >= EXACT))
            cur_acc = float(np.mean(cur >= EXACT))
            recomputed_dacc = cur_acc - base_acc
            W("1. mean_reward & accuracy deltas (recomputed from per_sample vs JSON):")
            W(f"   base mean={fmt(base.mean())}  {role} mean={fmt(cur.mean())}")
            W(f"   delta_mean_reward  recomputed={fmt(recomputed_dmr)}  "
              f"JSON={fmt(stated_dmr) if stated_dmr is not None else 'n/a'}  "
              f"(diff {fmt(abs(recomputed_dmr - (stated_dmr or 0)),2+9)})")
            W(f"   base acc={fmt(base_acc,4)}  {role} acc={fmt(cur_acc,4)}")
            W(f"   delta_accuracy     recomputed={fmt(recomputed_dacc,4)}  "
              f"JSON={fmt(stated_dacc,4) if stated_dacc is not None else 'n/a'}  "
              f"(diff {fmt(abs(recomputed_dacc - (stated_dacc or 0)),2+9)})")

            # 2. paired per-sample win/loss
            improved = int(np.sum(delta > TOL))
            worsened = int(np.sum(delta < -TOL))
            unchanged = int(np.sum(np.abs(delta) <= TOL))
            gains = float(np.sum(delta[delta > TOL]))
            losses = float(np.sum(delta[delta < -TOL]))
            net = float(np.sum(delta))
            W("\n2. paired per-sample win/loss (reward-delta sign, tol 1e-9):")
            W(f"   improved={improved}  worsened={worsened}  unchanged={unchanged}")
            W(f"   sum_gains=+{fmt(gains)}  sum_losses={fmt(losses)}  "
              f"net={fmt(net)}  (net/n={fmt(net/base.size)})")

            # 4. wilcoxon + sign + bootstrap
            wil = wilcoxon_signed_rank(delta)
            sgn = sign_test(delta)
            boot = bootstrap_ci_mean(delta, n_boot=10000, seed=0)
            W("\n4. significance on paired delta:")
            W(f"   Wilcoxon signed-rank: n_eff={wil['n_effective']}  "
              f"W+={fmt(wil.get('W_plus',0),1)}  W-={fmt(wil.get('W_minus',0),1)}  "
              f"z={fmt(wil.get('z',0),4)}  p(2-sided)={fmt(wil['p_two_sided'],5)}")
            W(f"   sign test (exact binom): pos={sgn['pos']}  neg={sgn['neg']}  "
              f"p(2-sided)={fmt(sgn['p_two_sided'],5)}")
            W(f"   bootstrap mean-delta: point={fmt(boot['point'])}  "
              f"95% CI=[{fmt(boot['lo'])}, {fmt(boot['hi'])}]  "
              f"(boot SE={fmt(boot['boot_se'],5)}, 10k resamples seed=0)")
            ci_excludes_zero = (boot['lo'] > 0) or (boot['hi'] < 0)
            W(f"   -> 95% CI {'EXCLUDES' if ci_excludes_zero else 'INCLUDES'} zero")

            struct[task_name][role] = dict(
                recomputed_dmr=recomputed_dmr, json_dmr=stated_dmr,
                recomputed_dacc=recomputed_dacc, json_dacc=stated_dacc,
                improved=improved, worsened=worsened, unchanged=unchanged,
                gains=gains, losses=losses, net=net,
                wilcoxon_p=wil['p_two_sided'], wilcoxon_z=wil.get('z'),
                sign_p=sgn['p_two_sided'], sign_pos=sgn['pos'], sign_neg=sgn['neg'],
                boot_lo=boot['lo'], boot_hi=boot['hi'], boot_point=boot['point'],
                ci_excludes_zero=ci_excludes_zero,
            )

        # 5 & 6: extreme-mass transition analysis for pydantic (and ascii for completeness)
        W("\n" + "-" * 78)
        W(f"### 5/6. Extreme-mass transition analysis — {task_name}")
        W("-" * 78 + "\n")
        for role in ["GRPO", "ES"]:
            cur = np.asarray(roles[role]["per_sample_reward"], dtype=float)
            base_partial = (base > TOL) & (base < EXACT)
            base_zero = np.abs(base) <= TOL
            base_full = base >= EXACT
            cur_full = cur >= EXACT
            cur_zero = np.abs(cur) <= TOL

            # partial -> full (exact GAINED)
            exact_gained_idx = np.where(base_partial & cur_full)[0].tolist()
            # zero -> full
            zero_to_full_idx = np.where(base_zero & cur_full)[0].tolist()
            # partial -> zero (partial LOST to zero)
            partial_to_zero_idx = np.where(base_partial & cur_zero)[0].tolist()
            # full -> not full (regressions out of exact)
            full_lost_idx = np.where(base_full & ~cur_full)[0].tolist()
            # full -> zero specifically
            full_to_zero_idx = np.where(base_full & cur_zero)[0].tolist()

            # value-change accounting on the trade
            gain_exact_mass = float(np.sum(cur[exact_gained_idx] - base[exact_gained_idx])) \
                if exact_gained_idx else 0.0
            lost_partial_mass = float(np.sum(cur[partial_to_zero_idx] - base[partial_to_zero_idx])) \
                if partial_to_zero_idx else 0.0

            W(f"[{role}]  ({task_name})")
            W(f"   base buckets: zero={int(base_zero.sum())} "
              f"partial={int(base_partial.sum())} full={int(base_full.sum())}")
            W(f"   exact GAINED (0<base<1 -> =1.0):  n={len(exact_gained_idx)}  "
              f"idx={exact_gained_idx}")
            W(f"   zero -> full   (=0 -> =1.0):      n={len(zero_to_full_idx)}  "
              f"idx={zero_to_full_idx}")
            W(f"   partial LOST   (0<base<1 -> =0):  n={len(partial_to_zero_idx)}  "
              f"idx={partial_to_zero_idx}")
            W(f"   exact LOST     (base=1 -> <1):    n={len(full_lost_idx)}  "
              f"idx={full_lost_idx}")
            W(f"      of which full -> 0 exactly:    n={len(full_to_zero_idx)}  "
              f"idx={full_to_zero_idx}")
            W(f"   reward-mass from exact-gained = +{fmt(gain_exact_mass)} ; "
              f"reward-mass from partial->zero = {fmt(lost_partial_mass)}")
            net_full = (len(exact_gained_idx) + len(zero_to_full_idx)) - len(full_lost_idx)
            W(f"   net change in #full (=accuracy*n): "
              f"+{len(exact_gained_idx)+len(zero_to_full_idx)} gained "
              f"-{len(full_lost_idx)} lost = {net_full:+d}")
            W("")

            struct.setdefault(task_name, {}).setdefault("transitions", {})[role] = dict(
                exact_gained=exact_gained_idx, zero_to_full=zero_to_full_idx,
                partial_to_zero=partial_to_zero_idx, full_lost=full_lost_idx,
                full_to_zero=full_to_zero_idx,
                gain_exact_mass=gain_exact_mass, lost_partial_mass=lost_partial_mass,
                net_full=net_full,
            )

    report = "\n".join(lines) + "\n"
    out = PRIME / "ANALYSIS_paired_stats.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"\n[written] {out}")
    return struct


if __name__ == "__main__":
    analyze()
