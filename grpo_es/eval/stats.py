"""Paired significance tests for method comparisons.

Both methods are scored on the *same* held-out prompts (pinned data seed), so
the right tests are paired ones: resample or permute per-prompt differences,
never the pooled scores. Everything is seeded — a paper table must reproduce
bit-for-bit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list, q in [0, 1]."""
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def paired_bootstrap_ci(
    baseline: list[float],
    treatment: list[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(mean_diff, ci_low, ci_high)`` for treatment - baseline.

    Resamples prompts (with replacement), which keeps each pair intact —
    that's what makes the interval a statement about this prompt population.
    """
    if len(baseline) != len(treatment):
        raise ValueError("paired test needs equal-length score lists")
    diffs = [t - b for b, t in zip(baseline, treatment)]
    n = len(diffs)
    rng = random.Random(seed)
    boot_means = sorted(
        _mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
    )
    return (
        _mean(diffs),
        _percentile(boot_means, alpha / 2),
        _percentile(boot_means, 1 - alpha / 2),
    )


def paired_permutation_test(
    baseline: list[float],
    treatment: list[float],
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided p-value under H0 "no effect": per-prompt difference signs are
    exchangeable, so flip them at random and see how often the permuted mean
    is at least as extreme as the observed one."""
    if len(baseline) != len(treatment):
        raise ValueError("paired test needs equal-length score lists")
    diffs = [t - b for b, t in zip(baseline, treatment)]
    observed = abs(_mean(diffs))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        permuted = _mean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(permuted) >= observed:
            hits += 1
    # +1 smoothing: an empirical p-value of exactly 0 overstates the evidence.
    return (hits + 1) / (n_perm + 1)


@dataclass
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_boot: int
    alpha: float


def compare_paired(
    baseline: list[float],
    treatment: list[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    mean_diff, ci_low, ci_high = paired_bootstrap_ci(
        baseline, treatment, n_boot=n_boot, alpha=alpha, seed=seed
    )
    p = paired_permutation_test(baseline, treatment, n_perm=n_boot, seed=seed)
    return PairedResult(
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        n=len(baseline),
        n_boot=n_boot,
        alpha=alpha,
    )
