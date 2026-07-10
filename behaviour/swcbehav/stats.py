"""Reference implementations of the population-level inference tools (Part 2).

Mirrors ``models.py``/``design.py``: notebooks 9-11 have students re-derive each
technique, and import these versions for anything a later (possibly skipped)
section depends on. Nothing here refits the behavioural model -- these all operate
on per-session summary numbers, so the inference notebooks stay fast.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Correlation / linear fit (Fig 2E)                                           #
# --------------------------------------------------------------------------- #
def linear_fit(x, y):
    """Least-squares line y ~ a*x + b plus Pearson r and R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return dict(slope=slope, intercept=intercept, r=r, r2=r ** 2)


# --------------------------------------------------------------------------- #
# Variance partitioning + permutation test (Fig 2H)                           #
# --------------------------------------------------------------------------- #
def variance_explained(values, groups):
    """Fraction of the variance in ``values`` explained by group identity (eta^2).

    Between-group sum of squares over total sum of squares: 0 means the group
    means are identical, 1 means all variation is between groups.
    """
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    ss_between = sum(len(v) * (v.mean() - grand) ** 2
                     for v in (values[groups == g] for g in np.unique(groups)))
    return ss_between / ss_total


def permutation_variance_explained(values, groups, n_perm=2000, seed=0):
    """Permutation test for variance_explained: shuffle group labels to build a null."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    observed = variance_explained(values, groups)
    null = np.array([variance_explained(values, rng.permutation(groups))
                     for _ in range(n_perm)])
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return observed, null, p


# --------------------------------------------------------------------------- #
# Multiple comparisons: Benjamini-Hochberg FDR (Fig 4 tooling)                 #
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR control. Returns a boolean 'rejected' mask.

    Sort the p-values; the largest rank k with p_(k) <= (k/m) * alpha sets the
    cutoff, and every test at or below it is rejected.
    """
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = pvals[order] <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if passing.any():
        kmax = np.max(np.flatnonzero(passing))
        rejected[order[: kmax + 1]] = True
    return rejected


# --------------------------------------------------------------------------- #
# Hierarchical bootstrap for nested data (Fig 4 tooling)                       #
# --------------------------------------------------------------------------- #
def hierarchical_bootstrap(values_by_group, n_boot=2000, statistic=np.mean, seed=0):
    """Bootstrap that respects nesting: resample groups, then units within groups.

    ``values_by_group`` is a list of arrays (e.g. one array of session values per
    mouse). Each iteration draws groups with replacement, then draws units with
    replacement within each chosen group, and applies ``statistic`` to the pooled
    values. Returns the bootstrap distribution of the statistic.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, float) for g in values_by_group]
    n_groups = len(groups)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pooled = []
        for gi in rng.integers(0, n_groups, n_groups):
            g = groups[gi]
            pooled.append(g[rng.integers(0, len(g), len(g))])
        boot[b] = statistic(np.concatenate(pooled))
    return boot
