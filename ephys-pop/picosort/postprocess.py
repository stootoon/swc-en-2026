"""Merging and cleanup -- catch the mistakes clustering and matching leave behind.

Two spike trains that are really the *same* neuron (an over-split unit, or a
template that matched a neighbour's spikes) fire together far more than chance. The
tool for spotting this is the **cross-correlogram**: a histogram of the time
differences between two units' spikes. A tall spike at zero lag means duplicates ->
merge them. The **auto-correlogram** of a single clean unit, by contrast, has a
hole at zero: a real neuron can't fire twice within its ~1-2 ms **refractory
period**, so a filled-in hole is a warning that a "unit" is actually two.
"""

from __future__ import annotations

import numpy as np


def correlogram(times_a, times_b, fs, bin_ms: float = 0.5, window_ms: float = 25.0,
                exclude_zero: bool = False):
    """Histogram of (b - a) spike-time differences, in milliseconds.

    Returns ``(bin_centers_ms, counts)``. With ``times_a is times_b`` and
    ``exclude_zero=True`` this is the auto-correlogram.
    """
    ta = np.sort(np.asarray(times_a)) / fs * 1e3
    tb = np.sort(np.asarray(times_b)) / fs * 1e3
    w = window_ms
    edges = np.arange(-w, w + bin_ms, bin_ms)
    counts = np.zeros(len(edges) - 1)
    for t in ta:
        lo = np.searchsorted(tb, t - w)
        hi = np.searchsorted(tb, t + w)
        d = tb[lo:hi] - t
        if exclude_zero:
            d = d[np.abs(d) > 1e-9]
        counts += np.histogram(d, edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def refractory_violations(times, fs, refractory_ms: float = 1.5) -> float:
    """Fraction of inter-spike intervals shorter than the refractory period."""
    if len(times) < 2:
        return 0.0
    isi_ms = np.diff(np.sort(times)) / fs * 1e3
    return float(np.mean(isi_ms < refractory_ms))


def coincidence_fraction(times_a, times_b, fs, window_ms: float = 0.5) -> float:
    """Fraction of the smaller train that coincides (within +-window) with the other.

    Near 1 for duplicate units, near 0 for genuinely different neurons.
    """
    ta, tb = np.sort(times_a), np.sort(times_b)
    if len(ta) == 0 or len(tb) == 0:
        return 0.0
    tol = window_ms * 1e-3 * fs
    idx = np.searchsorted(tb, ta)
    idx = np.clip(idx, 1, len(tb) - 1)
    nearest = np.minimum(np.abs(tb[idx] - ta), np.abs(tb[idx - 1] - ta))
    hits = np.sum(nearest <= tol)
    return float(hits / min(len(ta), len(tb)))


def find_duplicate_pairs(spike_trains, fs, threshold: float = 0.4, window_ms: float = 0.5):
    """List (i, j, coincidence) for unit pairs whose trains coincide above threshold."""
    pairs = []
    n = len(spike_trains)
    for i in range(n):
        for j in range(i + 1, n):
            c = coincidence_fraction(spike_trains[i], spike_trains[j], fs, window_ms)
            if c >= threshold:
                pairs.append((i, j, c))
    return sorted(pairs, key=lambda p: -p[2])
