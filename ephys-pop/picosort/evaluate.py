"""Scoring -- the thing you can only do with synthetic data.

We know every true spike and its unit, so we can grade the sort. Match each sorted
unit to the ground-truth unit it best explains, then count, spike by spike:

* a **hit** (true positive): a sorted spike within ``tol`` of a true spike,
* a **miss** (false negative): a true spike with no sorted spike nearby,
* a **false positive**: a sorted spike with no true spike nearby.

From these come **recall** (fraction of true spikes found), **precision** (fraction
of sorted spikes that are real), and an overall **agreement** score. This is exactly
the report you wish you had on a real recording -- and the reason we built one where
we could.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def _nearest_within(query, reference, tol):
    """How many points in ``query`` have a ``reference`` point within ``tol``."""
    if len(query) == 0 or len(reference) == 0:
        return 0
    reference = np.sort(reference)
    idx = np.clip(np.searchsorted(reference, query), 1, len(reference) - 1)
    nearest = np.minimum(np.abs(reference[idx] - query), np.abs(reference[idx - 1] - query))
    return int(np.sum(nearest <= tol))


@dataclass
class UnitScore:
    sorted_id: int
    true_id: int
    n_sorted: int
    n_true: int
    true_found: int      # true spikes with a sorted spike nearby (-> recall)
    sorted_real: int     # sorted spikes with a true spike nearby (-> precision)

    @property
    def recall(self):
        return self.true_found / self.n_true if self.n_true else 0.0

    @property
    def precision(self):
        return self.sorted_real / self.n_sorted if self.n_sorted else 0.0

    @property
    def agreement(self):
        # Kilosort-style: hits / (misses + hits + false positives)
        misses = self.n_true - self.true_found
        false_pos = self.n_sorted - self.sorted_real
        denom = self.true_found + misses + false_pos
        return self.true_found / denom if denom else 0.0


def match_to_truth(sorted_times, sorted_labels, gt_times, gt_labels, fs,
                   tol_ms: float = 0.5):
    """Optimally match sorted units to ground-truth units and score each pair.

    Returns ``(scores, overlap)`` where ``scores`` is a list of ``UnitScore`` (one
    per matched pair) and ``overlap`` is the (n_sorted x n_true) hit-count matrix
    used for the assignment. Unmatched true units show up as missed pairs.
    """
    tol = tol_ms * 1e-3 * fs
    s_ids = np.unique(sorted_labels)
    t_ids = np.unique(gt_labels)
    s_trains = {i: np.sort(sorted_times[sorted_labels == i]) for i in s_ids}
    t_trains = {j: np.sort(gt_times[gt_labels == j]) for j in t_ids}

    overlap = np.zeros((len(s_ids), len(t_ids)), dtype=int)
    for a, i in enumerate(s_ids):
        for b, j in enumerate(t_ids):
            overlap[a, b] = _nearest_within(t_trains[j], s_trains[i], tol)  # true found

    # optimal one-to-one assignment maximising total true spikes recovered
    rows, cols = linear_sum_assignment(-overlap)
    scores = []
    for a, b in zip(rows, cols):
        i, j = s_ids[a], t_ids[b]
        scores.append(UnitScore(
            sorted_id=int(i), true_id=int(j),
            n_sorted=len(s_trains[i]), n_true=len(t_trains[j]),
            true_found=int(overlap[a, b]),
            sorted_real=_nearest_within(s_trains[i], t_trains[j], tol)))
    return scores, overlap


def summary(scores):
    """Aggregate mean recall / precision / agreement across matched units."""
    if not scores:
        return dict(mean_recall=0.0, mean_precision=0.0, mean_agreement=0.0, n_units=0)
    return dict(
        mean_recall=float(np.mean([s.recall for s in scores])),
        mean_precision=float(np.mean([s.precision for s in scores])),
        mean_agreement=float(np.mean([s.agreement for s in scores])),
        n_units=len(scores),
    )
