"""The whole picosort pipeline in one call -- the canonical stage order.

Each notebook builds one stage; this ties them together so Notebook 8 can score a
complete sort and any notebook has a working reference to fall back on. The stages
are exactly those of the module:

    preprocess -> detect -> extract -> localize + drop noise -> cluster
    -> templates -> matching pursuit

The one non-obvious constant is ``min_amplitude``: threshold crossings below it are
noise and collision fragments, not real spikes, and dropping them keeps the
clustering clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import preprocess
from .detect import detect_spikes, extract_snippets
from .features import spike_features
from .cluster import cluster_spikes, templates_from_labels
from .match import matching_pursuit


@dataclass
class SortResult:
    """Every intermediate of a picosort run, for inspection and scoring."""

    whitened: np.ndarray
    filtered: np.ndarray
    det_times: np.ndarray            # detected spike times (pre-clustering)
    peak_channels: np.ndarray
    snippets: np.ndarray
    features: np.ndarray
    parts: dict
    keep: np.ndarray                 # which detections passed the amplitude floor
    labels: np.ndarray               # cluster label per detection (-1 = dropped)
    templates: np.ndarray
    template_ids: np.ndarray
    spike_times: np.ndarray          # final (matching-pursuit) spike times
    spike_labels: np.ndarray
    spike_amplitudes: np.ndarray


def run_picosort(rec, threshold: float = 5.0, min_amplitude: float = 100.0,
                 n_units: int = 6, amp_threshold: float = 0.5, seed: int = 0) -> SortResult:
    """Run every stage on ``rec`` and return all intermediates as a ``SortResult``."""
    whitened, filtered, W = preprocess(rec)
    det_times, peak_channels = detect_spikes(whitened, rec.probe, rec.fs, threshold=threshold)
    snippets, det_times = extract_snippets(filtered, det_times, peak_channels=peak_channels)
    features, parts = spike_features(snippets, rec.probe)

    keep = parts["amplitude"] > min_amplitude
    labels = np.full(len(det_times), -1)
    labels[keep] = cluster_spikes(features[keep], n_units, seed=seed)
    templates, template_ids = templates_from_labels(snippets, labels)

    spike_times, spike_labels, spike_amps = matching_pursuit(
        filtered, templates, amp_threshold=amp_threshold)

    return SortResult(
        whitened=whitened, filtered=filtered, det_times=det_times,
        peak_channels=peak_channels, snippets=snippets, features=features, parts=parts,
        keep=keep, labels=labels, templates=templates, template_ids=template_ids,
        spike_times=spike_times, spike_labels=spike_labels, spike_amplitudes=spike_amps)
