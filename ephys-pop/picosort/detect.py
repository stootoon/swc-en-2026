"""Spike detection and snippet extraction.

On the preprocessed (whitened) traces a spike is a sharp **negative** deflection
that crosses several noise standard deviations. ``detect_spikes`` finds those
troughs on every channel, then collapses the copies of one spike (it appears on a
band of channels at once) into a single event at its peak channel.
``extract_snippets`` then cuts a short, trough-aligned, multi-channel clip around
each detected spike -- the raw material for the feature and clustering steps.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .generate import TEMPLATE_SAMPLES


def channel_noise(traces: np.ndarray) -> np.ndarray:
    """Robust per-channel noise sigma via the median absolute deviation."""
    return np.median(np.abs(traces), axis=0) / 0.6745


def detect_spikes(traces, probe, fs, threshold: float = 5.0, merge_ms: float = 0.3,
                  merge_radius_um: float = 50.0):
    """Detect spikes as threshold-crossing troughs, deduplicated across channels.

    Returns ``(times, peak_channels)`` -- integer sample indices and, for each, the
    channel where the trough was largest. A spike lands on several channels, so we
    first find candidate troughs per channel, then greedily keep the biggest and
    drop any other candidate close to it in both time and depth.
    """
    sd = channel_noise(traces)
    min_distance = max(1, int(merge_ms * 1e-3 * fs))

    # candidate troughs on every channel
    cand_t, cand_c, cand_amp = [], [], []
    for c in range(traces.shape[1]):
        peaks, props = find_peaks(-traces[:, c], height=threshold * sd[c],
                                  distance=min_distance)
        cand_t.append(peaks)
        cand_c.append(np.full(len(peaks), c))
        cand_amp.append(props["peak_heights"])
    cand_t = np.concatenate(cand_t)
    cand_c = np.concatenate(cand_c)
    cand_amp = np.concatenate(cand_amp)
    if len(cand_t) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # greedy spatiotemporal dedup: keep the largest, suppress nearby candidates
    order = np.argsort(cand_amp)[::-1]
    taken_t, taken_c = [], []
    alive = np.ones(len(cand_t), dtype=bool)
    for i in order:
        if not alive[i]:
            continue
        taken_t.append(cand_t[i])
        taken_c.append(cand_c[i])
        close_time = np.abs(cand_t - cand_t[i]) <= min_distance
        close_depth = np.abs(probe.y[cand_c] - probe.y[cand_c[i]]) <= merge_radius_um
        alive[close_time & close_depth] = False
    taken_t, taken_c = np.array(taken_t), np.array(taken_c)
    order = np.argsort(taken_t)
    return taken_t[order], taken_c[order]


def extract_snippets(traces, times, n_samples: int = TEMPLATE_SAMPLES,
                     align: bool = True, peak_channels=None, search: int = 5):
    """Cut a ``(n_spikes, n_channels, n_samples)`` array of clips around ``times``.

    If ``align`` and ``peak_channels`` are given, each clip is re-centred on the
    local trough of its peak channel (within +-``search`` samples), so the same
    part of the waveform lands at the same offset in every snippet -- essential for
    averaging clips into a clean template later.
    """
    half = n_samples // 2
    edge = half + search
    times = np.clip(np.asarray(times), edge, traces.shape[0] - edge - 1)
    if align and peak_channels is not None:
        aligned = []
        for t, pc in zip(times, peak_channels):
            local = traces[t - search:t + search + 1, pc]
            aligned.append(t - search + int(np.argmin(local)))
        times = np.array(aligned)
    # guard the edges
    times = np.clip(times, half, traces.shape[0] - half - 1)
    idx = times[:, None] + np.arange(-half, half + 1)[None, :]
    snippets = traces[idx]                       # (n_spikes, n_samples, n_channels)
    return np.transpose(snippets, (0, 2, 1)), times
