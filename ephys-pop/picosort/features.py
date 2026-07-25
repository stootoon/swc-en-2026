"""Feature extraction -- describe each spike with a few meaningful numbers.

A snippet is a big object (channels x samples), mostly noise and redundancy. We
boil each spike down to a handful of features that actually distinguish units:

* **where** it is -- its ``depth`` on the probe (centre of mass of the footprint),
* **how big** it is -- its ``amplitude`` on the peak channel,
* **what shape** it has -- the top few **PCA** scores of its peak-channel waveform.

Depth and amplitude are read straight off the snippet. For shape we use PCA: the
spikes of one neuron are near-copies, so a couple of principal components capture
almost all of a waveform's variation. Clustering then runs in this low-dimensional
**feature space** rather than on raw multichannel snippets.
"""

from __future__ import annotations

import numpy as np


def localize(snippets: np.ndarray, probe, radius_um: float = 60.0):
    """Per-spike peak channel, amplitude, and depth from the multichannel snippet.

    Returns ``(peak_channel, amplitude, depth)``. Depth is the peak-to-peak-weighted
    centre of mass, computed over only the channels within ``radius_um`` of the peak
    channel. Restricting to that neighbourhood keeps probe-wide noise (whose
    peak-to-peak can rival a weak channel's signal) from dragging the estimate
    toward the probe centre.
    """
    p2p = snippets.max(axis=2) - snippets.min(axis=2)          # (n_spikes, n_channels)
    peak_channel = np.argmax(p2p, axis=1)
    amplitude = p2p[np.arange(len(p2p)), peak_channel]
    dy = np.abs(probe.y[None, :] - probe.y[peak_channel][:, None])
    w = p2p * (dy <= radius_um)                                # only the local footprint
    depth = (w * probe.y[None, :]).sum(axis=1) / w.sum(axis=1)
    return peak_channel, amplitude, depth


def peak_waveforms(snippets: np.ndarray, peak_channel=None) -> np.ndarray:
    """The waveform on each spike's own peak channel: ``(n_spikes, n_samples)``."""
    if peak_channel is None:
        p2p = snippets.max(axis=2) - snippets.min(axis=2)
        peak_channel = np.argmax(p2p, axis=1)
    return snippets[np.arange(len(snippets)), peak_channel, :]


def pca_features(waveforms: np.ndarray, n_components: int = 3):
    """Project peak-channel waveforms onto their top principal components.

    Returns ``(scores, components, mean)`` with ``scores`` of shape
    ``(n_spikes, n_components)``. The components are waveform shapes; the scores say
    how much of each shape a given spike contains.
    """
    X = np.asarray(waveforms, float)
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    scores = Xc @ components.T
    return scores, components, mean


def explained_variance(waveforms: np.ndarray, n_components: int = 10):
    """Fraction of waveform variance captured by each of the first PCs."""
    X = np.asarray(waveforms, float)
    Xc = X - X.mean(axis=0)
    S = np.linalg.svd(Xc, compute_uv=False)
    var = S ** 2
    return (var / var.sum())[:n_components]


def spike_features(snippets: np.ndarray, probe, n_pcs: int = 2):
    """Assemble the standardized clustering features and the named parts.

    Returns ``(features, parts)``. ``features`` is ``(n_spikes, 2)``, the z-scored
    ``[depth, log10 amplitude]`` -- the two physical properties that cleanly separate
    well-isolated units, and what we cluster on. ``parts`` carries the raw
    ``peak_channel``, ``amplitude``, ``depth``, and the waveform ``pca`` scores /
    ``components`` used to *describe* and *visualise* spikes (Notebook 4) and to
    separate units that share a location.
    """
    peak_channel, amplitude, depth = localize(snippets, probe)
    waveforms = peak_waveforms(snippets, peak_channel)
    scores, components, mean = pca_features(waveforms, n_pcs)
    raw = np.column_stack([depth, np.log10(amplitude)])
    features = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    parts = dict(peak_channel=peak_channel, amplitude=amplitude, depth=depth,
                 pca=scores, components=components, waveforms=waveforms)
    return features, parts
