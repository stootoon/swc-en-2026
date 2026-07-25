"""Preprocessing -- clean the raw voltage before any spike is detected.

Three nuisances, three tools, applied in order:

1. **High-pass filter** (``highpass_filter``) removes slow LFP / drift below a few
   hundred Hz, leaving the fast spikes.
2. **Common average reference** (``common_average_reference``) subtracts the
   across-channel average at each moment, killing signals shared *identically* by
   every channel (movement, reference artefacts).
3. **Whitening** (``whiten``) removes the *spatial* correlation of the remaining
   noise, so a fixed threshold means the same thing on every channel and nearby
   channels stop reporting the same fluctuation.

``preprocess`` runs all three. Everything downstream (detection, clustering,
matching pursuit) operates on the preprocessed traces.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def highpass_filter(traces: np.ndarray, fs: float, cutoff: float = 300.0,
                    order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth high-pass, applied to each channel independently."""
    b, a = butter(order, cutoff / (fs / 2.0), btype="high")
    return filtfilt(b, a, traces, axis=0).astype(np.float32)


def common_average_reference(traces: np.ndarray) -> np.ndarray:
    """Subtract the median across channels at each time sample.

    The median (rather than the mean) is robust: a big spike on one channel does
    not drag the reference. This removes any signal common to all channels.
    """
    return (traces - np.median(traces, axis=1, keepdims=True)).astype(np.float32)


def noise_covariance(traces: np.ndarray) -> np.ndarray:
    """Channel-by-channel noise covariance (spikes are sparse, so this is ~noise)."""
    x = traces - traces.mean(axis=0, keepdims=True)
    return (x.T @ x) / (len(x) - 1)


def whitening_matrix(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Symmetric (ZCA) whitening matrix ``W = C^{-1/2}`` from a covariance ``C``.

    ZCA keeps the channels in place (unlike PCA whitening), so a whitened trace
    still looks channel-by-channel like the original -- just decorrelated.
    """
    vals, vecs = np.linalg.eigh(cov)
    reg = eps * vals.max()
    return (vecs @ np.diag(1.0 / np.sqrt(vals + reg)) @ vecs.T)


def whiten(traces: np.ndarray, W: np.ndarray | None = None):
    """Whiten traces across channels. Returns ``(whitened, W)``.

    If ``W`` is not given it is estimated from these traces. Whitened traces are
    in units of noise standard deviations, so a threshold of "5" means 5 sigma on
    every channel.
    """
    if W is None:
        W = whitening_matrix(noise_covariance(traces))
    return (traces @ W).astype(np.float32), W


def preprocess(rec, cutoff: float = 300.0):
    """Full pipeline: high-pass -> common average reference -> whiten.

    Returns ``(whitened, filtered, W)``:

    * ``filtered`` -- high-passed and common-average-referenced traces, in
      microvolts, with the spatial footprint of each spike intact. Detection cuts
      its snippets, learns templates, and runs matching pursuit here.
    * ``whitened`` -- ``filtered`` with the noise decorrelated across channels, so a
      fixed detection threshold means the same thing everywhere. Spike *detection*
      runs here.
    * ``W`` -- the whitening matrix.
    """
    filtered = common_average_reference(highpass_filter(rec.traces, rec.fs, cutoff=cutoff))
    whitened, W = whiten(filtered)
    return whitened, filtered, W
