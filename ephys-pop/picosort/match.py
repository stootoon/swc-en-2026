"""Template matching -- explain the trace as a sum of templates (matching pursuit).

Detection + clustering give us a **template** per unit, but they stumble on
**collisions**: when two spikes overlap, their waveforms add and neither looks like
a clean template. Matching pursuit resolves this greedily. Repeatedly:

1. slide every template along the residual trace and find the (unit, time) whose
   template best fits what is left,
2. subtract that scaled template from the residual, recording a spike,

until nothing fits well enough. Overlapping spikes are peeled off one at a time.

The engine below is the efficient, Kilosort-style bookkeeping version: it
pre-computes each template's matched-filter output over the whole trace and, after
each subtraction, only patches the scores near the spike it removed (using the
templates' cross-correlations). Notebook 6 builds the *idea* on a small example;
this runs it fast on the full recording.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _matched_filter(traces, templates):
    """conv[j, t] = inner product of template j with the trace window at t."""
    n_units, n_ch, L = templates.shape
    conv = np.zeros((n_units, traces.shape[0]))
    for j in range(n_units):
        acc = np.zeros(traces.shape[0])
        for c in range(n_ch):
            acc += fftconvolve(traces[:, c], templates[j, c, ::-1], mode="same")
        conv[j] = acc
    return conv


def _template_cross(templates):
    """xcorr[j, k] : correlation of template j with template k across channels."""
    n_units, n_ch, L = templates.shape
    xc = np.zeros((n_units, n_units, 2 * L - 1))
    for j in range(n_units):
        for k in range(n_units):
            acc = np.zeros(2 * L - 1)
            for c in range(n_ch):
                acc += np.correlate(templates[j, c], templates[k, c], mode="full")
            xc[j, k] = acc
    return xc


def matching_pursuit(traces, templates, amp_threshold: float = 0.5,
                     max_spikes: int = 100000):
    """Deconvolve ``traces`` into spikes using ``templates``.

    Returns ``(times, labels, amplitudes)``. An amplitude near 1 means the template
    fit at full size; ``amp_threshold`` is the smallest fit (in template-norm units)
    worth calling a spike -- the stopping rule.
    """
    templates = np.asarray(templates, float)
    n_units, n_ch, L = templates.shape
    center = L // 2
    norm = (templates ** 2).sum(axis=(1, 2))            # ||template_j||^2
    conv = _matched_filter(traces, templates)
    xcorr = _template_cross(templates)                  # (units, units, 2L-1)
    lags = np.arange(-(L - 1), L)

    times, labels, amps = [], [], []
    for _ in range(max_spikes):
        score = conv / norm[:, None]                    # best amplitude at each (j,t)
        j, t = np.unravel_index(np.argmax(conv * score), conv.shape)
        a = conv[j, t] / norm[j]
        if a < amp_threshold:
            break
        times.append(int(t)); labels.append(int(j)); amps.append(float(a))
        # subtract a * template_j: patch conv near t for every template k
        lo, hi = t + lags[0], t + lags[-1] + 1
        c0 = max(0, -lo); c1 = (2 * L - 1) - max(0, hi - conv.shape[1])
        lo, hi = max(0, lo), min(conv.shape[1], hi)
        for k in range(n_units):
            conv[k, lo:hi] -= a * xcorr[k, j, c0:c1]

    order = np.argsort(times)
    return (np.array(times)[order], np.array(labels)[order], np.array(amps)[order])
