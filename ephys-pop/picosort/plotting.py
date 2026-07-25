"""Shared plotting helpers for the ephys-pop notebooks.

Plotting is not the learning objective, so it lives here rather than in the
notebooks. These reproduce the *flavour* of a spike-sorting GUI (probe maps,
voltage traces, waveform footprints), qualitatively -- not pixel-perfect.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_probe(probe, unit_xy=None, ax=None):
    """Scatter the channel positions; optionally overlay true unit locations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(2.4, 6))
    ax.scatter(probe.x, probe.y, s=40, c="0.4", marker="s", label="channels")
    for i, (xi, yi) in enumerate(zip(probe.x, probe.y)):
        ax.text(xi - 6, yi, str(i), va="center", ha="right", fontsize=6, color="0.5")
    if unit_xy is not None:
        unit_xy = np.atleast_2d(unit_xy)
        ax.scatter(unit_xy[:, 0], unit_xy[:, 1], s=90, marker="*",
                   c="tab:red", zorder=3, label="true units")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("depth y (µm)")
    ax.set_title("probe geometry")
    ax.invert_yaxis()
    return ax


def plot_template(template, probe, ax=None, cmap="RdBu_r"):
    """Heatmap of one unit's spatiotemporal template: channels (rows) x time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))
    vmax = np.abs(template).max()
    im = ax.imshow(template, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   extent=[0, template.shape[1], probe.n_channels - 0.5, -0.5])
    ax.set_xlabel("time (samples)"); ax.set_ylabel("channel")
    ax.set_title("spatiotemporal template")
    plt.colorbar(im, ax=ax, label="µV", fraction=0.046)
    return ax


def plot_footprint(template, probe, ax=None):
    """Peak-to-peak amplitude of a template across channels -- its spatial footprint."""
    if ax is None:
        _, ax = plt.subplots(figsize=(2.6, 5))
    ptp = template.max(axis=1) - template.min(axis=1)
    ax.plot(ptp, probe.y, "-o", ms=3, color="tab:purple")
    ax.set_xlabel("peak-to-peak (µV)"); ax.set_ylabel("depth y (µm)")
    ax.set_title("footprint")
    ax.invert_yaxis()
    return ax


def plot_unit_spike(recording, unit, which=0, n_channels=13, pad_ms=(6.0, 8.0),
                    ax=None):
    """Zoom in on one ground-truth spike of ``unit``, on the channel band around it.

    A whole-probe view over a long window turns spikes into mush; this centres on a
    single spike and its neighbouring channels so the spatial spread is visible.
    """
    gt = recording.ground_truth
    fs = recording.fs
    times = gt.spike_times[gt.spike_labels == unit]
    t_spk = int(times[which])
    pc = int(np.argmax(gt.templates[unit].max(1) - gt.templates[unit].min(1)))
    lo = max(0, pc - n_channels // 2)
    hi = min(recording.n_channels, lo + n_channels)
    ax = plot_traces(recording, channels=range(lo, hi),
                     t0=(t_spk - pad_ms[0] * 1e-3 * fs) / fs,
                     t1=(t_spk + pad_ms[1] * 1e-3 * fs) / fs,
                     mark_spikes=gt.spike_times, ax=ax)
    ax.set_title(f"unit {unit}: one spike, peak channel {pc}")
    return ax


def plot_traces(recording, channels=None, t0=0.0, t1=0.05, spacing=None,
                mark_spikes=None, ax=None):
    """Stacked voltage traces over a short window, one line per channel.

    ``t0``/``t1`` are in seconds. ``mark_spikes`` is an optional array of spike
    sample-times (e.g. ground truth) to draw as vertical ticks.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    fs = recording.fs
    i0, i1 = int(t0 * fs), int(t1 * fs)
    if channels is None:
        channels = range(recording.n_channels)
    seg = recording.traces[i0:i1]
    if spacing is None:
        spacing = 4 * np.std(seg)
    t = np.arange(i0, i1) / fs * 1e3  # milliseconds
    for row, ch in enumerate(channels):
        ax.plot(t, seg[:, ch] + row * spacing, lw=0.6, color="k")
    if mark_spikes is not None:
        mark = np.asarray(mark_spikes)
        mark = mark[(mark >= i0) & (mark < i1)]
        for m in mark:
            ax.axvline(m / fs * 1e3, color="tab:red", alpha=0.25, lw=1)
    ax.set_xlabel("time (ms)")
    ax.set_yticks([r * spacing for r in range(len(list(channels)))])
    ax.set_yticklabels(list(channels))
    ax.set_ylabel("channel")
    ax.set_title(f"raw voltage, {t0*1e3:.0f}–{t1*1e3:.0f} ms")
    return ax
