"""Shared plotting helpers for the ephys-pop notebooks.

Plotting is not the learning objective, so it lives here rather than in the
notebooks. These reproduce the *flavour* of a spike-sorting GUI (probe maps,
voltage traces, waveform footprints), qualitatively -- not pixel-perfect.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# A stable per-unit colour cycle, so a unit keeps its colour across figures.
UNIT_CMAP = "tab10"


def unit_color(i):
    return plt.get_cmap(UNIT_CMAP)(i % 10)


def plot_probe(probe, unit_xy=None, ax=None, orientation="horizontal"):
    """Scatter the channel positions; optionally overlay true unit locations.

    ``orientation="horizontal"`` lays the probe along the x-axis (depth left→right),
    which reads far better on a page than a tall, thin vertical strip.
    """
    horizontal = orientation == "horizontal"
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.4) if horizontal else (2.4, 6))
    cx, cy = (probe.y, probe.x) if horizontal else (probe.x, probe.y)
    ax.scatter(cx, cy, s=45, c="0.4", marker="s", label="channels")
    for i in range(probe.n_channels):
        if horizontal:
            ax.text(probe.y[i], probe.x[i] - 7, str(i), va="top", ha="center",
                    fontsize=6, color="0.5")
        else:
            ax.text(probe.x[i] - 6, probe.y[i], str(i), va="center", ha="right",
                    fontsize=6, color="0.5")
    if unit_xy is not None:
        unit_xy = np.atleast_2d(unit_xy)
        ux, uy = (unit_xy[:, 1], unit_xy[:, 0] + 12) if horizontal else (unit_xy[:, 0], unit_xy[:, 1])
        ax.scatter(ux, uy, s=110, marker="*", c="tab:red", zorder=3, label="true units")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("probe geometry")
    if horizontal:
        ax.set_xlabel("depth along probe (µm)"); ax.set_ylabel("x (µm)")
        ax.set_ylim(-25, 25)
    else:
        ax.set_xlabel("x (µm)"); ax.set_ylabel("depth y (µm)")
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


def plot_signal(traces, fs, channels=None, t0=0.0, t1=0.05, spacing=None,
                mark_spikes=None, ax=None, title=None):
    """Stacked voltage traces from a plain ``(n_samples, n_channels)`` array.

    ``t0``/``t1`` are in seconds. ``mark_spikes`` is an optional array of spike
    sample-times (e.g. ground truth) drawn as vertical ticks.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    i0, i1 = int(t0 * fs), int(t1 * fs)
    if channels is None:
        channels = range(traces.shape[1])
    channels = list(channels)
    seg = traces[i0:i1]
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
    ax.set_yticks([r * spacing for r in range(len(channels))])
    ax.set_yticklabels(channels)
    ax.set_ylabel("channel")
    ax.set_title(title if title is not None else f"voltage, {t0*1e3:.0f}–{t1*1e3:.0f} ms")
    return ax


def plot_traces(recording, channels=None, t0=0.0, t1=0.05, spacing=None,
                mark_spikes=None, ax=None):
    """Stacked voltage traces of a ``Recording`` over a short window."""
    return plot_signal(recording.traces, recording.fs, channels=channels, t0=t0,
                       t1=t1, spacing=spacing, mark_spikes=mark_spikes, ax=ax,
                       title=f"raw voltage, {t0*1e3:.0f}–{t1*1e3:.0f} ms")


def plot_covariance(cov, ax=None, title="channel covariance", labels=None,
                    annotate=False):
    """Heatmap of a channel-by-channel covariance (or correlation) matrix.

    ``labels`` sets the tick labels; ``annotate=True`` writes each entry's value in
    its cell (for small matrices like the 2x2 case).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.4))
    vmax = np.abs(cov).max()
    im = ax.imshow(cov, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    if labels is not None:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    else:
        ax.set_xlabel("channel"); ax.set_ylabel("channel")
    if annotate:
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                ax.text(j, i, f"{cov[i, j]:.0f}", ha="center", va="center", fontsize=11,
                        color="white" if abs(cov[i, j]) > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_feature_space(depth, amplitude, labels=None, ax=None, title="feature space"):
    """Scatter spikes in the depth-amplitude plane, optionally coloured by unit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    if labels is None:
        ax.scatter(depth, amplitude, s=10, c="0.5")
    else:
        labels = np.asarray(labels)
        for u in np.unique(labels):
            m = labels == u
            c = "0.7" if u < 0 else unit_color(int(u))
            ax.scatter(depth[m], amplitude[m], s=12, color=c,
                       label=("noise" if u < 0 else f"unit {u}"))
        ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel("depth (µm)"); ax.set_ylabel("amplitude (µV)"); ax.set_title(title)
    return ax


def plot_templates(templates, probe, ncols=None, height=3.0):
    """Grid of template footprints (peak-to-peak vs depth), one panel per unit."""
    n = len(templates)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.4 * ncols, height * nrows),
                             squeeze=False, sharey=True)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off"); continue
        ptp = templates[i].max(axis=1) - templates[i].min(axis=1)
        ax.plot(ptp, probe.y, "-", color=unit_color(i))
        ax.fill_betweenx(probe.y, 0, ptp, color=unit_color(i), alpha=0.3)
        ax.set_title(f"unit {i}", fontsize=9); ax.invert_yaxis()
    fig.supxlabel("peak-to-peak (µV)"); fig.supylabel("depth (µm)")
    fig.tight_layout()
    return fig, axes


def plot_correlogram(centers, counts, ax=None, refractory_ms=1.5, color="0.3",
                     title="correlogram"):
    """Bar plot of a cross/auto-correlogram with the refractory window shaded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3))
    width = centers[1] - centers[0]
    ax.bar(centers, counts, width=width, color=color)
    ax.axvspan(-refractory_ms, refractory_ms, color="tab:red", alpha=0.12)
    ax.set_xlabel("lag (ms)"); ax.set_ylabel("count"); ax.set_title(title)
    return ax


def plot_confusion(overlap, sorted_ids=None, true_ids=None, ax=None):
    """Heatmap of the sorted-vs-true spike overlap matrix (hits per pair)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(overlap, cmap="Blues", aspect="auto")
    ax.set_xlabel("true unit"); ax.set_ylabel("sorted unit")
    if true_ids is not None:
        ax.set_xticks(range(len(true_ids))); ax.set_xticklabels(true_ids)
    if sorted_ids is not None:
        ax.set_yticks(range(len(sorted_ids))); ax.set_yticklabels(sorted_ids)
    for a in range(overlap.shape[0]):
        for b in range(overlap.shape[1]):
            ax.text(b, a, int(overlap[a, b]), ha="center", va="center", fontsize=7,
                    color="white" if overlap[a, b] > overlap.max() / 2 else "black")
    ax.set_title("overlap: sorted vs true (spike hits)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_raster(times, labels, fs, t0=0.0, t1=None, ax=None, title="spike raster"):
    """Raster of spike times, one row per unit, coloured per unit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3))
    times = np.asarray(times) / fs
    labels = np.asarray(labels)
    t1 = t1 if t1 is not None else times.max()
    for row, u in enumerate(np.unique(labels)):
        m = (labels == u) & (times >= t0) & (times <= t1)
        ax.scatter(times[m], np.full(m.sum(), row), s=6, color=unit_color(int(u)))
    ax.set_xlabel("time (s)"); ax.set_ylabel("unit"); ax.set_title(title)
    ax.set_yticks(range(len(np.unique(labels))))
    return ax


# --------------------------------------------------------------------------- #
# Explanatory schematics (used in the roadmap / intro notebooks)              #
# --------------------------------------------------------------------------- #
def draw_probe_schematic(ax=None):
    """A stylised Neuropixels-style probe: a long thin shank studded with sites."""
    import matplotlib.patches as mpatches
    if ax is None:
        _, ax = plt.subplots(figsize=(2.6, 6))
    W, H = 1.0, 10.0                       # shank half-width, length (arbitrary units)
    # shank body + pointed tip
    ax.add_patch(mpatches.Rectangle((-W, 0), 2 * W, H, facecolor="0.85",
                                    edgecolor="0.4", lw=1.2))
    ax.add_patch(mpatches.Polygon([(-W, 0), (W, 0), (0, -0.9)], facecolor="0.85",
                                  edgecolor="0.4", lw=1.2))
    # base / flex at the top
    ax.add_patch(mpatches.Rectangle((-1.6, H), 3.2, 1.6, facecolor="0.7",
                                    edgecolor="0.4", lw=1.2))
    ax.text(0, H + 0.8, "base", ha="center", va="center", fontsize=8)
    # staggered recording sites
    n = 20
    ys = np.linspace(0.4, H - 0.3, n)
    for i, y in enumerate(ys):
        xoff = -0.45 if i % 2 else 0.45
        ax.add_patch(mpatches.Rectangle((xoff - 0.22, y - 0.11), 0.44, 0.22,
                                        facecolor="tab:orange", edgecolor="none"))
    ax.annotate("recording\nsites", xy=(0.45, ys[12]), xytext=(3.0, ys[15]),
                fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.annotate("shank\n(~1 cm long,\n~70 µm wide)", xy=(-W, H / 2), xytext=(-4.2, H / 2),
                fontsize=8, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xlim(-6, 6); ax.set_ylim(-1.5, H + 2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("a Neuropixels probe (schematic)", fontsize=10)
    return ax


def draw_spike_on_probe(ax=None, n_channels=11, pitch=25.0, unit_depth=140.0):
    """Cartoon: one neuron near a channel column, its spike shrinking with distance.

    Uses the real forward model so the spatial fall-off is honest, not hand-drawn.
    """
    from .probe import make_probe
    from .generate import spike_waveform, make_template
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))
    probe = make_probe(n_channels, pitch=pitch)
    wf = spike_waveform()
    tmpl = make_template(probe, [0, unit_depth], amplitude=1.0, waveform=wf)
    L = tmpl.shape[1]
    t = np.linspace(0, pitch * 0.8, L)
    scale = pitch * 1.1
    for c in range(n_channels):
        y = probe.y[c]
        ax.plot([0, pitch * 0.9], [y, y], color="0.85", lw=0.8, zorder=0)
        ax.add_patch(plt.Rectangle((-3, y - 2), 6, 4, color="tab:orange", zorder=1))
        ax.plot(t, y + tmpl[c] * scale, color="k", lw=1.3, zorder=2)
    # the neuron
    ax.scatter([-45], [unit_depth], s=340, color="tab:red", marker="o",
               edgecolor="k", zorder=3)
    ax.text(-45, unit_depth, "neuron", ha="center", va="center", fontsize=7, color="white")
    for c in range(n_channels):
        ax.plot([-40, -3], [unit_depth, probe.y[c]], color="tab:red", alpha=0.15, lw=1)
    ax.set_xlim(-70, pitch); ax.invert_yaxis()
    ax.set_xlabel("← channels record voltage over time →", fontsize=8)
    ax.set_ylabel("depth along probe (µm)")
    ax.set_title("one spike lands on many channels\n(biggest on the nearest)", fontsize=10)
    ax.set_yticks(probe.y[::2])
    return ax


def draw_pipeline(ax=None):
    """A left-to-right flow of the picosort stages, one box per notebook."""
    import matplotlib.patches as mpatches
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2.2))
    stages = ["raw\nvoltage", "preprocess\n(NB2)", "detect\n(NB3)", "features\n(NB4)",
              "cluster\n(NB5)", "template\nmatch (NB6)", "merge/QC\n(NB7)", "score\n(NB8)"]
    colors = ["0.85"] + ["#dCE8f5"] * 6 + ["#d6f0d6"]
    for i, (s, col) in enumerate(zip(stages, colors)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (i * 1.5, 0), 1.2, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=col, edgecolor="0.4"))
        ax.text(i * 1.5 + 0.6, 0.5, s, ha="center", va="center", fontsize=8.5)
        if i < len(stages) - 1:
            ax.annotate("", xy=(i * 1.5 + 1.42, 0.5), xytext=(i * 1.5 + 1.2, 0.5),
                        arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xlim(-0.2, len(stages) * 1.5); ax.set_ylim(-0.2, 1.2)
    ax.axis("off"); ax.set_title("the picosort pipeline", fontsize=11)
    return ax
