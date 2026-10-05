"""Plotting helpers and schematics for the ephys-one notebooks.

Everything here is plain matplotlib. The schematics (channel cartoon, patch-clamp
rig, pipeline) are licence-clean drawings used in Notebook 0.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

CLOSED_COLOR = "0.3"
OPEN_COLOR = "tab:orange"


# --------------------------------------------------------------------------- #
# Kinetic schemes                                                              #
# --------------------------------------------------------------------------- #
def plot_scheme(ax, scheme, V=None, title=None):
    """Draw a kinetic scheme as boxes and arrows; rates shown if ``V`` is given."""
    from .kinetics import get_scheme
    scheme = get_scheme(scheme)
    n = scheme.n_states
    Q = scheme.Q(V) if V is not None else None
    xs = np.arange(n) * 2.2
    for k, (x, name) in enumerate(zip(xs, scheme.states)):
        face = "#ffe0b2" if scheme.conducting[k] else "#e0e0e0"
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.45, -0.4), 0.9, 0.8,
                                             boxstyle="round,pad=0.05", fc=face, ec="k", lw=1.5))
        ax.text(x, 0, name, ha="center", va="center", fontsize=16, fontweight="bold")
        ax.text(x, -0.7, "conducts" if scheme.conducting[k] else "no current",
                ha="center", va="top", fontsize=9, color="0.4")
    for k in range(n - 1):
        x0, x1 = xs[k] + 0.5, xs[k + 1] - 0.5
        fwd = Q is not None and Q[k, k + 1] > 0
        bwd = Q is not None and Q[k + 1, k] > 0
        if Q is None or fwd:
            ax.annotate("", xy=(x1, 0.15), xytext=(x0, 0.15),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
        if Q is None or bwd:
            ax.annotate("", xy=(x0, -0.15), xytext=(x1, -0.15),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
        if Q is not None:
            ax.text((x0 + x1) / 2, 0.28, f"{Q[k, k+1]:.0f}/s", ha="center", va="bottom", fontsize=9)
            ax.text((x0 + x1) / 2, -0.28, f"{Q[k+1, k]:.0f}/s", ha="center", va="top", fontsize=9)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8); ax.set_ylim(-1.1, 0.9)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or scheme.name + (f"   at V = {V:g} mV" if V is not None else ""))


# --------------------------------------------------------------------------- #
# Traces                                                                       #
# --------------------------------------------------------------------------- #
def plot_single_channel(t_ms, current, ax=None, i=None, title=None, color="k", lw=0.8):
    """A single-channel record: a square-ish trace hopping between two levels."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.4))
    ax.plot(t_ms, current, color=color, lw=lw, drawstyle="steps-post")
    if i is not None:
        ax.axhline(0, color=CLOSED_COLOR, ls=":", lw=0.8)
        ax.axhline(i, color=OPEN_COLOR, ls=":", lw=0.8)
        ax.text(t_ms[-1], 0, " closed", va="center", color=CLOSED_COLOR, fontsize=9)
        ax.text(t_ms[-1], i, " open", va="center", color=OPEN_COLOR, fontsize=9)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("current (pA)")
    if title:
        ax.set_title(title)
    return ax


def plot_voltage(ax, t_ms, V, color="tab:purple"):
    ax.plot(t_ms, V, color=color, lw=1.5, drawstyle="steps-post")
    ax.set_ylabel("V (mV)")
    ax.set_yticks(sorted(set(np.round(V, 1))))
    ax.tick_params(labelbottom=False)


def plot_sweeps(exp, n_show=20, ax=None, isochrones_ms=None, alpha=0.6, lw=0.6,
                color="k", title=None, show_voltage=True):
    """Overlay the first ``n_show`` sweeps; mark isochrones as vertical lines."""
    if ax is None:
        if show_voltage:
            fig, (axv, ax) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True,
                                          gridspec_kw=dict(height_ratios=[1, 4]))
            plot_voltage(axv, exp.t_ms, exp.V)
        else:
            _, ax = plt.subplots(figsize=(10, 3.4))
    n_show = min(n_show, exp.n_sweeps)
    for k in range(n_show):
        ax.plot(exp.t_ms, exp.sweeps[k], color=color, lw=lw, alpha=alpha)
    if isochrones_ms is not None:
        for j, tm in enumerate(np.atleast_1d(isochrones_ms)):
            ax.axvline(tm, color="tab:red", lw=1.2, ls="--",
                       label="isochrone" if j == 0 else None)
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("current (pA)")
    ax.set_title(title or f"{n_show} of {exp.n_sweeps} sweeps overlaid")
    return ax


def plot_sweep_image(exp, ax=None, cmap="viridis", title=None):
    """All sweeps as an image: rows = sweeps, columns = time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(exp.sweeps, aspect="auto", cmap=cmap, origin="lower",
                   extent=[exp.t_ms[0], exp.t_ms[-1], 0, exp.n_sweeps])
    plt.colorbar(im, ax=ax, label="current (pA)")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("sweep #")
    ax.set_title(title or "every sweep (one row each)")
    return ax


def plot_mean_variance(t_ms, mean, var, truth=None, axes=None, i_label="pA"):
    """Two stacked panels: mean current vs time and variance vs time."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    a0, a1 = axes
    a0.plot(t_ms, mean, color="k", lw=1.2, label="mean across sweeps")
    a1.plot(t_ms, var, color="tab:blue", lw=1.0, label="variance across sweeps")
    if truth is not None:
        p = truth.p_open; N, i = truth.N, truth.i
        a0.plot(t_ms, N * i * p, color="tab:red", ls="--", lw=1, label="truth  N·i·p(t)")
        a1.plot(t_ms, N * i ** 2 * p * (1 - p) + truth.noise_sd ** 2, color="tab:red", ls="--",
                lw=1, label="truth  N·i²·p(1−p)")
    a0.set_ylabel(f"mean ({i_label})"); a1.set_ylabel(f"variance ({i_label}²)")
    a1.set_xlabel("time (ms)")
    a0.legend(fontsize=9); a1.legend(fontsize=9)
    return axes


# --------------------------------------------------------------------------- #
# The variance-mean parabola                                                   #
# --------------------------------------------------------------------------- #
def true_parabola(truth, mean=None):
    """The exact parabola var = i*mean - mean^2/N (+ noise floor) for a Truth."""
    top = truth.i * truth.N
    if mean is None:
        mean = np.linspace(0, top, 200)
    return mean, truth.i * mean - mean ** 2 / truth.N + truth.noise_sd ** 2


def plot_variance_mean(mean, var, fit=None, truth=None, ax=None, t_ms=None,
                       s=10, title=None, full_parabola=True, label=None, color=None):
    """Scatter of variance against mean (one dot per isochrone) + fitted/true curves.

    If ``t_ms`` is given the dots are coloured by time, so you can see the path the
    experiment traces along the parabola.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    if t_ms is not None and color is None:
        sc = ax.scatter(mean, var, c=t_ms, cmap="viridis", s=s, label=label or "isochrones")
        plt.colorbar(sc, ax=ax, label="time (ms)")
    else:
        ax.scatter(mean, var, s=s, color=color or "k", alpha=0.7, label=label or "isochrones")
    if truth is not None:
        m, v = true_parabola(truth)
        if not full_parabola:
            keep = m <= 1.05 * np.max(mean); m, v = m[keep], v[keep]
        ax.plot(m, v, color="tab:red", ls="--", lw=1.5,
                label=f"truth: i={truth.i:g} pA, N={truth.N}")
    if fit is not None:
        m, v = fit.curve()
        if not full_parabola:
            keep = m <= 1.05 * np.max(mean); m, v = m[keep], v[keep]
        ax.plot(m, v, color="tab:blue", lw=2,
                label=f"fit: i={fit.i:.2f} pA, N={fit.N:.0f}")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("mean current (pA)"); ax.set_ylabel("variance (pA²)")
    ax.set_title(title or "variance vs mean, one dot per isochrone")
    ax.legend(fontsize=9)
    return ax


def annotate_parabola(ax, i, N, color="tab:red"):
    """Mark what each feature of the parabola tells you."""
    top = i * N
    m = np.linspace(0, top, 200)
    ax.plot(m, i * m - m ** 2 / N, color=color, lw=2)
    # slope at the origin
    m0 = np.linspace(0, 0.25 * top, 10)
    ax.plot(m0, i * m0, color="tab:blue", ls="--", lw=1.5)
    ax.annotate("initial slope = i\n(unitary current)", xy=(0.2 * top, 0.2 * top * i),
                xytext=(0.25 * top, 0.95 * i * i * N / 4), fontsize=10, color="tab:blue",
                arrowprops=dict(arrowstyle="->", color="tab:blue"))
    # root
    ax.plot([top], [0], "o", color="tab:green", ms=8)
    ax.annotate("variance back to zero when\nmean = i·N (every channel open)",
                xy=(top, 0), xytext=(0.62 * top, 0.35 * i * i * N / 4), fontsize=10,
                color="tab:green", ha="center", arrowprops=dict(arrowstyle="->", color="tab:green"))
    # peak
    ax.plot([top / 2], [i * i * N / 4], "o", color="tab:purple", ms=8)
    ax.annotate("peak at mean = i·N/2  (half the channels open)\nheight i²·N/4",
                xy=(top / 2, i * i * N / 4), xytext=(0.5 * top, 1.18 * i * i * N / 4),
                fontsize=10, color="tab:purple", ha="center",
                arrowprops=dict(arrowstyle="->", color="tab:purple"))
    ax.set_ylim(-0.05 * i * i * N / 4, 1.35 * i * i * N / 4)
    ax.set_xlim(-0.02 * top, 1.08 * top)
    ax.set_xlabel("mean current (pA)"); ax.set_ylabel("variance (pA²)")


def plot_bootstrap(i_boot, N_boot, truth=None, axes=None):
    """Histograms of bootstrap estimates for i and N, with the truth marked."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, vals, name, tv in zip(axes, [i_boot, N_boot], ["i (pA)", "N (channels)"],
                                  [getattr(truth, "i", None), getattr(truth, "N", None)]):
        vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=30, color="tab:blue", alpha=0.7)
        ax.axvline(np.median(vals), color="k", lw=1.5, label="median estimate")
        if tv is not None:
            ax.axvline(tv, color="tab:red", lw=2, ls="--", label="truth")
        ax.set_xlabel(name); ax.set_ylabel("bootstrap samples"); ax.legend(fontsize=9)
    return axes


def plot_scorecard(estimates, truths, ax=None):
    """Estimates ± CI against truth for several patches, as % error.

    ``estimates``: dict name -> CountResult; ``truths``: dict name -> Truth.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.8))
    names = list(estimates)
    x = np.arange(len(names))
    for off, (attr, boot, color) in enumerate([("i", "i_boot", "tab:blue"), ("N", "N_boot", "tab:orange")]):
        for k, name in enumerate(names):
            r, t = estimates[name], truths[name]
            tv = getattr(t, attr)
            est = getattr(r.fit, attr)
            b = np.asarray(getattr(r, boot)); b = b[np.isfinite(b)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            err = 100 * (est - tv) / tv
            ax.errorbar(x[k] + (off - 0.5) * 0.25, err,
                        yerr=[[err - 100 * (lo - tv) / tv], [100 * (hi - tv) / tv - err]],
                        fmt="o", color=color, capsize=4, label=attr if k == 0 else None)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("error vs truth (%)"); ax.legend(title="estimate")
    ax.set_title("how far off were we? (dot = estimate, bar = 95% bootstrap interval)")
    return ax


# --------------------------------------------------------------------------- #
# Schematics (Notebook 0)                                                      #
# --------------------------------------------------------------------------- #
def _bilayer(ax, x0, x1, y, head_r=0.09, tail=0.22, n=None, color="#90a4ae"):
    """A stretch of lipid bilayer between x0 and x1 centred on height y."""
    n = n or int((x1 - x0) / (2.2 * head_r))
    xs = np.linspace(x0 + head_r, x1 - head_r, n)
    for x in xs:
        for sign in (+1, -1):
            yh = y + sign * (tail + head_r)
            ax.add_patch(mpatches.Circle((x, yh), head_r, fc=color, ec="none"))
            ax.plot([x, x], [y + sign * tail, y + sign * 0.02], color=color, lw=1.2)


def draw_channel_cartoon(ax):
    """A membrane with one channel closed (left) and one open (right), ions flowing."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.axis("off")
    y = 2.5
    # bilayer with two gaps for the channels
    _bilayer(ax, 0.2, 2.2, y); _bilayer(ax, 3.6, 6.6, y); _bilayer(ax, 8.0, 9.8, y)
    # closed channel: two halves touching
    for xo in (2.25, 3.05):
        ax.add_patch(mpatches.FancyBboxPatch((xo, y - 0.65), 0.5, 1.3, boxstyle="round,pad=0.03",
                                             fc="#bcaaa4", ec="k", lw=1.2))
    ax.text(2.9, y + 1.2, "closed", ha="center", fontsize=12, fontweight="bold", color=CLOSED_COLOR)
    ax.text(2.9, y - 1.25, "no current", ha="center", fontsize=10, color=CLOSED_COLOR)
    # open channel: two halves with a pore
    for xo in (6.65, 7.45):
        ax.add_patch(mpatches.FancyBboxPatch((xo, y - 0.65), 0.45, 1.3, boxstyle="round,pad=0.03",
                                             fc="#bcaaa4", ec="k", lw=1.2))
    ax.text(7.3, y + 1.2, "open", ha="center", fontsize=12, fontweight="bold", color=OPEN_COLOR)
    ax.text(7.3, y - 1.25, "unitary current  i", ha="center", fontsize=10, color=OPEN_COLOR)
    # ions moving through the pore
    for k, yy in enumerate([y + 1.0, y + 0.35, y - 0.35, y - 1.0]):
        ax.add_patch(mpatches.Circle((7.3, yy), 0.1, fc=OPEN_COLOR, ec="k", lw=0.5))
    ax.annotate("", xy=(7.3, y - 1.9), xytext=(7.3, y + 1.9),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=OPEN_COLOR))
    ax.text(0.3, y + 1.7, "outside", fontsize=10, color="0.4")
    ax.text(0.3, y - 1.9, "inside", fontsize=10, color="0.4")
    ax.set_title("an ion channel is a switch: closed (nothing) or open (a fixed current i)")


def draw_patch_clamp(ax):
    """A cell, a patch pipette on its membrane, and the amplifier that clamps voltage."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_aspect("equal"); ax.axis("off")
    # cell
    ax.add_patch(mpatches.Ellipse((3.2, 2.4), 4.6, 3.6, fc="#fff3e0", ec="#bf8f00", lw=2))
    ax.text(3.2, 2.4, "cell", ha="center", va="center", fontsize=13, color="#8a6d00")
    # a few channels dotted in the membrane under the pipette
    for x in np.linspace(4.95, 5.35, 4):
        ax.add_patch(mpatches.Rectangle((x - 0.05, 3.55 + 0.9 * (x - 5.15)), 0.1, 0.25,
                                        angle=-50, fc="#6d4c41", ec="none"))
    # pipette (tapered)
    tip = np.array([[5.25, 3.9], [5.55, 3.75], [8.4, 5.8], [7.6, 5.95]])
    ax.add_patch(mpatches.Polygon(tip, closed=True, fc="#e3f2fd", ec="#1565c0", lw=1.5))
    ax.text(7.2, 5.2, "patch pipette", fontsize=10, color="#1565c0", rotation=35)
    ax.annotate("patch of membrane\nwith N channels", xy=(5.15, 3.65), xytext=(6.3, 2.2),
                fontsize=9, ha="center", color="#6d4c41",
                arrowprops=dict(arrowstyle="->", color="#6d4c41", lw=1))
    # amplifier
    ax.add_patch(mpatches.FancyBboxPatch((8.2, 2.3), 1.6, 1.4, boxstyle="round,pad=0.05",
                                         fc="#eceff1", ec="k", lw=1.5))
    ax.text(9.0, 3.0, "amplifier", ha="center", va="center", fontsize=10)
    ax.plot([8.0, 8.4, 9.0, 9.0], [5.85, 5.85, 5.85, 3.7], color="k", lw=1.2)
    ax.annotate("", xy=(9.0, 3.7), xytext=(9.0, 4.0), arrowprops=dict(arrowstyle="-"))
    ax.text(9.95, 3.35, "sets V\n(command)", fontsize=9, va="center", color="tab:purple")
    ax.text(9.95, 2.6, "records I\n(current)", fontsize=9, va="center", color="k")
    # ground / bath electrode
    ax.plot([0.9, 0.9], [0.3, 1.3], color="k", lw=1.2)
    for w, yy in zip([0.5, 0.35, 0.2], [0.3, 0.18, 0.06]):
        ax.plot([0.9 - w / 2, 0.9 + w / 2], [yy, yy], color="k", lw=1.2)
    ax.text(1.2, 0.7, "bath electrode", fontsize=9, color="0.4")
    ax.set_title("voltage clamp: hold the membrane at a chosen voltage, record the current it takes")


def draw_pipeline(ax):
    """The analysis as a chain of boxes."""
    steps = [("voltage\nsteps", "repeat the\nsame step\nmany times"),
             ("stack of\nsweeps", "one current\ntrace per\nrepeat"),
             ("isochrone\nmean & variance", "across sweeps,\nat each time"),
             ("variance vs\nmean plot", "one dot per\ntime point"),
             ("fit the\nparabola", "var = i·m − m²/N"),
             ("N,  i,  p_max", "channels,\nunitary current,\nopen probability")]
    ax.set_xlim(0, len(steps) * 2.0); ax.set_ylim(0, 2.2); ax.axis("off")
    for k, (title, sub) in enumerate(steps):
        x = k * 2.0 + 0.15
        last = k == len(steps) - 1
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.9), 1.7, 1.0, boxstyle="round,pad=0.05",
                                             fc="#e8f5e9" if last else "#e3f2fd",
                                             ec="#2e7d32" if last else "#1565c0", lw=1.5))
        ax.text(x + 0.85, 1.4, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.85, 0.55, sub, ha="center", va="center", fontsize=8.5, color="0.35")
        if not last:
            ax.annotate("", xy=(x + 2.0, 1.4), xytext=(x + 1.75, 1.4),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("nonstationary noise analysis, step by step", fontsize=11)


def plot_open_raster(ax, open_matrix, t_ms, n_show=40, color=OPEN_COLOR, title=None):
    """One row per sweep; a bar wherever the channel is open (``open_matrix`` is bool)."""
    n_show = min(n_show, open_matrix.shape[0])
    dt = t_ms[1] - t_ms[0]
    for k in range(n_show):
        x = np.asarray(open_matrix[k]).astype(np.int8)
        change = np.flatnonzero(np.diff(x) != 0) + 1
        starts = np.concatenate([[0], change]); stops = np.concatenate([change, [len(x)]])
        bars = [(t_ms[a], (b - a) * dt) for a, b in zip(starts, stops) if x[a] == 1]
        if bars:
            ax.broken_barh(bars, (k + 0.1, 0.8), color=color, lw=0)
    ax.set_ylim(0, n_show); ax.set_ylabel("sweep #")
    if title:
        ax.set_title(title)
    return ax
