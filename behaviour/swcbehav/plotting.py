"""Shared plotting helpers for the behaviour notebooks.

Plotting is not the learning objective, so it lives here rather than in the
notebooks. These functions reproduce the *flavor* of the paper's panels
(qualitative, not pixel-perfect).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .task import IMAGE_DURATION
from .generate import WEIGHT_COLORS


def _draw_design_rows(ax, X, col_names, i0, i1, x_left, x_right):
    """Draw each design-matrix row in its paper strategy color (white -> color)."""
    for i, name in enumerate(col_names):
        cmap = LinearSegmentedColormap.from_list("", ["white", WEIGHT_COLORS.get(name, "black")])
        ax.imshow(X[i0:i1, i][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest", extent=[x_left, x_right, i + 0.5, i - 0.5])
    ax.set_ylim(len(col_names) - 0.5, -0.5)
    ax.set_yticks(range(len(col_names)))
    ax.set_yticklabels(col_names)
    for tick, name in zip(ax.get_yticklabels(), col_names):
        tick.set_color(WEIGHT_COLORS.get(name, "black"))


def plot_session_raster(sess, t0=560.0, t1=580.0, ax=None):
    """Licks, bouts, and task events over a time window (Fig 1B / 2I flavor).

    Blue lines = image changes, cyan dashed = omissions, red dots = rewarded
    changes (hits), grey spans = detected licking bouts, black ticks = licks.
    """
    from .design import segment_bouts

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.2))
    t = sess.table
    win = t[(t["time"] >= t0) & (t["time"] <= t1)]

    for onset in win.loc[win["is_change"], "time"]:
        ax.axvline(onset, color="tab:blue", lw=1.5, zorder=1)
    for onset in win.loc[win["is_omission"], "time"]:
        ax.axvline(onset, color="tab:cyan", ls="--", lw=1, zorder=1)

    bouts = segment_bouts(sess.lick_times)
    for b0, b1 in bouts:
        if b1 >= t0 and b0 <= t1:
            ax.axvspan(b0, b1, color="0.85", zorder=0)

    licks = sess.lick_times[(sess.lick_times >= t0) & (sess.lick_times <= t1)]
    ax.vlines(licks, 0.0, 1.0, color="k", lw=0.8, zorder=2)

    hits = win.loc[win["is_hit"], "time"]
    ax.plot(hits, np.full(len(hits), 1.15), "v", color="tab:red", ms=6, zorder=3)

    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    return ax


def plot_design_matrix(X, col_names, table, start=0, n=40, ax=None):
    """Show the design-matrix columns over a slice of images (Fig 1C flavor)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.0))
    _draw_design_rows(ax, X, col_names, start, start + n, -0.5, n - 0.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("image (relative)")
    return ax


def plot_regressors_and_licks(sess, X, col_names, start=0, n=40):
    """Stack the annotated lick raster over the design-matrix slice (shared x).

    Top panel: licks, bouts, and task events over a window of images (same
    annotations as ``plot_session_raster``). Bottom panel: the design-matrix
    columns for the *same* images. Lets students see how each regressor lines up
    with the behavior it is meant to explain. Events are drawn at the centre of
    their image column so they sit over the corresponding regressor cell.
    """
    from .design import segment_bouts

    i0, i1 = start, start + n
    fig, (ax_r, ax_m) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                                     gridspec_kw={"height_ratios": [1.0, 1.2]})
    t = sess.table
    win = t[(t["image_index"] >= i0) & (t["image_index"] < i1)]

    # --- top: raster, x in image-index units (image i spans [i, i+1)) ---
    for idx in win.loc[win["is_change"], "image_index"]:
        ax_r.axvline(idx + 0.5, color="tab:blue", lw=1.5, zorder=1)
    for idx in win.loc[win["is_omission"], "image_index"]:
        ax_r.axvline(idx + 0.5, color="tab:cyan", ls="--", lw=1, zorder=1)
    for b0, b1 in segment_bouts(sess.lick_times):
        x0, x1 = b0 / IMAGE_DURATION, b1 / IMAGE_DURATION
        if x1 >= i0 and x0 <= i1:
            ax_r.axvspan(x0, x1, color="0.85", zorder=0)
    licks = sess.lick_times / IMAGE_DURATION
    licks = licks[(licks >= i0) & (licks <= i1)]
    ax_r.vlines(licks, 0.0, 1.0, color="k", lw=0.8, zorder=2)
    hits = win.loc[win["is_hit"], "image_index"].to_numpy()
    ax_r.plot(hits + 0.5, np.full(len(hits), 1.15), "v", color="tab:red", ms=6, zorder=3)
    ax_r.set_ylim(-0.1, 1.3)
    ax_r.set_yticks([])
    ax_r.set_ylabel("licks")

    # --- bottom: design matrix, one row per strategy in its paper color ---
    _draw_design_rows(ax_m, X, col_names, i0, i1, i0, i1)
    ax_m.set_xlabel("image")
    ax_r.set_xlim(i0, i1)
    ax_m.set_xlim(i0, i1)
    return ax_r, ax_m


def plot_weights(true_weights=None, fit_weights=None, col_names=None, time=None,
                ax=None):
    """Overlay true and/or fitted strategy-weight trajectories over a session."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.2))
    n = (true_weights if true_weights is not None else fit_weights).shape[0]
    x = time if time is not None else np.arange(n)
    for k, name in enumerate(col_names):
        color = WEIGHT_COLORS.get(name, "black")
        if true_weights is not None:
            ax.plot(x, true_weights[:, k], color=color, lw=2, alpha=0.5,
                    label=f"{name} (true)")
        if fit_weights is not None:
            ax.plot(x, fit_weights[:, k], color=color, lw=1.2, ls="--",
                    label=f"{name} (fit)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("image")
    ax.set_ylabel("weight")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    return ax