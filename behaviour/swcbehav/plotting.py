"""Shared plotting helpers for the behaviour notebooks.

Plotting is not the learning objective, so it lives here rather than in the
notebooks. These functions reproduce the *flavor* of the paper's panels
(qualitative, not pixel-perfect).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .task import FLASH_DURATION


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
    """Show the design-matrix columns over a slice of flashes (Fig 1C flavor)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.0))
    sl = slice(start, start + n)
    im = ax.imshow(X[sl].T, aspect="auto", cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(col_names)))
    ax.set_yticklabels(col_names)
    ax.set_xlabel("image flash (relative)")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="regressor value")
    return ax


def plot_weights(true_weights=None, fit_weights=None, col_names=None, time=None,
                ax=None):
    """Overlay true and/or fitted strategy-weight trajectories over a session."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.2))
    n = (true_weights if true_weights is not None else fit_weights).shape[0]
    x = time if time is not None else np.arange(n)
    colors = plt.cm.tab10(np.arange(len(col_names)))
    for k, name in enumerate(col_names):
        if true_weights is not None:
            ax.plot(x, true_weights[:, k], color=colors[k], lw=2, alpha=0.5,
                    label=f"{name} (true)")
        if fit_weights is not None:
            ax.plot(x, fit_weights[:, k], color=colors[k], lw=1.2, ls="--",
                    label=f"{name} (fit)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("image flash")
    ax.set_ylabel("weight")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    return ax