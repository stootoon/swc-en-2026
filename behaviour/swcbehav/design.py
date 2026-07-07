"""Turning raw licks into model inputs: bouts, per-flash outcomes, design matrix.

These are *reference implementations* of the steps students write themselves in
notebook 1. Later notebooks import them so each notebook can run standalone
without depending on earlier ones having been executed.

The design matrix built here is column-for-column the same construction the
generative model uses (``generate.py``), so fitting it back recovers the
weights that produced the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generate import WEIGHT_NAMES, timing_feature, TIMING_MIDPOINT, TIMING_SLOPE

BOUT_ILI_THRESHOLD = 0.7  # seconds; licks closer than this belong to one bout


def segment_bouts(lick_times, ili_threshold: float = BOUT_ILI_THRESHOLD):
    """Group a sorted array of lick times into bouts.

    A new bout starts whenever the gap since the previous lick exceeds
    ``ili_threshold``. Returns an (n_bouts, 2) array of [start_time, end_time].
    """
    lick_times = np.asarray(lick_times)
    if lick_times.size == 0:
        return np.empty((0, 2))
    lick_times = np.sort(lick_times)
    gaps = np.diff(lick_times)
    breaks = np.flatnonzero(gaps > ili_threshold)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(lick_times) - 1]))
    return np.column_stack((lick_times[starts], lick_times[ends]))


def assign_bout_starts(session_table, bouts, flash_duration: float = 0.75):
    """Boolean per flash: did a licking bout *start* during this flash's window?

    Each bout's start time is mapped to the flash whose [onset, onset+flash)
    window contains it.
    """
    onset = session_table["time"].to_numpy()
    n = len(session_table)
    bout_start = np.zeros(n, dtype=bool)
    if len(bouts) == 0:
        return bout_start
    idx = np.floor((bouts[:, 0] - onset[0]) / flash_duration).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    bout_start[idx] = True
    return bout_start


def images_since_bout(bout_start, initial: float = TIMING_MIDPOINT):
    """Images elapsed since the last bout start, evaluated *before* each flash.

    Matches the counter used during generation: it resets to 0 on the flash
    where a bout starts and increments otherwise, and the value used on flash t
    is the count entering that flash.
    """
    bout_start = np.asarray(bout_start, dtype=bool)
    n = len(bout_start)
    since = np.zeros(n)
    c = float(initial)
    for t in range(n):
        since[t] = c
        c = 0.0 if bout_start[t] else c + 1.0
    return since


def build_design_matrix(session_table, bout_start=None,
                       midpoint: float = TIMING_MIDPOINT,
                       slope: float = TIMING_SLOPE):
    """Assemble the (n_images, 5) strategy design matrix.

    Columns: [bias, visual, omission, post_omission, timing].

    * bias         : constant 1 (overall drive to lick).
    * visual       : 1 on change flashes.
    * omission     : 1 on omitted flashes.
    * post_omission: 1 on the flash after an omission.
    * timing       : sigmoid of images-since-last-bout (needs ``bout_start``).

    If ``bout_start`` is None, the observed ``bout_start`` column of the table is
    used. Returns (X, column_names).
    """
    if bout_start is None:
        bout_start = session_table["bout_start"].to_numpy()
    since = images_since_bout(bout_start, initial=midpoint)
    X = np.column_stack([
        np.ones(len(session_table)),
        session_table["is_change"].to_numpy().astype(float),
        session_table["is_omission"].to_numpy().astype(float),
        session_table["is_post_omission"].to_numpy().astype(float),
        timing_feature(since, midpoint, slope),
    ])
    return X, list(WEIGHT_NAMES)