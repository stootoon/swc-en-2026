"""The generative model of licking -- the ground truth for the whole course.

A "mouse" is a rule for turning the stimulus stream into licking. We use exactly
the model Piet et al. (2024) *fit*, run in the forward direction:

    p(lick bout starts on image t) = sigmoid( w(t) . x(t) )

where ``x(t)`` is the five-strategy design vector for image t

    [ bias, visual, omission, post_omission, timing ]

and ``w(t)`` are the strategy weights. For a *static* mouse ``w(t)`` is constant;
for a *dynamic* mouse it drifts across the session. Because we choose ``w(t))``
ourselves, every downstream analysis can be checked against known truth.

The one subtlety: the ``timing`` regressor is a sigmoid of "images since the last
licking bout", which depends on the licking the model is *generating*. So the
simulation is inherently sequential -- at each image we build the design vector
from the history so far, sample a bout, and update the counter. Reconstructing
the design matrix from the finished session (see ``design.py``) reproduces this
timing column exactly, which is why parameter recovery works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import task as task_module

WEIGHT_NAMES = ["bias", "visual", "omission", "post_omission", "timing"]

# Timing-strategy sigmoid: licking is suppressed right after a bout, crosses 0.5
# at ``TIMING_MIDPOINT`` images, and saturates high after that.
TIMING_MIDPOINT = 4.0
TIMING_SLOPE = 1.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def timing_feature(images_since_bout, midpoint=TIMING_MIDPOINT, slope=TIMING_SLOPE):
    """Value of the timing regressor given images elapsed since the last bout."""
    return sigmoid((images_since_bout - midpoint) / slope)


# --------------------------------------------------------------------------- #
# Weight trajectories                                                         #
# --------------------------------------------------------------------------- #
def constant_weights(bias, visual, omission=0.0, post_omission=0.0, timing=0.0):
    """A static mouse: weights are the same on every image."""
    w = np.array([bias, visual, omission, post_omission, timing], dtype=float)
    return lambda t: w


def crossover_weights(n_images, rng, bias=-3.0, start=(4.0, -1.0), end=(-1.0, 4.0),
                      rw_sigma=0.03):
    """A dynamic mouse whose (visual, timing) weights drift across the session.

    Visual weight ramps from ``start[0]`` to ``end[0]`` and timing from
    ``start[1]`` to ``end[1]`` (a smooth linear crossover), plus a small random
    walk so the trajectory is noisy rather than perfectly smooth. This gives a
    mouse whose dominant strategy flips partway through -- the case the dynamic
    model exists to capture.
    """
    frac = np.linspace(0.0, 1.0, n_images)
    visual = start[0] + frac * (end[0] - start[0])
    timing = start[1] + frac * (end[1] - start[1])
    # small correlated drift on top
    visual = visual + np.cumsum(rng.normal(0, rw_sigma, n_images))
    timing = timing + np.cumsum(rng.normal(0, rw_sigma, n_images))
    W = np.zeros((n_images, 5))
    W[:, 0] = bias
    W[:, 1] = visual
    W[:, 4] = timing
    return (lambda t: W[t]), W


# --------------------------------------------------------------------------- #
# Engagement latent (optional; used by the final notebook)                    #
# --------------------------------------------------------------------------- #
def sample_engagement(n_images, rng, engaged_mean=320, disengaged_mean=120,
                     start_engaged=True):
    """A slow two-state (engaged/disengaged) telegraph process.

    This is generated *independently* of the strategy weights -- that is the
    whole point of the engagement notebook: strategy and engagement are separate
    axes. Mean run lengths give ~72% time engaged, matching the paper.
    """
    state = bool(start_engaged)
    out = np.zeros(n_images, dtype=bool)
    p_leave_engaged = 1.0 / engaged_mean
    p_leave_disengaged = 1.0 / disengaged_mean
    for t in range(n_images):
        out[t] = state
        if state and rng.random() < p_leave_engaged:
            state = False
        elif (not state) and rng.random() < p_leave_disengaged:
            state = True
    return out


@dataclass
class Session:
    """Everything about one simulated session.

    ``table`` holds one row per image. Observable columns (what a student's
    analysis is allowed to use) are the stimulus columns plus ``bout_start``.
    Hidden ground-truth columns (prefixed ``true_``) are included for checking
    recovery: ``p_lick``, ``true_engaged``, and the ``true_w_*`` weight columns.
    """

    table: pd.DataFrame
    lick_times: np.ndarray
    true_weights: np.ndarray  # (n_images, 5)
    weight_names: list = field(default_factory=lambda: list(WEIGHT_NAMES))
    params: dict = field(default_factory=dict)


def simulate(
    session_table: pd.DataFrame,
    weight_fn,
    engaged=None,
    disengaged_gain: float = 0.05,
    midpoint: float = TIMING_MIDPOINT,
    slope: float = TIMING_SLOPE,
    response_latency: float = 0.4,
    seed: int | None = 1,
) -> Session:
    """Run the generative model forward over a stimulus stream.

    Parameters
    ----------
    session_table : output of ``task.make_session``.
    weight_fn : callable ``t -> length-5 weight vector``.
    engaged : optional boolean array (one per image). Where False, the lick
        probability is scaled by ``disengaged_gain``.
    disengaged_gain : multiplicative suppression of licking when disengaged.
    response_latency : mean delay (s) from image onset to the first lick of a bout.
    seed : RNG seed.
    """
    rng = np.random.default_rng(seed)
    change = session_table["is_change"].to_numpy()
    omission = session_table["is_omission"].to_numpy()
    post_omission = session_table["is_post_omission"].to_numpy()
    onset = session_table["time"].to_numpy()
    n = len(session_table)

    bout_start = np.zeros(n, dtype=bool)
    p_lick = np.zeros(n)
    W = np.zeros((n, 5))
    since_bout = midpoint  # neutral start: no bout yet

    for t in range(n):
        tf = timing_feature(since_bout, midpoint, slope)
        x = np.array([1.0, float(change[t]), float(omission[t]),
                      float(post_omission[t]), tf])
        w = np.asarray(weight_fn(t), dtype=float)
        W[t] = w
        p = float(sigmoid(w @ x))
        if engaged is not None and not engaged[t]:
            p *= disengaged_gain
        p_lick[t] = p
        if rng.random() < p:
            bout_start[t] = True
            since_bout = 0.0
        else:
            since_bout += 1.0

    is_hit = bout_start & change
    is_miss = change & ~bout_start
    reward = is_hit  # open-loop: a bout on a change image is rewarded

    lick_times = _expand_bouts_to_licks(onset, bout_start, reward, response_latency, rng)

    table = session_table.copy()
    table["bout_start"] = bout_start
    table["is_hit"] = is_hit
    table["is_miss"] = is_miss
    table["reward"] = reward
    table["true_p_lick"] = p_lick
    if engaged is not None:
        table["true_engaged"] = engaged
    for k, name in enumerate(WEIGHT_NAMES):
        table[f"true_w_{name}"] = W[:, k]

    return Session(
        table=table,
        lick_times=lick_times,
        true_weights=W,
        params=dict(midpoint=midpoint, slope=slope, seed=seed,
                    disengaged_gain=disengaged_gain),
    )


def _expand_bouts_to_licks(onset, bout_start, reward, response_latency, rng,
                          ili=0.11):
    """Turn each per-image bout into a train of individual lick timestamps.

    Rewarded bouts last longer (the mouse consumes water); unrewarded bouts are
    brief. Inter-lick intervals stay well under the 700 ms bout threshold so the
    segmentation exercise in notebook 1 can recover these same bouts.
    """
    licks = []
    for t in np.flatnonzero(bout_start):
        start = onset[t] + response_latency + rng.normal(0, 0.03)
        duration = rng.uniform(1.0, 2.5) if reward[t] else rng.uniform(0.15, 0.8)
        n_licks = max(1, int(duration / ili))
        for k in range(n_licks):
            licks.append(start + k * ili + rng.normal(0, 0.01))
    return np.sort(np.asarray(licks))


# --------------------------------------------------------------------------- #
# Convenience: named archetypes                                               #
# --------------------------------------------------------------------------- #
def make_mouse(archetype: str, n_images: int = 4800, seed: int = 0,
              engaged: bool = False) -> Session:
    """Build a Session for a named mouse archetype.

    archetypes
    ----------
    "visual"  : licks almost only on image changes.
    "timing"  : licks rhythmically, tracking time since its last bout.
    "mixed"   : uses both strategies.
    "dynamic" : strategy drifts from visual-dominant to timing-dominant.
    """
    task_seed, mouse_seed = seed, seed + 10_000
    stim = task_module.make_session(n_images=n_images, seed=task_seed)
    rng = np.random.default_rng(mouse_seed)

    if archetype == "visual":
        weight_fn = constant_weights(bias=-3.0, visual=5.0, timing=0.0)
    elif archetype == "timing":
        weight_fn = constant_weights(bias=-3.0, visual=0.0, timing=5.0)
    elif archetype == "mixed":
        weight_fn = constant_weights(bias=-3.0, visual=3.0, timing=3.0)
    elif archetype == "dynamic":
        weight_fn, _ = crossover_weights(len(stim), rng)
    else:
        raise ValueError(f"unknown archetype {archetype!r}")

    engaged_arr = sample_engagement(len(stim), rng) if engaged else None
    return simulate(stim, weight_fn, engaged=engaged_arr, seed=mouse_seed + 1)