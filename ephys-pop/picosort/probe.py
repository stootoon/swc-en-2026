"""Probe geometry -- where the recording sites sit in space.

A real Neuropixels probe has ~384 sites in a dense two-column staggered layout.
For teaching we use a simpler single-column linear probe: ``n_channels`` sites
stacked vertically at a fixed ``pitch`` (micrometres apart). The one fact that
matters for spike sorting is that channels have *positions*, so a spike near one
site also shows up -- more weakly -- on its neighbours. Localising a unit along
this depth axis is what makes the "which channel is this spike on?" question well
posed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Neuropixels AP-band sampling rate. Every sample index in the module is at this
# rate, so ``t_seconds = sample / FS``.
FS = 30_000.0


@dataclass
class Probe:
    """Channel positions in micrometres. ``x`` and ``y`` are 1-D, one per channel."""

    x: np.ndarray
    y: np.ndarray

    @property
    def n_channels(self) -> int:
        return len(self.y)

    def positions(self) -> np.ndarray:
        """(n_channels, 2) array of [x, y] positions, in micrometres."""
        return np.column_stack([self.x, self.y])

    def distances_to(self, xy) -> np.ndarray:
        """Euclidean distance (micrometres) from every channel to a point ``xy``."""
        xy = np.asarray(xy, float)
        return np.hypot(self.x - xy[0], self.y - xy[1])


def make_probe(n_channels: int = 32, pitch: float = 20.0, x: float = 0.0) -> Probe:
    """A single-column linear probe: ``n_channels`` sites ``pitch`` micrometres apart.

    Channel 0 sits at the top (``y = 0``) and depth increases down the probe.
    """
    y = np.arange(n_channels, dtype=float) * pitch
    x = np.full(n_channels, float(x))
    return Probe(x=x, y=y)
