"""Voltage protocols: what the amplifier clamps the membrane to, sample by sample.

In **voltage clamp** the experimenter dictates the membrane potential and records
the current needed to hold it there. The standard experiment is a **step**: rest at
a negative holding potential (channels closed), jump to a depolarised test
potential (channels open), record the current, repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 50_000.0   # Hz -- sampling rate; dt = 20 us


@dataclass
class Protocol:
    """Sampled command voltage. ``V`` is in mV, ``t`` in seconds."""

    t: np.ndarray
    V: np.ndarray
    fs: float = FS

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def n_samples(self) -> int:
        return len(self.t)

    @property
    def t_ms(self) -> np.ndarray:
        return self.t * 1e3

    @property
    def segments(self):
        """Runs of constant voltage as (start, stop, V) -- rates only change there."""
        change = np.flatnonzero(np.diff(self.V) != 0) + 1
        starts = np.concatenate([[0], change])
        stops = np.concatenate([change, [len(self.V)]])
        return [(int(a), int(b), float(self.V[a])) for a, b in zip(starts, stops)]

    @property
    def step_index(self) -> int:
        """Sample at which the first voltage change happens (start of the test step)."""
        change = np.flatnonzero(np.diff(self.V) != 0)
        return int(change[0] + 1) if len(change) else 0


def step_protocol(v_hold: float = -80.0, v_step: float = 65.0, t_pre_ms: float = 2.0,
                  t_step_ms: float = 12.0, t_post_ms: float = 0.0, fs: float = FS) -> Protocol:
    """Hold at ``v_hold``, step to ``v_step`` for ``t_step_ms``, optionally return."""
    n_pre = int(round(t_pre_ms * 1e-3 * fs))
    n_step = int(round(t_step_ms * 1e-3 * fs))
    n_post = int(round(t_post_ms * 1e-3 * fs))
    V = np.concatenate([np.full(n_pre, v_hold), np.full(n_step, v_step), np.full(n_post, v_hold)])
    t = np.arange(len(V)) / fs
    return Protocol(t=t, V=V, fs=fs)


def constant_protocol(v: float = 45.0, duration_ms: float = 200.0, fs: float = FS) -> Protocol:
    """Sit at one voltage -- for steady-state (stationary) recordings."""
    n = int(round(duration_ms * 1e-3 * fs))
    return Protocol(t=np.arange(n) / fs, V=np.full(n, float(v)), fs=fs)
