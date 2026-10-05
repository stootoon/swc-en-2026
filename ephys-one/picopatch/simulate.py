"""The forward model: from a kinetic scheme to a stack of recorded sweeps.

Noise analysis runs *backwards* -- from the current's fluctuations to the number of
channels and the current through each. To learn it we run the model *forwards* from
a truth we choose, so every estimate can be checked against the answer.

A recorded sweep is built as::

    I(t) = i * n_open(t)  +  background noise           (then optionally filtered,
                                                          quantised, run down)

where ``n_open(t)`` is how many of the ``N`` channels are open at time ``t``. The
nuisances (noise, rundown, low-pass filtering, digitisation) are all **off by
default** and switched on one at a time in the "imperfect recordings" notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinetics import Scheme, get_scheme
from .protocol import FS, Protocol, step_protocol


# --------------------------------------------------------------------------- #
# Containers                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Truth:
    """Everything the analysis is *not* allowed to see, kept for scoring."""

    N: int                       # number of channels in the patch
    i: float                     # unitary current (pA) at the test potential
    scheme: str                  # name of the kinetic scheme
    p_open: np.ndarray           # (n_samples,) exact open probability vs time
    open_counts: np.ndarray      # (n_sweeps, n_samples) channels open, per sweep
    n_active: np.ndarray         # (n_sweeps,) channels present in each sweep (rundown)
    noise_sd: float = 0.0
    filter_fc: float | None = None
    adc_step: float | None = None

    @property
    def p_max(self) -> float:
        return float(self.p_open.max())


@dataclass
class Experiment:
    """A stack of sweeps from one voltage protocol. ``sweeps`` is (n_sweeps, n_samples), pA."""

    sweeps: np.ndarray
    protocol: Protocol
    truth: Truth | None = None
    label: str = ""

    @property
    def fs(self) -> float:
        return self.protocol.fs

    @property
    def t(self) -> np.ndarray:
        return self.protocol.t

    @property
    def t_ms(self) -> np.ndarray:
        return self.protocol.t_ms

    @property
    def V(self) -> np.ndarray:
        return self.protocol.V

    @property
    def n_sweeps(self) -> int:
        return self.sweeps.shape[0]

    @property
    def n_samples(self) -> int:
        return self.sweeps.shape[1]


# --------------------------------------------------------------------------- #
# Single-channel simulation                                                    #
# --------------------------------------------------------------------------- #
def simulate_two_state(alpha: float, beta: float, dt: float, n_steps: int,
                       rng=None, start_open: bool = False) -> np.ndarray:
    """The coin-flip simulation of one two-state channel (Notebook 1, Exercise 1).

    At every time step a closed channel opens with probability ``alpha*dt`` and an
    open channel closes with probability ``beta*dt``. Returns a 0/1 array (1 = open).
    Requires ``alpha*dt`` and ``beta*dt`` well below 1.
    """
    rng = np.random.default_rng(rng)
    state = np.empty(n_steps, dtype=np.int8)
    s = 1 if start_open else 0
    u = rng.random(n_steps)
    for k in range(n_steps):
        if s == 0 and u[k] < alpha * dt:
            s = 1
        elif s == 1 and u[k] < beta * dt:
            s = 0
        state[k] = s
    return state


def simulate_single(scheme, protocol: Protocol, n_sweeps: int = 1, rng=None) -> np.ndarray:
    """Exact state trajectories of ONE channel under a voltage protocol.

    Returns ``states`` of shape (n_sweeps, n_samples) with integer state indices.
    Each sweep starts from the equilibrium at the initial voltage. Vectorised over
    sweeps, so a few hundred repeats are cheap.
    """
    scheme = get_scheme(scheme)
    rng = np.random.default_rng(rng)
    n = protocol.n_samples
    states = np.empty((n_sweeps, n), dtype=np.int8)
    pi0 = scheme.stationary(protocol.V[0])
    s = rng.choice(scheme.n_states, size=n_sweeps, p=pi0)
    for a, b, V in protocol.segments:
        P = scheme.transition_matrix(V, protocol.dt)
        cdf = np.cumsum(P, axis=1)
        for k in range(a, b):
            states[:, k] = s
            u = rng.random(n_sweeps)
            s = (u[:, None] > cdf[s]).sum(axis=1)       # inverse-CDF draw per sweep
            s = np.minimum(s, scheme.n_states - 1)
    return states


def current_from_states(scheme, states: np.ndarray, i: float = 1.0) -> np.ndarray:
    """Convert state indices to current: ``i`` whenever the state conducts, else 0."""
    scheme = get_scheme(scheme)
    return i * scheme.conducting[states].astype(float)


# --------------------------------------------------------------------------- #
# Many channels: simulate the *counts* in each state                           #
# --------------------------------------------------------------------------- #
def simulate_counts(scheme, protocol: Protocol, n_channels, n_sweeps: int = 1,
                    rng=None) -> np.ndarray:
    """How many of the channels are OPEN at each time, for each sweep.

    Rather than tracking 1000 channels one by one, we track how many sit in each
    state and move them between states with multinomial draws -- the same
    statistics, a thousand times faster. ``n_channels`` may be an int or an
    (n_sweeps,) array (different channel counts per sweep, for rundown).
    Returns ``open_counts`` of shape (n_sweeps, n_samples).
    """
    scheme = get_scheme(scheme)
    rng = np.random.default_rng(rng)
    n_channels = np.broadcast_to(np.asarray(n_channels, dtype=np.int64), (n_sweeps,))
    S = scheme.n_states
    pi0 = scheme.stationary(protocol.V[0])
    counts = rng.multinomial(n_channels, pi0)                 # (n_sweeps, S)
    open_counts = np.empty((n_sweeps, protocol.n_samples), dtype=np.int32)
    cond = scheme.conducting
    for a, b, V in protocol.segments:
        P = scheme.transition_matrix(V, protocol.dt)
        P = np.clip(P, 0, None); P /= P.sum(axis=1, keepdims=True)
        for k in range(a, b):
            open_counts[:, k] = counts[:, cond].sum(axis=1)
            new = np.zeros_like(counts)
            for s in range(S):
                new += rng.multinomial(counts[:, s], P[s])
            counts = new
    return open_counts


def exact_open_probability(scheme, protocol: Protocol) -> np.ndarray:
    """The deterministic open probability p(t) implied by the scheme + protocol."""
    scheme = get_scheme(scheme)
    p = np.empty(protocol.n_samples)
    pi = scheme.stationary(protocol.V[0])
    for a, b, V in protocol.segments:
        P = scheme.transition_matrix(V, protocol.dt)
        for k in range(a, b):
            p[k] = pi[scheme.conducting].sum()
            pi = pi @ P
    return p


# --------------------------------------------------------------------------- #
# Recording nuisances                                                          #
# --------------------------------------------------------------------------- #
def lowpass_bessel(x: np.ndarray, fs: float, fc: float, order: int = 8) -> np.ndarray:
    """Causal 8-pole Bessel low-pass filter along the last axis (like the amplifier's)."""
    from scipy.signal import bessel, sosfilt
    sos = bessel(order, fc, btype="low", fs=fs, output="sos", norm="mag")
    return sosfilt(sos, x, axis=-1)


def quantize(x: np.ndarray, step: float) -> np.ndarray:
    """Analog-to-digital conversion: round every sample to a multiple of ``step`` pA."""
    return np.round(x / step) * step


# --------------------------------------------------------------------------- #
# The generator                                                                #
# --------------------------------------------------------------------------- #
def make_experiment(N: int = 1000, i: float = 1.0, scheme="two_state",
                    protocol: Protocol | None = None, n_sweeps: int = 200,
                    noise_sd: float = 0.0, rundown: float = 0.0,
                    filter_fc: float | None = None, adc_step: float | None = None,
                    seed: int | None = 0, label: str = "") -> Experiment:
    """Simulate ``n_sweeps`` repeats of a voltage protocol on a patch of ``N`` channels.

    Parameters
    ----------
    N, i        : number of channels and unitary current (pA) -- the ground truth.
    scheme      : "two_state" (default), "inactivating", "flickering", or a Scheme.
    protocol    : voltage protocol; default is a step from -80 to +65 mV.
    n_sweeps    : how many repeats.
    noise_sd    : background (amplifier + seal) noise, white, in pA.      [NB4]
    rundown     : fraction of channels lost by the last sweep (0 = none). [NB4]
    filter_fc   : -3 dB cutoff (Hz) of an 8-pole Bessel low-pass; None = off. [NB6]
    adc_step    : digitiser resolution in pA; None = off.                 [NB4]
    """
    rng = np.random.default_rng(seed)
    scheme = get_scheme(scheme)
    protocol = protocol or step_protocol()
    frac = 1.0 - rundown * np.arange(n_sweeps) / max(n_sweeps - 1, 1)
    n_active = np.maximum(np.round(N * frac), 1).astype(np.int64)
    open_counts = simulate_counts(scheme, protocol, n_active, n_sweeps, rng)
    sweeps = i * open_counts.astype(float)
    if noise_sd > 0:
        sweeps = sweeps + rng.normal(0.0, noise_sd, sweeps.shape)
    if filter_fc is not None:
        sweeps = lowpass_bessel(sweeps, protocol.fs, filter_fc)
    if adc_step is not None:
        sweeps = quantize(sweeps, adc_step)
    truth = Truth(N=int(N), i=float(i), scheme=scheme.name,
                  p_open=exact_open_probability(scheme, protocol),
                  open_counts=open_counts, n_active=n_active, noise_sd=noise_sd,
                  filter_fc=filter_fc, adc_step=adc_step)
    return Experiment(sweeps=sweeps, protocol=protocol, truth=truth, label=label)
