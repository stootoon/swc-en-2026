"""Channel gating as a Markov chain -- the ground truth of the whole module.

An ion channel is a protein that is either **open** (passing a unitary current
``i``) or **closed** (passing nothing). It flips between those states at random,
with a fixed *probability per unit time* -- a **rate constant** -- for each
transition. A **kinetic scheme** is the list of states and the rates between them::

                alpha
        C  <----------->  O            (two-state channel)
                beta

Everything in this module follows from that picture: a single channel is a
random switch; a patch of ``N`` channels is ``N`` independent switches; the
macroscopic current is their sum; and the *fluctuations* of that sum carry the
information we want (how many switches, how much current per switch).

A scheme is stored as a **rate matrix** ``Q`` (units 1/s): ``Q[a, b]`` is the rate
of the ``a -> b`` transition and each diagonal entry is minus its row sum. Rates may
depend on membrane voltage, so a :class:`Scheme` holds a function ``Q(V)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import expm


# --------------------------------------------------------------------------- #
# The scheme container                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Scheme:
    """A kinetic scheme: named states, which ones conduct, and ``Q(V)``."""

    name: str
    states: tuple
    conducting: np.ndarray          # bool per state: does it pass current?
    rates: Callable[[float], np.ndarray]   # V (mV) -> Q matrix (1/s)

    @property
    def n_states(self) -> int:
        return len(self.states)

    def Q(self, V: float) -> np.ndarray:
        return self.rates(float(V))

    def transition_matrix(self, V: float, dt: float) -> np.ndarray:
        """P[a, b] = probability of being in b after ``dt`` seconds, starting in a.

        Exact for a continuous-time Markov chain: P = exp(Q dt). This means the
        simulation is *sampled* exactly at the clock ticks however fast the rates
        are -- flickers faster than ``dt`` are seen as whatever state the channel
        happens to be in at each tick (as in a real digitised recording).
        """
        return expm(self.Q(V) * dt)

    def stationary(self, V: float) -> np.ndarray:
        """Equilibrium occupancy of each state at voltage ``V`` (solves pi Q = 0)."""
        Q = self.Q(V)
        n = self.n_states
        A = np.vstack([Q.T, np.ones(n)])
        b = np.zeros(n + 1); b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.clip(pi, 0, None) / np.clip(pi, 0, None).sum()

    def open_probability(self, V: float) -> float:
        """Steady-state probability of being in a conducting state."""
        return float(self.stationary(V)[self.conducting].sum())


def _Q_from_rates(n: int, pairs: Sequence[tuple]) -> np.ndarray:
    """Build Q from a list of (from, to, rate) triples (rates in 1/s)."""
    Q = np.zeros((n, n))
    for a, b, k in pairs:
        Q[a, b] += k
    Q[np.diag_indices(n)] = -Q.sum(axis=1)
    return Q


# --------------------------------------------------------------------------- #
# The two-state channel (the workhorse)                                        #
# --------------------------------------------------------------------------- #
# Voltage dependence after Alvarez, Gonzalez & Latorre (2002):
#     alpha(V) = alpha0 * exp(+2 (V - V_half) / 25)      opening rate, speeds up with depolarisation
#     beta(V)  = alpha0 * exp(-2 (V - V_half) / 25)      closing rate, slows down
# so the open probability alpha/(alpha+beta) is a Boltzmann curve centred on V_half,
# and the relaxation time constant 1/(alpha+beta) is longest at V_half.
ALPHA0 = 150.0        # 1/s  -> tau = 1/(alpha+beta) = 3.3 ms at V_half, 1.3 ms at +65 mV
V_HALF = 45.0         # mV
V_SLOPE = 25.0        # mV


def alpha_rate(V, alpha0: float = ALPHA0, v_half: float = V_HALF, slope: float = V_SLOPE):
    """Opening rate (1/s) as a function of voltage (mV)."""
    return alpha0 * np.exp(2.0 * (np.asarray(V, float) - v_half) / slope)


def beta_rate(V, alpha0: float = ALPHA0, v_half: float = V_HALF, slope: float = V_SLOPE):
    """Closing rate (1/s) as a function of voltage (mV)."""
    return alpha0 * np.exp(-2.0 * (np.asarray(V, float) - v_half) / slope)


def two_state(alpha0: float = ALPHA0, v_half: float = V_HALF, slope: float = V_SLOPE) -> Scheme:
    """C <-> O with voltage-dependent rates. States: 0 = closed, 1 = open."""
    def rates(V):
        a = float(alpha_rate(V, alpha0, v_half, slope))
        b = float(beta_rate(V, alpha0, v_half, slope))
        return _Q_from_rates(2, [(0, 1, a), (1, 0, b)])
    return Scheme("two-state  C <-> O", ("C", "O"), np.array([False, True]), rates)


def two_state_fixed(alpha: float, beta: float) -> Scheme:
    """C <-> O with voltage-*independent* rates (1/s) -- for toy examples."""
    def rates(V):
        return _Q_from_rates(2, [(0, 1, alpha), (1, 0, beta)])
    return Scheme(f"two-state  alpha={alpha:g}/s  beta={beta:g}/s", ("C", "O"),
                  np.array([False, True]), rates)


# --------------------------------------------------------------------------- #
# Schemes used in the "failure mode" notebooks                                 #
# --------------------------------------------------------------------------- #
def inactivating(alpha: float = 1000.0, beta: float = 100.0,
                 gamma: float = 2000.0, delta: float = 200.0,
                 v_threshold: float = 0.0) -> Scheme:
    """C <-> O <-> I: channels open on depolarisation, then inactivate.

    Rates (1/s) follow Fig. 10 of the paper. Below ``v_threshold`` (i.e. at the
    holding potential) opening is switched off and everything funnels back to C,
    so each sweep starts with all channels closed and available.
    States: 0 = closed, 1 = open, 2 = inactivated (non-conducting).
    """
    def rates(V):
        if V >= v_threshold:
            pairs = [(0, 1, alpha), (1, 0, beta), (1, 2, gamma), (2, 1, delta)]
        else:   # at rest: no opening, fast closing, recovery from inactivation
            pairs = [(1, 0, 20 * beta), (2, 1, 10 * delta)]
        return _Q_from_rates(3, pairs)
    return Scheme("inactivating  C <-> O <-> I", ("C", "O", "I"),
                  np.array([False, True, False]), rates)


def flickering(alpha: float = 1000.0, flicker: float = 1.0e5,
               v_threshold: float = 0.0) -> Scheme:
    """C1 -> C2 <-> O: after a ~1 ms latency the channel flickers very fast.

    Follows Fig. 12 of the paper: C2 <-> O rates are 100 /ms (``flicker``), equal
    both ways, so the open probability within a burst is 0.5. States:
    0 = C1 (resting), 1 = C2 (closed, in burst), 2 = O.
    """
    def rates(V):
        if V >= v_threshold:
            pairs = [(0, 1, alpha), (1, 2, flicker), (2, 1, flicker)]
        else:
            pairs = [(1, 0, 20 * alpha), (2, 1, flicker)]
        return _Q_from_rates(3, pairs)
    return Scheme("flickering  C1 -> C2 <-> O", ("C1", "C2", "O"),
                  np.array([False, False, True]), rates)


SCHEMES = {"two_state": two_state, "inactivating": inactivating, "flickering": flickering}


def get_scheme(scheme) -> Scheme:
    """Accept a Scheme, or one of the names in ``SCHEMES``."""
    if isinstance(scheme, Scheme):
        return scheme
    if scheme in SCHEMES:
        return SCHEMES[scheme]()
    raise ValueError(f"unknown scheme {scheme!r}; choose from {list(SCHEMES)}")
