"""The whole analysis in one call, plus the "mystery patch" used for scoring.

``count_channels`` is the reference pipeline (and the escape hatch for Notebook 8):
sweeps -> isochrone mean/variance -> baseline subtraction -> parabola fit ->
bootstrap error bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import (NoiseFit, baseline_variance, bootstrap_fit, fit_parabola,
                       isochrone_stats, successive_difference_variance)
from .protocol import step_protocol
from .simulate import Experiment, make_experiment


@dataclass
class CountResult:
    fit: NoiseFit
    i_boot: np.ndarray
    N_boot: np.ndarray

    @property
    def i_ci(self):
        return tuple(np.percentile(self.i_boot, [2.5, 97.5]))

    @property
    def N_ci(self):
        return tuple(np.percentile(self.N_boot, [2.5, 97.5]))

    def __repr__(self):
        lo, hi = self.i_ci; Nlo, Nhi = self.N_ci
        return (f"i = {self.fit.i:.2f} pA  [{lo:.2f}, {hi:.2f}]   "
                f"N = {self.fit.N:.0f}  [{Nlo:.0f}, {Nhi:.0f}]   "
                f"p_max = {self.fit.p_max:.2f}")


def count_channels(exp: Experiment, subtract_baseline: bool = True,
                   rundown_correction: bool = False, n_boot: int = 200,
                   seed: int = 0) -> CountResult:
    """Run the full nonstationary noise analysis on an experiment."""
    var_fn = successive_difference_variance if rundown_correction else None
    mean = exp.sweeps.mean(axis=0)
    var = var_fn(exp.sweeps) if var_fn else isochrone_stats(exp.sweeps)[1]
    base = baseline_variance(exp.sweeps, exp.protocol, var_fn or isochrone_stats) \
        if subtract_baseline else 0.0
    fit = fit_parabola(mean, var, base)
    i_b, N_b = bootstrap_fit(exp.sweeps, n_boot=n_boot, baseline=base, var_fn=var_fn, rng=seed)
    return CountResult(fit=fit, i_boot=i_b, N_boot=N_b)


# --------------------------------------------------------------------------- #
# Mystery patches for Notebook 8                                               #
# --------------------------------------------------------------------------- #
_MYSTERIES = {
    "A": dict(N=400, i=1.6, noise_sd=1.5, n_sweeps=300,
              protocol=step_protocol(v_step=75.0), label="patch A"),
    "B": dict(N=2500, i=0.6, noise_sd=2.0, rundown=0.25, n_sweeps=400,
              protocol=step_protocol(v_step=70.0), label="patch B"),
    "C": dict(N=150, i=2.5, noise_sd=1.0, n_sweeps=300,
              protocol=step_protocol(v_step=40.0), label="patch C"),
}


def make_mystery(which: str = "A", seed: int = 2026) -> Experiment:
    """A patch whose N, i and open probability are hidden (``truth`` is None)."""
    kw = dict(_MYSTERIES[which]); kw["seed"] = seed + ord(which)
    exp = make_experiment(**kw)
    return Experiment(sweeps=exp.sweeps, protocol=exp.protocol, truth=None, label=kw["label"])


def reveal(which: str = "A", seed: int = 2026):
    """The hidden truth for ``make_mystery(which)`` -- look only after you've fitted!"""
    kw = dict(_MYSTERIES[which]); kw["seed"] = seed + ord(which)
    return make_experiment(**kw).truth
