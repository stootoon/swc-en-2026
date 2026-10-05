"""Nonstationary noise analysis -- reference implementations of every step.

The chain is: sweeps -> mean and variance at each time point (across sweeps) ->
plot variance against mean -> fit the parabola

    var = i * mean - mean**2 / N

-> read off the unitary current ``i`` and the channel count ``N``, and from the
largest mean current the maximum open probability ``p_max = mean_max / (i N)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Single-channel statistics (Notebook 1)                                       #
# --------------------------------------------------------------------------- #
def dwell_times(open_state: np.ndarray, dt: float):
    """Durations (s) of every open and closed stretch in a 0/1 trajectory.

    Returns ``(open_durations, closed_durations)``. The first and last runs are
    dropped because we did not see where they started or ended.
    """
    x = np.asarray(open_state).astype(np.int8)
    change = np.flatnonzero(np.diff(x) != 0) + 1
    if len(change) < 2:
        return np.array([]), np.array([])
    starts, stops = change[:-1], change[1:]
    lengths = (stops - starts) * dt
    is_open = x[starts] == 1
    return lengths[is_open], lengths[~is_open]


# --------------------------------------------------------------------------- #
# Isochrone statistics (Notebook 2)                                            #
# --------------------------------------------------------------------------- #
def isochrone_stats(sweeps: np.ndarray):
    """Mean and variance across sweeps at every time point. Returns ``(mean, var)``."""
    sweeps = np.asarray(sweeps, float)
    return sweeps.mean(axis=0), sweeps.var(axis=0, ddof=1)


def successive_difference_variance(sweeps: np.ndarray) -> np.ndarray:
    """Variance from differences between consecutive sweeps -- immune to slow drift.

    If the current runs down from sweep to sweep, the plain variance across sweeps
    is inflated by that drift. Differences between *neighbouring* sweeps barely see
    it. With y_k = (x_k - x_{k+1}) / 2 the variance is 2/(n-1) * sum (y_k - ybar)^2.
    """
    y = np.diff(np.asarray(sweeps, float), axis=0) / 2.0
    n = sweeps.shape[0]
    return 2.0 / (n - 1) * ((y - y.mean(axis=0)) ** 2).sum(axis=0)


def baseline_variance(sweeps: np.ndarray, protocol, var_fn=isochrone_stats) -> float:
    """Variance of the current *before* the step -- the background noise floor."""
    n_pre = protocol.step_index
    if n_pre < 2:
        return 0.0
    if var_fn is isochrone_stats:
        _, v = isochrone_stats(sweeps[:, :n_pre])
    else:
        v = var_fn(sweeps[:, :n_pre])
    return float(np.mean(v))


# --------------------------------------------------------------------------- #
# The parabola fit (Notebook 3)                                                #
# --------------------------------------------------------------------------- #
@dataclass
class NoiseFit:
    """Result of fitting var = baseline + i*mean - mean^2/N."""

    i: float
    N: float
    baseline: float
    mean: np.ndarray
    var: np.ndarray

    @property
    def p_max(self) -> float:
        return float(self.mean.max() / (self.i * self.N))

    def curve(self, mean=None):
        """The fitted parabola evaluated at ``mean`` (default: 0 .. i*N)."""
        if mean is None:
            top = self.i * self.N if np.isfinite(self.N) and self.N > 0 else 1.2 * self.mean.max()
            mean = np.linspace(0, top, 200)
        return mean, self.baseline + self.i * mean - mean ** 2 / self.N

    def __repr__(self):
        return (f"NoiseFit(i={self.i:.3f} pA, N={self.N:.0f}, "
                f"p_max={self.p_max:.2f}, baseline={self.baseline:.3g} pA^2)")


def fit_parabola(mean: np.ndarray, var: np.ndarray, baseline: float = 0.0) -> NoiseFit:
    """Least-squares fit of var - baseline = i*mean - mean^2/N.

    The model is *linear* in the two unknowns (i, 1/N), so it is an ordinary
    linear regression with two regressors, mean and -mean^2, and no intercept.
    """
    mean = np.asarray(mean, float); var = np.asarray(var, float)
    A = np.column_stack([mean, -mean ** 2])
    coef, *_ = np.linalg.lstsq(A, var - baseline, rcond=None)
    i, inv_N = coef
    N = 1.0 / inv_N if inv_N > 0 else np.inf
    return NoiseFit(i=float(i), N=float(N), baseline=float(baseline), mean=mean, var=var)


def fit_slope_only(mean, var, baseline: float = 0.0, max_fraction: float = 0.3) -> float:
    """Unitary current from the initial slope alone (uses the lowest means only)."""
    mean = np.asarray(mean, float); var = np.asarray(var, float) - baseline
    keep = mean <= max_fraction * mean.max()
    return float(np.sum(mean[keep] * var[keep]) / np.sum(mean[keep] ** 2))


def open_probability(mean: np.ndarray, fit: NoiseFit) -> np.ndarray:
    """Open probability time course implied by a fit: p(t) = mean(t) / (i N)."""
    return np.asarray(mean) / (fit.i * fit.N)


# --------------------------------------------------------------------------- #
# Uncertainty (Notebook 4)                                                     #
# --------------------------------------------------------------------------- #
def bootstrap_fit(sweeps: np.ndarray, n_boot: int = 200, baseline: float = 0.0,
                  var_fn=None, rng=None):
    """Refit on ``n_boot`` resamples of the sweeps (with replacement).

    Returns ``(i_samples, N_samples)``. The spread of these is the uncertainty of
    the estimates given THIS many sweeps. If ``var_fn`` is
    :func:`successive_difference_variance`, *consecutive pairs* of sweeps are
    resampled (so the pairing that cancels drift is preserved).
    """
    rng = np.random.default_rng(rng)
    sweeps = np.asarray(sweeps, float)
    n = sweeps.shape[0]
    i_s, N_s = np.empty(n_boot), np.empty(n_boot)
    paired = var_fn is successive_difference_variance
    if paired:
        y_all = np.diff(sweeps, axis=0) / 2.0                 # (n-1, n_samples)
        m_all = (sweeps[:-1] + sweeps[1:]) / 2.0
    for b in range(n_boot):
        if paired:
            idx = rng.integers(0, n - 1, n - 1)
            y = y_all[idx]
            m = m_all[idx].mean(axis=0)
            v = 2.0 / (n - 1) * ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        else:
            sub = sweeps[rng.integers(0, n, n)]
            if var_fn is None:
                m, v = isochrone_stats(sub)
            else:
                m = sub.mean(axis=0); v = var_fn(sub)
        f = fit_parabola(m, v, baseline)
        i_s[b], N_s[b] = f.i, f.N
    return i_s, N_s


# --------------------------------------------------------------------------- #
# Stationary noise (Notebook 7)                                                #
# --------------------------------------------------------------------------- #
def power_spectrum(x: np.ndarray, fs: float, nperseg: int = 2048):
    """One-sided power spectral density of a (steady-state) record via Welch's method."""
    from scipy.signal import welch
    x = np.asarray(x, float)
    x = x - x.mean()
    f, S = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    if S.ndim > 1:
        S = S.mean(axis=0)
    return f, S


def lorentzian(f, S0, fc):
    """S(f) = S0 / (1 + (f/fc)^2): the spectrum of a two-state switch."""
    return S0 / (1.0 + (np.asarray(f) / fc) ** 2)


def fit_lorentzian(f, S, f_min: float = 0.0, f_max: float | None = None):
    """Fit S(f) = S0 / (1 + (f/fc)^2) to a spectrum. Returns ``(S0, fc)``.

    Fitted in log-power so every decade of frequency counts equally. The top of
    the band (default: above a fifth of the highest frequency) is left out, where
    sampling effects bend the spectrum away from a Lorentzian.
    """
    from scipy.optimize import curve_fit
    f = np.asarray(f, float); S = np.asarray(S, float)
    if f_max is None:
        f_max = 0.2 * f.max()
    keep = (f > f_min) & (f <= f_max) & (S > 0)
    ff, SS = f[keep], S[keep]
    S0_guess = np.median(SS[:4])
    half = np.flatnonzero(SS < S0_guess / 2)
    fc_guess = ff[half[0]] if len(half) else ff[len(ff) // 2]
    def model(x, logS0, logfc):
        return np.log(lorentzian(x, np.exp(logS0), np.exp(logfc)))
    popt, _ = curve_fit(model, ff, np.log(SS), p0=[np.log(S0_guess), np.log(fc_guess)], maxfev=20000)
    return float(np.exp(popt[0])), float(np.exp(popt[1]))


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of a record for lags 0..max_lag."""
    x = np.asarray(x, float); x = x - x.mean()
    n = len(x)
    ac = np.array([np.dot(x[:n - k], x[k:]) / (n - k) for k in range(max_lag + 1)])
    return ac / ac[0]
