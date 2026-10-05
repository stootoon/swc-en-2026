"""picopatch -- backend for the SWC ENC 2026 ephys-one (counting channels) module.

Students estimate how many ion channels sit in a patch of membrane, and how much
current flows through each, from the *fluctuations* of the macroscopic current
(nonstationary noise analysis, after Alvarez, Gonzalez & Latorre 2002). This package
holds the simulator (the ground truth) and reference implementations of every
analysis step, so each notebook runs standalone and a stuck exercise has a
``pp.<name>`` fallback.

Imported in the notebooks as::

    import picopatch as pp
"""

from .kinetics import (Scheme, two_state, two_state_fixed, inactivating, flickering,
                       get_scheme, alpha_rate, beta_rate, ALPHA0, V_HALF, V_SLOPE)
from .protocol import FS, Protocol, step_protocol, constant_protocol
from .simulate import (Truth, Experiment, simulate_two_state, simulate_single,
                       current_from_states, simulate_counts, exact_open_probability,
                       lowpass_bessel, quantize, make_experiment)
from .analysis import (dwell_times, isochrone_stats, successive_difference_variance,
                       baseline_variance, NoiseFit, fit_parabola, fit_slope_only,
                       open_probability, bootstrap_fit, power_spectrum, lorentzian,
                       fit_lorentzian, autocorrelation)
from .pipeline import count_channels, CountResult, make_mystery, reveal
from . import plotting

__all__ = [
    # kinetics
    "Scheme", "two_state", "two_state_fixed", "inactivating", "flickering", "get_scheme",
    "alpha_rate", "beta_rate", "ALPHA0", "V_HALF", "V_SLOPE",
    # protocols
    "FS", "Protocol", "step_protocol", "constant_protocol",
    # simulation
    "Truth", "Experiment", "simulate_two_state", "simulate_single", "current_from_states",
    "simulate_counts", "exact_open_probability", "lowpass_bessel", "quantize", "make_experiment",
    # analysis
    "dwell_times", "isochrone_stats", "successive_difference_variance", "baseline_variance",
    "NoiseFit", "fit_parabola", "fit_slope_only", "open_probability", "bootstrap_fit",
    "power_spectrum", "lorentzian", "fit_lorentzian", "autocorrelation",
    # pipeline
    "count_channels", "CountResult", "make_mystery", "reveal",
    "plotting",
]
