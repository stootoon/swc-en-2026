"""The forward model -- the ground truth for the whole module.

Spike sorting runs *backwards*: from a wall of voltage to "which neuron fired
when." To learn it, we run the model *forwards* from a truth we choose, so every
stage can be checked against the answer.

A recording is built by superposition:

    voltage(t, channel) = sum over spikes of  template[unit](channel, t - t_spike)
                          + noise(t, channel)

Each **unit** is one neuron with:

  * a position on the probe (a depth ``y``),
  * a **temporal waveform** -- the canonical biphasic extracellular spike (a
    sharp negative trough then a slower positive rebound), and
  * a **spatial footprint** -- the waveform's amplitude decays with distance from
    the unit, so a spike lands on several neighbouring channels at once. The
    product of the two is the unit's **spatiotemporal template**.

Spikes are drawn as independent Poisson trains, dropped into the traces, and
buried in **spatially correlated noise** plus a slow **common-mode** fluctuation.
Those two nuisances are not incidental: the correlated noise is what channel
whitening removes, and the common mode is what high-pass filtering and common
average referencing remove. Because we know every spike time, unit label, and
template, the sorter's output can be scored exactly (see ``evaluate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .probe import FS, Probe, make_probe

# Template window: ~2 ms, centred on the trough. Long enough to hold the whole
# biphasic waveform at 30 kHz.
TEMPLATE_SAMPLES = 61


# --------------------------------------------------------------------------- #
# Ground-truth containers                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class GroundTruth:
    """Everything the sorter is *not* allowed to see, kept for scoring."""

    spike_times: np.ndarray      # (n_spikes,) sample index of each spike, sorted
    spike_labels: np.ndarray     # (n_spikes,) which unit produced each spike
    templates: np.ndarray        # (n_units, n_channels, TEMPLATE_SAMPLES)
    unit_xy: np.ndarray          # (n_units, 2) unit positions in micrometres
    unit_amplitude: np.ndarray   # (n_units,) peak amplitude in microvolts
    fs: float = FS

    @property
    def n_units(self) -> int:
        return len(self.templates)


@dataclass
class Recording:
    """A synthetic extracellular recording. ``traces`` is (n_samples, n_channels)."""

    traces: np.ndarray           # microvolts, float32
    fs: float
    probe: Probe
    ground_truth: GroundTruth | None = None

    @property
    def n_samples(self) -> int:
        return self.traces.shape[0]

    @property
    def n_channels(self) -> int:
        return self.traces.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


# --------------------------------------------------------------------------- #
# Waveforms and templates                                                      #
# --------------------------------------------------------------------------- #
def spike_waveform(n: int = TEMPLATE_SAMPLES, trough_width: float = 3.0,
                   peak_width: float = 7.0, peak_delay: float = 8.0,
                   peak_ratio: float = 0.4) -> np.ndarray:
    """A canonical extracellular action potential, normalised to a trough of -1.

    A sharp negative trough (the sodium current) at the centre, followed by a
    broader positive rebound (repolarisation). Varying the widths and rebound
    across units is what makes some pairs easy and others hard to tell apart.
    """
    t = np.arange(n) - n // 2
    trough = -np.exp(-0.5 * (t / trough_width) ** 2)
    peak = peak_ratio * np.exp(-0.5 * ((t - peak_delay) / peak_width) ** 2)
    w = trough + peak
    return (w / -w.min()).astype(np.float64)


def spatial_decay(probe: Probe, xy, space_constant: float = 25.0) -> np.ndarray:
    """Per-channel amplitude of a unit at position ``xy`` (exponential falloff)."""
    return np.exp(-probe.distances_to(xy) / space_constant)


def make_template(probe: Probe, xy, amplitude: float, waveform: np.ndarray,
                  space_constant: float = 25.0) -> np.ndarray:
    """Spatiotemporal template: (n_channels, n_samples), in microvolts.

    The outer product of the spatial footprint (which channels) and the temporal
    waveform (what shape), scaled to the unit's peak ``amplitude``.
    """
    footprint = amplitude * spatial_decay(probe, xy, space_constant)
    return np.outer(footprint, waveform)


# --------------------------------------------------------------------------- #
# The generator                                                                #
# --------------------------------------------------------------------------- #
def _correlated_noise(rng, n_samples, probe, noise_sd, noise_space_constant):
    """Gaussian noise with exponential spatial correlation across channels.

    Nearby channels see similar noise -- exactly the structure whitening undoes.
    """
    dy = np.abs(probe.y[:, None] - probe.y[None, :])
    cov = np.exp(-dy / noise_space_constant)
    chol = np.linalg.cholesky(cov + 1e-6 * np.eye(probe.n_channels))
    white = rng.standard_normal((n_samples, probe.n_channels))
    return (white @ chol.T) * noise_sd


def _common_mode(rng, n_samples, fs, amplitude):
    """A slow 1/f-like drift/LFP shared across all channels.

    Real drift and local field potential have a ``1/f`` spectrum -- lots of power at
    low frequencies falling off with frequency -- which is the low-frequency
    "mountain" seen in a channel's power spectrum. High-pass filtering removes it.
    """
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    freqs[0] = freqs[1]                              # avoid divide-by-zero at DC
    spectrum = spectrum / freqs                      # 1/f amplitude -> ~1/f^2 power
    drift = np.fft.irfft(spectrum, n=n_samples)
    return amplitude * drift / drift.std()


def footprint(template: np.ndarray) -> np.ndarray:
    """Peak-to-peak amplitude of a template on each channel (its spatial footprint)."""
    return template.max(axis=1) - template.min(axis=1)


def peak_channel(template: np.ndarray) -> int:
    """Channel index where the template's footprint is largest."""
    return int(np.argmax(footprint(template)))


def average_waveform(rec: "Recording", unit: int, channel: int, half: int = 30):
    """Recover a unit's waveform on one channel by averaging ground-truth snippets.

    Returns ``(mean_waveform, snippets)``. Uses the recording's ground-truth spike
    times -- this is the reference for Notebook 1's exercise, not part of the sorter.
    """
    gt = rec.ground_truth
    times = gt.spike_times[gt.spike_labels == unit]
    snippets = np.array([rec.traces[t - half:t + half + 1, channel] for t in times])
    return snippets.mean(axis=0), snippets


def make_recording(n_units: int = 6, duration_s: float = 20.0, fs: float = FS,
                   n_channels: int = 32, pitch: float = 20.0,
                   rate_range=(1.5, 5.0), amplitude_range=(90.0, 200.0),
                   noise_sd: float = 12.0, space_constant: float = 25.0,
                   noise_space_constant: float = 30.0, common_mode: float = 60.0,
                   common_broadband: float = 10.0, seed: int = 0) -> Recording:
    """Generate a synthetic recording with full ground truth.

    Units are placed at random depths, each with its own firing rate, amplitude,
    and slightly jittered waveform. Their spikes are superposed into the traces,
    then three nuisances are layered on -- each cleaned up by a different tool in
    Notebook 2:

    * ``common_mode``: a slow LFP-like signal shared across channels -> high-pass
    * ``common_broadband``: a fast signal shared identically across channels
      (e.g. movement/EMG artefact) -> common average reference
    * spatially correlated noise (nearby channels share noise) -> whitening

    Collisions between spikes happen naturally -- that is what template matching
    later has to resolve.
    """
    rng = np.random.default_rng(seed)
    probe = make_probe(n_channels=n_channels, pitch=pitch)
    n_samples = int(round(duration_s * fs))
    half = TEMPLATE_SAMPLES // 2

    # --- units: position, amplitude, and an idiosyncratic waveform ---------- #
    # Space the units out along the probe (with jitter) so they are separable by
    # depth. Two neurons at the same site with the same waveform are genuinely
    # unsortable; the difficulty we want to teach comes from noise, collisions and
    # duplicates, not from degenerate placement.
    margin = 2 * pitch
    edges = np.linspace(probe.y.min() + margin, probe.y.max() - margin, n_units + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    spacing = centers[1] - centers[0] if n_units > 1 else 1.0
    unit_y = centers + rng.uniform(-0.25, 0.25, n_units) * spacing
    unit_xy = np.column_stack([np.zeros(n_units), unit_y])
    # Distinct amplitudes (spread across the range, in random order vs depth) so the
    # units occupy well-separated regions of the depth-amplitude plane.
    unit_amp = rng.permutation(np.linspace(amplitude_range[0], amplitude_range[1], n_units))
    templates = np.stack([
        make_template(
            probe, unit_xy[u], unit_amp[u],
            spike_waveform(trough_width=rng.uniform(2.0, 4.0),
                           peak_width=rng.uniform(5.0, 10.0),
                           peak_delay=rng.uniform(6.0, 11.0),
                           peak_ratio=rng.uniform(0.2, 0.6)),
            space_constant=space_constant)
        for u in range(n_units)
    ])

    # --- Poisson spike trains ---------------------------------------------- #
    rates = rng.uniform(*rate_range, n_units)
    times_list, labels_list = [], []
    for u in range(n_units):
        n_sp = rng.poisson(rates[u] * duration_s)
        t = rng.integers(half, n_samples - half - 1, size=n_sp)
        times_list.append(t)
        labels_list.append(np.full(n_sp, u))
    spike_times = np.concatenate(times_list)
    spike_labels = np.concatenate(labels_list)
    order = np.argsort(spike_times)
    spike_times, spike_labels = spike_times[order], spike_labels[order]

    # --- superpose templates into the traces ------------------------------- #
    traces = np.zeros((n_samples, probe.n_channels))
    for t, u in zip(spike_times, spike_labels):
        traces[t - half:t + half + 1] += templates[u].T

    # --- nuisances: correlated noise + shared low- and broadband signals --- #
    traces += _correlated_noise(rng, n_samples, probe, noise_sd, noise_space_constant)
    traces += _common_mode(rng, n_samples, fs, common_mode)[:, None]
    traces += common_broadband * rng.standard_normal(n_samples)[:, None]

    gt = GroundTruth(spike_times=spike_times, spike_labels=spike_labels,
                     templates=templates, unit_xy=unit_xy,
                     unit_amplitude=unit_amp, fs=fs)
    return Recording(traces=traces.astype(np.float32), fs=fs, probe=probe,
                     ground_truth=gt)
