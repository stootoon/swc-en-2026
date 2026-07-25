"""picosort -- backend for the SWC ENC 2026 ephys-pop (spike sorting) module.

Students build a miniature spike sorter, "picosort", loosely following the
stages of Kilosort4. This package holds the synthetic ground-truth generator and
reference implementations of every stage, so each notebook runs standalone and a
stuck exercise has a `ps.<name>` fallback.

Imported in the notebooks as::

    import picosort as ps

Modules are added here as the module is built out, one notebook at a time.
"""

from .probe import FS, Probe, make_probe
from .generate import (
    Recording,
    GroundTruth,
    make_recording,
    make_template,
    spike_waveform,
    spatial_decay,
    footprint,
    peak_channel,
    average_waveform,
    TEMPLATE_SAMPLES,
)
from . import plotting

__all__ = [
    "FS",
    "Probe",
    "make_probe",
    "Recording",
    "GroundTruth",
    "make_recording",
    "make_template",
    "spike_waveform",
    "spatial_decay",
    "footprint",
    "peak_channel",
    "average_waveform",
    "TEMPLATE_SAMPLES",
    "plotting",
]
