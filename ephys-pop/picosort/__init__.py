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
from .preprocess import (
    highpass_filter,
    common_average_reference,
    noise_covariance,
    whitening_matrix,
    whiten,
    preprocess,
)
from .detect import channel_noise, detect_spikes, extract_snippets
from .features import (
    localize,
    peak_waveforms,
    pca_features,
    explained_variance,
    spike_features,
)
from .cluster import (
    graph_cluster,
    cluster_spikes,
    bic_curve,
    templates_from_labels,
    tsne_embedding,
)
from .match import matching_pursuit
from .pipeline import run_picosort, SortResult
from .postprocess import (
    correlogram,
    refractory_violations,
    coincidence_fraction,
    find_duplicate_pairs,
)
from .evaluate import match_to_truth, summary, UnitScore
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
    # preprocessing
    "highpass_filter",
    "common_average_reference",
    "noise_covariance",
    "whitening_matrix",
    "whiten",
    "preprocess",
    # detection & features
    "channel_noise",
    "detect_spikes",
    "extract_snippets",
    "localize",
    "peak_waveforms",
    "pca_features",
    "explained_variance",
    "spike_features",
    # clustering
    "graph_cluster",
    "cluster_spikes",
    "bic_curve",
    "templates_from_labels",
    "tsne_embedding",
    # matching pursuit
    "matching_pursuit",
    # full pipeline
    "run_picosort",
    "SortResult",
    # postprocessing
    "correlogram",
    "refractory_violations",
    "coincidence_fraction",
    "find_duplicate_pairs",
    # evaluation
    "match_to_truth",
    "summary",
    "UnitScore",
    "plotting",
]
