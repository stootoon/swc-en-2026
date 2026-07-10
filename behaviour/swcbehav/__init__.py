"""swcbehav -- backend for the SWC ENC 2026 behaviour module.

Reference implementations and the synthetic ground-truth generator behind the
Piet et al. (2024) analysis notebooks. Notebooks import prerequisites from here
so each one runs standalone.
"""

from .task import make_session, IMAGE_DURATION
from .generate import (
    Session,
    make_mouse,
    simulate,
    sigmoid,
    timing_feature,
    constant_weights,
    crossover_weights,
    sample_engagement,
    CohortSession,
    make_cohort,
    WEIGHT_NAMES,
    WEIGHT_COLORS,
    TIMING_MIDPOINT,
    TIMING_SLOPE,
)
from .stats import (
    linear_fit,
    variance_explained,
    permutation_variance_explained,
    benjamini_hochberg,
    hierarchical_bootstrap,
)
from .design import (
    segment_bouts,
    assign_bout_starts,
    images_since_bout,
    build_design_matrix,
    BOUT_ILI_THRESHOLD,
)
from .models import (
    neg_log_likelihood,
    fit_static,
    predict_prob,
    roc_curve,
    auc_score,
    cross_val_auc,
    mean_log_likelihood,
    cross_val_loglik,
    ablation_loglik_deltas,
    log_evidence_laplace,
    dynamic_neg_log_posterior,
    fit_dynamic,
)
from . import plotting

__all__ = [
    "make_session",
    "IMAGE_DURATION",
    "Session",
    "make_mouse",
    "simulate",
    "sigmoid",
    "timing_feature",
    "constant_weights",
    "crossover_weights",
    "sample_engagement",
    "WEIGHT_NAMES",
    "TIMING_MIDPOINT",
    "TIMING_SLOPE",
    "segment_bouts",
    "assign_bout_starts",
    "images_since_bout",
    "build_design_matrix",
    "BOUT_ILI_THRESHOLD",
]