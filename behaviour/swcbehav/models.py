"""Reference implementations of the modeling steps.

Mirrors ``design.py``: notebooks 4+ import these as prerequisites, while each
notebook still has students re-derive *its own* new step in the blanks.

Contains the static logistic-regression fit (Notebook 3) and its evaluation
tools (Notebook 4). Evidence/ablation and the dynamic smoother are added
alongside the notebooks that introduce them.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from .generate import sigmoid


# --------------------------------------------------------------------------- #
# Static logistic regression (Notebook 3)                                     #
# --------------------------------------------------------------------------- #
def neg_log_likelihood(w, X, y, l2=1e-4):
    """Negative Bernoulli log-likelihood with a weak ridge (Gaussian) prior."""
    z = X @ w
    log_lik = np.sum(y * z - np.logaddexp(0.0, z))
    return -log_lik + l2 * np.sum(w ** 2)


def fit_static(X, y, l2=1e-4):
    """MAP fit of the static strategy weights (convex; global optimum)."""
    result = minimize(neg_log_likelihood, np.zeros(X.shape[1]),
                      args=(X, np.asarray(y, float), l2), method="L-BFGS-B")
    return result.x


def predict_prob(w, X):
    """Model probability that a bout starts on each image."""
    return sigmoid(X @ w)


# --------------------------------------------------------------------------- #
# Evaluation: ROC, AUC, cross-validation (Notebook 4)                         #
# --------------------------------------------------------------------------- #
def roc_curve(scores, y):
    """False-positive and true-positive rates as the decision threshold sweeps."""
    scores = np.asarray(scores)
    y = np.asarray(y).astype(float)
    order = np.argsort(-scores)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    tpr = np.concatenate([[0.0], tps / tps[-1]])
    fpr = np.concatenate([[0.0], fps / fps[-1]])
    return fpr, tpr


def auc_score(scores, y):
    """Area under the ROC curve, via the rank-sum identity (handles ties)."""
    scores = np.asarray(scores)
    y = np.asarray(y).astype(bool)
    n_pos = y.sum()
    n_neg = (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cross_val_auc(X, y, n_folds=5, l2=1e-4):
    """Cross-validated AUC using contiguous time-block folds.

    Contiguous (not random) folds respect the session's temporal structure, so
    held-out performance isn't inflated by leakage between adjacent images.
    """
    y = np.asarray(y, float)
    n = len(y)
    folds = np.array_split(np.arange(n), n_folds)
    aucs = []
    for i in range(n_folds):
        test = folds[i]
        train = np.setdiff1d(np.arange(n), test)
        w = fit_static(X[train], y[train], l2)
        aucs.append(auc_score(predict_prob(w, X[test]), y[test]))
    return np.array(aucs)


# --------------------------------------------------------------------------- #
# Which strategies matter: ablation (Notebook 5)                              #
# --------------------------------------------------------------------------- #
def mean_log_likelihood(w, X, y):
    """Average per-image Bernoulli log-likelihood (nats). Higher is better."""
    z = X @ w
    return float(np.mean(y * z - np.logaddexp(0.0, z)))


def cross_val_loglik(X, y, n_folds=5, l2=1e-4):
    """Held-out predictive log-likelihood per image, averaged over folds."""
    y = np.asarray(y, float)
    n = len(y)
    folds = np.array_split(np.arange(n), n_folds)
    lls = []
    for i in range(n_folds):
        test = folds[i]
        train = np.setdiff1d(np.arange(n), test)
        w = fit_static(X[train], y[train], l2)
        lls.append(mean_log_likelihood(w, X[test], y[test]))
    return float(np.mean(lls))


def ablation_loglik_deltas(X, y, col_names, n_folds=5):
    """Drop in cross-validated log-likelihood when each non-bias strategy is removed."""
    full = cross_val_loglik(X, y, n_folds)
    deltas = {}
    for k, name in enumerate(col_names):
        if name == "bias":
            continue
        deltas[name] = full - cross_val_loglik(np.delete(X, k, axis=1), y, n_folds)
    return deltas


def dynamic_neg_log_posterior(W_flat, X, y, sigma, T, K):
    """Negative log posterior of a drifting-weight (random-walk) logistic model.

    Objective = Bernoulli NLL (each image t uses its own weights w_t)
              + (1 / 2 sigma^2) * sum_t || w_{t+1} - w_t ||^2   (random-walk prior).

    Returns (objective, gradient) so the optimizer can use the analytic gradient.
    """
    W = W_flat.reshape(T, K)
    z = np.sum(W * X, axis=1)
    data = -np.sum(y * z - np.logaddexp(0.0, z))
    D = np.diff(W, axis=0)                       # w_{t+1} - w_t
    penalty = np.sum(D * D) / (2 * sigma ** 2)

    p = sigmoid(z)
    grad = (p - y)[:, None] * X                  # data-term gradient (logistic)
    gp = np.zeros_like(W)                        # penalty-term gradient (2nd difference)
    gp[:-1] -= D
    gp[1:] += D
    grad += gp / sigma ** 2
    return data + penalty, grad.ravel()


def fit_dynamic(X, y, sigma=0.05, w_init=None, maxiter=500):
    """MAP fit of the drifting-weight model. Returns a (T, K) weight trajectory.

    ``sigma`` is the random-walk step size: small sigma -> stiff (approaches the
    single static fit), large sigma -> flexible. Warm-started from the static fit.
    """
    y = np.asarray(y, float)
    T, K = X.shape
    if w_init is None:
        w_static = fit_static(X, y)
        w_init = np.tile(w_static, (T, 1))
    res = minimize(dynamic_neg_log_posterior, w_init.ravel(),
                   args=(X, y, sigma, T, K), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return res.x.reshape(T, K)


def log_evidence_laplace(X, y, prior_var=100.0):
    """Laplace approximation to the log marginal likelihood log p(y | model).

    Integrates the weights out under a Gaussian prior N(0, prior_var * I). The
    result automatically rewards fit and penalizes unnecessary parameters (an
    Occam factor) -- the quantity the paper uses for its strategy indices.
    """
    y = np.asarray(y, float)
    d = X.shape[1]
    l2 = 0.5 / prior_var                      # matches -log N(0, prior_var) penalty
    w = fit_static(X, y, l2=l2)
    z = X @ w
    p = sigmoid(z)
    log_lik = np.sum(y * z - np.logaddexp(0.0, z))
    log_prior = -0.5 * d * np.log(2 * np.pi * prior_var) - 0.5 * np.dot(w, w) / prior_var
    H = (X * (p * (1 - p))[:, None]).T @ X + np.eye(d) / prior_var  # neg-Hessian of log-joint
    _, logdet = np.linalg.slogdet(H)
    return float(log_lik + log_prior + 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet)
