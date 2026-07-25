"""Clustering -- group spikes into units, and average each group into a template.

Kilosort clusters spike features with a graph algorithm; we use a simpler,
transparent version with the same spirit. Build a graph on the spikes in feature
space -- connect each spike to its near neighbours -- and read off the **connected
components**: islands of mutually-close spikes are the putative units. Averaging
the snippets in a cluster gives that unit's **template** (the same
noise-averages-away trick from Notebook 1, now with clusters we discovered rather
than labels we were handed).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def graph_cluster(features: np.ndarray, n_neighbors: int = 10, eps: float | None = None,
                  eps_factor: float = 2.0, min_cluster_size: int = 15) -> np.ndarray:
    """Connected-components clustering of points in feature space.

    Each point is joined to its ``n_neighbors`` nearest neighbours that lie within
    ``eps``; the connected components of that graph are the clusters. ``eps``
    defaults to ``eps_factor`` times the typical near-neighbour distance. Clusters
    smaller than ``min_cluster_size`` are labelled ``-1`` (unassigned noise).
    Returns an integer label per spike, relabelled ``0..K-1`` by descending size.
    """
    n = len(features)
    tree = cKDTree(features)
    dists, idx = tree.query(features, k=min(n_neighbors + 1, n))
    if eps is None:
        eps = float(np.median(dists[:, 1:])) * eps_factor

    uf = _UnionFind(n)
    for i in range(n):
        for j, d in zip(idx[i, 1:], dists[i, 1:]):
            if d <= eps:
                uf.union(i, int(j))
    roots = np.array([uf.find(i) for i in range(n)])

    # keep clusters above the size floor, relabel by descending size
    labels = np.full(n, -1)
    uniq, counts = np.unique(roots, return_counts=True)
    big = uniq[counts >= min_cluster_size]
    big = big[np.argsort([-(roots == r).sum() for r in big])]
    for new, r in enumerate(big):
        labels[roots == r] = new
    return labels


def cluster_spikes(features: np.ndarray, n_units: int, seed: int = 0) -> np.ndarray:
    """Fit a Gaussian mixture of ``n_units`` clusters and return a label per spike.

    A Gaussian mixture models each unit as an ellipse in feature space, so it copes
    with clusters of different size and shape (unlike k-means, which assumes equal
    round blobs and can merge two close units while splitting a third). Choose
    ``n_units`` from the BIC curve (see ``bic_curve``) or by eye from the t-SNE plot.
    """
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_units, n_init=10, random_state=seed).fit_predict(features)


def bic_curve(features: np.ndarray, k_values=range(2, 11), seed: int = 0) -> dict:
    """BIC of a Gaussian mixture for each candidate cluster count ``k``.

    BIC rewards fit and penalises extra clusters. It drops steeply until ``k``
    reaches the true number of units, then flattens -- the "elbow" tells you how many
    units are really there. Returns ``{k: bic}``.
    """
    from sklearn.mixture import GaussianMixture
    return {int(k): float(GaussianMixture(k, n_init=5, random_state=seed).fit(features).bic(features))
            for k in k_values if k < len(features)}


def templates_from_labels(snippets: np.ndarray, labels: np.ndarray):
    """Average snippets within each cluster into a template.

    Returns ``(templates, cluster_ids)`` where ``templates`` is
    ``(n_clusters, n_channels, n_samples)`` and ``cluster_ids`` lists the labels
    used (excluding ``-1``).
    """
    ids = np.array(sorted(u for u in np.unique(labels) if u >= 0))
    templates = np.stack([snippets[labels == u].mean(axis=0) for u in ids])
    return templates, ids


def tsne_embedding(features: np.ndarray, seed: int = 0, perplexity: float = 30.0):
    """2-D t-SNE layout of the feature space, for eyeballing cluster separation."""
    from sklearn.manifold import TSNE
    perplexity = min(perplexity, max(5.0, (len(features) - 1) / 3.0))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    return tsne.fit_transform(features)
