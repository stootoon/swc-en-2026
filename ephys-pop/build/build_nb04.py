"""Notebook 4 -- Feature extraction: localization and PCA."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 4 — Feature extraction

*SWC ENC 2026 · ephys-pop module*

We have a pile of unlabelled snippets. To tell the neurons apart we describe each
spike by a few **features** — numbers that differ between units but stay similar
within a unit. Three natural ones:

- **where** the spike is — its **depth** on the probe,
- **how big** it is — its **amplitude**,
- **what shape** it has — captured by **PCA** of the waveform.

**In this notebook you will:**
1. **Localize** each spike: peak channel, amplitude, depth.
2. See units separate in the depth–amplitude plane.
3. Use **PCA** to capture waveform shape — and see when you need it.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picosort as ps

rec = ps.make_recording(n_units=6, duration_s=20.0, seed=0)
whitened, filtered, W = ps.preprocess(rec)
times, peak_channels = ps.detect_spikes(whitened, rec.probe, rec.fs, threshold=5.0)
snippets, times = ps.extract_snippets(filtered, times, peak_channels=peak_channels)
print("snippets:", snippets.shape)
""",),
    md(r"""
## 1. Localize: depth and amplitude

A spike's **peak-to-peak amplitude** on each channel is its footprint. The channel
where it's biggest is the **peak channel**, and its size there is the spike's
**amplitude**. Its **depth** is the footprint-weighted average channel position —
the centre of mass — using only channels near the peak, so probe-wide noise doesn't
drag the estimate.

**Exercise 1** *(~7 min)*. Complete `localize`: from the `(n_spikes, n_channels, n_samples)`
snippets, compute peak-to-peak per channel, then peak channel, amplitude, and the
centre-of-mass depth over channels within `radius_um` of the peak.

> **Check / unstuck.** Amplitudes should span ~120–320 µV; depths should cover the
> probe (roughly 80–560 µm). Stuck? Use `ps.localize(snippets, rec.probe)`.
""",),
    code(
        solution=r"""
def localize(snippets, probe, radius_um=60.0):
    p2p = snippets.max(axis=2) - snippets.min(axis=2)      # (n_spikes, n_channels)
    peak_channel = np.argmax(p2p, axis=1)
    amplitude = p2p[np.arange(len(p2p)), peak_channel]
    dy = np.abs(probe.y[None, :] - probe.y[peak_channel][:, None])
    w = p2p * (dy <= radius_um)
    depth = (w * probe.y[None, :]).sum(axis=1) / w.sum(axis=1)
    return peak_channel, amplitude, depth

peak_channel, amplitude, depth = localize(snippets, rec.probe)
print(f"amplitude range: {amplitude.min():.0f}–{amplitude.max():.0f} µV")
print(f"depth range: {depth.min():.0f}–{depth.max():.0f} µm")
""",
        student=r"""
def localize(snippets, probe, radius_um=60.0):
    p2p = snippets.max(axis=2) - snippets.min(axis=2)      # (n_spikes, n_channels)
    peak_channel = np.argmax(p2p, axis=1)
    amplitude = p2p[np.arange(len(p2p)), peak_channel]
    # YOUR CODE HERE: depth = centre of mass of p2p over channels within radius_um
    # of the peak channel. Build a mask dy<=radius_um from probe.y, weight p2p by it.
    raise NotImplementedError

peak_channel, amplitude, depth = localize(snippets, rec.probe)
print(f"amplitude range: {amplitude.min():.0f}–{amplitude.max():.0f} µV")
print(f"depth range: {depth.min():.0f}–{depth.max():.0f} µm")
""",
    ),
    md(r"""
## 2. The feature space

Plot every spike as a point in the **depth–amplitude plane**. Even though we have no
labels, the units jump out as separated clouds: each neuron sits at its own depth
and fires at its own amplitude. This is what makes the sorting possible.
""",),
    code(r"""
ps.plotting.plot_feature_space(depth, amplitude, title="every detected spike (unlabelled)")
plt.show()
""",),
    md(r"""
Those clouds are the units, waiting to be found — that's Notebook 5. But first, two
more features to make the clouds even cleaner, and to handle a case depth and
amplitude *can't*.

## 3. Shape, via PCA

Two neurons can sit at the same depth and fire at the same amplitude yet have
different **waveform shapes** — a narrow spike vs a broad one. Depth and amplitude
can't tell them apart, but shape can. The trouble is a waveform is 61 numbers; we
want a couple. **Principal component analysis** finds the few directions along which
waveforms actually vary and re-expresses each waveform by its **scores** on them.

**The idea in two dimensions first.** Remember the tilted cloud of two channels' noise
in Notebook 2? A cloud like that has a **long axis** (the direction it spreads most)
and a short axis at right angles. PCA finds those axes. The long axis is **PC1**; a
point's position along it — its **score** — captures most of what makes that point
different from the others, in a single number.
""",),
    code(r"""
# A cloud of 2-D points, stretched and tilted (like the neighbour-channel cloud in NB2).
rng = np.random.default_rng(1)
cloud = rng.normal(size=(400, 2)) @ np.array([[2.4, 1.3], [0.0, 0.7]])
cloud = cloud - cloud.mean(0)
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(cloud)          # PCA finds the cloud's principal axes

plt.figure(figsize=(4.8, 4.8))
plt.scatter(cloud[:, 0], cloud[:, 1], s=8, alpha=0.4)
for k, color in zip(range(2), ["tab:red", "tab:orange"]):
    v = pca.components_[k] * np.sqrt(pca.explained_variance_[k]) * 2   # axis × 2 std
    plt.arrow(0, 0, v[0], v[1], color=color, width=0.06, length_includes_head=True)
    plt.text(v[0] * 1.15, v[1] * 1.15, f"PC{k+1}", color=color, fontweight="bold")
plt.gca().set_aspect("equal"); plt.xlabel("feature 1"); plt.ylabel("feature 2")
plt.title("PCA finds the axes of most spread"); plt.show()
""",),
    md(r"""
PC1 (red) points along the length of the cloud — the direction of greatest spread; PC2
(orange) is the leftover, perpendicular direction. Describing each point by its PC1
score alone throws away only the little bit of width in the PC2 direction.

Our waveforms are points in **61 dimensions**, not two — but PCA does exactly the same
thing: find the handful of axes of greatest spread, and describe each waveform by its
position along them. Since the spikes of one neuron are near-copies, a couple of axes
capture almost all the variation. Look at the **scree plot** — the fraction of variance
each component explains:

<details>
<summary><b>▸ Go deeper: what PCA actually computes (optional)</b></summary>

Stack the (centred) waveforms as rows of a matrix $X$ of size $n \times p$
($n$ spikes, $p = 61$ samples). We want the single direction $v$ (a unit vector in
waveform space) along which the projected waveforms $Xv$ vary the most:

$$\max_{\lVert v\rVert = 1}\ \mathrm{Var}(Xv) = \max_{\lVert v\rVert=1}\ v^\top C\, v,
\qquad C = \tfrac{1}{n} X^\top X$$

where $C$ is the $p\times p$ covariance of the waveforms. The Lagrangian
$v^\top C v - \lambda(v^\top v - 1)$ has stationary points where

$$C\,v = \lambda\, v,$$

so the principal directions are the **eigenvectors of the covariance**, and each
one's variance is its **eigenvalue** $\lambda$. The second PC is the next eigenvector,
and so on — orthogonal directions of decreasing variance.

You don't compute any of this by hand — **`sklearn`'s `PCA`** finds those eigenvectors
for you (via a numerically stable **singular value decomposition** internally) and
returns the shape directions (`components_`) and each spike's **scores**. The variance
explained by component $k$ is $\lambda_k / \sum_j \lambda_j$ — the scree plot. Why a
*few* components suffice is the **Eckart–Young theorem**: keeping the top $k$ gives the
best possible rank-$k$ approximation of the data (smallest reconstruction error).
Because every spike of a neuron is nearly the same waveform, a couple of components
rebuild it almost perfectly and the rest is noise.
</details>
""",),
    code(r"""
waveforms = ps.peak_waveforms(snippets)          # (n_spikes, n_samples)
ev = ps.explained_variance(waveforms, n_components=8)
plt.figure(figsize=(5, 3.4))
plt.bar(np.arange(1, 9), ev)
plt.xlabel("principal component"); plt.ylabel("variance explained")
plt.title(f"first 2 PCs capture {ev[:2].sum():.0%} of waveform variation"); plt.show()
""",),
    md(r"""
**Exercise 2** *(~3 min · easy)*. Complete `pca_scores` using **sklearn's `PCA`**: fit it with
`k` components to the waveforms and return each spike's scores (its `k` numbers).

> **Check / unstuck.** `scores` should be `(n_spikes, 2)`. Stuck? it's
> `PCA(n_components=k).fit_transform(waveforms)`.
""",),
    code(
        solution=r"""
from sklearn.decomposition import PCA

def pca_scores(waveforms, k=2):
    return PCA(n_components=k).fit_transform(waveforms)

scores = pca_scores(waveforms, 2)
print("each of the", len(scores), "spikes is now just", scores.shape[1], "numbers instead of",
      waveforms.shape[1])
""",
        student=r"""
from sklearn.decomposition import PCA

def pca_scores(waveforms, k=2):
    # YOUR CODE HERE: fit sklearn's PCA with k components to the waveforms and return
    # the transformed scores (one row of k numbers per spike).
    raise NotImplementedError

scores = pca_scores(waveforms, 2)
print("each of the", len(scores), "spikes is now just", scores.shape[1], "numbers instead of",
      waveforms.shape[1])
""",
    ),
    md(r"""
What *is* this "shape space"? Two things worth seeing. **(left)** what the axes mean:
start from the average waveform and move along PC1, or along PC2, to see the shape
change each one controls. **(right)** every spike placed at its `(PC1, PC2)` — its
whole 61-sample waveform boiled down to two numbers.
""",),
    code(r"""
pca = PCA(n_components=2).fit(waveforms)          # the same PCA as in the exercise
components, mean_wave = pca.components_, pca.mean_  # shape modes and the average waveform
tt = np.arange(waveforms.shape[1])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# (left) what each PC means, as a shape: the mean waveform, nudged +/- along each PC
for k, col in [(0, "tab:red"), (1, "tab:green")]:
    step = 2 * scores[:, k].std()
    ax[0].plot(tt, mean_wave + step * components[k], color=col, lw=1, label=f"mean + PC{k+1}")
    ax[0].plot(tt, mean_wave - step * components[k], color=col, lw=1, ls="--", label=f"mean − PC{k+1}")
ax[0].plot(tt, mean_wave, "k", lw=2.5, label="mean waveform")
ax[0].set_title("what the PC axes mean, as shapes"); ax[0].set_xlabel("sample")
ax[0].set_ylabel("µV"); ax[0].legend(fontsize=7)
# (right) every spike as a point in shape space
sc = ax[1].scatter(scores[:, 0], scores[:, 1], s=8, c=amplitude, cmap="viridis")
plt.colorbar(sc, ax=ax[1], label="amplitude (µV)")
ax[1].set_title("every spike as 2 numbers (its shape)"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
plt.tight_layout(); plt.show()
""",),
    md(r"""
Reading the left panel: moving along **PC1** (red) mostly scales the whole spike up and
down — it captures **overall size**. **PC2** (green) tweaks the **shape** (the width and
rebound). Each spike is now a point in this plane.

On the right, the spikes do fall into groups along **PC1** — but look at the colour:
PC1 is essentially tracking **amplitude** (dark = small, yellow = large). So the
separation you see along PC1 is the *same* separation amplitude already gave us, just
relabelled. The genuinely *shape*-based axis is **PC2**, and here it barely separates
anything, because our six units all make similar biphasic spikes. That's fine — depth
and amplitude already did the job. **PC2 (pure shape) earns its keep only when two units
share a location** — which is exactly the case we build next.

## 4. When shape is the only clue *(optional)*

*Skippable.* Build two units at the **same depth** and **same amplitude**, differing
only in their waveform **width**. Depth and amplitude now see one blob — but their
*shapes* differ, so PCA splits them. First, the two units' actual waveforms (the clue),
then the two feature spaces:
""",),
    code(r"""
probe = rec.probe
def one_unit(width, seed):
    wf = ps.spike_waveform(trough_width=width)
    tmpl = ps.make_template(probe, [0, 300], amplitude=150, waveform=wf)
    rng = np.random.default_rng(seed)
    return np.array([tmpl + rng.normal(0, 8, tmpl.shape) for _ in range(120)])

snips = np.concatenate([one_unit(2.2, 1), one_unit(3.8, 2)])
lab = np.r_[np.zeros(120), np.ones(120)].astype(int)
_, amp2, dep2 = ps.localize(snips, probe)
sc2 = pca_scores(ps.peak_waveforms(snips), 2)     # the sklearn PCA from Exercise 2

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
# (1) the actual waveforms -- the shape difference PCA will detect
for u, width in [(0, 2.2), (1, 3.8)]:
    axes[0].plot(ps.spike_waveform(trough_width=width), color=ps.plotting.unit_color(u),
                 lw=2, label=f"unit {u} (width {width})")
axes[0].set_title("the two units' waveforms"); axes[0].set_xlabel("sample")
axes[0].set_ylabel("amplitude (norm.)"); axes[0].legend(fontsize=8)
# (2) + (3) the two feature spaces
for u in [0, 1]:
    m = lab == u
    axes[1].scatter(dep2[m], amp2[m], s=10, color=ps.plotting.unit_color(u))
    axes[2].scatter(sc2[m, 0], sc2[m, 1], s=10, color=ps.plotting.unit_color(u), label=f"unit {u}")
axes[1].set_title("depth–amplitude: one blob"); axes[1].set_xlabel("depth"); axes[1].set_ylabel("amp")
axes[2].set_title("PC (shape) space: two clusters"); axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")
axes[2].legend(); plt.tight_layout(); plt.show()
""",),
    md(r"""
## Wrap-up

Each spike is now a handful of numbers: depth, amplitude, and a couple of shape PCs.
In the depth–amplitude plane the units already stand out as separate clouds, and PCA
is there for the harder cases where they don't.

**Next (Notebook 5 — clustering):** turn those clouds into labelled units, and
average each cluster into a template.
""",),
]

student, solution = build("04_features", cells)
print("wrote:", student)
print("wrote:", solution)
