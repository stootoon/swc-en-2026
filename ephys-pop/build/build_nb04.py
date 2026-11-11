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
import matplotlib.pyplot as plt, picosort as ps
ps.plotting.draw_pipeline(highlight=4); plt.show()      # where we are in the pipeline
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
Those clouds are the units, waiting to be found — that's Notebook 5. But depth and
amplitude have a blind spot. Two neurons can sit at the **same depth** and fire at the
**same amplitude**. Here are two such units, built to order: same place, same
peak-to-peak size, differing only in how big the **rebound** after the trough is. To
depth and amplitude they are one blob.
""",),
    code(r"""
probe = rec.probe
def one_unit(rebound, seed):
    wf = ps.spike_waveform(peak_ratio=rebound)              # bigger rebound after the trough
    tmpl = ps.make_template(probe, [0, 300], amplitude=230 / (wf.max() - wf.min()),  # same peak-to-peak
                            waveform=wf)
    rng = np.random.default_rng(seed)
    return np.array([tmpl + rng.normal(0, 8, tmpl.shape) for _ in range(120)])

snips = np.concatenate([one_unit(0.2, 1), one_unit(0.6, 2)])   # 120 spikes of each unit
lab = np.r_[np.zeros(120), np.ones(120)].astype(int)
_, amp2, dep2 = ps.localize(snips, probe)
pair_waves = ps.peak_waveforms(snips)             # each spike's waveform on its peak channel

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for u in [0, 1]:
    m = lab == u
    axes[0].plot(pair_waves[m][:20].T, color=ps.plotting.unit_color(u), lw=0.5, alpha=0.4)
    axes[0].plot(pair_waves[m].mean(0), color=ps.plotting.unit_color(u), lw=2.5, label=f"unit {u}")
    axes[1].scatter(dep2[m], amp2[m], s=10, color=ps.plotting.unit_color(u), label=f"unit {u}")
axes[0].set_title("two units' spikes (20 each) and their averages")
axes[0].set_xlabel("sample"); axes[0].set_ylabel("µV"); axes[0].legend()
axes[1].set_title("depth–amplitude: one blob")
axes[1].set_xlabel("depth (µm)"); axes[1].set_ylabel("amplitude (µV)"); axes[1].legend()
plt.tight_layout(); plt.show()
""",),
    md(r"""
The *shapes* are plainly different, so shape is the feature we're missing. But a
waveform is 61 numbers. Which **aspect** of the shape should we measure — the width?
the size of the rebound? where the trough falls? Rather than guess, we'll let the
data tell us which aspects of shape actually vary from spike to spike.

## 3. Shape, via PCA

Here are real waveforms from our recording, drawn with time running downwards and
laid out in two rows. Within each row the waveforms are ordered along one
**direction of variation**: in the top row they change in one way as you go from left
to right, in the bottom row in another. The two units from above are slotted into
each row at the position their shape puts them (one actual spike of each). The rows
were ordered by a method you'll build yourself in a moment.
""",),
    code(r"""
waveforms = ps.peak_waveforms(snippets)          # (n_spikes, 61): every real spike, peak channel
scores0, comps0, mean0 = ps.pca_features(waveforms, 2)   # the backend's version of what Exercise 2 builds
pair_mean = np.array([pair_waves[lab == u].mean(0) for u in [0, 1]])
pair_example = np.array([pair_waves[lab == u][0] for u in [0, 1]])   # one real spike of each unit
pair_scores0 = (pair_example - mean0) @ comps0.T          # the two spikes, in the same coordinates

def pick_along(axis, cand, n=6):
    # n spikes from `cand`, spread evenly along one direction
    targets = np.quantile(scores0[cand, axis], np.linspace(0.03, 0.97, n))
    return [int(cand[np.argmin(np.abs(scores0[cand, axis] - t))]) for t in targets]

typical = np.where(np.abs(scores0[:, 1] - np.median(scores0[:, 1])) < scores0[:, 1].std())[0]
biggest = np.argsort(amplitude)[-40:]              # the 40 largest spikes: cleanest shapes
rows = [pick_along(0, typical), pick_along(1, biggest)]

gallery = {}                                      # letter -> spike index, used again below
tt = np.arange(waveforms.shape[1])
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
for axis, ax, letters in [(0, axes[0], "abcdef"), (1, axes[1], "ghijkl")]:
    items = [(scores0[j, axis], waveforms[j], "k", L) for j, L in zip(rows[axis], letters)]
    items += [(pair_scores0[u, axis], pair_example[u], ps.plotting.unit_color(u), f"unit {u}") for u in [0, 1]]
    scale = 1.3 * max(np.abs(w).max() for _, w, _, _ in items)
    for k, (sc, w, col, L) in enumerate(sorted(items, key=lambda it: it[0])):
        ax.plot(k + w / scale, tt, color=col, lw=2 if col != "k" else 1.2)
        ax.text(k, 63, L, ha="center", va="top", color=col, fontweight="bold")
        if L.isalpha() and len(L) == 1:
            gallery[L] = int(np.where((waveforms == w).all(1))[0][0])
    ax.set_ylim(70, -2); ax.set_xticks([]); ax.set_ylabel("time (samples)")
    ax.set_title(f"waveforms ordered along direction {axis + 1}", loc="left")
plt.tight_layout(); plt.show()
""",),
    md(r"""
Look at what changes along each row. **Top row:** the waveforms mostly change in
**size**, and the two units sit side by side (same amplitude, so this direction
barely tells them apart). **Bottom row:** the size stays similar and the **shape**
changes — the balance between trough and rebound, and where the trough falls — and
now the two units land well apart. Those two directions are the first two
**principal components** of the waveforms, and they are all the shape features
we'll need.

**What PCA does.** It treats each waveform as a point in 61 dimensions and finds,
from the data, (1) the **mean waveform** and (2) a few **directions** in waveform
space along which the spikes differ most from each other. Every waveform is then
re-expressed as

$$\text{waveform} \approx \text{mean} + s_1 \times \text{PC1} + s_2 \times \text{PC2},$$

so each spike is described by its **scores** $s_1, s_2$ — two numbers instead of 61.
Since the spikes of one neuron are near-copies, a couple of directions capture almost
all the variation. **Left:** the **scree plot** — the fraction of variance across all
spikes that each component explains. **Right:** what that means for one spike (the
unit 1 spike from the gallery): its deviation from the mean, then that deviation
rebuilt from one, two and three components, each labelled with the share of the
spike's variance it captures. Each component adds a piece the previous ones missed.

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
ev = ps.explained_variance(waveforms, n_components=8)
_, comps3, mean3 = ps.pca_features(waveforms, 3)         # three components this time
x = pair_example[1] - mean3                               # the unit 1 spike, mean subtracted
s3 = x @ comps3.T                                         # its scores on PC1..3

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.1])
ax = fig.add_subplot(gs[:, 0])
ax.bar(np.arange(1, 9), ev)
ax.set_xlabel("principal component"); ax.set_ylabel("variance explained (all spikes)")
ax.set_title(f"scree plot: first 2 PCs capture {ev[:2].sum():.0%}")
panels = [fig.add_subplot(gs[i, 1]) for i in range(4)]
panels[0].plot(tt, x, color=ps.plotting.unit_color(1), lw=1.5)
panels[0].set_title("unit 1 spike, mean subtracted", loc="left", fontsize=10)
for k in range(1, 4):
    rebuilt = s3[:k] @ comps3[:k]
    captured = 1 - ((x - rebuilt) ** 2).sum() / (x ** 2).sum()
    panels[k].plot(tt, x, color=ps.plotting.unit_color(1), lw=1, alpha=0.4)
    panels[k].plot(tt, rebuilt, "k", lw=1.5)
    panels[k].set_title(f"rebuilt from PC1..{k}: captures {captured:.0%} of this spike's variance",
                        loc="left", fontsize=10)
for p_ in panels:
    p_.axhline(0, color="0.8", lw=0.5); p_.set_ylabel("µV")
    if p_ is not panels[-1]: p_.tick_params(labelbottom=False)
panels[-1].set_xlabel("sample")
plt.tight_layout(); plt.show()
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
Now the pieces. **Left column:** the mean waveform, PC1 and PC2 as shapes, and one
real spike from the gallery rebuilt as mean + $s_1$ × PC1 + $s_2$ × PC2 (what's left
over is noise). **Right:** every spike placed at its $(s_1, s_2)$ — its whole
61-sample waveform boiled down to two numbers — with the gallery waveforms lettered,
the two units from section 2 as stars, and a few of the actual waveforms drawn in
miniature next to their points.
""",),
    code(r"""
pca = PCA(n_components=2).fit(waveforms)          # the same PCA as in the exercise
pc1, pc2, mean_wave = pca.components_[0], pca.components_[1], pca.mean_
pair_scores = pca.transform(pair_example)         # the two units' spikes, in these coordinates

ex = gallery["l"]                                 # one gallery spike to take apart
s1, s2 = scores[ex]
fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(4, 2, width_ratios=[0.7, 1.6])
right_xlim = (-260, 330)                            # trim empty space on the right
left = [fig.add_subplot(gs[i, 0]) for i in range(4)]
for ax, y, col, name in [(left[0], mean_wave, "k", "mean waveform (µV)"),
                         (left[1], pc1, "tab:red", "PC1 (a shape; unit length)"),
                         (left[2], pc2, "tab:green", "PC2 (a shape; unit length)")]:
    ax.plot(tt, y, color=col, lw=2); ax.set_title(name, loc="left", fontsize=10)
    ax.axhline(0, color="0.8", lw=0.5); ax.tick_params(labelbottom=False)
left[3].plot(tt, waveforms[ex], "k", lw=2, label="spike l, as recorded")
left[3].plot(tt, mean_wave + s1 * pc1 + s2 * pc2, color="tab:purple", lw=2, ls="--",
             label="mean + s1·PC1 + s2·PC2")
left[3].set_title(f"spike l (µV) = mean {s1:+.0f}·PC1 {s2:+.0f}·PC2 + rest", loc="left", fontsize=10)
left[3].set_xlabel("sample"); left[3].legend(fontsize=8)

right = fig.add_subplot(gs[:, 1])
sc = right.scatter(scores[:, 0], scores[:, 1], s=8, c=amplitude, cmap="viridis")
plt.colorbar(sc, ax=right, label="amplitude (µV)", fraction=0.03, pad=0.01)
for L, j in gallery.items():
    right.text(scores[j, 0], scores[j, 1], L, fontsize=12, fontweight="bold", ha="center", va="center",
               bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="k", lw=0.8))
for u in [0, 1]:
    right.scatter(*pair_scores[u], marker="*", s=350, color=ps.plotting.unit_color(u),
                  edgecolor="k", lw=0.8, label=f"unit {u} (section 2)", zorder=5)
right.scatter(*scores[ex], s=450, facecolor="none", edgecolor="tab:purple", lw=3,
              label="spike l, rebuilt at left", zorder=4)

def mini_trace(ax, wave, at, color, dx, dy, width=80, height=50, tether="start"):
    # Draw `wave` as a little trace at data position at + (dx, dy), tethered to `at`
    # from the trace's start or end, whichever is nearer.
    x = at[0] + dx + tt / tt[-1] * width
    y = at[1] + dy + wave / 200 * height                  # ±200 µV -> ±height
    ax.plot(x, y, color=color, lw=1.2, zorder=6)
    k = 0 if tether == "start" else -1
    ax.plot([at[0], x[k]], [at[1], y[k]], color=color, lw=0.6, ls=":", zorder=6)

mini_trace(right, pair_example[0], pair_scores[0], ps.plotting.unit_color(0), dx=15, dy=-50)
mini_trace(right, pair_example[1], pair_scores[1], ps.plotting.unit_color(1), dx=15, dy=30)
mini_trace(right, waveforms[ex], scores[ex], "tab:purple", dx=20, dy=-70)
mini_trace(right, waveforms[gallery["b"]], scores[gallery["b"]], "k", dx=-105, dy=40, tether="end")
mini_trace(right, waveforms[gallery["e"]], scores[gallery["e"]], "k", dx=-30, dy=-150)
mini_trace(right, waveforms[gallery["f"]], scores[gallery["f"]], "k", dx=-30, dy=-80)

right.set_xlim(*right_xlim)
right.set_title("every spike as 2 numbers (s1, s2)"); right.set_xlabel("s1 (score on PC1)")
right.set_ylabel("s2 (score on PC2)"); right.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.show()
""",),
    md(r"""
Reading the left column: **PC1** looks like a spike, so adding or subtracting a
multiple of it scales the whole spike up or down — it captures **overall size**.
**PC2** is a lopsided wiggle around the trough, so it shifts the balance between
trough and rebound and moves the trough — it captures **shape**. A real spike is the
mean plus a multiple of each; what's left over is noise.

On the right, gallery letters a–f lie in order along the $s_1$ axis (the top row of
the gallery) and g–l in order along $s_2$ (the bottom row). The spikes do fall into
groups along $s_1$ — but look at the colour: $s_1$ is essentially tracking
**amplitude** (dark = small, yellow = large). So the separation along PC1 is the
*same* separation amplitude already gave us, just relabelled. The genuinely
*shape*-based axis is PC2, and for the recording's six units it barely separates
anything, because they all make similar biphasic spikes. That's fine — depth and
amplitude already did the job for them. The two stars, which depth and amplitude
could not separate, are where PC2 earns its keep.

## 4. Resolving the ambiguity

Back to the two units from section 2. Depth and amplitude saw one blob. Put every
one of *their* spikes into the same shape space (the recording's PC1 and PC2, the
axes of the scatter above) and look:
""",),
    code(r"""
sc2 = pca.transform(pair_waves)                   # scores on the recording's PC1 and PC2

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for u, rebound in [(0, 0.2), (1, 0.6)]:
    m = lab == u
    axes[0].plot(pair_mean[u], color=ps.plotting.unit_color(u), lw=2, label=f"unit {u} (rebound {rebound})")
    axes[1].scatter(dep2[m], amp2[m], s=10, color=ps.plotting.unit_color(u))
    axes[2].scatter(sc2[m, 0], sc2[m, 1], s=10, color=ps.plotting.unit_color(u), label=f"unit {u}")
axes[0].set_title("the two units' waveforms"); axes[0].set_xlabel("sample")
axes[0].set_ylabel("µV"); axes[0].legend(fontsize=8)
axes[1].set_title("depth–amplitude: one blob"); axes[1].set_xlabel("depth"); axes[1].set_ylabel("amp")
axes[2].set_title("shape space: two clusters"); axes[2].set_xlabel("s1 (score on PC1)"); axes[2].set_ylabel("s2 (score on PC2)")
axes[2].legend(); plt.tight_layout(); plt.show()
""",),
    md(r"""
Two clean clusters, separated along $s_2$ just as the two stars were. Depth and
amplitude, plus a couple of shape scores, is the feature set we'll cluster in the
next notebook.
""",),
    md(r"""
## Wrap-up

Each spike is now a handful of numbers: depth, amplitude, and a couple of shape
scores. In the depth–amplitude plane most units already stand out as separate
clouds, and the shape scores are there for the ones that don't.

**Next (Notebook 5 — clustering):** turn those clouds into labelled units, and
average each cluster into a template.
""",),
]

student, solution = build("04_features", cells)
print("wrote:", student)
print("wrote:", solution)
