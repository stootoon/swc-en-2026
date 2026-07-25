"""Notebook 2 -- Preprocessing: high-pass, common average reference, whitening."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 2 — Preprocessing

*SWC ENC 2026 · ephys-pop module*

The raw voltage is a mess of nuisances that bury the spikes. Before we detect
anything we clean it up, following Kilosort's preprocessing: **three problems, three
tools**, applied in order.

| nuisance | what it is | tool |
|---|---|---|
| slow drift / LFP | low-frequency signal shared across channels | **high-pass filter** |
| shared artefact | a fast signal identical on every channel (movement, reference) | **common average reference** |
| correlated noise | nearby channels share noise, so a threshold means different things on different channels | **whitening** |

**In this notebook you will:**
1. High-pass filter out the slow drift.
2. Common-average-reference away the shared artefact.
3. **Whiten** to remove the spatial noise correlations — the statistical heart of the notebook.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picosort as ps

rec = ps.make_recording(n_units=6, duration_s=20.0, seed=0)
raw = rec.traces
print("raw traces:", raw.shape)
""",),
    md(r"""
## 1. High-pass filtering

Spikes are fast events (~1 ms); the drift and LFP are slow (a few Hz). A **high-pass
filter** keeps the fast part and removes the slow part. We'll use a Butterworth
filter at a 300 Hz cutoff — the standard choice for the Neuropixels AP band. The
clearest way to see what it does is the **power spectrum**: the low-frequency
mountain disappears.
""",),
    code(r"""
from scipy.signal import welch

filt_hp = ps.highpass_filter(raw, rec.fs, cutoff=300)

f, praw = welch(raw[:, 10], fs=rec.fs, nperseg=4096)
f, php = welch(filt_hp[:, 10], fs=rec.fs, nperseg=4096)
plt.figure(figsize=(6, 3.6))
plt.semilogy(f, praw, label="raw")
plt.semilogy(f, php, label="high-passed")
plt.axvline(300, color="k", ls="--", lw=0.8, label="300 Hz cutoff")
plt.xlim(0, 2000); plt.xlabel("frequency (Hz)"); plt.ylabel("power")
plt.legend(); plt.title("high-pass removes the low-frequency drift"); plt.show()
""",),
    md(r"""
## 2. Common average reference

Some artefacts appear *identically* on every channel — a movement, a reference
fluctuation. Because they're the same everywhere, we can estimate them by the
**across-channel average** at each moment and subtract it. We use the **median**
rather than the mean so that a big spike on one channel can't drag the reference.

**Exercise 1** *(~3 min · easy)*. Complete `common_average_reference`: subtract the median
across channels (axis 1) at every time sample.

> **Check / unstuck.** After CAR, the per-sample median across channels should be
> ~0. Stuck? Use `ps.common_average_reference(filt_hp)`.
""",),
    code(
        solution=r"""
def common_average_reference(traces):
    return traces - np.median(traces, axis=1, keepdims=True)

filt = common_average_reference(filt_hp)
print("median across channels before CAR:", np.median(np.abs(np.median(filt_hp, axis=1))).round(2))
print("median across channels after  CAR:", np.median(np.abs(np.median(filt, axis=1))).round(4))
""",
        student=r"""
def common_average_reference(traces):
    # YOUR CODE HERE: subtract the across-channel median at each time sample.
    # Hint: np.median(traces, axis=1, keepdims=True)
    raise NotImplementedError

filt = common_average_reference(filt_hp)
print("median across channels before CAR:", np.median(np.abs(np.median(filt_hp, axis=1))).round(2))
print("median across channels after  CAR:", np.median(np.abs(np.median(filt, axis=1))).round(4))
""",
    ),
    md(r"""
## 3. Whitening

One nuisance remains, and it's the subtle one. The background noise is **spatially
correlated**: nearby channels tend to wiggle together. Look at the **noise
covariance** — the matrix of how much each pair of channels co-varies. It has a
bright band along the diagonal, meaning neighbours are correlated.
""",),
    code(r"""
cov = ps.noise_covariance(filt)
ps.plotting.plot_covariance(cov, title="noise covariance (before whitening)")
plt.show()
""",),
    md(r"""
Correlated noise is a problem for detection: a fluctuation shared by five channels
looks like a big multi-channel event, easy to mistake for a spike, and a threshold
of "5 sigma" means different things on different channels. **Whitening** fixes this.
It applies a linear transform $W$ that makes the noise covariance the identity —
equal variance on every channel, zero correlation between them.

The transform is $W = C^{-1/2}$, built from the eigen-decomposition of the
covariance $C = V \Lambda V^\top$: take $W = V \Lambda^{-1/2} V^\top$. (This
symmetric "ZCA" form keeps channels in place, so a whitened trace still reads
channel-by-channel like the original — just decorrelated.)

<details>
<summary><b>▸ The math: why <i>W = C</i><sup>−1/2</sup> whitens (optional)</b></summary>

Write the noise on the array at one time as a vector $n$ with covariance
$C = \mathbb{E}[n\,n^\top]$. We want a linear map $W$ so the transformed noise
$z = Wn$ has **identity** covariance — unit variance on every channel, zero
correlation between them:

$$\mathrm{Cov}(z) = \mathbb{E}[Wn\,n^\top W^\top] = W\,C\,W^\top = I.$$

Any $W$ with $W^\top W = C^{-1}$ works, so the solution is only fixed up to a
rotation. Using the eigen-decomposition $C = V\Lambda V^\top$ (with $V$ orthonormal
and $\Lambda$ the positive eigenvalues), take the **symmetric** root

$$W = C^{-1/2} = V\,\Lambda^{-1/2}\,V^\top .$$

Then $W C W^\top = V\Lambda^{-1/2}V^\top\,V\Lambda V^\top\,V\Lambda^{-1/2}V^\top =
V\,\Lambda^{-1/2}\Lambda\,\Lambda^{-1/2}\,V^\top = VV^\top = I.$ ✓

Two footnotes. **(i) Why symmetric (ZCA) and not PCA whitening?** $W = \Lambda^{-1/2}V^\top$
*also* whitens, but it rotates the data into the eigenbasis, scrambling channel
identity; the symmetric root is the whitening matrix **closest to the identity**
(it minimises $\lVert W - I\rVert$), so a whitened trace still looks channel-by-channel
like the original. **(ii) The $\epsilon$** in the code regularises tiny eigenvalues:
$1/\sqrt{\lambda}$ blows up when $\lambda\to 0$, amplifying directions that are all
noise, so we add a small floor.

This also connects forward to **Notebook 6**: under Gaussian noise with covariance
$C$, the statistically optimal detector for a template $s$ is the *whitened* matched
filter $s^\top C^{-1} x$. Whitening the data first lets us then use a plain matched
filter and get the optimal detector for free.
</details>

**Exercise 2** *(~8 min · meaty)*. Complete `whitening_matrix`: eigen-decompose the covariance
with `np.linalg.eigh`, and form $V \,\mathrm{diag}(1/\sqrt{\lambda+\epsilon})\, V^\top$.
Then we apply it and check the covariance is now diagonal.

> **Check / unstuck.** After whitening, the covariance should be ~the identity:
> bright diagonal, near-zero off-diagonal. Stuck? Use `ps.whitening_matrix(cov)`.
""",),
    code(
        solution=r"""
def whitening_matrix(cov, eps=1e-6):
    vals, vecs = np.linalg.eigh(cov)
    reg = eps * vals.max()
    return vecs @ np.diag(1.0 / np.sqrt(vals + reg)) @ vecs.T

W = whitening_matrix(cov)
whitened = filt @ W
cov_after = ps.noise_covariance(whitened)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ps.plotting.plot_covariance(cov, ax=axes[0], title="before whitening")
ps.plotting.plot_covariance(cov_after, ax=axes[1], title="after whitening")
plt.tight_layout(); plt.show()
print("mean |off-diagonal| before:", round(np.abs(cov - np.diag(np.diag(cov))).mean(), 3))
print("mean |off-diagonal| after :", round(np.abs(cov_after - np.diag(np.diag(cov_after))).mean(), 3))
""",
        student=r"""
def whitening_matrix(cov, eps=1e-6):
    vals, vecs = np.linalg.eigh(cov)
    reg = eps * vals.max()
    # YOUR CODE HERE: return V @ diag(1/sqrt(vals + reg)) @ V.T
    raise NotImplementedError

W = whitening_matrix(cov)
whitened = filt @ W
cov_after = ps.noise_covariance(whitened)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ps.plotting.plot_covariance(cov, ax=axes[0], title="before whitening")
ps.plotting.plot_covariance(cov_after, ax=axes[1], title="after whitening")
plt.tight_layout(); plt.show()
print("mean |off-diagonal| before:", round(np.abs(cov - np.diag(np.diag(cov))).mean(), 3))
print("mean |off-diagonal| after :", round(np.abs(cov_after - np.diag(np.diag(cov_after))).mean(), 3))
""",
    ),
    md(r"""
The off-diagonal correlations collapse: whitened noise is spatially flat, so a
detection threshold means the same thing on every channel.

## Putting it together

The whole pipeline — high-pass → CAR → whiten — is `ps.preprocess`. It returns two
things we'll both use: the **whitened** traces (for *detecting* spikes, where a
uniform threshold is what we want) and the **filtered** traces (high-passed and
CAR'd but not whitened, in real microvolts, where a spike's spatial footprint is
undistorted — the right place to *measure* waveforms and build templates).
""",),
    code(r"""
whitened, filtered, W = ps.preprocess(rec)
print("whitened:", whitened.shape, " filtered:", filtered.shape)

# The same spike, now clearly above the flat whitened noise.
gt = rec.ground_truth
u = int(np.argmax(gt.unit_amplitude)); t = gt.spike_times[gt.spike_labels == u][5]
pc = ps.peak_channel(gt.templates[u]); lo = max(0, pc - 6)
ps.plotting.plot_signal(whitened, rec.fs, channels=range(lo, lo + 13),
                        t0=(t - 200) / rec.fs, t1=(t + 250) / rec.fs,
                        title="a spike in the whitened traces")
plt.show()
""",),
    md(r"""
## Wrap-up

Three nuisances, three tools: high-pass killed the drift, CAR removed the shared
artefact, and whitening flattened the spatial noise correlations. The traces are now
clean enough that a spike is a sharp deflection standing clear of the noise.

**Next (Notebook 3 — detection):** put a threshold on these whitened traces, find
the crossings, and cut out a snippet around each detected spike.
""",),
]

student, solution = build("02_preprocessing", cells)
print("wrote:", student)
print("wrote:", solution)
