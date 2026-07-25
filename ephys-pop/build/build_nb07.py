"""Notebook 7 -- Merging and cleanup with correlograms and template similarity."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 7 — Merging & cleanup

*SWC ENC 2026 · ephys-pop module*

A sorter's raw output is rarely final. A common error is **over-splitting**: one
neuron gets divided into two "units" (say its bigger and smaller spikes fall in
different clusters). We need to catch that and **merge** them. Two tools, both built
on spike *timing* and *shape*:

- the **correlogram** — a histogram of the time gaps between two units' spikes. A real
  neuron obeys a **refractory period** (~1–2 ms of silence after each spike), so an
  over-split pair, being one neuron, shows a tell-tale **hole at zero lag**;
- **template similarity** — two clusters of the same neuron have near-identical
  templates; two different neurons don't.

**In this notebook you will:**
1. Build a **correlogram** and see the refractory hole of a clean unit.
2. Diagnose an **over-split** from its cross-correlogram and template similarity.
3. Contrast it with two genuinely different units, which must *not* be merged.

We use a longer (60 s) recording here so the correlograms are well populated.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picosort as ps

rec = ps.make_recording(n_units=6, duration_s=60.0, seed=0)
res = ps.run_picosort(rec)
times, labels, amps = res.spike_times, res.spike_labels, res.spike_amplitudes
filtered, fs = res.filtered, rec.fs

def template_of(spike_times):
    # average template of any set of spike times, from the filtered traces
    snippets, _ = ps.extract_snippets(filtered, spike_times, align=False)
    return snippets.mean(axis=0)

print(f"sorted {len(times)} spikes into {len(np.unique(labels))} units")
""",),
    md(r"""
## 1. The refractory period and the auto-correlogram

The **correlogram** of two spike trains histograms every gap (spike in one) − (spike
in the other). A unit's **auto-correlogram** (itself vs itself) reveals the
refractory period as a **hole at zero lag**: the neuron never fires twice within
~1–2 ms.

<details>
<summary><b>▸ The math: what a correlogram measures, and the flat baseline (optional)</b></summary>

Think of each unit as a **point process** — a random set of spike times. The
cross-correlogram estimates the **cross-intensity**: given a spike of unit A, the
rate of unit B's spikes at a lag $\tau$ later. If the two units are **independent
Poisson** processes with rates $\lambda_A, \lambda_B$, then B's spikes near an A spike
are just B's ordinary spikes — no relationship — so the expected count in a bin of
width $\Delta$ over a recording of length $T$ is flat:

$$\mathbb{E}[\text{count in bin at lag }\tau] \;=\; \lambda_A\,\lambda_B\,\Delta\,T,$$

independent of $\tau$. That flat line is the **chance baseline**; structure is any
departure from it. The **refractory hole** is the sharpest such departure: a real
neuron's own biophysics forbid a second spike within $\sim$1–2 ms, so its
auto-correlogram is pinned near zero at small lags — something no pair of *distinct*
neurons produces. Two spike trains whose cross-correlogram has that hole therefore
can't be two neurons: they're one, over-split (the merge rule of section 2). A pooled
train's **refractory-violation rate** — the fraction of inter-spike intervals below
$\sim$1.5 ms — is the same idea reduced to a single number.
</details>

**Exercise 1** *(~7 min)*. Complete `correlogram`: for each spike in `ta`, add the
differences to all spikes of `tb` that fall within `±window_ms` into a histogram.
Return the bin centres (ms) and counts.

> **Check / unstuck.** A clean unit's auto-correlogram has a clear dip around 0.
> Stuck? Use `ps.correlogram(ta, tb, fs)`.
""",),
    code(
        solution=r"""
def correlogram(ta, tb, fs, bin_ms=0.5, window_ms=25.0, exclude_zero=False):
    ta = np.sort(ta) / fs * 1e3
    tb = np.sort(tb) / fs * 1e3
    edges = np.arange(-window_ms, window_ms + bin_ms, bin_ms)
    counts = np.zeros(len(edges) - 1)
    for t in ta:
        d = tb[(tb >= t - window_ms) & (tb <= t + window_ms)] - t
        if exclude_zero:
            d = d[np.abs(d) > 1e-9]
        counts += np.histogram(d, edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts

u = np.bincount(labels).argmax()
t_u = times[labels == u]
centers, counts = correlogram(t_u, t_u, fs, exclude_zero=True)
ps.plotting.plot_correlogram(centers, counts, title=f"unit {u} auto-correlogram (hole at 0)")
plt.show()
print(f"unit {u}: refractory violations = {ps.refractory_violations(t_u, fs):.3f}")
""",
        student=r"""
def correlogram(ta, tb, fs, bin_ms=0.5, window_ms=25.0, exclude_zero=False):
    ta = np.sort(ta) / fs * 1e3
    tb = np.sort(tb) / fs * 1e3
    edges = np.arange(-window_ms, window_ms + bin_ms, bin_ms)
    counts = np.zeros(len(edges) - 1)
    for t in ta:
        # YOUR CODE HERE: differences (tb within +-window_ms of t) minus t; if
        # exclude_zero drop the ~0 self-match; accumulate np.histogram(d, edges)[0].
        raise NotImplementedError
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts

u = np.bincount(labels).argmax()
t_u = times[labels == u]
centers, counts = correlogram(t_u, t_u, fs, exclude_zero=True)
ps.plotting.plot_correlogram(centers, counts, title=f"unit {u} auto-correlogram (hole at 0)")
plt.show()
print(f"unit {u}: refractory violations = {ps.refractory_violations(t_u, fs):.3f}")
""",
    ),
    md(r"""
## 2. Catching an over-split

Real sorters often *over*-cluster on purpose and merge afterwards — splitting is easy
to undo, but un-merging a true collision is not. Let's manufacture the classic
mistake: split unit `u` into two "units" by amplitude — its bigger spikes (**A**) and
smaller spikes (**B**), exactly what a slightly-too-eager clusterer would do.
""",),
    code(r"""
a_u = amps[labels == u]; med = np.median(a_u)
tA, tB = t_u[a_u >= med], t_u[a_u < med]
print(f"split unit {u} -> A ({len(tA)} spikes) and B ({len(tB)} spikes)")

# Two tells that A and B are the SAME neuron.
cA, nA = ps.correlogram(tA, tB, fs)                    # cross-correlogram
simAB = np.corrcoef(template_of(tA).ravel(), template_of(tB).ravel())[0, 1]

ps.plotting.plot_correlogram(cA, nA, title=f"A × B cross-correlogram (hole at 0 -> same neuron)")
plt.show()
print(f"template similarity A vs B = {simAB:.3f}  (near 1 -> same neuron)")
""",),
    md(r"""
Both tells agree: the A×B cross-correlogram has a **hole at zero** (A and B never fire
within a refractory period of each other — impossible for two neurons, expected for
one), and their **templates are nearly identical**. Merge them.

## 3. Don't over-merge: two genuinely different units

The same two tests must say *no* for two different neurons. Pick another unit and
compare. Its cross-correlogram with unit `u` has **no hole** (independent neurons do
fire close together by chance), and its template is **different**.

**Exercise 2** *(~5 min)*. Compute the template similarity between unit `u` and a different
unit `u2`, and compare to the A–B value. Decide which pair to merge.

> **Check / unstuck.** A–B similarity ≈ 0.98 (merge); u–u2 similarity is much lower,
> often near 0 or negative (keep separate).
""",),
    code(
        solution=r"""
v = np.unique(labels); u2 = v[v != u][0]
t_u2 = times[labels == u2]

sim_diff = np.corrcoef(template_of(t_u).ravel(), template_of(t_u2).ravel())[0, 1]
cD, nD = ps.correlogram(t_u, t_u2, fs)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ps.plotting.plot_correlogram(cA, nA, ax=axes[0], title=f"A × B: hole (merge)")
ps.plotting.plot_correlogram(cD, nD, ax=axes[1], color="tab:orange",
                             title=f"unit {u} × unit {u2}: no hole (keep)")
plt.tight_layout(); plt.show()

print(f"template similarity  A vs B          = {simAB:.3f}  -> merge")
print(f"template similarity  unit {u} vs unit {u2} = {sim_diff:.3f}  -> keep separate")
""",
        student=r"""
v = np.unique(labels); u2 = v[v != u][0]
t_u2 = times[labels == u2]

# YOUR CODE HERE: sim_diff = correlation between template_of(t_u) and template_of(t_u2)
sim_diff = ...
cD, nD = ps.correlogram(t_u, t_u2, fs)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ps.plotting.plot_correlogram(cA, nA, ax=axes[0], title=f"A × B: hole (merge)")
ps.plotting.plot_correlogram(cD, nD, ax=axes[1], color="tab:orange",
                             title=f"unit {u} × unit {u2}: no hole (keep)")
plt.tight_layout(); plt.show()

print(f"template similarity  A vs B          = {simAB:.3f}  -> merge")
print(f"template similarity  unit {u} vs unit {u2} = {sim_diff:.3f}  -> keep separate")
""",
    ),
    md(r"""
The two tests agree and cleanly separate the cases: **merge when templates match and
the cross-correlogram has a refractory hole; keep separate otherwise.** That is the
logic behind the manual **curation** every real spike-sorting pipeline still relies
on — and, increasingly, behind its automated merge steps.

## Wrap-up

Timing and shape statistics — the correlogram, the refractory period, template
similarity — let you audit a sort with no ground truth in sight: catch over-splits,
merge them, and flag contaminated units.

**Next (Notebook 8 — scoring):** the one thing we *can* do because our data is
synthetic — grade the whole sort against the truth.
""",),
]

student, solution = build("07_merging_cleanup", cells)
print("wrote:", student)
print("wrote:", solution)
