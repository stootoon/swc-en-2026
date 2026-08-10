"""Notebook 7 -- Merging and cleanup with correlograms and template similarity."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 7 — Merging & cleanup

*SWC ENC 2026 · ephys-pop module*

A sorter's raw output is rarely final. A common error is **over-splitting**: one neuron
gets divided into two "units" (say its bigger and smaller spikes fall in different
clusters). We need to catch that and **merge** them. The tools are spike **timing** and
spike **shape** — and we'll build the timing tool, the **correlogram**, from scratch.

**In this notebook you will:**
1. See a unit as a **spike train**, and learn to read a **correlogram** from toy examples.
2. Build the correlogram yourself and find the **refractory hole** of a clean unit.
3. Diagnose an **over-split** and merge it; and see when *not* to merge.

We use a longer (60 s) recording so the correlograms are well populated.
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

u = int(np.bincount(labels).argmax())          # a well-sampled unit to work with
t_u = times[labels == u]
print(f"sorted {len(times)} spikes into {len(np.unique(labels))} units; unit {u} has {len(t_u)}")
""",),
    md(r"""
## 1. A unit is a spike train

The end product of sorting is, for each unit, a **spike train** — simply the list of
times it fired. Drawn as a raster (one tick per spike), unit `u` looks like this:
""",),
    code(r"""
plt.figure(figsize=(11, 1.5))
plt.eventplot(t_u / fs, colors="k", lineoffsets=0, linelengths=1)
plt.xlim(0, 10); plt.yticks([]); plt.xlabel("time (s)")
plt.title(f"unit {u}'s spike train — each tick is one spike (first 10 s)")
plt.show()
""",),
    md(r"""
## 2. The correlogram: reading the gaps between spikes

To find structure in a spike train we look at the **time gaps** between spikes. The
**correlogram** of two trains does this systematically: for every spike in train A, it
measures the time difference to each nearby spike in train B, and **histograms** all
those differences. A train compared with *itself* gives the **auto-correlogram** — the
distribution of gaps between a unit's own spikes.

The shape tells you a lot. Two toy trains make the two key readings concrete: a purely
**random** (Poisson) train, and the same train with a **refractory period** imposed —
no spike allowed within 2 ms of the previous one, as for a real neuron.
""",),
    code(r"""
# a purely random train, and one with a 2 ms refractory period enforced
rng = np.random.default_rng(1)
T = rec.duration_s
random_train = np.sort(rng.integers(0, int(T * fs), rng.poisson(15 * T)))    # ~15 Hz, random
refr = int(0.002 * fs)
keep = [random_train[0]]
for t in random_train[1:]:
    if t - keep[-1] > refr:                     # drop spikes too soon after the last kept one
        keep.append(t)
refractory_train = np.array(keep)

fig, ax = plt.subplots(2, 2, figsize=(11, 5))
for j, (train, name, col) in enumerate([(random_train, "random (Poisson)", "0.4"),
                                        (refractory_train, "with 2 ms refractory period", "tab:green")]):
    ts = train / fs; m = (ts >= 10) & (ts < 11)
    ax[0, j].eventplot(ts[m], colors=col, lineoffsets=0, linelengths=1)
    ax[0, j].set_xlim(10, 11); ax[0, j].set_yticks([]); ax[0, j].set_xlabel("time (s)")
    ax[0, j].set_title(f"{name} — spike train (1 s)")
    c, n = ps.correlogram(train, train, fs, exclude_zero=True)
    ps.plotting.plot_correlogram(c, n, ax=ax[1, j], color=col, title="its auto-correlogram")
plt.tight_layout(); plt.show()
""",),
    md(r"""
Read the two auto-correlograms. The **random** train's is **flat**: knowing one spike
tells you nothing about when the next comes, so every lag is equally likely — that flat
line is the *chance* level. The **refractory** train's has a **hole at zero**: because
no two spikes come within 2 ms, there are simply no small-lag pairs to count. That hole
is the fingerprint of a *single real neuron*, and it's what we'll hunt for.

Now build the correlogram yourself.

<details>
<summary><b>▸ Go deeper: what a correlogram measures, and the flat baseline (optional)</b></summary>

Each unit is a **point process** — a random set of spike times. The cross-correlogram
estimates the **cross-intensity**: given a spike of unit A, the rate of unit B's spikes
at a lag $\tau$ later. If the two are **independent Poisson** processes with rates
$\lambda_A, \lambda_B$, the expected count in a bin of width $\Delta$ over a recording of
length $T$ is flat,

$$\mathbb{E}[\text{count at lag }\tau] \;=\; \lambda_A\,\lambda_B\,\Delta\,T,$$

independent of $\tau$ — the chance baseline. The **refractory hole** is the sharpest
departure from it: a neuron's biophysics forbid a second spike within $\sim$1–2 ms, so
its auto-correlogram is pinned near zero at small lags — something no pair of *distinct*
neurons produces. A pooled train's **refractory-violation rate** (fraction of
inter-spike intervals below $\sim$1.5 ms) is the same idea as a single number.
</details>

**Exercise 1** *(~7 min)*. Complete `correlogram`: for each spike in `ta`, take the
differences to the spikes of `tb` within `±window_ms`, and accumulate them into a
histogram. Return the bin centres (ms) and counts. We then apply it to unit `u`.

> **Check / unstuck.** Unit `u`'s auto-correlogram should show the refractory hole at 0,
> just like the toy. Stuck? Use `ps.correlogram(ta, tb, fs)`.
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
    return (edges[:-1] + edges[1:]) / 2, counts

# show unit u's spikes (the data) and its auto-correlogram (the result) together
fig, ax = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={"height_ratios": [1, 3]})
ax[0].eventplot(t_u / fs, colors="k", lineoffsets=0, linelengths=1)
ax[0].set_xlim(20, 25); ax[0].set_yticks([]); ax[0].set_title(f"unit {u} spikes (5 s)")
centers, counts = correlogram(t_u, t_u, fs, exclude_zero=True)
ps.plotting.plot_correlogram(centers, counts, ax=ax[1], title=f"unit {u} auto-correlogram")
plt.tight_layout(); plt.show()
print(f"unit {u}: refractory violations = {ps.refractory_violations(t_u, fs):.3f}")
""",
        student=r"""
def correlogram(ta, tb, fs, bin_ms=0.5, window_ms=25.0, exclude_zero=False):
    ta = np.sort(ta) / fs * 1e3
    tb = np.sort(tb) / fs * 1e3
    edges = np.arange(-window_ms, window_ms + bin_ms, bin_ms)
    counts = np.zeros(len(edges) - 1)
    for t in ta:
        # YOUR CODE HERE: differences of tb (within +-window_ms of t) minus t; if
        # exclude_zero drop the ~0 self-match; accumulate np.histogram(d, edges)[0].
        raise NotImplementedError
    return (edges[:-1] + edges[1:]) / 2, counts

fig, ax = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={"height_ratios": [1, 3]})
ax[0].eventplot(t_u / fs, colors="k", lineoffsets=0, linelengths=1)
ax[0].set_xlim(20, 25); ax[0].set_yticks([]); ax[0].set_title(f"unit {u} spikes (5 s)")
centers, counts = correlogram(t_u, t_u, fs, exclude_zero=True)
ps.plotting.plot_correlogram(centers, counts, ax=ax[1], title=f"unit {u} auto-correlogram")
plt.tight_layout(); plt.show()
print(f"unit {u}: refractory violations = {ps.refractory_violations(t_u, fs):.3f}")
""",
    ),
    md(r"""
Just like the toy: unit `u` has the refractory hole, so it's a single, clean neuron.

## 3. Catching an over-split

Real sorters often *over*-cluster on purpose and merge afterwards — splitting is easy to
undo, but un-merging a true collision is not. Let's manufacture the classic mistake:
split unit `u` into two "units" by amplitude — its bigger spikes (**A**) and smaller
spikes (**B**), exactly what a slightly-too-eager clusterer would do. **First, look at
the two spike trains:**
""",),
    code(r"""
med = np.median(amps[labels == u])
tA, tB = t_u[amps[labels == u] >= med], t_u[amps[labels == u] < med]

plt.figure(figsize=(11, 1.8))
plt.eventplot([tB / fs, tA / fs], colors=["tab:orange", "tab:blue"], lineoffsets=[0, 1], linelengths=0.8)
plt.xlim(20, 25); plt.yticks([0, 1], ["B (small)", "A (big)"]); plt.xlabel("time (s)")
plt.title(f'unit {u} split into A and B — they interleave, but never fire at the same instant')
plt.show()
print(f"A: {len(tA)} spikes,  B: {len(tB)} spikes")
""",),
    md(r"""
Look carefully at the raster: A ticks and B ticks are never on top of each other. They
can't be — they're the same neuron, which can't fire twice at once. That's what the
**cross-correlogram** (A vs B) makes quantitative: it should have the same **refractory
hole** at zero. And because A and B are the same neuron, their **templates** should be
nearly identical. Here's the data behind both tests:
""",),
    code(r"""
pc = ps.peak_channel(template_of(tA))
cAB, nAB = ps.correlogram(tA, tB, fs)
simAB = np.corrcoef(template_of(tA).ravel(), template_of(tB).ravel())[0, 1]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(template_of(tA)[pc], color="tab:blue", label="A template")
ax[0].plot(template_of(tB)[pc], color="tab:orange", label="B template")
ax[0].set_title(f"templates on channel {pc} — nearly identical"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("sample"); ax[0].set_ylabel("µV")
ps.plotting.plot_correlogram(cAB, nAB, ax=ax[1], title="A × B cross-correlogram — hole at 0")
plt.tight_layout(); plt.show()
print(f"template similarity A vs B = {simAB:.3f}  (near 1 -> same neuron -> MERGE)")
""",),
    md(r"""
Both tests agree: the A×B cross-correlogram has the refractory **hole**, and the
templates lie on top of each other. A and B are one neuron, over-split — **merge them.**

## 4. Don't over-merge: two genuinely different units

The same tests must say *no* for two *different* neurons. Take unit `u` and another unit
`u2`. **Again, start with the spike trains** — now the two rasters are unrelated, and
they *do* occasionally coincide (independent neurons fire together by chance):
""",),
    code(r"""
u2 = int([v for v in np.unique(labels) if v != u][0])
t_u2 = times[labels == u2]

plt.figure(figsize=(11, 1.8))
plt.eventplot([t_u2 / fs, t_u / fs], colors=["tab:red", "k"], lineoffsets=[0, 1], linelengths=0.8)
plt.xlim(20, 25); plt.yticks([0, 1], [f"unit {u2}", f"unit {u}"]); plt.xlabel("time (s)")
plt.title(f"two different neurons — unrelated trains that sometimes coincide")
plt.show()
""",),
    md(r"""
**Exercise 2** *(~5 min)*. Compute the **template similarity** between units `u` and `u2`
(correlation of `template_of(t_u)` and `template_of(t_u2)`), and compare it to the A–B
value. The figure shows both units' templates, and the A×B vs u×u2 cross-correlograms.

> **Check / unstuck.** A–B similarity ≈ 0.98 (merge); u–u2 similarity is much lower —
> often near 0 or negative (keep separate). Stuck? `np.corrcoef(x.ravel(), y.ravel())[0, 1]`.
""",),
    code(
        solution=r"""
sim_diff = np.corrcoef(template_of(t_u).ravel(), template_of(t_u2).ravel())[0, 1]
cD, nD = ps.correlogram(t_u, t_u2, fs)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(template_of(t_u)[ps.peak_channel(template_of(t_u))], "k", label=f"unit {u}")
ax[0].plot(template_of(t_u2)[ps.peak_channel(template_of(t_u2))], "tab:red", label=f"unit {u2}")
ax[0].set_title("templates — different shapes/positions"); ax[0].legend(fontsize=8); ax[0].set_xlabel("sample")
ps.plotting.plot_correlogram(cD, nD, ax=ax[1], color="tab:red",
                             title=f"unit {u} × unit {u2}: no hole")
plt.tight_layout(); plt.show()
print(f"template similarity  A vs B          = {simAB:.3f}  -> merge")
print(f"template similarity  unit {u} vs unit {u2} = {sim_diff:.3f}  -> keep separate")
""",
        student=r"""
# YOUR CODE HERE: sim_diff = correlation of template_of(t_u) and template_of(t_u2)
sim_diff = ...
cD, nD = ps.correlogram(t_u, t_u2, fs)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(template_of(t_u)[ps.peak_channel(template_of(t_u))], "k", label=f"unit {u}")
ax[0].plot(template_of(t_u2)[ps.peak_channel(template_of(t_u2))], "tab:red", label=f"unit {u2}")
ax[0].set_title("templates — different shapes/positions"); ax[0].legend(fontsize=8); ax[0].set_xlabel("sample")
ps.plotting.plot_correlogram(cD, nD, ax=ax[1], color="tab:red",
                             title=f"unit {u} × unit {u2}: no hole")
plt.tight_layout(); plt.show()
print(f"template similarity  A vs B          = {simAB:.3f}  -> merge")
print(f"template similarity  unit {u} vs unit {u2} = {sim_diff:.3f}  -> keep separate")
""",
    ),
    md(r"""
The tests cleanly separate the cases: **merge when the templates match and the
cross-correlogram has a refractory hole; keep separate otherwise.** That is the logic
behind the manual **curation** every real spike-sorting pipeline still relies on — and,
increasingly, behind its automated merge steps.

## Wrap-up

Timing and shape statistics — the correlogram, the refractory period, template
similarity — let you audit a sort with no ground truth in sight: catch over-splits,
merge them, and flag contaminated units.

**Next (Notebook 8 — scoring):** the one thing we *can* do because our data is synthetic
— grade the whole sort against the truth.
""",),
]

student, solution = build("07_merging_cleanup", cells)
print("wrote:", student)
print("wrote:", solution)
