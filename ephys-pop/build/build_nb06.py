"""Notebook 6 -- Template matching by matching pursuit."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 6 — Template matching

*SWC ENC 2026 · ephys-pop module*

Clustering assigns each detected spike to one unit — but when two neurons fire almost
at once, their waveforms **add** into a shape that belongs to neither, and detection
either mislabels it or misses one of the two. **Matching pursuit** fixes this. Armed
with the templates from Notebook 5, it explains the trace as a **sum of templates**,
peeling them off one at a time, so overlapping spikes are separated.

**In this notebook you will:**
1. See a **collision** — two overlapping spikes — that clustering gets wrong.
2. Build the **matched filter**: find a known shape hidden in a trace.
3. Run **matching pursuit** — greedily subtract the best-fitting template — and watch
   the collision come apart.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picosort as ps

rec = ps.make_recording(n_units=6, duration_s=20.0, seed=0)
whitened, filtered, W = ps.preprocess(rec)
# Reuse the pipeline through clustering to get templates (built in Notebook 5).
res = ps.run_picosort(rec)
templates = res.templates
print(f"{len(templates)} templates, each {templates.shape[1]}×{templates.shape[2]} (channels×samples)")
""",),
    md(r"""
## 1. A collision

Picture two neurons sitting close together on the probe (so they share channels), with
**different** spike shapes. Most of the time they fire seconds apart and detection sees
two clean spikes. But every so often they fire within a fraction of a millisecond — and
then their waveforms **overlap and add** into a single lump that matches *neither*
template. Here is exactly that, built from two shapes so we can see each ingredient:
""",),
    code(r"""
probe = rec.probe
L = templates.shape[2]            # template length in samples
half = L // 2

# two neurons at the SAME location (so they share channels) but with different shapes
tmplA = ps.make_template(probe, [0.0, 300.0], amplitude=170,
                         waveform=ps.spike_waveform(trough_width=2.2, peak_ratio=0.35))
tmplB = ps.make_template(probe, [0.0, 300.0], amplitude=130,
                         waveform=ps.spike_waveform(trough_width=4.0, peak_ratio=0.6))
pc = ps.peak_channel(tmplA)       # the shared peak channel

rng = np.random.default_rng(0)
seg = rng.normal(0, 8.0, size=(160, probe.n_channels)).astype(float)   # a noisy blank patch
tA, tB = 40, 52                   # the two spikes fire ~0.4 ms apart -> they overlap
seg[tA:tA + L] += tmplA.T
seg[tB:tB + L] += tmplB.T

compA = np.zeros(160); compA[tA:tA + L] = tmplA[pc]
compB = np.zeros(160); compB[tB:tB + L] = tmplB[pc]
fig, ax = plt.subplots(3, 1, figsize=(9, 5), sharex=True, sharey=True)
ax[0].plot(compA, color=ps.plotting.unit_color(0)); ax[0].set_title("spike from neuron A (narrow)")
ax[1].plot(compB, color=ps.plotting.unit_color(1)); ax[1].set_title("spike from neuron B (broad), 0.4 ms later")
ax[2].plot(seg[:, pc], "k"); ax[2].set_title("recorded = A + B + noise — a lump that matches neither")
ax[-1].set_xlabel("sample"); [a.set_ylabel("µV") for a in ax]
plt.tight_layout(); plt.show()
""",),
    md(r"""
The bottom trace is what the electrode actually records: the **sum**. Clustering, which
must give this one lump a single label, is stuck. Matching pursuit will take it apart —
but first it needs a way to recognise a known shape inside a trace.

## 2. The matched filter

**How do you find a known shape hidden in a signal?** You **slide** a copy of it along,
and at each position measure how well the two line up — by multiplying them point by
point and adding up. That sum is the **dot product** (a *correlation*): large where the
signal looks like the shape, small where it doesn't. Let's build that up in four steps:
""",),
    code(r"""
shape = tmplA[pc]                       # the shape we'll look for (neuron A's waveform)
Lw = len(shape)
trough = int(np.argmin(shape))          # where the spike's trough sits inside the template
loc0 = 250                              # we'll place the trough here
start = loc0 - trough                    # so the template starts here
rng = np.random.default_rng(1)
clean = np.zeros(500); clean[start:start + Lw] = shape         # (1) the shape on its own
noisy = clean + rng.normal(0, shape.std() * 1.3, 500)        # (2) buried in noise

# (4) sliding correlation: raw[i] is the fit for a template STARTING at i; it peaks when
# the template lines up, i.e. i = start. To plot it against the spike's *trough*
# position (which is what the red line marks), shift the x-axis by `trough`.
raw = np.array([np.dot(noisy[i:i + Lw], shape) / np.dot(shape, shape)
                for i in range(len(noisy) - Lw)])
corr_x = np.arange(len(raw)) + trough

fig, ax = plt.subplots(4, 1, figsize=(9, 7.5))
for a in (ax[0], ax[1], ax[3]):
    a.axvline(loc0, color="tab:red", ls="--"); a.set_xlim(0, 500)
ax[0].plot(clean, "k"); ax[0].set_title("(1) the spike we're looking for, on its own")
ax[1].plot(noisy, "k", lw=0.7)
ax[1].set_title("(2) the same spike, now buried in noise — the red line shows where it is")
ax[2].plot(shape, "tab:green"); ax[2].set_xlim(0, Lw)
ax[2].set_title("(3) the template we slide along: our stored copy of the shape")
ax[3].plot(corr_x, raw, "tab:blue")
ax[3].set_title("(4) the sliding correlation — it peaks exactly where the spike is buried")
ax[3].set_xlabel("position (sample)")
plt.tight_layout(); plt.show()
""",),
    md(r"""
The spike is invisible to the eye in panel (2), yet the correlation in panel (4) spikes
right at its hidden location. That sliding dot product **is** the matched filter, and
it's how we find each unit's spikes: correlate the recording with a unit's template, and
the peaks are its spike times.

One refinement turns the raw correlation into an actual *amplitude* — how big a copy of
the template fits. How well does template $j$ fit the trace at time $t$? Slide it there,
take the **inner product** with the trace window, and normalise. The best-fitting
amplitude is

$$a \;=\; \frac{\langle \text{trace window},\ \text{template}_j\rangle}{\lVert \text{template}_j\rVert^2}.$$

An $a$ near 1 means the template fits at full size — a spike of unit $j$ is there. The
**score** (how much subtracting it reduces the leftover error) is $a^2\lVert
\text{template}_j\rVert^2$.

<details>
<summary><b>▸ Go deeper: where these formulas come from (optional)</b></summary>

**The amplitude is least squares.** Model the trace window as a scaled template plus
noise, $x = a\,s + \text{noise}$, and pick $a$ to minimise the squared error
$\lVert x - a s\rVert^2$. Setting the derivative to zero,
$\frac{d}{da}\lVert x - as\rVert^2 = -2\,\langle x, s\rangle + 2a\lVert s\rVert^2 = 0$,
gives

$$a = \frac{\langle x, s\rangle}{\lVert s\rVert^2}.$$

Substituting back, the error *drops* by exactly $a^2\lVert s\rVert^2$ — that's the
**score**, so "best score" = "explains the most trace."

**Why correlation is the right detector.** Among all linear filters $w$, which one best
flags the presence of $s$ in white noise? Maximising the signal-to-noise ratio
$\langle w, s\rangle^2 / \lVert w\rVert^2$ is answered by the **Cauchy–Schwarz
inequality**: it's largest when $w \propto s$. So the optimal detector *is* correlation
with the template — the **matched filter**. (If the noise is coloured with covariance
$C$, the optimum becomes $w \propto C^{-1}s$ — matched filtering on *whitened* data,
which is exactly why Notebook 2 whitened first.)

**Matching pursuit is greedy sparse coding.** We're approximating the whole recording as
a sparse sum of templates, $x \approx \sum_i a_i\, s_{k_i}(t - \tau_i)$ — ideally the
fewest spikes that explain the trace. Finding the truly sparsest set is NP-hard, so
matching pursuit takes the **greedy** route: repeatedly add the single (template, time)
with the largest score, subtract it, and recurse on the residual. Each step is the
least-squares fit above; peeling in order of score is what lets it separate two
overlapping spikes that a single-label clusterer cannot.
</details>

**Exercise 1** *(~5 min)*. Complete `fit_amplitude`: given a trace window and a template
(both `(n_channels, n_samples)`), return $a$. We'll fit neuron A's template to a *clean*
copy of its spike.

> **Check / unstuck.** A template fitted to its own spike gives $a \approx 1$ (full
> size). Stuck? $a = \sum(\text{window} \cdot \text{template}) / \sum(\text{template}^2)$.
""",),
    code(
        solution=r"""
def fit_amplitude(window, template):
    return np.sum(window * template) / np.sum(template ** 2)

clean_A = tmplA + np.random.default_rng(2).normal(0, 8.0, tmplA.shape)   # A's spike alone + noise
a = fit_amplitude(clean_A, tmplA)
print(f"neuron A's template fits a clean copy of its spike with amplitude a = {a:.2f}")
""",
        student=r"""
def fit_amplitude(window, template):
    # YOUR CODE HERE: inner product of window and template, divided by the template's
    # squared norm. Both are (n_channels, n_samples); sum over both axes.
    raise NotImplementedError

clean_A = tmplA + np.random.default_rng(2).normal(0, 8.0, tmplA.shape)   # A's spike alone + noise
a = fit_amplitude(clean_A, tmplA)
print(f"neuron A's template fits a clean copy of its spike with amplitude a = {a:.2f}")
""",
    ),
    md(r"""
## 3. Greedy peeling: the collision comes apart

Matching pursuit repeats one idea until nothing fits well:

1. over all templates and all times, find the **best** fit (largest score),
2. **subtract** that scaled template from the trace, and record a spike,
3. repeat on the leftover **residual**.

Run it on our collision, with A's and B's templates as the two candidates. Each panel is
the leftover trace; the template about to be subtracted is drawn in red. Watch the lump
resolve into two separate spikes:
""",),
    code(r"""
candidates = [("A", tmplA, ps.plotting.unit_color(0)), ("B", tmplB, ps.plotting.unit_color(1))]

def best_fit(residual, candidates):
    # the greedy step: over both templates and all times, find the highest-scoring fit
    best = None
    for name, tmpl, col in candidates:
        norm = np.sum(tmpl ** 2)
        for t in range(residual.shape[0] - L):
            a = np.sum(residual[t:t + L] * tmpl.T) / norm
            score = a * a * norm
            if a > 0.3 and (best is None or score > best[0]):
                best = (score, name, tmpl, col, t, a)
    return best

residual = seg.copy()
fig, ax = plt.subplots(3, 1, figsize=(9, 5), sharex=True, sharey=True)
ax[0].plot(residual[:, pc], "k"); ax[0].set_title("the collision (recorded)")
for step in range(2):
    score, name, tmpl, col, t, a = best_fit(residual, candidates)
    fitted = np.zeros_like(residual); fitted[t:t + L] = a * tmpl.T
    ax[step].plot(np.where(fitted[:, pc] == 0, np.nan, fitted[:, pc]), color="tab:red", lw=1.6)
    residual = residual - fitted
    ax[step + 1].plot(residual[:, pc], "k")
    ax[step + 1].set_title(f"after peeling neuron {name}  (amplitude a = {a:.2f})")
ax[-1].set_xlabel("sample"); [a_.set_ylabel("µV") for a_ in ax]
plt.tight_layout(); plt.show()
""",),
    md(r"""
Whichever spike scores highest is peeled first; once it's gone the other stands alone
and is peeled next. After both subtractions only noise is left — the collision has been
correctly split into **two** spikes, one per neuron, where clustering would have reported
one. That is the whole point of matching pursuit.

## 4. Matching pursuit on the whole recording

`ps.matching_pursuit` runs this same peeling across the entire recording — over 20
seconds our six neurons fire a **few hundred** times in total, so it peels off a few
hundred spikes, one per real spike:
""",),
    code(r"""
spike_times, spike_labels, spike_amps = ps.matching_pursuit(filtered, templates, amp_threshold=0.5)
print(f"the recording contains ~{len(rec.ground_truth.spike_times)} true spikes; "
      f"matching pursuit peeled off {len(spike_times)}")
""",),
    md(r"""
Watch it clear a **single channel** — its busiest stretch. Each panel subtracts one more
spike (the red marks are the spike times); by the last panel the channel is just noise:
""",),
    code(r"""
pk = np.array([ps.peak_channel(templates[l]) for l in spike_labels])   # each spike's channel

# find the busiest ~160 ms stretch on one channel
wl = int(0.16 * rec.fs)
best_n, best_k = 0, 0
for k in range(0, len(spike_times), 2):
    t0, ch = spike_times[k], pk[k]
    n = np.sum((spike_times >= t0 - wl // 2) & (spike_times < t0 + wl // 2) & (pk == ch))
    if n > best_n:
        best_n, best_k = n, k
ch, t0 = pk[best_k], spike_times[best_k]
w0 = int(t0 - wl // 2); w1 = w0 + wl
sel = np.where((spike_times >= w0 + half) & (spike_times < w1 - half) & (pk == ch))[0]
sel = sel[np.argsort(spike_times[sel])][:5]

resid = filtered[w0:w1].copy()
tt = (np.arange(w0, w1) - w0) / rec.fs * 1e3
fig, axes = plt.subplots(len(sel) + 1, 1, figsize=(9, 1.2 * (len(sel) + 1)), sharex=True, sharey=True)
axes[0].plot(tt, resid[:, ch], "k", lw=0.8)
for m in sel:
    axes[0].axvline((spike_times[m] - w0) / rec.fs * 1e3, color="tab:red", alpha=0.5, lw=1)
axes[0].set_title(f"recorded (channel {ch}) — {len(sel)} spikes marked", fontsize=9)
for step, m in enumerate(sel):
    t = spike_times[m]
    resid[t - half - w0:t + half + 1 - w0] -= spike_amps[m] * templates[spike_labels[m]].T
    axes[step + 1].plot(tt, resid[:, ch], "k", lw=0.8)
    axes[step + 1].set_title(f"after peeling spike {step + 1}", fontsize=9)
axes[-1].set_xlabel("time (ms)"); [a.set_ylabel("µV") for a in axes]
plt.tight_layout(); plt.show()
""",),
    md(r"""
One by one the spikes are subtracted and vanish. The same happens across the whole
recording. To summarise the progress, here is the fraction of the **spike signal** still
unexplained — measured on the samples *around* spikes, where the action is — as they're
peeled off (measuring over the whole trace instead would hide it, since noise, not
spikes, dominates the recording's total energy):
""",),
    code(r"""
recon = np.zeros_like(filtered)
near = np.zeros(len(filtered), bool)
for t in spike_times:
    near[t - half:t + half + 1] = True               # the samples around spikes
spike_energy = np.sum(filtered[near] ** 2)
frac = [1.0]
for n, (t, lab, a) in enumerate(zip(spike_times, spike_labels, spike_amps), 1):
    recon[t - half:t + half + 1] += a * templates[lab].T
    if n % 20 == 0:
        frac.append(np.sum((filtered[near] - recon[near]) ** 2) / spike_energy)
plt.figure(figsize=(6, 3.4))
plt.plot(np.arange(len(frac)) * 20, frac, "o-", ms=3)
plt.xlabel("number of spikes peeled off"); plt.ylabel("spike signal still unexplained")
plt.title(f"peeling accounts for the spikes — {len(spike_times)} in all"); plt.ylim(0, 1.02)
plt.show()
""",),
    md(r"""
The unexplained fraction falls steadily as the spikes are peeled (it settles above zero
because noise remains even where spikes were). Matching pursuit has identified which
template fired, when, and how big, everywhere — a full spike train.

## Wrap-up

Matching pursuit turned the templates into a complete list of spike times and labels,
pulling apart overlaps that clustering alone could not. We now have a complete sort.

**Next (Notebook 7 — cleanup):** not every unit the sorter returns is real. We use
**refractory periods** and **correlograms** to catch units that should be merged.
""",),
]

student, solution = build("06_template_matching", cells)
print("wrote:", student)
print("wrote:", solution)
