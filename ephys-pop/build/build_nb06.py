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
loc0 = 250                              # where we'll hide it
rng = np.random.default_rng(1)
clean = np.zeros(500); clean[loc0:loc0 + Lw] = shape          # (1) the shape on its own
noisy = clean + rng.normal(0, shape.std() * 1.3, 500)        # (2) buried in noise
corr = np.array([np.dot(noisy[i:i + Lw], shape) / np.dot(shape, shape)  # (4) sliding correlation
                 for i in range(len(noisy) - Lw)])

fig, ax = plt.subplots(4, 1, figsize=(9, 7.5))
ax[0].plot(clean, "k"); ax[0].set_xlim(0, 500)
ax[0].set_title("(1) the spike we're looking for, on its own")
ax[1].plot(noisy, "k", lw=0.7); ax[1].axvline(loc0, color="tab:red", ls="--"); ax[1].set_xlim(0, 500)
ax[1].set_title("(2) the same spike, now buried in noise — the red line shows where it is")
ax[2].plot(shape, "tab:green"); ax[2].set_xlim(0, Lw)
ax[2].set_title("(3) the template we slide along: our stored copy of the shape")
ax[3].plot(corr, "tab:blue"); ax[3].axvline(loc0, color="tab:red", ls="--"); ax[3].set_xlim(0, 500)
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

`ps.matching_pursuit` runs this same peeling across the entire recording. Over 20
seconds our six neurons fire a **few hundred** times in total, so it peels off a few
hundred spikes — one per real spike:
""",),
    code(r"""
spike_times, spike_labels, spike_amps = ps.matching_pursuit(filtered, templates, amp_threshold=0.5)
print(f"the recording contains ~{len(rec.ground_truth.spike_times)} true spikes; "
      f"matching pursuit peeled off {len(spike_times)}")

# residual energy: the fraction of the recording still unexplained after N spikes removed
recon = np.zeros_like(filtered)
total = np.sum(filtered ** 2)
resid_energy = [1.0]
for n, (t, lab, a) in enumerate(zip(spike_times, spike_labels, spike_amps), 1):
    recon[t - half:t + half + 1] += a * templates[lab].T
    if n % 20 == 0:
        resid_energy.append(np.sum((filtered - recon) ** 2) / total)
plt.figure(figsize=(6, 3.4))
plt.plot(np.arange(len(resid_energy)) * 20, resid_energy, "o-", ms=3)
plt.xlabel("number of spikes peeled off"); plt.ylabel("fraction of trace unexplained")
plt.title(f"each spike explains a bit more — {len(spike_times)} spikes in all"); plt.ylim(0, 1)
plt.show()
""",),
    md(r"""
Every subtracted spike removes a little more signal, so the unexplained fraction falls
steadily as the few-hundred spikes are peeled. And on the real traces the reconstruction
(the sum of all fitted templates) tracks the recording closely — here on the channel of
the largest spike:
""",),
    code(r"""
i = int(np.argmax(spike_amps))                       # the biggest spike MP found
t0, pc2 = spike_times[i], ps.peak_channel(templates[spike_labels[i]])
w = slice(t0 - 60, t0 + 60)
tt = np.arange(w.start, w.stop) / rec.fs * 1e3
plt.figure(figsize=(8, 3))
plt.plot(tt, filtered[w, pc2], "k", lw=1.2, label="recorded")
plt.plot(tt, recon[w, pc2], "tab:red", lw=1.2, ls="--", label="reconstruction (fitted template)")
plt.xlabel("time (ms)"); plt.ylabel("µV"); plt.legend()
plt.title(f"channel {pc2}: a real spike, explained by its template"); plt.show()
""",),
    md(r"""
The dashed reconstruction sits right on top of the recorded spike: matching pursuit has
correctly identified which template fired, when, and how big. Do that everywhere and you
have a full spike train.

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
