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
1. See a **collision** that clustering gets wrong.
2. Build the **matched filter**: how well does a template fit at a given time?
3. Run **matching pursuit** — greedily subtract the best-fitting template — and watch
   the residual melt away.
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

Find a moment where two true spikes from different units land within a fraction of a
millisecond. Their waveforms superimpose — the recorded trace is the **sum**, not
either template alone. A single-label method has to pick one unit and gets the other
wrong.
""",),
    code(r"""
gt = rec.ground_truth
order = np.argsort(gt.spike_times)
st, sl = gt.spike_times[order], gt.spike_labels[order]
gaps = np.diff(st)
i = np.where((gaps > 0) & (gaps < 20) & (sl[:-1] != sl[1:]))[0][0]  # <~0.7 ms apart, different units
t_col = st[i]
print(f"collision at {t_col/rec.fs:.3f}s: units {sl[i]} and {sl[i+1]}, {gaps[i]} samples apart")

pc = ps.peak_channel(gt.templates[sl[i]])
ps.plotting.plot_signal(filtered, rec.fs, channels=range(max(0, pc-5), pc+6),
                        t0=(t_col-60)/rec.fs, t1=(t_col+60)/rec.fs,
                        title="two overlapping spikes add together")
plt.show()
""",),
    md(r"""
## 2. The matched filter

How well does template $j$ fit the trace at time $t$? Slide the template there and
take the **inner product** with the trace window. The best-fitting amplitude is

$$a \;=\; \frac{\langle \text{trace window},\ \text{template}_j\rangle}{\lVert \text{template}_j\rVert^2}.$$

An $a$ near 1 means the template fits at full size — a spike of unit $j$ is there. The
**score** (how much subtracting it reduces the leftover error) is $a^2\lVert
\text{template}_j\rVert^2$.

**Exercise 1** *(~6 min)*. Complete `fit_amplitude`: given a trace window and a template
(both `(n_channels, n_samples)`), return $a$.

> **Check / unstuck.** At the collision, the true unit's template should fit with
> $a \approx 1$. Stuck? $a = \sum(\text{window} \cdot \text{template}) / \sum(\text{template}^2)$.
""",),
    code(
        solution=r"""
half = templates.shape[2] // 2

def fit_amplitude(window, template):
    return np.sum(window * template) / np.sum(template ** 2)

# match the true unit's template at the collision (map true unit -> learned template)
tmpl_id = int(np.argmax([np.corrcoef(templates[a].ravel(), gt.templates[sl[i]].ravel())[0,1]
                         for a in range(len(templates))]))
window = filtered[t_col - half:t_col + half + 1].T           # (channels, samples)
a = fit_amplitude(window, templates[tmpl_id])
print(f"template {tmpl_id} fits at the collision with amplitude a = {a:.2f}")
""",
        student=r"""
half = templates.shape[2] // 2

def fit_amplitude(window, template):
    # YOUR CODE HERE: inner product of window and template, divided by the template's
    # squared norm. Both are (n_channels, n_samples); sum over both axes.
    raise NotImplementedError

tmpl_id = int(np.argmax([np.corrcoef(templates[a].ravel(), gt.templates[sl[i]].ravel())[0,1]
                         for a in range(len(templates))]))
window = filtered[t_col - half:t_col + half + 1].T           # (channels, samples)
a = fit_amplitude(window, templates[tmpl_id])
print(f"template {tmpl_id} fits at the collision with amplitude a = {a:.2f}")
""",
    ),
    md(r"""
## 3. Greedy peeling

Matching pursuit repeats one idea until nothing fits well:

1. over all templates and all times, find the **best** fit (largest score),
2. **subtract** that scaled template from the trace, and record a spike,
3. repeat on the leftover **residual**.

Each subtraction removes one spike — including one member of a collision, so the
other is exposed on the next pass. `ps.matching_pursuit` runs this efficiently over
the whole recording. Watch the residual energy fall as spikes are peeled off:
""",),
    code(r"""
spike_times, spike_labels, spike_amps = ps.matching_pursuit(filtered, templates, amp_threshold=0.5)
print(f"matching pursuit found {len(spike_times)} spikes")

# residual energy: how much of the trace is left unexplained as we add spikes back
recon = np.zeros_like(filtered)
half = templates.shape[2] // 2
total = np.sum(filtered ** 2)
resid_energy = []
for n, (t, lab, a) in enumerate(zip(spike_times, spike_labels, spike_amps)):
    recon[t - half:t + half + 1] += a * templates[lab].T
    if n % 20 == 0:
        resid_energy.append(np.sum((filtered - recon) ** 2) / total)
plt.figure(figsize=(6, 3.4))
plt.plot(np.arange(len(resid_energy)) * 20, resid_energy)
plt.xlabel("spikes peeled off"); plt.ylabel("residual energy (fraction)")
plt.title("each subtracted spike explains a bit more of the trace"); plt.show()
""",),
    md(r"""
## 4. The collision, resolved

Back to the collision. Matching pursuit should have placed **both** spikes — one of
each unit — where clustering could label only one. Overlay the reconstruction
(sum of the fitted templates) on the real trace at that moment:
""",),
    code(r"""
near = spike_times[np.abs(spike_times - t_col) < 15]
near_lab = spike_labels[np.abs(spike_times - t_col) < 15]
print(f"matching pursuit placed {len(near)} spikes at the collision: units {list(near_lab)}")

pc = ps.peak_channel(gt.templates[sl[i]])
fig, ax = plt.subplots(figsize=(8, 3))
w = slice(t_col - 60, t_col + 60)
tt = np.arange(w.start, w.stop) / rec.fs * 1e3
ax.plot(tt, filtered[w, pc], "k", lw=1.2, label="recorded")
ax.plot(tt, recon[w, pc], "tab:red", lw=1.2, ls="--", label="reconstruction (sum of templates)")
ax.set_xlabel("time (ms)"); ax.set_ylabel("µV"); ax.set_title(f"channel {pc}: the collision explained")
ax.legend(); plt.show()
""",),
    md(r"""
The reconstruction tracks the recorded trace through the collision: both spikes are
accounted for. That's the payoff of template matching over plain clustering.

## Wrap-up

Matching pursuit turned the templates into a full list of spike times and labels,
pulling apart overlaps that clustering alone could not. We now have a complete sort.

**Next (Notebook 7 — cleanup):** not every unit the sorter returns is real. We use
**refractory periods** and **correlograms** to catch units that should be merged.
""",),
]

student, solution = build("06_template_matching", cells)
print("wrote:", student)
print("wrote:", solution)
