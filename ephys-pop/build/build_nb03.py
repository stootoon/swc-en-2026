"""Notebook 3 -- Spike detection and snippet extraction."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 3 — Spike detection

*SWC ENC 2026 · ephys-pop module*

The traces are clean; now we find the spikes. A spike is a sharp **negative**
deflection that crosses a threshold. The two wrinkles are (1) setting the threshold
in a principled way, and (2) not counting one spike many times, since it lands on a
whole band of channels at once.

**In this notebook you will:**
1. Set a detection **threshold** in units of the noise, and find crossings.
2. **Deduplicate** across channels so each spike is counted once.
3. **Extract** an aligned snippet around every detected spike.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picosort as ps

rec = ps.make_recording(n_units=6, duration_s=20.0, seed=0)
whitened, filtered, W = ps.preprocess(rec)
""",),
    md(r"""
## 1. A threshold in noise units

After whitening, every channel has noise of roughly unit standard deviation, so we
can set one threshold — say **5 sigma** — that means the same thing everywhere. We
estimate each channel's noise robustly with the **median absolute deviation** (the
MAD, which a few big spikes can't inflate) and mark every trough that dips below
$-5\sigma$.

<details>
<summary><b>▸ The math: the MAD, and why 5σ (optional)</b></summary>

**Why the MAD, and the 0.6745.** The ordinary standard deviation is wrecked by the
very spikes we're hunting — a handful of large outliers inflate it. The **median
absolute deviation**, $\mathrm{MAD} = \mathrm{median}(|x|)$, ignores them (the median
doesn't care about the tails). For Gaussian noise the MAD and the true $\sigma$ are
related by a fixed constant: $\mathrm{MAD} = \Phi^{-1}(0.75)\,\sigma \approx 0.6745\,\sigma$,
so $\hat\sigma = \mathrm{MAD}/0.6745$ recovers $\sigma$ robustly (this is the
`/0.6745` in `channel_noise`).

**Why 5σ.** The threshold trades misses against false alarms. Under Gaussian noise,
the chance a single sample dips below $-\theta\sigma$ is $\Phi(-\theta)$. At
$\theta = 5$ that's $\Phi(-5) \approx 3\times10^{-7}$; over a 20-second, 32-channel
recording ($\sim\!2\times10^7$ samples per channel) you'd expect only a handful of
noise crossings per channel — rare enough that a threshold crossing is almost always
a real spike, while still low enough to catch all but the faintest units. Lower it and
false positives explode; raise it and you lose small spikes.
</details>
""",),
    code(r"""
sd = ps.channel_noise(whitened)          # robust noise sigma per channel
ch = 15
x = whitened[:, ch]
i0, i1 = int(0.30 * rec.fs), int(0.45 * rec.fs)
t = np.arange(i0, i1) / rec.fs * 1e3
plt.figure(figsize=(11, 3))
plt.plot(t, x[i0:i1], color="k", lw=0.7)
plt.axhline(-5 * sd[ch], color="tab:red", ls="--", label="-5 sigma threshold")
plt.xlabel("time (ms)"); plt.ylabel("whitened voltage"); plt.legend()
plt.title(f"channel {ch}: troughs below threshold are candidate spikes"); plt.show()
""",),
    md(r"""
**Exercise 1** *(~6 min)*. Complete `detect_on_channel`: return the sample indices of local
minima that dip below `threshold` sigma. `scipy.signal.find_peaks` finds **peaks**,
so feed it the negated signal; `height=threshold*sigma` sets the depth, and
`distance` enforces a minimum spacing so one trough isn't split in two.

> **Check / unstuck.** On channel 15 you should find a few hundred crossings over the
> 20 s. Stuck? `ps.detect_spikes` does the full (multi-channel) version.
""",),
    code(
        solution=r"""
from scipy.signal import find_peaks

def detect_on_channel(x, sigma, threshold=5.0, min_distance=9):
    peaks, _ = find_peaks(-x, height=threshold * sigma, distance=min_distance)
    return peaks

crossings = detect_on_channel(whitened[:, ch], sd[ch])
print(f"channel {ch}: {len(crossings)} threshold crossings")
""",
        student=r"""
from scipy.signal import find_peaks

def detect_on_channel(x, sigma, threshold=5.0, min_distance=9):
    # YOUR CODE HERE: use find_peaks on -x with height=threshold*sigma and
    # distance=min_distance; return the array of peak indices.
    raise NotImplementedError

crossings = detect_on_channel(whitened[:, ch], sd[ch])
print(f"channel {ch}: {len(crossings)} threshold crossings")
""",
    ),
    md(r"""
## 2. One spike, many channels → deduplicate

A single spike appears on a band of neighbouring channels, so if we detect on every
channel independently we count it several times. `ps.detect_spikes` handles this: it
finds crossings on all channels, then **greedily** keeps the largest and suppresses
any other crossing close to it in both time and depth. The result is one detection
per spike, at its **peak channel**.
""",),
    code(r"""
times, peak_channels = ps.detect_spikes(whitened, rec.probe, rec.fs, threshold=5.0)
gt = rec.ground_truth
print(f"detected {len(times)} spikes  (ground truth has {len(gt.spike_times)})")

# How many true spikes did we catch? (match each true spike to a detection within 0.5 ms)
tol = int(0.5e-3 * rec.fs)
det = np.sort(times)
idx = np.clip(np.searchsorted(det, gt.spike_times), 1, len(det) - 1)
near = np.minimum(np.abs(det[idx] - gt.spike_times), np.abs(det[idx - 1] - gt.spike_times))
print(f"detection recall: {(near <= tol).mean():.1%} of true spikes found")
""",),
    md(r"""
Detection finds the large majority of true spikes. It misses a few (spikes buried in
a collision, or the smallest ones) and picks up a few false crossings — both get
sorted out later. Overlay the detections on the raw traces to see them land on the
spikes:
""",),
    code(r"""
ps.plotting.plot_signal(whitened, rec.fs, channels=range(0, 14),
                        t0=0.90, t1=0.99, mark_spikes=times,
                        title="detected spikes (red) on the whitened traces")
plt.show()
""",),
    md(r"""
## 3. Extract aligned snippets

For each detected spike we cut a short **snippet**: a `(n_channels, n_samples)` clip
around the spike time. `ps.extract_snippets` also **re-aligns** each clip on the
trough of its peak channel, so the same part of the waveform lands at the same
offset in every snippet — essential for averaging them into a clean template later.

**Exercise 2** *(~5 min)*. Extract the snippets, then average all snippets on their peak
channel to check they're aligned: the average should be a crisp spike, not a smeared
blur.

> **Check / unstuck.** A well-aligned average is a sharp biphasic waveform. If it's
> smeared, alignment failed. Stuck? `ps.extract_snippets(filtered, times, peak_channels=peak_channels)`.
""",),
    code(
        solution=r"""
snippets, times = ps.extract_snippets(filtered, times, peak_channels=peak_channels)
print("snippets:", snippets.shape, "(spikes, channels, samples)")

# average each snippet on ITS peak channel, then overlay all of them
peak_wave = ps.peak_waveforms(snippets)          # (n_spikes, n_samples)
plt.figure(figsize=(6, 4))
for w in peak_wave[:150]:
    plt.plot(w, color="0.85", lw=0.5)
plt.plot(peak_wave.mean(0), color="tab:blue", lw=2, label="mean")
plt.title("peak-channel snippets are aligned"); plt.xlabel("sample"); plt.ylabel("µV")
plt.legend(); plt.show()
""",
        student=r"""
snippets, times = ps.extract_snippets(filtered, times, peak_channels=peak_channels)
print("snippets:", snippets.shape, "(spikes, channels, samples)")

# average each snippet on ITS peak channel, then overlay all of them
peak_wave = ps.peak_waveforms(snippets)          # (n_spikes, n_samples)
plt.figure(figsize=(6, 4))
for w in peak_wave[:150]:
    plt.plot(w, color="0.85", lw=0.5)
plt.plot(peak_wave.mean(0), color="tab:blue", lw=2, label="mean")
plt.title("peak-channel snippets are aligned"); plt.xlabel("sample"); plt.ylabel("µV")
plt.legend(); plt.show()
""",
    ),
    md(r"""
The snippets stack up neatly — but notice they're clearly a *mix* of different
shapes and sizes, because they come from different neurons all jumbled together. We
detected spikes, but we don't yet know **which neuron** each one came from.

## Wrap-up

You set a principled threshold, deduplicated multi-channel detections into one spike
each, and cut aligned snippets. We now have a pile of spikes with no labels.

**Next (Notebook 4 — features):** describe each snippet with a few numbers — where it
is, how big it is, what shape it has — so we can tell the neurons apart.
""",),
]

student, solution = build("03_detection", cells)
print("wrote:", student)
print("wrote:", solution)
