"""Notebook 8 -- Scoring: the full analysis on three mystery patches, graded against truth."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 8 — Scoring

*SWC ENC 2026 · ephys-one module*

Time to find out whether it works. Three **mystery patches** have been recorded for
you — different numbers of channels, different unitary currents, different test
voltages, and each with its own share of real-world problems. Their truth is hidden.
Your job: look at each recording, decide which corrections it needs, produce estimates
of $N$, $i$ and $p_{max}$ **with error bars**, commit — and only then reveal the truth
and see how you did.

Everything you need was built in Notebooks 3–5. If you skipped one, the backend has
it: `pp.isochrone_stats`, `pp.baseline_variance`, `pp.successive_difference_variance`,
`pp.fit_parabola`, `pp.bootstrap_fit` — or the all-in-one `pp.count_channels`.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picopatch as pp

patches = {name: pp.make_mystery(name) for name in ["A", "B", "C"]}
for name, e in patches.items():
    print(f"patch {name}: {e.n_sweeps} sweeps of {e.n_samples} samples, step to {e.V.max():+.0f} mV;  truth hidden: {e.truth is None}")
""",),
    md(r"""
## 1. Look before you fit

Never fit blind. For each patch, four pictures tell you what you're dealing with:

1. **the sweeps** — how big is the current, how noisy the baseline?
2. **plateau current vs sweep number** — is there rundown?
3. **mean and variance vs time** — is there a pre-step baseline to subtract?
4. **variance vs mean** — how far along the parabola did the experiment get? Does it
   bend back down, or is it still rising at the end?
""",),
    code(r"""
def look(exp, name):
    mean, var = pp.isochrone_stats(exp.sweeps)
    plateau = exp.sweeps[:, -100:].mean(axis=1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for k in range(15):
        axes[0].plot(exp.t_ms, exp.sweeps[k], color="k", lw=0.5, alpha=0.6)
    axes[0].set_title("15 sweeps"); axes[0].set_xlabel("time (ms)"); axes[0].set_ylabel("pA")
    axes[1].plot(plateau, ".", ms=3, color="k"); axes[1].set_title("plateau current per sweep"); axes[1].set_xlabel("sweep #")
    axes[2].plot(exp.t_ms, mean, "k", label="mean"); axes[2].set_ylabel("mean (pA)")
    ax2 = axes[2].twinx(); ax2.plot(exp.t_ms, var, color="tab:blue", lw=0.7); ax2.set_ylabel("variance (pA²)", color="tab:blue")
    axes[2].set_title("mean & variance vs time"); axes[2].set_xlabel("time (ms)")
    axes[3].scatter(mean, var, c=exp.t_ms, cmap="viridis", s=6); axes[3].set_title("variance vs mean")
    axes[3].set_xlabel("mean (pA)"); axes[3].set_ylabel("variance (pA²)")
    fig.suptitle(f"patch {name}", fontweight="bold", x=0.02, ha="left"); plt.tight_layout(); plt.show()

for name, e in patches.items():
    look(e, name)
""",),
    md(r"""
**Exercise 1** *(~6 min)*. Turn those looks into numbers. Complete `diagnose`, which
returns for a patch:

- `baseline`: the across-sweep variance before the step (background noise floor);
- `rundown`: the fractional drop in plateau current from the first 10% of sweeps to
  the last 10% (positive = current shrinking);
- `turns_down`: whether the variance at the largest mean current is clearly below the
  peak variance (say, under 80% of it) — a sign the parabola bends back and $N$ is
  estimable.

> **Check / unstuck.** One patch should show a rundown of ~20–25%; two should have
> `turns_down = False` — for one of them because the parabola genuinely never bends
> (a weak step), for the other because rundown has inflated the plateau variance and
> hidden the bend (re-check after correcting). All three have a non-zero baseline.
""",),
    code(
        solution=r"""
def diagnose(exp):
    mean, var = pp.isochrone_stats(exp.sweeps)
    baseline = pp.baseline_variance(exp.sweeps, exp.protocol)
    plateau = exp.sweeps[:, -100:].mean(axis=1)
    n10 = max(exp.n_sweeps // 10, 1)
    rundown = 1 - plateau[-n10:].mean() / plateau[:n10].mean()
    turns_down = var[np.argmax(mean)] < 0.8 * var.max()
    return dict(baseline=baseline, rundown=rundown, turns_down=turns_down)

diagnostics = {name: diagnose(e) for name, e in patches.items()}
for name, d in diagnostics.items():
    print(f"patch {name}:  baseline var = {d['baseline']:6.2f} pA²   rundown = {d['rundown']:+.1%}   parabola turns down: {d['turns_down']}")
""",
        student=r"""
def diagnose(exp):
    mean, var = pp.isochrone_stats(exp.sweeps)
    # YOUR CODE HERE: baseline (pre-step variance), rundown (fractional plateau drop,
    # first 10% vs last 10% of sweeps), turns_down (var at max mean < 0.8 * max var)
    raise NotImplementedError
    return dict(baseline=baseline, rundown=rundown, turns_down=turns_down)

diagnostics = {name: diagnose(e) for name, e in patches.items()}
for name, d in diagnostics.items():
    print(f"patch {name}:  baseline var = {d['baseline']:6.2f} pA²   rundown = {d['rundown']:+.1%}   parabola turns down: {d['turns_down']}")
""",
    ),
    md(r"""
## 2. Analyse, with the right corrections

**Exercise 2** *(~8 min)*. Complete `analyse`. Given a patch and a flag
`rundown_correction`:

1. compute the mean across sweeps, and the variance either the ordinary way or by
   successive differences (if correcting for rundown);
2. measure and subtract the pre-step baseline variance (`pp.baseline_variance` takes
   an optional `var_fn`, so you can measure the baseline consistently);
3. fit the parabola;
4. bootstrap the fit (`pp.bootstrap_fit(sweeps, n_boot, baseline, var_fn)` resamples
   consecutive pairs when `var_fn=pp.successive_difference_variance`);
5. return the fit and the 95% intervals for $i$ and $N$.

Then run it on each patch, switching on the rundown correction where your
diagnostics say so.

> **Check / unstuck.** `pp.count_channels(exp, rundown_correction=...)` does all five
> steps and returns an object with `.fit`, `.i_ci`, `.N_ci`.
""",),
    code(
        solution=r"""
def analyse(exp, rundown_correction=False, n_boot=200):
    var_fn = pp.successive_difference_variance if rundown_correction else None
    mean = exp.sweeps.mean(axis=0)
    var = var_fn(exp.sweeps) if var_fn else pp.isochrone_stats(exp.sweeps)[1]
    baseline = pp.baseline_variance(exp.sweeps, exp.protocol, var_fn or pp.isochrone_stats)
    fit = pp.fit_parabola(mean, var, baseline)
    i_b, N_b = pp.bootstrap_fit(exp.sweeps, n_boot=n_boot, baseline=baseline, var_fn=var_fn, rng=0)
    N_b = N_b[np.isfinite(N_b)]
    return dict(fit=fit, mean=mean, var=var,
                i_ci=np.percentile(i_b, [2.5, 97.5]),
                N_ci=np.percentile(N_b, [2.5, 97.5]) if len(N_b) > 1 else (np.nan, np.nan),
                i_boot=i_b, N_boot=N_b)

results = {}
for name, e in patches.items():
    use_sd = diagnostics[name]["rundown"] > 0.05
    results[name] = analyse(e, rundown_correction=use_sd)
    r = results[name]
    print(f"patch {name} (rundown correction: {use_sd}):  i = {r['fit'].i:.2f} pA [{r['i_ci'][0]:.2f}, {r['i_ci'][1]:.2f}]   "
          f"N = {r['fit'].N:.0f} [{r['N_ci'][0]:.0f}, {r['N_ci'][1]:.0f}]   p_max = {r['fit'].p_max:.2f}")
""",
        student=r"""
def analyse(exp, rundown_correction=False, n_boot=200):
    # YOUR CODE HERE: steps 1-5 from the text. Return a dict with keys
    # fit, mean, var, i_ci, N_ci, i_boot, N_boot
    raise NotImplementedError

results = {}
for name, e in patches.items():
    use_sd = diagnostics[name]["rundown"] > 0.05
    results[name] = analyse(e, rundown_correction=use_sd)
    r = results[name]
    print(f"patch {name} (rundown correction: {use_sd}):  i = {r['fit'].i:.2f} pA [{r['i_ci'][0]:.2f}, {r['i_ci'][1]:.2f}]   "
          f"N = {r['fit'].N:.0f} [{r['N_ci'][0]:.0f}, {r['N_ci'][1]:.0f}]   p_max = {r['fit'].p_max:.2f}")
""",
    ),
    code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, r) in zip(axes, results.items()):
    pp.plotting.plot_variance_mean(r["mean"], r["var"] - r["fit"].baseline,
                                   fit=pp.fit_parabola(r["mean"], r["var"] - r["fit"].baseline),
                                   ax=ax, t_ms=patches[name].t_ms, title=f"patch {name}: baseline-subtracted fit")
plt.tight_layout(); plt.show()
""",),
    md(r"""
Before revealing anything, write down what you believe. For each patch: $i$ with its
interval, $N$ with its interval, $p_{max}$ — and, honestly, whether you trust $N$ at
all (look at `turns_down` and at the width of the interval).

## 3. The reveal
""",),
    code(r"""
truths = {name: pp.reveal(name) for name in patches}
print(f"{'patch':>5}  {'i est':>8} {'i true':>7}   {'N est':>7} {'N true':>7}   {'p_max est':>9} {'p_max true':>10}")
for name in patches:
    r, t = results[name], truths[name]
    print(f"{name:>5}  {r['fit'].i:8.2f} {t.i:7.2f}   {r['fit'].N:7.0f} {t.N:7d}   {r['fit'].p_max:9.2f} {t.p_max:10.2f}")

# the scorecard wants CountResult-like objects: wrap our dicts
class _R:
    def __init__(self, r): self.fit, self.i_boot, self.N_boot = r["fit"], r["i_boot"], r["N_boot"]
pp.plotting.plot_scorecard({n: _R(r) for n, r in results.items()}, truths)
plt.show()
""",),
    md(r"""
What was hiding in each patch:

- **Patch A** — a well-behaved recording with background noise and a strong step
  ($p_{max} \approx 0.99$). Baseline subtraction is all it needed; both estimates land
  within a few percent, with tight intervals.
- **Patch B** — rundown: a quarter of the channels were lost over the 400 sweeps. The
  ordinary variance would have given a nonsensical $N$; successive differences recover
  the parabola. Note that the $N$ you get is the *average* number of channels present
  during the experiment (about 2200 of the original 2500) — the method can't tell you
  how many there were before you started.
- **Patch C** — a weak step: only ~30% of the channels ever open. The parabola never
  turns down, so $i$ is fine but $N$ is barely constrained, and the honest answer is
  the wide interval, not the point estimate. $p_{max}$ inherits the same uncertainty.

If your $N$ for patch C was far off but your interval included the truth, you did it
right.
""",),
    md(r"""
## 4. What you've built

Starting from a single random switch, you derived why the sum of $N$ switches has
variance $Ni^2p(1-p)$, eliminated the unknown open probability to get a parabola, fitted
it to recover the unitary current and the channel count, and then learned to recognise
and repair the ways real recordings bend that parabola: background noise (subtract the
baseline), rundown (successive differences), too few sweeps (bootstrap and report the
interval), incomplete coverage (trust $i$, doubt $N$), and the amplifier's filter
(keep it fast, check it doesn't matter). That is nonstationary noise analysis as it is
actually practised — the method behind the paper's real Shaker and K_Ca measurements
(1.4 pA and 6000 channels; 21 pA and 94 channels), and behind many of the
single-channel conductances in the textbooks before anyone had seen a single channel.
""",),
]

if __name__ == "__main__":
    print(build("08_scoring", cells))
