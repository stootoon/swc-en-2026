"""Notebook 0 -- Roadmap for the ephys-one (counting channels) module (no exercises)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 0 — Roadmap

*SWC ENC 2026 · ephys-one module*

Welcome. In the previous module the signal was the extracellular voltage picked up by
a probe sitting *near* hundreds of neurons. In this one we go the other way: a
**patch-clamp** electrode sealed onto a *single cell*, recording the current that
flows through the **ion channels** in its membrane. The question we'll answer sounds
impossible at first: from a smooth macroscopic current, carried by thousands of
channels at once, **how many channels are there, and how much current does each one
pass?**

The trick is to look not at the current itself but at its **fluctuations** — the noise.
This is **nonstationary noise analysis** (Sigworth 1980; Heinemann & Conti 1992), and
we'll follow the tutorial by Alvarez, Gonzalez & Latorre (2002), *Counting Channels*,
which — like this module — teaches it with simulations. This first notebook has no
exercises; it's a map.
""",),
    md(r"""
## The physical picture

An **ion channel** is a protein pore in the membrane. At any instant it is either
**closed** (no current) or **open**, passing a small, fixed **unitary current** `i` —
on the order of a picoampere (pA). It flips between the two at random. A patch of
membrane holds `N` such channels; the current you record is the sum of all of them.

In **voltage clamp** the amplifier holds the membrane potential at a value *you*
choose and records the current needed to hold it there. Depolarise the membrane and
voltage-gated channels open; the current grows. Repeat the same step many times and
you get a stack of **sweeps**.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picopatch as pp

fig = plt.figure(figsize=(13, 4))
pp.plotting.draw_channel_cartoon(plt.subplot(1, 2, 1))
pp.plotting.draw_patch_clamp(plt.subplot(1, 2, 2))
plt.tight_layout(); plt.show()
""",),
    md(r"""
## The idea in one picture

Here is the whole module in three panels, made from a simulated patch of 1000
channels (1 pA each) stepped from −80 to +65 mV, 300 times.

- **Left:** the sweeps. The current rises smoothly as channels open, but every sweep
  wiggles a little differently — that wiggle is the noise we'll analyse.
- **Middle:** at every time point, the **mean** and the **variance** of the current
  across sweeps.
- **Right:** plot variance against mean, one dot per time point. The dots trace a
  **parabola** — and its shape encodes the answer. The initial slope is the current
  through one channel; the point where the variance returns to zero is where every
  channel is open, i.e. the mean current equals `i × N`.

Don't worry about *why* yet — that's Notebooks 1–3. Just notice that a smooth-looking
current carries hidden information about the switches that make it up.
""",),
    code(r"""
exp = pp.make_experiment(N=1000, i=1.0, n_sweeps=300, seed=0)
mean, var = pp.isochrone_stats(exp.sweeps)
fit = pp.fit_parabola(mean, var)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
pp.plotting.plot_sweeps(exp, n_show=20, ax=axes[0], show_voltage=False, title="20 sweeps")
axes[1].plot(exp.t_ms, mean, color="k", label="mean"); axes[1].set_ylabel("mean current (pA)", color="k")
ax2 = axes[1].twinx(); ax2.plot(exp.t_ms, var, color="tab:blue", lw=0.8, label="variance")
ax2.set_ylabel("variance (pA²)", color="tab:blue"); axes[1].set_xlabel("time (ms)")
axes[1].set_title("mean and variance across sweeps")
pp.plotting.plot_variance_mean(mean, var, fit=fit, ax=axes[2], t_ms=exp.t_ms,
                               title="variance vs mean: a parabola")
plt.tight_layout(); plt.show()
print(f"fitted: {fit}    (truth: i = 1.0 pA, N = 1000)")
""",),
    md(r"""
## Why synthetic data?

We *simulate* the channels: we choose the number of channels, the unitary current and
the gating kinetics, generate the sweeps a real amplifier would record, and then try
to recover what we put in. You learn a method best when you can check it against a
known answer — and on a real patch you never can. So every notebook ends by asking
*did we recover the truth?*, and Notebook 8 scores your whole analysis on three
"mystery patches" whose truth is hidden until you've committed to an estimate.

The simulator lives in a small backend package, **`picopatch`** (imported as `pp`),
which also holds a reference implementation of every analysis step — so any notebook
runs on its own, and a stuck exercise has a `pp.<name>` fallback to keep you moving.
""",),
    md(r"""
## The analysis, notebook by notebook

Notebooks **1–3 are the core**: by the end of Notebook 3 you will have the complete
basic analysis working. Notebooks **4–7 are independent of each other** — each takes
one way a real recording departs from the ideal, shows how it bends the parabola, and
builds the fix. Do them in any order, as time allows. Notebook 8 puts it all together.

| # | Notebook | What you build | What you'll learn |
|---|----------|----------------|-------------------|
| 1 | One channel | a coin-flip simulation of a single channel | rate constants, open probability, dwell times, voltage dependence |
| 2 | Many channels | isochrone mean & variance across sweeps | why the sum of N switches has variance N·i²·p(1−p) |
| 3 | **The parabola** | the variance–mean fit → **N, i, p_max** | the core analysis, end to end |
| 4 | Imperfect recordings | baseline subtraction, successive differences, bootstrap | background noise, rundown, how many sweeps you need |
| 5 | Incomplete parabolas | fitting with a limited open-probability range | what you can and can't trust; inactivating channels |
| 6 | Beware the filter | low-pass filtering vs gating speed | why filtering shrinks `i`; flickering channels |
| 7 | Stationary noise *(optional)* | the Lorentzian power spectrum | getting kinetics from steady-state noise |
| 8 | Scoring | your pipeline on three mystery patches | estimates with error bars, graded against truth |
""",),
    code(r"""
fig, ax = plt.subplots(figsize=(13, 2.4))
pp.plotting.draw_pipeline(ax); plt.show()
""",),
    md(r"""
## How the notebooks work

**Read at your own depth.** The material comes in three layers, so you can go as deep
as you like on each technique:

1. **The big picture** — the plain-language prose and the code cells. Enough to
   understand what each step does and run it. If that's all you want, skim and go.
2. **The intuition** — worked examples and figures that build up *why* a method works
   (e.g. watching the variance of ten coin flips before we talk about a thousand channels).
3. **The math** — collapsible **▸ Go deeper** blocks with the underlying equations, for
   those who want them. They're folded away by default; open them if you're curious,
   skip them freely if not.

- Each notebook comes in two versions: a **student** copy with `# YOUR CODE HERE`
  blanks, and a **solutions** copy with everything worked out.
- The backend package **`picopatch`** simulates the patch and holds reference
  implementations of every step. Each notebook imports what it needs, so you can run
  any notebook on its own.
- Wherever we can, an exercise ends by **checking your result against ground truth**,
  and tells you roughly what to expect so you can spot a bug.
- **Stuck on an exercise?** Don't let it block your day. Every step also lives in the
  backend as `pp.<name>` (and fully worked out in the solutions copy), so you can drop
  that in, keep pace, and circle back later.

Run the cell below to confirm your environment is set up (select the
**"SWC Ephys-One (.venv)"** kernel if prompted). If it prints the shape of a stack of
sweeps and a plot appears, you're ready for Notebook 1.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picopatch as pp

exp = pp.make_experiment(N=50, i=1.0, n_sweeps=5, seed=0)
print("environment OK — simulated an experiment")
print(f"  sweeps: {exp.sweeps.shape}  (sweeps x samples), {exp.n_samples / exp.fs * 1e3:.0f} ms at {exp.fs/1e3:.0f} kHz")
print(f"  truth:  N = {exp.truth.N} channels, i = {exp.truth.i} pA, scheme = {exp.truth.scheme}")
pp.plotting.plot_sweeps(exp, n_show=5, title="5 sweeps from a patch of 50 channels")
plt.show()
""",),
    md(r"""
## Glossary — terms you'll meet

You don't need these yet; come back when a word trips you up. Each is defined
properly where it first appears (in parentheses).

| term | in plain words |
|------|----------------|
| **patch clamp** | a glass pipette sealed onto a cell membrane, recording the current through the channels under (or around) it (NB0) |
| **voltage clamp** | the amplifier holds the membrane voltage fixed and records the current needed to do so (NB0) |
| **unitary current, `i`** | the current through one open channel, ~pA (NB1) |
| **rate constant** | probability per unit time of a transition, e.g. closed → open; units 1/s (NB1) |
| **open probability, `p`** | the fraction of time a channel spends open, or equivalently the fraction of channels open at an instant (NB1) |
| **dwell time** | how long a channel stays in one state before flipping (NB1) |
| **kinetic scheme** | the list of states a channel can be in and the rates between them, e.g. C ⇌ O (NB1) |
| **macroscopic current** | the summed current of all N channels in the patch (NB2) |
| **sweep** | one repeat of the voltage protocol and its recorded current (NB2) |
| **isochrone** | the set of current values at one time point, taken across all sweeps (NB2) |
| **variance** | the average squared deviation from the mean; the square of the standard deviation (NB2) |
| **binomial** | the distribution of the number of successes in N independent yes/no trials (NB2) |
| **variance–mean parabola** | var = i·mean − mean²/N, the relation the whole analysis rests on (NB3) |
| **p_max** | the largest open probability reached: mean_max / (i·N) (NB3) |
| **rundown** | a gradual loss of active channels over the course of an experiment (NB4) |
| **bootstrap** | re-fitting on resampled sweeps to measure how uncertain an estimate is (NB4) |
| **inactivation** | a channel entering a closed state it can't open from until the membrane repolarises (NB5) |
| **low-pass filter** | removes fast wiggles; every amplifier has one (NB6) |
| **Lorentzian** | the power spectrum of a random two-state switch (NB7) |

**The paper:** Alvarez O, Gonzalez C, Latorre R (2002). *Counting Channels: a tutorial
guide on ion channel fluctuation analysis.* Advances in Physiology Education 26: 327–341.
""",),
]

if __name__ == "__main__":
    print(build("00_roadmap", cells))
