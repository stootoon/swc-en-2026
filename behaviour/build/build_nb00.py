"""Notebook 0 -- Roadmap for the behaviour module (no exercises)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 0 — Roadmap

*SWC ENC 2026 · behaviour module*

Welcome. Over this day you'll rebuild, from the ground up, the statistical toolkit
behind **Figures 1–3 of Piet et al. (2024), *Neuron*** — "Behavioral strategy
shapes activation of the Vip-Sst disinhibitory circuit in visual cortex." This
first notebook has no exercises; it's a map, so you know where the day is going and
why each step matters.
"""),
    md(r"""
## The question

Mice performing the same task can solve it in different ways — different
**strategies** — and those strategies can drift over the course of a session. The
paper's task is **visual change detection**: a stream of images is shown, and the
mouse is rewarded for licking when the image *changes*. A mouse might solve this by
**visually comparing** each image to the last, or by **timing** — learning roughly
how often changes come and licking on schedule.

<img src="../assets/paper/fig1a.png" width="460">

*The change-detection task (Fig. 1A).*

The central problem of this module: **given only the licking, how do we read out
which strategy a mouse is using, and how it changes over time?**
"""),
    md(r"""
## The approach

Piet et al. answer it with one model of **when a licking bout starts**, a logistic
regression built from five candidate "strategy" regressors (a constant bias, a
visual term, a timing term, and two omission terms). Fitting that model to a
session recovers *how much* the mouse leans on each strategy; letting the weights
drift recovers *how that changes* through the hour.

**Why synthetic data?** The real dataset is awkward to access, but there's a deeper
pedagogical reason to avoid it: we learn a method best when we can check it against
a known answer. So all day you'll analyze **synthetic mice generated from the
paper's own model**, whose true strategy we chose. Every technique you build gets
tested the same way — *can we recover the truth we put in?* That question is the
spine of the whole module.
"""),
    md(r"""
## The day, notebook by notebook

Each notebook is a short, self-contained step: a bit of statistics, an exercise or
two where you implement the key idea, and a figure from the paper reproduced at the
end. Notebooks 3–6 build a **fit → evaluate → ablate** cycle on mice with a fixed
strategy; Notebook 7 extends it to a mouse whose strategy drifts.

| # | Notebook | What you build | Paper |
|---|----------|----------------|-------|
| 1 | Task → design matrix | licking bouts and the five-strategy design matrix | Fig 1A, 1C |
| 2 | Meet the mice | how each strategy looks in behavior | Fig 1B |
| 3 | Static model | logistic regression from its likelihood; weight recovery | Fig 1D |
| 4 | Evaluation | ROC, AUC, cross-validation | Fig 2A |
| 5 | Evidence & ablation | which strategies matter; the strategy index | Fig 2B, 2D |
| 6 | Mixtures | the cycle on mice that blend strategies | Fig 2F |
| 7 | Dynamic model | the drifting-weight (random-walk) model | Fig 1D |
| 8 | Engagement *(optional)* | strategy vs. task engagement | Fig 3B |
"""),
    md(r"""
## How the notebooks work

- Each notebook comes in two versions: a **student** copy with `# YOUR CODE HERE`
  blanks for you to fill, and a **solutions** copy with everything worked out.
- A small backend package, **`swcbehav`**, generates the synthetic mice and holds
  reference implementations of every technique. Each notebook imports what it needs
  from it, so you can run any notebook on its own.
- Wherever we can, an exercise ends by **checking your result against ground
  truth** — the recover-the-truth spine again.

Run the cell below to confirm your environment is set up (select the
**"SWC Behaviour (.venv)"** kernel if prompted). If it prints a mouse's weights
and a plot appears, you're ready for Notebook 1.
"""),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import swcbehav as sb

# Generate one synthetic mouse and take a quick look -- no exercise here,
# just a check that everything is installed and importable.
sess = sb.make_mouse("visual", seed=0)
print("environment OK — generated a session with", len(sess.table), "image presentations")
print("this mouse's true strategy weights:", np.round(sess.true_weights[0], 1))
print("order: [bias, visual, omission, post_omission, timing]")

sb.plotting.plot_session_raster(sess, t0=560, t1=585)
plt.show()
"""),
    md(r"""
## Ready

That's the whole arc: from raw licks to a design matrix, to a fitted model, to
knowing which strategy a mouse uses and how it drifts — each step checked against a
truth you control. On to **Notebook 1**.
"""),
]

student, solution = build("00_roadmap", cells)
print("wrote:", student)
print("wrote:", solution)
