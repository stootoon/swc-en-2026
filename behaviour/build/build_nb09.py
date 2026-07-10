"""Notebook 9 -- Individual differences: variance partitioning + permutation test."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 9 — Individual differences

*SWC ENC 2026 · behaviour module · Part 2 (population-level inference)*

This is the first of three notebooks on **inference across a population**. They're
modular — each stands alone, and any section can be skipped and left for self-study.

Here we reproduce **Figure 2H**: mice have their own strategy *preferences*, stable
across sessions. Two questions, two tools:

- *How much* of the variation in strategy is between mice vs. within a mouse? →
  **variance partitioning**.
- Is that more than you'd expect by chance? → a **permutation test**.

Both hinge on the fact that our data are **nested** — several sessions per mouse —
which is the theme of this whole part.

---
**The paper panel this notebook reproduces**

<img src="../assets/paper/fig2h.png" width="640">

*Fig. 2H — each mouse's sessions (dots) sorted by the mouse's average strategy
index; the spread within a column is within-mouse, the trend across columns is
between-mouse. Mouse identity explained ~72% of the variance, vs ~22% when the
labels were shuffled.*
"""),
    md(r"""
## Setup

We generate a **cohort**: several sessions from each of many mice, where every
mouse has its own *stable* strategy plus session-to-session wobble. For each session
we fit the static model (Notebook 3) and summarize its strategy by a single fast
number, the recovered **visual − timing** weight. This cell always runs; the
sections after it are independent.
"""),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import swcbehav as sb

cohort = sb.make_cohort(n_mice=12, sessions_per_mouse=4, seed=1)

def session_strategy(cs):
    X, cols = sb.build_design_matrix(cs.session.table)
    y = cs.session.table["bout_start"].to_numpy().astype(float)
    w = sb.fit_static(X, y)
    return w[1] - w[4]                 # visual weight minus timing weight

strategy = np.array([session_strategy(cs) for cs in cohort])
mouse = np.array([cs.mouse_id for cs in cohort])
print(f"{len(cohort)} sessions from {mouse.max() + 1} mice")
"""),
    md(r"""
## 1. Look at the data

Sort the mice by their average strategy and plot every session, colored by mouse.
Sessions from the same mouse cluster together (a short vertical spread), while the
mouse averages march steadily across — the visual signature of a trait that is more
between-mouse than within-mouse. This is our Figure 2H.
"""),
    code(r"""
means = np.array([strategy[mouse == m].mean() for m in np.unique(mouse)])
order = np.unique(mouse)[np.argsort(means)]
xpos = np.array([list(order).index(m) for m in mouse])
jitter = (np.arange(len(mouse)) % 7 - 3) * 0.03      # spread sessions within a column

plt.figure(figsize=(7.5, 4))
plt.scatter(xpos + jitter, strategy, c=mouse, cmap="tab20", s=35)
for i, m in enumerate(order):
    plt.hlines(strategy[mouse == m].mean(), i - 0.3, i + 0.3, color="k", lw=2)
plt.axhline(0, color="0.7", lw=0.8)
plt.xlabel("mouse (sorted by average strategy)")
plt.ylabel("session strategy (visual − timing)")
plt.title("sessions cluster by mouse (cf. Fig 2H)")
plt.show()
"""),
    md(r"""
## 2. Variance partitioning

We can make "clusters by mouse" quantitative. Split the total spread of the session
values into a part **between** mice (how far each mouse's average sits from the
grand average) and the rest **within** mice, and report the between-mouse share:

$$\text{variance explained by mouse} \;=\;
\frac{\sum_m n_m\,(\bar{x}_m - \bar{x})^2}{\sum_i (x_i - \bar{x})^2}
\;=\; \frac{\text{between-mouse sum of squares}}{\text{total sum of squares}}.$$

It runs from 0 (all mice identical on average) to 1 (no within-mouse spread at all).
This quantity is $\eta^2$ — the same thing a one-way ANOVA reports.

**Exercise 1.** Complete `variance_explained`.

> **Check / unstuck.** Expect **≈ 0.79** (matches `sb.variance_explained`). Stuck?
> Use `sb.variance_explained(strategy, mouse)`.
"""),
    code(
        solution=r"""
def variance_explained(values, groups):
    values, groups = np.asarray(values, float), np.asarray(groups)
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    ss_between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2
                     for g in np.unique(groups))
    return ss_between / ss_total

observed = variance_explained(strategy, mouse)
print(f"variance explained by mouse identity: {observed:.2f}")
print("backend check:", round(sb.variance_explained(strategy, mouse), 2))
""",
        student=r"""
def variance_explained(values, groups):
    values, groups = np.asarray(values, float), np.asarray(groups)
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    # YOUR CODE HERE: between-mouse sum of squares = sum over mice of
    # (n_m) * (mouse_mean - grand)^2 ; then return ss_between / ss_total
    raise NotImplementedError

observed = variance_explained(strategy, mouse)
print(f"variance explained by mouse identity: {observed:.2f}")
print("backend check:", round(sb.variance_explained(strategy, mouse), 2))
"""),
    md(r"""
## 3. Is it more than chance? A permutation test

A number like 0.8 *sounds* big, but we need a null: how much variance would mouse
identity "explain" if the labels were **meaningless**? A **permutation test**
answers this without any distributional assumptions. The recipe:

1. Compute the statistic on the real data (done above).
2. **Shuffle** the mouse labels — destroying any real link between session and
   mouse — and recompute the statistic. Repeat many times to trace out the null
   distribution: *what the statistic looks like when the structure is fake*.
3. The p-value is the fraction of shuffles that match or beat the real value.

**Exercise 2.** Complete `permutation_test`. Use `sb.variance_explained` as the
statistic (so this section works even if you skipped Exercise 1).

> **Check / unstuck.** Expect **observed ≈ 0.79, shuffled ≈ 0.23, p ≈ 0.0005**.
> Stuck? Use `sb.permutation_variance_explained(strategy, mouse)`.
"""),
    code(
        solution=r"""
def permutation_test(values, groups, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    observed = sb.variance_explained(values, groups)
    null = np.array([sb.variance_explained(values, rng.permutation(groups))
                     for _ in range(n_perm)])
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return observed, null, p

observed, null, p = permutation_test(strategy, mouse)
print(f"observed = {observed:.2f},  shuffled mean = {null.mean():.2f},  p = {p:.4f}")
""",
        student=r"""
def permutation_test(values, groups, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    observed = sb.variance_explained(values, groups)
    # YOUR CODE HERE: build `null` by recomputing sb.variance_explained on
    # rng.permutation(groups) n_perm times, then p = fraction of null >= observed
    # (use the +1/+1 correction). Return observed, null, p.
    raise NotImplementedError

observed, null, p = permutation_test(strategy, mouse)
print(f"observed = {observed:.2f},  shuffled mean = {null.mean():.2f},  p = {p:.4f}")
"""),
    code(r"""
plt.figure(figsize=(6, 3.6))
plt.hist(null, bins=30, color="0.7", label="shuffled (null)")
plt.axvline(observed, color="tab:red", lw=2, label=f"observed = {observed:.2f}")
plt.xlabel("variance explained by mouse identity"); plt.ylabel("count")
plt.title(f"permutation test (p = {p:.4f})"); plt.legend()
plt.show()
"""),
    md(r"""
The observed value sits far out in the tail of the null, so mouse identity explains
much more than chance — strategy is a **stable individual trait**, just as in the
paper. Note where the null piles up: with $k$ mice and $N$ sessions, shuffled labels
explain about $(k-1)/(N-1)$ of the variance on average — here $11/47 \approx 0.23$,
matching the paper's ~22%. That's the baseline any real effect must clear.
"""),
    md(r"""
## 4. (Optional) Check against the ground truth

*Skippable.* Because we built the cohort, we know each session's **true** strategy
tilt and each mouse's stable latent. Partitioning the *true* tilts tells us how much
structure is really there, and we can confirm the value we recovered from noisy
model fits lands in the same place.
"""),
    code(r"""
true_tilt = np.array([cs.session_tilt for cs in cohort])
print("variance explained using TRUE tilts     :", round(sb.variance_explained(true_tilt, mouse), 2))
print("variance explained using RECOVERED fits  :", round(sb.variance_explained(strategy, mouse), 2))
print("(close -> the fit recovered the individual structure, not just noise)")
"""),
    md(r"""
## Wrap-up

You quantified individual differences with variance partitioning and tested them
with a permutation test — a general, assumption-light tool you'll reach for
constantly. The key structural fact was that sessions are **nested within mice**.
The next two notebooks take that nesting seriously:

- **Notebook 10** — when you run *many* tests (one per strategy, say), how to keep
  false positives under control.
- **Notebook 11** — how to put honest error bars on a nested-data average with the
  hierarchical bootstrap.
"""),
]

student, solution = build("09_individual_differences", cells)
print("wrote:", student)
print("wrote:", solution)
