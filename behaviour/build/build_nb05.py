"""Notebook 5 -- Which strategies matter? Ablation (and, optionally, evidence)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 5 — Which strategies matter?

*SWC ENC 2026 · behaviour module*

The AUC in Notebook 4 told us *how well* the model predicts, but not *which of
the five strategies* is doing the work. To find out, we **ablate**: remove a
strategy, refit, and see how much the model's predictions suffer. The strategies
whose removal hurts most are the ones the mouse is really using.

We'll measure "how much it hurts" with the **held-out log-likelihood** you
already built in Notebook 4 — a fair, overfitting-resistant currency. (The paper
uses a more elaborate quantity, *model evidence*; that's the optional advanced
section at the end, for anyone who wants the full picture.)

**In this notebook you will:**
1. Score a model by its cross-validated **log-likelihood**.
2. **Ablate** each strategy and read off how much it contributes.
3. Turn those contributions into the paper's **strategy index** (Fig 2D–F).
4. *(Optional, advanced)* Redo the ablation with **model evidence**.
"""),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import swcbehav as sb

def get_Xy(sess):
    # Observable design matrix, per-flash outcome, and column names.
    X, cols = sb.build_design_matrix(sess.table)
    y = sess.table["bout_start"].to_numpy().astype(float)
    return X, y, cols
"""),
    md(r"""
## 1. Scoring a model by held-out log-likelihood

Cross-validated **log-likelihood** measures how much probability the model
assigns, on average, to licking it *didn't* see during fitting. It's reported
here as nats per flash — higher (closer to zero) is better. Unlike AUC it's a
*proper* score: it rewards well-calibrated probabilities, not just correct
ranking, which makes differences between nested models meaningful.
"""),
    code(r"""
sess = sb.make_mouse("visual", seed=0)
X, y, cols = get_Xy(sess)
full_ll = sb.cross_val_loglik(X, y)
print(f"full model, cross-validated log-likelihood: {full_ll:.4f} nats/flash")
"""),
    md(r"""
## 2. Ablation

To ask what a strategy contributes, drop its column from the design matrix,
refit, and re-score. The **drop in held-out log-likelihood**,

$$\Delta_{\text{strategy}} = \text{LL}_{\text{full}} - \text{LL}_{\text{without strategy}},$$

is how much worse the model predicts without it. Large positive $\Delta$ = the
mouse really relies on that strategy; near zero = it's not being used.

**Exercise 1.** Complete `ablation_deltas`: for each non-bias strategy, build the
design matrix without that column and compute $\Delta$.
*Hint:* `np.delete(X, k, axis=1)` removes column `k`.
"""),
    code(
        solution=r"""
def ablation_deltas(X, y, cols, n_folds=5):
    full_ll = sb.cross_val_loglik(X, y, n_folds)
    deltas = {}
    for k, name in enumerate(cols):
        if name == "bias":
            continue
        X_without = np.delete(X, k, axis=1)
        deltas[name] = full_ll - sb.cross_val_loglik(X_without, y, n_folds)
    return deltas
""",
        student=r"""
def ablation_deltas(X, y, cols, n_folds=5):
    full_ll = sb.cross_val_loglik(X, y, n_folds)
    deltas = {}
    for k, name in enumerate(cols):
        if name == "bias":
            continue
        # YOUR CODE HERE: build X without column k, then set
        # deltas[name] = full_ll - cross-validated log-likelihood of that reduced model
        raise NotImplementedError
    return deltas
""",
    ),
    code(r"""
deltas = ablation_deltas(X, y, cols)
names = list(deltas)
plt.figure(figsize=(5, 3.2))
plt.bar(names, [deltas[n] for n in names], color="tab:red")
plt.ylabel("drop in held-out LL"); plt.title("visual mouse: strategy contributions")
plt.xticks(rotation=45, ha="right"); plt.axhline(0, color="k", lw=0.8)
plt.tight_layout(); plt.show()
for n in names:
    print(f"  {n:14s} {deltas[n]:+.4f}")
"""),
    md(r"""
Exactly what we'd hope: removing **visual** collapses the model's predictions,
while removing omission, post-omission, or timing barely registers. On a
single-strategy mouse the ablation points, unambiguously, at the one strategy we
built in — the clean result that makes this the right place to trust the method.

## 3. The strategy index

Piet et al. summarize each session with two numbers — how much the model leans on
the **visual** vs the **timing** strategy — and their difference, the
**strategy index**:

$$\text{strategy index} = \Delta_{\text{visual}} - \Delta_{\text{timing}}.$$

Positive = visual-dominant, negative = timing-dominant. Let's compute it across a
population of single-strategy mice and reproduce the layout of Figure 2F.
"""),
    code(
        solution=r"""
def strategy_index(sess, n_folds=5):
    X, y, cols = get_Xy(sess)
    d = ablation_deltas(X, y, cols, n_folds)
    return d["visual"], d["timing"], d["visual"] - d["timing"]
""",
        student=r"""
def strategy_index(sess, n_folds=5):
    X, y, cols = get_Xy(sess)
    d = ablation_deltas(X, y, cols, n_folds)
    # YOUR CODE HERE: return (visual_index, timing_index, strategy_index)
    raise NotImplementedError
""",
    ),
    code(r"""
plt.figure(figsize=(5.2, 4.4))
colors = {"visual": "tab:green", "timing": "tab:blue", "mixed": "tab:purple"}
for name in ["visual", "timing", "mixed"]:
    for seed in range(4):
        vi, ti, si = strategy_index(sb.make_mouse(name, seed=seed))
        plt.scatter(vi, ti, color=colors[name], s=40,
                    label=name if seed == 0 else None)
plt.xlabel("visual index (drop in LL)"); plt.ylabel("timing index (drop in LL)")
plt.title("strategy space (cf. Fig 2F)"); plt.legend()
plt.axline((0, 0), slope=1, color="0.7", ls="--")
plt.tight_layout(); plt.show()
"""),
    md(r"""
Visual mice sit in the bottom-right (high visual index, low timing), timing mice
in the top-left, and mixed mice near the diagonal — each strategy lands where it
should. In Notebook 6 we'll push this on mice that genuinely blend strategies.
"""),
    md(r"""
---
## 4. (Advanced, optional) Model evidence

*This section is for students who want the paper's actual method — skip it
without loss of continuity.*

Ablation by held-out log-likelihood is intuitive and reuses the cross-validation
you already know. The paper instead compares models by their **evidence** (the
marginal likelihood): the probability the model assigns to the data after
**integrating the weights out** under a prior,

$$p(y \mid \text{model}) = \int p(y \mid w)\, p(w)\, dw.$$

Integrating over all possible weights — rather than fitting one best set —
automatically penalizes models with spare parameters they don't need (a built-in
Occam's razor), so no separate cross-validation is required. The integral is
intractable for logistic regression, so we approximate it with a Gaussian around
the MAP fit (the **Laplace approximation**); `sb.log_evidence_laplace` does this,
combining fit quality, the prior, and a complexity (volume) term.

Redoing the ablation with evidence gives the same qualitative verdict:
"""),
    code(r"""
def evidence_deltas(X, y, cols, prior_var=100.0):
    full_ev = sb.log_evidence_laplace(X, y, prior_var)
    out = {}
    for k, name in enumerate(cols):
        if name == "bias":
            continue
        out[name] = full_ev - sb.log_evidence_laplace(np.delete(X, k, axis=1), y, prior_var)
    return out

ev = evidence_deltas(X, y, cols)
print("change in log-evidence when each strategy is removed:")
for n in names:
    print(f"  {n:14s} evidence {ev[n]:+9.1f}   held-out-LL {deltas[n]:+.4f}")
"""),
    md(r"""
Both methods crown **visual** by a wide margin and rank the rest near zero. The
evidence differences are large in absolute terms (they sum log-probabilities over
all 4800 flashes rather than averaging), which is why the paper reports them as a
**percent change in evidence** (Fig 2D). The takeaway is the same: on these mice,
one strategy carries the model.

## Wrap-up

You measured what each strategy contributes by ablation, and distilled it into
the strategy index that organizes the paper's Figure 2 — verified on mice whose
answer we already knew.

**Next (Notebook 6):** stress-test the whole fit → evaluate → ablate cycle on
mice that truly mix strategies, where the index has to earn its keep.
"""),
]

student, solution = build("05_evidence_and_ablation", cells)
print("wrote:", student)
print("wrote:", solution)
