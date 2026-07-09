"""Notebook 3 -- Modeling the licks: the static strategy model."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 3 — Modeling the licks

*SWC ENC 2026 · behaviour module*

We can now build the design matrix $X$ (Notebook 1) and we've seen strategies
leave fingerprints in behavior (Notebook 2). But so far we *peeked at ground
truth* to know each mouse's strategy. This notebook is where we stop peeking and
**infer** the strategy from behavior, by fitting the model the paper is built on:
**logistic regression**.

**In this notebook you will:**
1. Write down the model and its **likelihood**.
2. Fit it by **maximizing** that likelihood.
3. **Recover** the true strategy weights of single-strategy mice.
4. See what a single static fit does to the *dynamic* mouse — the crack that
   motivates the rest of the day.
"""),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import swcbehav as sb

def get_Xy(sess):
    # Observable design matrix X and per-image outcome y for a session.
    X, cols = sb.build_design_matrix(sess.table)
    y = sess.table["bout_start"].to_numpy().astype(float)
    return X, y, cols
"""),
    md(r"""
## 1. The model and its likelihood

The model predicts the probability that a licking bout starts on image $t$ from
the design row $x_t$ and a weight vector $w$:

$$p_t = \sigma(w \cdot x_t), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Each image is a coin flip (Bernoulli): a bout starts ($y_t = 1$) with probability
$p_t$. Assuming images are independent given $w$, the probability of the whole
observed lick pattern is the product over images, and its logarithm — the
**log-likelihood** — is a sum:

$$\log \mathcal{L}(w) = \sum_t \Big[\, y_t \log p_t + (1 - y_t)\log(1 - p_t)\,\Big]
 = \sum_t \Big[\, y_t\,(w\cdot x_t) - \log\!\big(1 + e^{\,w\cdot x_t}\big) \Big].$$

The right-hand form is the one to implement: it avoids computing $p_t$ and then
taking its log (which overflows for confident predictions). We use
`np.logaddexp(0, z)` for $\log(1 + e^z)$, which is numerically stable.

**Exercise 1.** Complete the log-likelihood term inside `neg_log_likelihood`.
We return the *negative* log-likelihood (plus a tiny ridge penalty) because
optimizers minimize.
"""),
    code(
        solution=r"""
def neg_log_likelihood(w, X, y, l2=1e-4):
    z = X @ w                                   # w . x for every image
    log_lik = np.sum(y * z - np.logaddexp(0.0, z))
    return -log_lik + l2 * np.sum(w ** 2)       # ridge = weak Gaussian prior (MAP)
""",
        student=r"""
def neg_log_likelihood(w, X, y, l2=1e-4):
    z = X @ w                                   # w . x for every image
    # YOUR CODE HERE: log_lik = sum_t [ y_t * z_t - log(1 + exp(z_t)) ]
    # use np.logaddexp(0.0, z) for the log(1 + exp(z)) term.
    raise NotImplementedError
    return -log_lik + l2 * np.sum(w ** 2)
""",
    ),
    code(r"""
# Sanity check: at w = 0 every p_t = 0.5, so the log-likelihood is n*log(0.5).
Xv, yv, cols = get_Xy(sb.make_mouse("visual", seed=0))
w0 = np.zeros(Xv.shape[1])
print("NLL at w=0:", round(neg_log_likelihood(w0, Xv, yv), 1))
print("expected  :", round(-len(yv) * np.log(0.5), 1))
"""),
    md(r"""
## 2. Fitting by maximum likelihood

Fitting means finding the $w$ that makes the observed licking most probable —
the $w$ that minimizes the negative log-likelihood. The function is convex, so a
generic gradient-based optimizer lands on the global optimum. (The small ridge
penalty makes this the *maximum a posteriori* estimate under a weak Gaussian
prior — exactly the static model in the paper.)

**Exercise 2.** Complete `fit_static` using `scipy.optimize.minimize`.
"""),
    code(
        solution=r"""
def fit_static(X, y, l2=1e-4):
    result = minimize(neg_log_likelihood, np.zeros(X.shape[1]),
                      args=(X, y, l2), method="L-BFGS-B")
    return result.x
""",
        student=r"""
def fit_static(X, y, l2=1e-4):
    # YOUR CODE HERE: minimize neg_log_likelihood over w, starting from zeros.
    # Pass (X, y, l2) via args=... and return the fitted weight vector.
    raise NotImplementedError
""",
    ),
    code(r"""
w_fit = fit_static(Xv, yv)
w_true = sb.make_mouse("visual", seed=0).true_weights[0]
print(f"{'strategy':14s}{'true':>7s}{'fit':>8s}")
for c, t, f in zip(cols, w_true, w_fit):
    print(f"{c:14s}{t:7.1f}{f:8.2f}")
"""),
    md(r"""
The fit puts nearly all the weight on **visual**, with bias near $-3$ and the
other strategies near zero — it has **recovered the strategy from behavior
alone**, without ever seeing the true weights. This "can we recover the truth?"
check is the backbone of the whole course.

## 3. Recovery across strategies

Does it work for every single-strategy mouse? Fit all three and compare fitted to
true weights.
"""),
    code(r"""
archetypes = ["visual", "timing", "mixed"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
xpos = np.arange(len(cols))
for ax, name in zip(axes, archetypes):
    sess = sb.make_mouse(name, seed=0)
    X, y, _ = get_Xy(sess)
    w_fit = fit_static(X, y)
    w_true = sess.true_weights[0]
    ax.bar(xpos - 0.2, w_true, 0.4, label="true", color="0.6")
    ax.bar(xpos + 0.2, w_fit, 0.4, label="fit", color="tab:red")
    ax.set_xticks(xpos); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8); ax.set_title(f"{name} mouse")
axes[0].set_ylabel("weight"); axes[0].legend()
plt.tight_layout(); plt.show()
"""),
    md(r"""
Fitted bars track the true bars across all three mice. A few things worth saying
out loud:

* The weights are **log-odds**. A bias of $-3$ means a baseline bout probability
  of $\sigma(-3) \approx 0.05$; adding a visual weight of $+5$ on a change image
  gives $\sigma(-3 + 5) = \sigma(2) \approx 0.88$ — the hit rate we measured in
  Notebook 2.
* The **omission** and **post-omission** weights wobble more around their true
  value of zero. Those events are rare, so there's little data to pin them down —
  the same reason the paper found the omission strategy weakly determined.
"""),
    md(r"""
## 4. A single fit meets the dynamic mouse

Everything above assumed the strategy is *fixed*. What happens if we fit one
static $w$ to the **dynamic** mouse, whose strategy drifts? The optimizer returns
the single weight vector that best explains the whole hour at once — a blurry
average that lands *between* visual and timing and matches the animal at no point
in particular.
"""),
    code(r"""
dyn = sb.make_mouse("dynamic", seed=1)
Xd, yd, _ = get_Xy(dyn)
w_static = fit_static(Xd, yd)

sb.plotting.plot_weights(true_weights=dyn.true_weights, col_names=sb.WEIGHT_NAMES)
for k, name in enumerate(sb.WEIGHT_NAMES):
    plt.axhline(w_static[k], color=plt.cm.tab10(k), ls=":", lw=1.5)
plt.title("dynamic mouse: drifting truth (solid) vs one static fit (dotted)")
plt.show()
print("static fit visual =", round(w_static[1], 2),
      " timing =", round(w_static[4], 2))
"""),
    md(r"""
The dotted static estimates sit flat while the true visual and timing weights
cross right past them. The model isn't wrong — it's answering the wrong question,
forcing one number onto a moving target.

## Wrap-up

You built the strategy model from its likelihood, fit it by maximizing that
likelihood, and recovered the true weights of single-strategy mice. You also saw
the static model's blind spot: a mouse whose strategy changes.

**Next (Notebook 4):** before trusting any of these fits, we need to *evaluate*
them — how well does the model actually predict held-out licking? That's the ROC
curve, AUC, and cross-validation.
"""),
]

student, solution = build("03_static_model", cells)
print("wrote:", student)
print("wrote:", solution)