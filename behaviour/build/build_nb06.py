"""Notebook 6 -- Stress-testing the cycle on mixed-strategy mice."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 6 — Mixtures

*SWC ENC 2026 · behaviour module*

Real mice don't use a single pure strategy. Notebooks 3–5 built the
**fit → evaluate → ablate** cycle and checked it on clean single-strategy mice.
Now we stress-test it on mice that genuinely *blend* the visual and timing
strategies, and ask the sharper question: does the **strategy index** track the
true blend?

Everything here reuses the backend you've already built — `fit_static`,
`cross_val_auc`, `ablation_loglik_deltas`. The only new thing is that we dial the
mouse's strategy continuously.

**In this notebook you will:**
1. Generate mice along a **continuum** from pure-visual to pure-timing.
2. Confirm the recovered **strategy index** tracks the true blend.
3. Populate the strategy-space cloud of Figure 2F.
"""),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import swcbehav as sb

def get_Xy(sess):
    X, cols = sb.build_design_matrix(sess.table)
    y = sess.table["bout_start"].to_numpy().astype(float)
    return X, y, cols
"""),
    md(r"""
## 1. A continuum of mice

A static mouse is just a weight vector, so we can build any blend we like with
the public generator: `make_session` for the stimulus, `constant_weights` for the
strategy, `simulate` to produce behavior. We parameterize the blend by
$\alpha \in [0, 1]$: $\alpha = 1$ is pure visual, $\alpha = 0$ is pure timing,
holding the total strategy strength fixed.

**Exercise 1.** Complete `make_blend_mouse`.
"""),
    code(
        solution=r"""
def make_blend_mouse(alpha, seed=0, total=5.0, bias=-3.0, n_images=4800):
    stim = sb.make_session(n_images=n_images, seed=seed)
    weights = sb.constant_weights(bias=bias, visual=total * alpha,
                                  timing=total * (1 - alpha))
    return sb.simulate(stim, weights, seed=seed + 1)
""",
        student=r"""
def make_blend_mouse(alpha, seed=0, total=5.0, bias=-3.0, n_images=4800):
    stim = sb.make_session(n_images=n_images, seed=seed)
    # YOUR CODE HERE: build constant weights with visual = total*alpha and
    # timing = total*(1-alpha), then simulate and return the Session.
    raise NotImplementedError
""",
    ),
    code(r"""
# Behavioral signatures shift smoothly as we slide from timing to visual.
print(f"{'alpha':>6s}{'hit rate':>10s}{'false alarm':>13s}")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    t = make_blend_mouse(alpha, seed=0).table
    ch = t["is_change"].to_numpy(); b = t["bout_start"].to_numpy()
    print(f"{alpha:6.2f}{b[ch].mean():10.2f}{b[~ch].mean():13.2f}")
"""),
    md(r"""
## 2. Does the strategy index track the blend?

Run the ablation cycle on each mouse and compute the strategy index
$\Delta_{\text{visual}} - \Delta_{\text{timing}}$. If the method works, the index
should rise monotonically with $\alpha$ — recovering the blend we dialed in.

**Exercise 2.** Complete `strategy_index` using `sb.ablation_loglik_deltas`.
"""),
    code(
        solution=r"""
def strategy_index(sess):
    X, y, cols = get_Xy(sess)
    d = sb.ablation_loglik_deltas(X, y, cols)
    return d["visual"], d["timing"], d["visual"] - d["timing"]
""",
        student=r"""
def strategy_index(sess):
    X, y, cols = get_Xy(sess)
    d = sb.ablation_loglik_deltas(X, y, cols)
    # YOUR CODE HERE: return (visual_index, timing_index, strategy_index)
    raise NotImplementedError
""",
    ),
    code(r"""
alphas = np.linspace(0, 1, 9)
indices = np.array([strategy_index(make_blend_mouse(a, seed=0))[2] for a in alphas])

plt.figure(figsize=(5, 3.6))
plt.plot(alphas, indices, "o-")
plt.axhline(0, color="0.7", lw=0.8)
plt.xlabel(r"true blend $\alpha$  (0 = timing, 1 = visual)")
plt.ylabel("recovered strategy index")
plt.title("the index tracks the true blend")
plt.show()
"""),
    md(r"""
The recovered index climbs steadily from negative (timing-dominant) through zero
(balanced) to positive (visual-dominant) as we slide $\alpha$ from 0 to 1. The
method reads the blend out of behavior — this is the recovery guarantee that lets
us trust the same index on *real* mice, where there is no $\alpha$ to check
against.

## 3. The strategy-space cloud (Figure 2F)

Finally, populate the visual-vs-timing plane with a spread of blends and seeds,
colored by the strategy index — the synthetic counterpart of the paper's
Figure 2F.
"""),
    code(r"""
plt.figure(figsize=(5.4, 4.6))
vis, tim, strat = [], [], []
for alpha in np.linspace(0, 1, 7):
    for seed in range(3):
        vi, ti, si = strategy_index(make_blend_mouse(alpha, seed=seed))
        vis.append(vi); tim.append(ti); strat.append(si)
sc = plt.scatter(vis, tim, c=strat, cmap="coolwarm", s=45, edgecolor="0.3", lw=0.3)
plt.colorbar(sc, label="strategy index")
plt.xlabel("visual index"); plt.ylabel("timing index")
plt.axline((0, 0), slope=1, color="0.7", ls="--")
plt.title("strategy space (cf. Fig 2F)")
plt.tight_layout(); plt.show()
"""),
    md(r"""
Mice fan out along an anti-diagonal band: as one strategy strengthens the other
weakens, and the strategy index (color) sweeps from blue to red across the cloud
— exactly the structure Piet et al. report across their real sessions.

One honest caveat worth raising with students: visual and timing are not perfectly
separable. A lick on a change that also happens to follow a long wait is evidence
for *either* strategy, so at intermediate blends the two indices trade off with
some noise. Held-out prediction keeps this in check, but it's a real limit of
inferring latent strategy from behavior alone.

## Wrap-up

The static cycle survives contact with mixed strategies: the strategy index
recovers the blend and lays out the Figure 2F cloud. But every mouse so far has
held its strategy *fixed* for the whole session.

**Next (Notebook 7 — the climax):** the dynamic mouse, whose strategy drifts. We
watch the static model fail, discover the fix by fitting it in sliding windows,
and then tie those windows together into the dynamic model.
"""),
]

student, solution = build("06_mixtures", cells)
print("wrote:", student)
print("wrote:", solution)
