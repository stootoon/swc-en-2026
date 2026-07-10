"""Notebook 2 -- Meet the mice: how strategies show up in behavior."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 2 — Meet the mice

*SWC ENC 2026 · behaviour module*

In Notebook 1 we treated `make_mouse` as a black box. Now we open it. The whole
course rests on synthetic mice whose **true strategy we choose**, so it's worth
seeing exactly how a strategy is encoded and what each one looks like in
behavior — *before* we try to infer it by fitting (Notebook 3 onward).

**In this notebook you will:**
1. See the generative model: a strategy is just a **weight vector**.
2. Read the behavioral **signature** of the visual and timing strategies.
3. Recover the timing strategy straight from data, without peeking at truth.
4. Meet the **dynamic** mouse whose strategy drifts across the session.

---
**The paper panel this notebook reproduces**

<img src="../assets/paper/fig1b.png" width="560">

*Fig. 1B — licking aligned to task events (image changes, omissions, and the number
of images since the last bout). Each strategy leaves a different fingerprint here;
this notebook builds the intuition for those signatures.*
"""),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import swcbehav as sb
"""),
    md(r"""
## 1. A strategy is a weight vector

Every mouse licks according to the same rule — the model from the paper, run
forward:

$$p(\text{bout starts on image } t) = \sigma\big(w \cdot x_t\big),$$

where $x_t$ is the five-strategy design row from Notebook 1 and
$w = [\,\text{bias},\ \text{visual},\ \text{omission},\ \text{post\_omission},\
\text{timing}\,]$ are the **strategy weights**. A *static* mouse uses the same
$w$ on every image. Different mice are just different $w$:
"""),
    code(r"""
for name in ["visual", "timing", "mixed"]:
    w = sb.make_mouse(name, seed=0).true_weights[0]   # constant across the session
    print(f"{name:7s}  w = {np.round(w, 1)}")
print("\norder: [bias, visual, omission, post_omission, timing]")
"""),
    md(r"""
Read those off: every mouse has a strongly negative **bias** (licking is rare by
default). The **visual** mouse adds weight only to the *visual* column; the
**timing** mouse only to the *timing* column; the **mixed** mouse to both. That
single difference is what we'll be trying to detect for the rest of the day.
"""),
    md(r"""
## 2. Behavioral signatures

Before any modeling, how does each strategy *look*? The two simplest summaries:

* **hit rate** — fraction of image changes the mouse licked to.
* **false-alarm rate** — fraction of non-change images it licked on anyway.

**Exercise 1.** Complete `strategy_signature`.

> **Check / unstuck.** Expect the visual mouse near **0.88 hits / 0.05 false
> alarms**, the timing mouse near **0.21 / 0.20**. Stuck? The solutions copy has it.
"""),
    code(
        solution=r"""
def strategy_signature(sess):
    t = sess.table
    change = t["is_change"].to_numpy()
    bout = t["bout_start"].to_numpy()
    hit_rate = bout[change].mean()
    false_alarm = bout[~change].mean()
    return hit_rate, false_alarm
""",
        student=r"""
def strategy_signature(sess):
    t = sess.table
    change = t["is_change"].to_numpy()
    bout = t["bout_start"].to_numpy()
    # YOUR CODE HERE: hit_rate = P(bout | change), false_alarm = P(bout | not change)
    raise NotImplementedError
""",
    ),
    code(r"""
print(f"{'mouse':7s}  {'hit rate':>8s}  {'false alarm':>11s}")
for name in ["visual", "timing", "mixed"]:
    hr, fa = strategy_signature(sb.make_mouse(name, seed=0))
    print(f"{name:7s}  {hr:8.2f}  {fa:11.2f}")
"""),
    md(r"""
The **visual** mouse is a crisp detector — high hits, few false alarms. The
**timing** mouse licks on a schedule, so it catches fewer changes *and* fires
often between them (high false-alarm rate). You can see the same thing in the
raw rasters: the visual mouse's licks cluster on the blue change lines, while the
timing mouse's are spread rhythmically across the stream.
"""),
    code(r"""
fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
sb.plotting.plot_session_raster(sb.make_mouse("visual", seed=0), 560, 585, ax=axes[0])
axes[0].set_title("visual mouse"); axes[0].set_xlabel("")
sb.plotting.plot_session_raster(sb.make_mouse("timing", seed=0), 560, 585, ax=axes[1])
axes[1].set_title("timing mouse")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 3. Recovering the timing strategy from data

The signatures above hint at the strategies but don't isolate them. The timing
strategy makes a sharp, testable prediction: **licking probability should climb
with the time waited since the last bout.** We can measure that directly from the
data — no ground truth needed — by binning images on *images since last bout*
and asking how often a bout starts in each bin.

**Exercise 2.** Complete `lick_prob_by_wait`.
*Hint:* `sb.images_since_bout(bout_start)` gives the waiting count for every image.

> **Check / unstuck.** The timing mouse's curve should climb with waiting time while
> the visual mouse's stays roughly flat. Stuck? The solutions copy has it.
"""),
    code(
        solution=r"""
def lick_prob_by_wait(sess, max_wait=10):
    bout = sess.table["bout_start"].to_numpy()
    since = sb.images_since_bout(bout)
    waits = np.arange(max_wait + 1)
    probs = np.array([bout[since == k].mean() if np.any(since == k) else np.nan
                      for k in waits])
    return waits, probs
""",
        student=r"""
def lick_prob_by_wait(sess, max_wait=10):
    bout = sess.table["bout_start"].to_numpy()
    since = sb.images_since_bout(bout)
    waits = np.arange(max_wait + 1)
    # YOUR CODE HERE: for each wait k, compute P(bout start | since == k)
    raise NotImplementedError
""",
    ),
    code(r"""
plt.figure(figsize=(6, 3.5))
for name in ["visual", "timing"]:
    waits, probs = lick_prob_by_wait(sb.make_mouse(name, seed=0))
    plt.plot(waits, probs, "o-", color=sb.WEIGHT_COLORS[name], label=name)
plt.xlabel("images since last bout"); plt.ylabel("P(bout starts)")
plt.title("the timing strategy, read from behavior"); plt.legend()
plt.show()
"""),
    md(r"""
There it is: the **timing** mouse's curve rises steeply with waiting time (the
sigmoid we built into the design matrix), while the **visual** mouse's stays
flat — its licking is driven by image changes, not by the clock. This is the
essence of what the full model will do, one strategy at a time and all at once.
"""),
    md(r"""
## 4. The dynamic mouse

Real mice don't hold one strategy forever. Our **dynamic** mouse starts
visual-dominant and drifts to timing-dominant. Because we generated it, we can
look at its *true* weights over the session (something we'll have to **infer**
for real data — that's the climax in Notebook 7):
"""),
    code(r"""
dyn = sb.make_mouse("dynamic", seed=1)
sb.plotting.plot_weights(true_weights=dyn.true_weights, col_names=sb.WEIGHT_NAMES)
plt.title("dynamic mouse: true strategy weights drift across the session")
plt.show()
"""),
    md(r"""
The visual weight falls and the timing weight rises — the dominant strategy
**flips** partway through. A single static fit (Notebooks 3–6) cannot express
this; watch what it does to one when we get there. For now, just confirm the
drift is visible in behavior by comparing early vs late in the session:
"""),
    code(r"""
n = len(dyn.table)
early, late = dyn.table.iloc[:n // 3], dyn.table.iloc[-n // 3:]
for label, seg in [("first third", early), ("last third", late)]:
    change = seg["is_change"].to_numpy(); bout = seg["bout_start"].to_numpy()
    print(f"{label:12s}  hit rate {bout[change].mean():.2f}   "
          f"false alarm {bout[~change].mean():.2f}")

fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=False)
sb.plotting.plot_session_raster(dyn, 40, 65, ax=axes[0])
axes[0].set_title("early: change-locked (visual)")
sb.plotting.plot_session_raster(dyn, 3500, 3525, ax=axes[1])
axes[1].set_title("late: rhythmic (timing)")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## Wrap-up

You've seen that a strategy is nothing more than a weight vector, and that
different weights leave different fingerprints in behavior — crisp change-locked
licking for the visual mouse, rhythmic licking for the timing mouse, and a moving
target for the dynamic mouse.

So far we've *peeked at ground truth* to know each mouse's strategy. The rest of
the day is about **inferring** it from behavior alone.

**Next (Notebook 3):** fit the static strategy model — write down its likelihood,
maximize it, and recover the weights of a single-strategy mouse.
"""),
]

student, solution = build("02_meet_the_mice", cells)
print("wrote:", student)
print("wrote:", solution)