"""Notebook 2 -- Many channels: the macroscopic current and its fluctuations."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 2 — Many channels

*SWC ENC 2026 · ephys-one module*

A real patch holds not one channel but tens to thousands. Each is the random switch
of Notebook 1, flipping independently of the others. The current you record is their
**sum**. This notebook is about what that sum looks like — and, above all, how much
it *fluctuates*, because the fluctuations are where the information is.

**In this notebook you will:**
1. Watch the macroscopic current emerge as N grows from 1 to 1000.
2. Run the **ensemble experiment**: repeat a voltage step, stack the sweeps, and
   compute the **mean and variance across sweeps** at each time point (**isochrones**).
3. Understand the two formulas the whole method rests on,
   $\langle I\rangle = N i p$ and $\sigma^2 = N i^2 p(1-p)$, from coin flips.
4. See why bigger patches look smoother: relative noise falls as $1/\sqrt{N}$.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picopatch as pp

scheme = pp.two_state()                                   # the voltage-dependent channel of NB1
protocol = pp.step_protocol(v_hold=-80, v_step=65)        # 2 ms at -80, then 12 ms at +65
""",),
    md(r"""
## 1. From one switch to a thousand

Put N channels in the patch and apply the same voltage step. At each instant every
channel is open with the same probability $p(t)$ — the curve we met at the end of
Notebook 1 — but each decides for itself. The current is $i$ times **the number of
channels open right now**. Watch what happens to one sweep as N grows:
""",),
    code(r"""
i_unit = 1.0
fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
for ax, N in zip(axes, [1, 3, 10, 100, 1000]):
    exp = pp.make_experiment(N=N, i=i_unit, scheme=scheme, protocol=protocol, n_sweeps=1, seed=N)
    ax.plot(exp.t_ms, exp.sweeps[0], color="k", lw=0.8, drawstyle="steps-post")
    ax.set_ylabel("pA"); ax.set_title(f"N = {N} channel{'s' if N > 1 else ''}", loc="left", fontsize=10)
    ax.axvline(2, color="tab:purple", ls=":", lw=1)
axes[-1].set_xlabel("time (ms)  (step to +65 mV at 2 ms)")
plt.tight_layout(); plt.show()
""",),
    md(r"""
With one channel you see the switch. With three, a staircase: the current takes
values 0, 1, 2, 3 pA as channels open and close. With a hundred the steps blur into a
rough rise; with a thousand the trace is a smooth, textbook **macroscopic current**
— but look closely at the plateau and it still wiggles. Those wiggles are a thousand
channels flickering, and they *shrink relative to the mean* as N grows. Hold that
thought for section 4.
""",),
    md(r"""
## 2. The ensemble experiment: sweeps and isochrones

One sweep is one random draw. To get at the statistics we repeat the identical step
many times — here 300 — and stack the recordings into a matrix of shape
`(n_sweeps, n_samples)`. Here is the stack, shown two ways: overlaid, and as an image
with one row per sweep.
""",),
    code(r"""
exp = pp.make_experiment(N=1000, i=1.0, scheme=scheme, protocol=protocol, n_sweeps=300, seed=0)
print("sweeps:", exp.sweeps.shape, "(n_sweeps, n_samples)")

pp.plotting.plot_sweeps(exp, n_show=30, isochrones_ms=[3.0, 12.0])
plt.show()
pp.plotting.plot_sweep_image(exp)
plt.show()
""",),
    md(r"""
An **isochrone** ("same time") is a vertical slice through the stack: the 300 current
values recorded at one particular time after the step, one from each sweep. The two
dashed lines above mark two of them. Along an isochrone the sweeps disagree with each
other a little — let's look at the spread of values on each:
""",),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, tm in zip(axes, [3.0, 12.0]):
    k = int(np.argmin(np.abs(exp.t_ms - tm)))
    vals = exp.sweeps[:, k]
    ax.hist(vals, bins=25, color="tab:blue", alpha=0.7)
    ax.axvline(vals.mean(), color="k", lw=2, label=f"mean = {vals.mean():.0f} pA")
    ax.axvspan(vals.mean() - vals.std(), vals.mean() + vals.std(), color="tab:red", alpha=0.15,
               label=f"± 1 SD = {vals.std():.1f} pA")
    ax.set_title(f"isochrone at t = {tm:.0f} ms: the 300 values across sweeps")
    ax.set_xlabel("current (pA)"); ax.set_ylabel("sweeps"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
""",),
    md(r"""
Two numbers summarise each isochrone:

- the **mean** across sweeps, $\langle I \rangle(t)$ — the smooth macroscopic current;
- the **variance** across sweeps, $\sigma^2(t)$ — the average *squared* distance of
  the sweeps from that mean. Its square root is the standard deviation (SD), the
  typical size of the wiggle, in pA; the variance itself is in pA².

Notice in the histograms that the spread is **larger at 3 ms** (where about half the
channels are open) **than at 12 ms** (where nearly all are) even though the mean is
smaller. That is not an accident; it's the whole story, and section 3 explains it.

**Exercise 1** *(~4 min)*. Complete `isochrone_stats`: return the mean and the
variance across sweeps at every time point (i.e. along axis 0). Use `ddof=1` for the
variance (the usual sample variance). Then plot both against time.

> **Check / unstuck.** Mean should rise from 0 to ~960 pA; variance should peak
> around 250 pA² during the rise and settle near 40 pA² on the plateau. Stuck?
> `pp.isochrone_stats(sweeps)`.
""",),
    code(
        solution=r"""
def isochrone_stats(sweeps):
    mean = sweeps.mean(axis=0)
    var = sweeps.var(axis=0, ddof=1)
    return mean, var

mean, var = isochrone_stats(exp.sweeps)
pp.plotting.plot_mean_variance(exp.t_ms, mean, var)
plt.show()
print(f"peak variance {var.max():.0f} pA²  at t = {exp.t_ms[np.argmax(var)]:.2f} ms;  plateau variance ~{var[-100:].mean():.0f} pA²")
""",
        student=r"""
def isochrone_stats(sweeps):
    # YOUR CODE HERE: mean and variance (ddof=1) across sweeps, at each time point
    raise NotImplementedError

mean, var = isochrone_stats(exp.sweeps)
pp.plotting.plot_mean_variance(exp.t_ms, mean, var)
plt.show()
print(f"peak variance {var.max():.0f} pA²  at t = {exp.t_ms[np.argmax(var)]:.2f} ms;  plateau variance ~{var[-100:].mean():.0f} pA²")
""",
    ),
    md(r"""
The variance has a shape of its own: zero before the step, a **bump during the rise**,
then down to a lower plateau. The bump is where the channels are most "undecided".

## 3. Why the variance looks like that: coin flips

Forget channels for a moment. Toss **N = 10 coins**, each landing heads with
probability $p$, and count the heads. Do it many times. The count is random; what are
its mean and variance?

- **Mean**: each coin contributes $p$ heads on average, so the mean count is $N p$.
- **Variance**: a single coin is a 0/1 variable with variance $p(1-p)$ — zero if the
  coin always lands the same way ($p = 0$ or $1$), largest when $p = 0.5$. Independent
  coins add their variances, so the variance of the count is $N p (1-p)$.

That count is called a **binomial** random variable. Let's check both claims by
simulation, sweeping $p$ from 0 to 1:
""",),
    code(r"""
rng = np.random.default_rng(1)
N_coins, n_rep = 10, 5000
ps = np.linspace(0.02, 0.98, 25)
mean_heads = []; var_heads = []
for p in ps:
    heads = (rng.random((n_rep, N_coins)) < p).sum(axis=1)       # number of heads in each of n_rep tosses
    mean_heads.append(heads.mean()); var_heads.append(heads.var(ddof=1))

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
# the distribution of the count at p = 0.5
heads = (rng.random((n_rep, N_coins)) < 0.5).sum(axis=1)
axes[0].hist(heads, bins=np.arange(-0.5, N_coins + 1.5), density=True, color="tab:blue", alpha=0.7, label="simulated")
from scipy.stats import binom
axes[0].plot(np.arange(N_coins + 1), binom.pmf(np.arange(N_coins + 1), N_coins, 0.5), "ko", label="binomial")
axes[0].set_title("10 coins, p = 0.5: number of heads"); axes[0].set_xlabel("heads"); axes[0].legend()
axes[1].plot(ps, mean_heads, "o", color="tab:blue", label="simulated"); axes[1].plot(ps, N_coins * ps, "k-", label="N·p")
axes[1].set_title("mean number of heads"); axes[1].set_xlabel("p"); axes[1].legend()
axes[2].plot(ps, var_heads, "o", color="tab:blue", label="simulated"); axes[2].plot(ps, N_coins * ps * (1 - ps), "k-", label="N·p·(1−p)")
axes[2].set_title("variance of the number of heads"); axes[2].set_xlabel("p"); axes[2].legend()
plt.tight_layout(); plt.show()
""",),
    md(r"""
The variance against $p$ is an upside-down parabola: nothing to fluctuate when the
coins are all heads or all tails, maximal uncertainty at $p = 0.5$. Now replace
"coin" with "channel", "heads" with "open", and multiply the count by the unitary
current $i$ (which multiplies the mean by $i$ and the variance by $i^2$):

$$\boxed{\;\langle I \rangle = N\, i\, p \qquad\qquad \sigma^2 = N\, i^2\, p\,(1-p)\;}$$

These two lines are the entire theory. During the rise after the step, $p(t)$ climbs
from 0 through 0.5 to 0.96, so the variance climbs to its peak and comes back down —
exactly the bump you measured. On the plateau $p = 0.96$, so the variance is small
but not zero: $1000 \times 1 \times 0.96 \times 0.04 \approx 38$ pA².

<details>
<summary><b>▸ Go deeper: the binomial mean and variance (optional)</b></summary>

Let $x_k \in \{0, 1\}$ indicate whether channel $k$ is open. Then $E[x_k] = p$ and,
since $x_k^2 = x_k$, $\mathrm{Var}[x_k] = E[x_k^2] - E[x_k]^2 = p - p^2 = p(1-p)$.
The current is $I = i \sum_k x_k$. Expectation is linear, so $E[I] = N i p$. For
*independent* channels variances add, so
$\mathrm{Var}[I] = i^2 \sum_k \mathrm{Var}[x_k] = N i^2 p(1-p)$.

Independence is the one real assumption. It fails if channels gate cooperatively, or
if something *common* to all of them fluctuates (e.g. the clamp voltage) — and
background instrument noise adds its own variance on top (Notebook 4).
</details>
""",),
    md(r"""
**Exercise 2** *(~5 min)*. Use the ground truth to predict the mean and variance you
measured. `exp.truth` holds `N`, `i` and `p_open` — the exact open probability $p(t)$
at every sample. Compute the predicted $N i p(t)$ and $N i^2 p(t)(1-p(t))$ and overlay
them on your measured curves.

> **Check / unstuck.** The predictions should run straight through the measured
> curves (the measured variance is noisier — it's estimated from 300 sweeps).
> Stuck? `pp.plotting.plot_mean_variance(exp.t_ms, mean, var, truth=exp.truth)`
> draws the same overlay.
""",),
    code(
        solution=r"""
N, i, p = exp.truth.N, exp.truth.i, exp.truth.p_open
pred_mean = N * i * p
pred_var = N * i**2 * p * (1 - p)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(exp.t_ms, mean, "k", label="measured mean"); axes[0].plot(exp.t_ms, pred_mean, "r--", label="N·i·p(t)")
axes[1].plot(exp.t_ms, var, color="tab:blue", lw=0.8, label="measured variance"); axes[1].plot(exp.t_ms, pred_var, "r--", label="N·i²·p(1−p)")
axes[0].set_ylabel("mean (pA)"); axes[1].set_ylabel("variance (pA²)"); axes[1].set_xlabel("time (ms)")
axes[0].legend(); axes[1].legend(); plt.tight_layout(); plt.show()
""",
        student=r"""
N, i, p = exp.truth.N, exp.truth.i, exp.truth.p_open
# YOUR CODE HERE: pred_mean and pred_var from N, i and p(t)
pred_mean = ...
pred_var = ...

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(exp.t_ms, mean, "k", label="measured mean"); axes[0].plot(exp.t_ms, pred_mean, "r--", label="N·i·p(t)")
axes[1].plot(exp.t_ms, var, color="tab:blue", lw=0.8, label="measured variance"); axes[1].plot(exp.t_ms, pred_var, "r--", label="N·i²·p(1−p)")
axes[0].set_ylabel("mean (pA)"); axes[1].set_ylabel("variance (pA²)"); axes[1].set_xlabel("time (ms)")
axes[0].legend(); axes[1].legend(); plt.tight_layout(); plt.show()
""",
    ),
    md(r"""
## 4. Why big patches look smooth: relative noise

Divide the standard deviation by the mean to get the **relative noise**, the wiggle as
a fraction of the current:

$$\frac{\sigma}{\langle I\rangle} = \frac{\sqrt{N i^2 p (1-p)}}{N i p} = \sqrt{\frac{1-p}{N p}} \;\propto\; \frac{1}{\sqrt N}.$$

Ten times more channels → the current looks about three times smoother. That is why
the N = 1000 trace in section 1 looked clean, and it is the first warning sign for the
method: with very many channels the fluctuations become tiny compared with the
current, and anything that limits how finely you can measure (Notebook 4) starts to
bite.

**Exercise 3** *(~6 min)*. For N in `[10, 30, 100, 300, 1000, 3000]`, simulate 100
sweeps, compute the relative noise $\sigma/\langle I\rangle$ on the plateau (average
the last 200 samples of mean and SD), and plot it against N on log–log axes together
with the prediction $\sqrt{(1-p)/(N p)}$ with $p = 0.96$.

> **Check / unstuck.** The points should fall on a straight line of slope −1/2.
> Stuck? `pp.isochrone_stats` for the mean/variance; the rest is arithmetic.
""",),
    code(
        solution=r"""
Ns = np.array([10, 30, 100, 300, 1000, 3000])
rel = []
for N in Ns:
    e = pp.make_experiment(N=int(N), i=1.0, scheme=scheme, protocol=protocol, n_sweeps=100, seed=int(N))
    m, v = pp.isochrone_stats(e.sweeps)
    rel.append(np.sqrt(v[-200:]).mean() / m[-200:].mean())
rel = np.array(rel)

p_plateau = 0.96
plt.figure(figsize=(5.5, 4))
plt.loglog(Ns, rel, "o", ms=8, color="tab:blue", label="simulated")
plt.loglog(Ns, np.sqrt((1 - p_plateau) / (Ns * p_plateau)), "k-", label=r"$\sqrt{(1-p)/(Np)}$")
plt.xlabel("number of channels N"); plt.ylabel("relative noise  SD / mean"); plt.legend()
plt.title("relative noise falls as 1/√N"); plt.show()
""",
        student=r"""
Ns = np.array([10, 30, 100, 300, 1000, 3000])
rel = []
for N in Ns:
    e = pp.make_experiment(N=int(N), i=1.0, scheme=scheme, protocol=protocol, n_sweeps=100, seed=int(N))
    # YOUR CODE HERE: mean/variance across sweeps; append plateau SD/mean to rel
    raise NotImplementedError
rel = np.array(rel)

p_plateau = 0.96
plt.figure(figsize=(5.5, 4))
plt.loglog(Ns, rel, "o", ms=8, color="tab:blue", label="simulated")
plt.loglog(Ns, np.sqrt((1 - p_plateau) / (Ns * p_plateau)), "k-", label=r"$\sqrt{(1-p)/(Np)}$")
plt.xlabel("number of channels N"); plt.ylabel("relative noise  SD / mean"); plt.legend()
plt.title("relative noise falls as 1/√N"); plt.show()
""",
    ),
    md(r"""
**Where we are.** We can record an ensemble of sweeps and reduce it to two curves,
mean and variance against time, and we know what both should be:
$\langle I\rangle = N i p$ and $\sigma^2 = N i^2 p(1-p)$. The catch is that $p(t)$ is
unknown and changing. Notebook 3 shows the trick that gets rid of it — and turns
these two curves into N and i.
""",),
]

if __name__ == "__main__":
    print(build("02_many_channels", cells))
