"""Notebook 7 (optional) -- Stationary noise: autocorrelation and the Lorentzian spectrum."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import md, code, build

cells = [
    md(r"""
# Notebook 7 — Stationary noise *(optional)*

*SWC ENC 2026 · ephys-one module · independent of Notebooks 4–6*

Everything so far used a voltage *step*, and the information came from how the
variance changed as the open probability swept from 0 upwards. There is an older,
complementary approach: hold the voltage **constant**, let the channels settle into a
steady state, and study the noise that remains. The mean and variance are now fixed
numbers, so the variance–mean trick is gone — but the noise has a new property to
exploit: its **speed**. Fast-gating channels produce fast noise, slow ones slow noise,
and the shape of that noise in time (or frequency) reveals the gating rates.

**In this notebook you will:**
1. Record steady-state channel noise and see what mean and variance alone can tell you.
2. Build the **autocorrelation function** and read a time constant off it.
3. Build the **power spectrum** and fit a **Lorentzian** to it.
4. Combine stationary and nonstationary results to get the rates $\alpha$ and $\beta$.
""",),
    code(r"""
import numpy as np
import matplotlib.pyplot as plt
import picopatch as pp

scheme = pp.two_state()
V_hold = 45.0                                              # p = 0.5 here, where the noise is largest
prot = pp.constant_protocol(V_hold, duration_ms=2000)      # a 2-second steady-state record
exp = pp.make_experiment(N=100, i=1.0, scheme=scheme, protocol=prot, n_sweeps=4, seed=0)
x = exp.sweeps[0]
""",),
    md(r"""
## 1. Steady-state noise

A hundred channels, 1 pA each, held at +45 mV where each is open half the time. The
current hovers around 50 pA and never settles: channels keep opening and closing.
Look at the whole 2 seconds, then zoom in.
""",),
    code(r"""
fig, axes = plt.subplots(2, 1, figsize=(11, 5))
axes[0].plot(exp.t_ms, x, color="k", lw=0.4); axes[0].set_xlabel("time (ms)"); axes[0].set_ylabel("current (pA)")
axes[0].set_title("2 s at +45 mV: the current fluctuates around a steady mean")
sl = slice(0, int(0.1 * exp.fs))
axes[1].plot(exp.t_ms[sl], x[sl], color="k", lw=0.8, drawstyle="steps-post"); axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("current (pA)")
axes[1].set_title("first 100 ms: the fluctuations have a characteristic speed — a few ms per excursion")
plt.tight_layout(); plt.show()
print(f"mean = {x.mean():.1f} pA,  variance = {x.var():.1f} pA²")
""",),
    md(r"""
What do mean and variance give us here? The two formulas of Notebook 2 still hold,
$\langle I\rangle = Nip$ and $\sigma^2 = Ni^2p(1-p)$, so their ratio is

$$\frac{\sigma^2}{\langle I\rangle} = i\,(1-p).$$

One equation, two unknowns. With $p = 0.5$ we'd read $i = 2 \times 25/50 = 1$ pA, but
we only know $p$ because we built the simulation. On a real patch, stationary mean
and variance alone give you $i(1-p)$ — a lower bound on $i$ — and nothing about $N$.
That's why the step experiment was invented. What stationary noise *does* have is
time structure.

## 2. How fast is the noise? Autocorrelation

Zoomed in, the trace wanders: once it's high it tends to stay high for a few
milliseconds. A quantity for "how long does the signal remember its value" is the
**autocorrelation function**: take the signal, shift a copy of it by a lag $\Delta$,
and ask how well the two agree (as a correlation, from 1 for perfect agreement to 0
for none). Do this for every lag:
""",),
    code(r"""
xc = x - x.mean()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for ax, lag_ms in zip(axes, [0.2, 2.0, 20.0]):
    lag = int(lag_ms * 1e-3 * exp.fs)
    a, b = xc[:-lag], xc[lag:]
    r = np.corrcoef(a, b)[0, 1]
    ax.plot(a[::4], b[::4], ".", ms=2, color="tab:blue", alpha=0.4)
    ax.set_xlabel("I(t) − mean"); ax.set_ylabel(f"I(t + {lag_ms:g} ms) − mean")
    ax.set_title(f"lag {lag_ms:g} ms:  correlation = {r:.2f}")
plt.suptitle("the signal against a shifted copy of itself"); plt.tight_layout(); plt.show()
""",),
    md(r"""
At a 0.2 ms lag the signal is almost unchanged (correlation near 1). At 2 ms it has
partly forgotten; at 20 ms the two copies are unrelated. For a two-state channel the
forgetting is exponential, with exactly the time constant of Notebook 1,

$$R(\Delta) = e^{-\Delta/\tau}, \qquad \tau = \frac{1}{\alpha + \beta},$$

because whatever a channel is doing now, the chance it has flipped by $\Delta$ later
grows as $1 - e^{-(\alpha+\beta)\Delta}$ (the master equation of Notebook 1, Go deeper).

**Exercise 1** *(~6 min)*. Complete `autocorrelation`: for each lag from 0 to
`max_lag` samples, compute the mean product of the mean-subtracted signal with its
shifted copy, then normalise so lag 0 gives 1. Plot it on a log y-axis and fit a
straight line to the first ~2 ms to get $\tau$.

> **Check / unstuck.** $\tau \approx 3.3$ ms, i.e. $1/(\alpha+\beta)$ with
> $\alpha = \beta = 150$/s. Stuck? `pp.autocorrelation(x, max_lag)`.
""",),
    code(
        solution=r"""
def autocorrelation(x, max_lag):
    xc = x - x.mean()
    n = len(xc)
    ac = np.array([np.dot(xc[:n - k], xc[k:]) / (n - k) for k in range(max_lag + 1)])
    return ac / ac[0]

max_lag = int(15e-3 * exp.fs)
lags_ms = np.arange(max_lag + 1) / exp.fs * 1e3
ac = autocorrelation(x, max_lag)

fit_to = lags_ms <= 2.0
slope, intercept = np.polyfit(lags_ms[fit_to], np.log(ac[fit_to]), 1)
tau_ac = -1 / slope
tau_true = 1e3 / (float(pp.alpha_rate(V_hold)) + float(pp.beta_rate(V_hold)))
print(f"tau from autocorrelation = {tau_ac:.2f} ms    (1/(alpha+beta) = {tau_true:.2f} ms)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(lags_ms, ac, "k"); axes[0].plot(lags_ms, np.exp(-lags_ms / tau_true), "r--", label="exp(−Δ/τ), τ = 1/(α+β)")
axes[0].set_xlabel("lag Δ (ms)"); axes[0].set_ylabel("autocorrelation"); axes[0].legend(); axes[0].set_title("linear axes")
axes[1].semilogy(lags_ms, np.clip(ac, 1e-3, None), "k"); axes[1].semilogy(lags_ms, np.exp(-lags_ms / tau_true), "r--")
axes[1].set_xlabel("lag Δ (ms)"); axes[1].set_title("log axis: a straight line of slope −1/τ")
plt.tight_layout(); plt.show()
""",
        student=r"""
def autocorrelation(x, max_lag):
    # YOUR CODE HERE: mean-subtract; for k in 0..max_lag, mean of xc[:n-k]*xc[k:]; normalise by lag-0 value
    raise NotImplementedError

max_lag = int(15e-3 * exp.fs)
lags_ms = np.arange(max_lag + 1) / exp.fs * 1e3
ac = autocorrelation(x, max_lag)

fit_to = lags_ms <= 2.0
slope, intercept = np.polyfit(lags_ms[fit_to], np.log(ac[fit_to]), 1)
tau_ac = -1 / slope
tau_true = 1e3 / (float(pp.alpha_rate(V_hold)) + float(pp.beta_rate(V_hold)))
print(f"tau from autocorrelation = {tau_ac:.2f} ms    (1/(alpha+beta) = {tau_true:.2f} ms)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(lags_ms, ac, "k"); axes[0].plot(lags_ms, np.exp(-lags_ms / tau_true), "r--", label="exp(−Δ/τ), τ = 1/(α+β)")
axes[0].set_xlabel("lag Δ (ms)"); axes[0].set_ylabel("autocorrelation"); axes[0].legend(); axes[0].set_title("linear axes")
axes[1].semilogy(lags_ms, np.clip(ac, 1e-3, None), "k"); axes[1].semilogy(lags_ms, np.exp(-lags_ms / tau_true), "r--")
axes[1].set_xlabel("lag Δ (ms)"); axes[1].set_title("log axis: a straight line of slope −1/τ")
plt.tight_layout(); plt.show()
""",
    ),
    md(r"""
## 3. The same thing in frequency: the power spectrum

The classical way to present channel noise is in the frequency domain. The idea in
one paragraph: any wiggly signal can be written as a sum of sine waves of different
frequencies, and the **power spectrum** $S(f)$ says how much of the signal's variance
is carried by each frequency. Slow wander shows up at low $f$, fast jitter at high
$f$, and the area under $S(f)$ is the total variance. A quick illustration with a
made-up signal — two sines plus a little noise:
""",),
    code(r"""
from scipy.signal import welch
tt = np.arange(0, 1.0, 1 / 5000)
demo = 1.0 * np.sin(2 * np.pi * 20 * tt) + 0.5 * np.sin(2 * np.pi * 150 * tt) + 0.2 * np.random.default_rng(0).normal(size=tt.size)
f_d, S_d = welch(demo, fs=5000, nperseg=2048)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].plot(tt[:1000], demo[:1000], "k", lw=0.8); axes[0].set_xlabel("time (s)"); axes[0].set_title("a 20 Hz sine + a smaller 150 Hz sine + noise")
axes[1].semilogy(f_d, S_d, "k"); axes[1].set_xlim(0, 400); axes[1].set_xlabel("frequency (Hz)"); axes[1].set_ylabel("power")
axes[1].set_title("its power spectrum: a peak at each frequency present")
plt.tight_layout(); plt.show()
""",),
    md(r"""
Channel noise has no single frequency; it has an exponential autocorrelation, and the
spectrum of an exponentially-correlated signal is a **Lorentzian**:

$$S(f) = \frac{S_0}{1 + (f/f_c)^2}, \qquad f_c = \frac{1}{2\pi\tau}.$$

Flat at low frequencies, falling as $1/f^2$ above the **corner frequency** $f_c$. The
corner is just $\tau$ in disguise: $\tau = 3.3$ ms ↔ $f_c \approx 48$ Hz. On log–log
axes the Lorentzian is a plateau with a knee, and the knee is where you read off the
kinetics.

<details>
<summary><b>▸ Go deeper: Wiener–Khinchin, and what S₀ tells you (optional)</b></summary>

The power spectrum and the autocorrelation are a Fourier-transform pair
(Wiener–Khinchin theorem). The transform of $\sigma^2 e^{-|\Delta|/\tau}$ is
$S(f) = 4\sigma^2\tau / (1 + (2\pi f\tau)^2)$ (one-sided), which is the Lorentzian
above with $S_0 = 4\sigma^2\tau$ and $f_c = 1/(2\pi\tau)$. So the plateau height times
the corner frequency gives back the variance: $\int_0^\infty S(f)\,df = S_0 f_c \pi/2
= \sigma^2$. With $\sigma^2 = Ni^2p(1-p)$ and $\tau = 1/(\alpha+\beta)$, the spectrum
of a two-state channel population is completely specified by $N$, $i$, $\alpha$ and
$\beta$. Channels with more than two states give sums of Lorentzians, one corner per
relaxation time — the spectral signature of a more complex scheme.
</details>

**Exercise 2** *(~6 min)*. Compute the power spectrum of the record with
`scipy.signal.welch(x - x.mean(), fs=exp.fs, nperseg=4096)` and fit a Lorentzian to
it with `scipy.optimize.curve_fit` — fit in log-power so every decade counts equally
(the `log_lorentzian` model is given). Read off $f_c$ and convert to $\tau$.

> **Check / unstuck.** $f_c \approx 48$ Hz, $\tau \approx 3.3$ ms. Stuck?
> `pp.power_spectrum(x, fs)` and `pp.fit_lorentzian(f, S)` do both steps.
""",),
    code(
        solution=r"""
from scipy.optimize import curve_fit

def log_lorentzian(f, log_S0, log_fc):
    return log_S0 - np.log(1 + (f / np.exp(log_fc)) ** 2)

f, S = welch(x - x.mean(), fs=exp.fs, nperseg=4096)
keep = (f > 0) & (f < 5000)                      # stay away from the very top of the band
popt, _ = curve_fit(log_lorentzian, f[keep], np.log(S[keep]), p0=[np.log(S[1]), np.log(50.0)])
S0, fc = np.exp(popt)
tau_spec = 1e3 / (2 * np.pi * fc)
print(f"Lorentzian fit: S0 = {S0:.3f} pA²/Hz, corner f_c = {fc:.1f} Hz  ->  tau = {tau_spec:.2f} ms  (truth {tau_true:.2f} ms)")

plt.figure(figsize=(7, 4.2))
plt.loglog(f[1:], S[1:], color="k", lw=0.8, label="measured spectrum")
plt.loglog(f[1:], S0 / (1 + (f[1:] / fc) ** 2), color="tab:red", lw=2, label=f"Lorentzian, f_c = {fc:.0f} Hz")
plt.axvline(fc, color="tab:red", ls=":")
plt.xlabel("frequency (Hz)"); plt.ylabel("power (pA²/Hz)"); plt.legend(); plt.title("channel noise is a Lorentzian: flat, then 1/f²")
plt.show()
""",
        student=r"""
from scipy.optimize import curve_fit

def log_lorentzian(f, log_S0, log_fc):
    return log_S0 - np.log(1 + (f / np.exp(log_fc)) ** 2)

# YOUR CODE HERE: f, S = welch(...); fit log_lorentzian to (f, log S) for 0 < f < 5000 Hz;
# S0, fc = np.exp(popt); tau_spec = 1e3 / (2*pi*fc)
f, S = ...
popt = ...
S0, fc = np.exp(popt)
tau_spec = 1e3 / (2 * np.pi * fc)
print(f"Lorentzian fit: S0 = {S0:.3f} pA²/Hz, corner f_c = {fc:.1f} Hz  ->  tau = {tau_spec:.2f} ms  (truth {tau_true:.2f} ms)")

plt.figure(figsize=(7, 4.2))
plt.loglog(f[1:], S[1:], color="k", lw=0.8, label="measured spectrum")
plt.loglog(f[1:], S0 / (1 + (f[1:] / fc) ** 2), color="tab:red", lw=2, label=f"Lorentzian, f_c = {fc:.0f} Hz")
plt.axvline(fc, color="tab:red", ls=":")
plt.xlabel("frequency (Hz)"); plt.ylabel("power (pA²/Hz)"); plt.legend(); plt.title("channel noise is a Lorentzian: flat, then 1/f²")
plt.show()
""",
    ),
    md(r"""
## 4. Putting the two methods together

Stationary noise gave us $\tau = 1/(\alpha+\beta)$ — the *sum* of the rates. The
nonstationary analysis of Notebook 3 gives $i$ and $N$, and hence
$p = \langle I\rangle/(iN)$ at this voltage. With both,

$$\alpha = \frac{p}{\tau}, \qquad \beta = \frac{1-p}{\tau}.$$

Let's close the loop: run a step experiment on the same patch for $i$ and $N$, use
them to turn the steady-state mean into $p$, and combine with $\tau$ from the spectrum.
""",),
    code(r"""
step = pp.make_experiment(N=100, i=1.0, scheme=scheme, n_sweeps=500, seed=1)       # a step to +65 mV
fit = pp.fit_parabola(*pp.isochrone_stats(step.sweeps))
p_45 = x.mean() / (fit.i * fit.N)
tau_s = tau_spec * 1e-3
alpha_hat, beta_hat = p_45 / tau_s, (1 - p_45) / tau_s
print(f"from the step:        i = {fit.i:.2f} pA, N = {fit.N:.0f}")
print(f"steady-state mean:    p(+45 mV) = {p_45:.2f}")
print(f"from the spectrum:    tau = {tau_spec:.2f} ms")
print(f"=> alpha = {alpha_hat:.0f}/s, beta = {beta_hat:.0f}/s      (truth: {float(pp.alpha_rate(45)):.0f}/s, {float(pp.beta_rate(45)):.0f}/s)")
""",),
    md(r"""
**Where we are.** Steady-state noise cannot count channels, but it measures the
*speed* of gating: the autocorrelation decays, and the power spectrum rolls off, with
the time constant $1/(\alpha+\beta)$. Combined with the step analysis it yields the
individual rate constants — the full two-state description of the channel from
macroscopic recordings alone. Historically this spectral method (Katz & Miledi; Neher
& Stevens 1977) came first, and the Lorentzian shape of synaptic and channel noise
was among the earliest evidence that currents flow through discrete, switching pores.
""",),
]

if __name__ == "__main__":
    print(build("07_stationary_noise", cells))
