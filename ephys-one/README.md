# ephys-one — counting channels (SWC ENC 2026)

Record from a single cell under **voltage clamp**, and work out **how many ion
channels** are in the membrane and **how much current flows through each one** —
from the *fluctuations* of the macroscopic current. This is **nonstationary noise
analysis**, taught the way Alvarez, Gonzalez & Latorre (2002, *Counting Channels*)
teach it: on **simulated patches with known ground truth**, so every estimate can be
graded against the answer.

Live previews: **https://stootoon.github.io/swc-en-2026/** · notebooks are in
[`notebooks/`](notebooks/) (student + `_solutions` copies).

## The arc

Notebooks **1–3 are the core** and deliver the complete basic analysis. Notebooks
**4–7 are independent** of one another — each covers one way real recordings depart
from the ideal — so in class they can be taken in any order, by priority, as time
allows. Notebook 8 scores the whole pipeline on three mystery patches.

| # | Notebook | What you build | Key idea |
|---|----------|----------------|----------|
| 0 | Roadmap | — | patch clamp, voltage clamp, the question |
| 1 | One channel | coin-flip simulation of a single channel | rates, open probability, dwell times |
| 2 | Many channels | isochrone mean & variance across sweeps | ⟨I⟩ = Nip, σ² = Ni²p(1−p) |
| 3 | **The parabola** | least-squares fit → **N, i, p_max** | σ² = i⟨I⟩ − ⟨I⟩²/N |
| 4 | Imperfect recordings | baseline subtraction, successive differences, bootstrap | noise, rundown, how many sweeps |
| 5 | Incomplete parabolas | fits with limited open-probability range | trust i, doubt N; inactivation |
| 6 | Beware the filter | Bessel low-pass, bias vs cutoff | filter τ < channel τ / 10; flicker |
| 7 | Stationary noise *(optional)* | autocorrelation, Lorentzian spectrum | kinetics from steady-state noise |
| 8 | Scoring | diagnostics + full analysis with error bars | three mystery patches vs truth |

## Run it locally

```bash
cd ephys-one
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .                       # installs the picopatch backend
python -m ipykernel install --user --name swc-ephys-one --display-name "SWC Ephys-One (.venv)"
jupyter lab                            # open notebooks/ and pick the kernel
```

## Editing the notebooks

The `.ipynb` files are generated from the builders in [`build/`](build/) — edit
`build/build_nbNN.py` (one source, paired student + solutions output), then:

```bash
python build/build_all.py              # regenerate all notebooks
build/execute.sh                       # execute the solutions copies in place
python build/check.py                  # errors / figures / exercises per notebook
```

The `picopatch` package in [`picopatch/`](picopatch/) holds the channel simulator
(`kinetics.py`, `protocol.py`, `simulate.py` — the ground truth), reference
implementations of every analysis step (`analysis.py`, `pipeline.py` — the
`pp.<name>` escape hatches), and the plotting helpers and schematics (`plotting.py`).

## The paper

Alvarez O, Gonzalez C, Latorre R (2002). *Counting Channels: a tutorial guide on ion
channel fluctuation analysis.* Advances in Physiology Education 26: 327–341.
([`counting-channels.pdf`](counting-channels.pdf))
