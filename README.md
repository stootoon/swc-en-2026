# SWC ENC 2026

Course material for the ENC 2026 day courses for incoming graduate students in
systems neuroscience.

## 📖 Read the notebooks online

**→ https://stootoon.github.io/swc-en-2026/**

Every notebook is rendered there — both the student version (with blanks) and the
worked solutions (with all figures) — and each page links on to the next. Nothing to
install: good for browsing before the course, or catching up afterwards.

## ▶️ Run the notebooks

To actually *do* the exercises, clone the repo and follow the module's own README:

```bash
git clone https://github.com/stootoon/swc-en-2026.git
cd swc-en-2026/behaviour     # see behaviour/README.md for setup
```

## Modules

| module | topic |
|---|---|
| [`behaviour/`](behaviour/) | Quantifying behavioural strategy — the statistics behind Figures 1–3 of Piet et al. (2024), *Neuron*, taught on synthetic data with known ground truth |
| `ephys/` | *(to come)* |
| `neuropixels/` | *(to come)* |

Each module is self-contained: its own notebooks, environment and README.

## How the previews are built

`tools/make_preview.py` renders every module's notebooks into a static HTML site.
A "module" is simply any top-level directory containing a `notebooks/` folder, so a
new module is picked up automatically — no configuration needed.

```bash
python tools/make_preview.py            # all modules -> ./_site
python tools/make_preview.py behaviour  # just one module
```

Open `_site/index.html` to review locally. On every push to `main` the
[Pages workflow](.github/workflows/pages.yml) runs the same command and publishes the
result. The notebooks are committed **with their outputs**, so the workflow only
*renders* them — it never executes anything (fast, no scientific stack needed), and
the generated HTML is never committed, keeping the repo small for the students who
clone it.
