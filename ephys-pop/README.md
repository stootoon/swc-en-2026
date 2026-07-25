# ephys-pop — spike sorting (SWC ENC 2026)

Build **picosort**, a miniature spike sorter loosely following the stages of
**Kilosort4**, and understand what a real sorter does to your Neuropixels data.
Everything runs on **synthetic recordings with known ground truth**, so every stage
can be graded against the truth it's trying to recover.

Live previews: **https://stootoon.github.io/swc-en-2026/** · notebooks are in
[`notebooks/`](notebooks/) (student + `_solutions` copies).

## The pipeline

| # | Notebook | Stage |
|---|----------|-------|
| 0 | Roadmap | the whole pipeline; what you build / learn |
| 1 | The recording | probe geometry + forward model |
| 2 | Preprocessing | high-pass filter → CAR → whitening |
| 3 | Spike detection | threshold crossings + snippet extraction |
| 4 | Feature extraction | PCA on waveforms |
| 5 | Clustering → templates | greedy/graph clustering + t-SNE QC |
| 6 | Template matching | matching-pursuit deconvolution |
| 7 | Merging & cleanup | cross-correlograms, duplicate removal |
| 8 | Scoring | match to ground truth; precision/recall |

The full pipeline sorts the synthetic data at ~100% recall/precision (it's a clean
teaching example); Notebooks 7–8 introduce controlled errors and stress tests to show
what the QC and scoring tools reveal.

## Run it locally

```bash
cd ephys-pop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # installs the picosort backend
python -m ipykernel install --user --name swc-ephys-pop --display-name "SWC Ephys-Pop (.venv)"
jupyter lab                            # open notebooks/ and pick the kernel
```

## Editing the notebooks

The `.ipynb` files are generated from the builders in [`build/`](build/) — edit
`build/build_nbNN.py` (one source, paired student + solutions output), then:

```bash
python build/build_all.py              # regenerate all notebooks
```

The `picosort` package in [`picosort/`](picosort/) holds the synthetic generator
and a reference implementation of every stage (the `ps.<name>` escape hatches).
