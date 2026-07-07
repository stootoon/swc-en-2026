# Behaviour module — SWC ENC 2026

A one-day, hands-on introduction to the statistics behind **Figures 1–3 of
Piet et al. (2024), *Neuron*** ("Behavioral strategy shapes activation of the
Vip-Sst disinhibitory circuit in visual cortex", `piet.pdf`).

Students work through a sequence of Jupyter notebooks, fitting the paper's
behavioral model to **synthetic mice whose true strategy is known**. Because the
data is generated from the model itself, every analysis can be checked against
ground truth — the organizing question of the day is *"can we recover the truth?"*

## Setup

```bash
cd behaviour
python3 -m venv .venv
./.venv/bin/pip install -e .           # installs swcbehav + numpy/scipy/matplotlib/pandas
./.venv/bin/pip install jupyterlab ipykernel
./.venv/bin/python -m ipykernel install --user \
    --name swc-behaviour --display-name "SWC Behaviour (.venv)"
./.venv/bin/jupyter lab
```

In each notebook, select the **"SWC Behaviour (.venv)"** kernel.

## The day

| # | Notebook | Technique | Paper |
|---|----------|-----------|-------|
| 1 | `01_task_and_design_matrix` | bouts, the five-strategy design matrix | Fig 1C |
| 2 | `02_meet_the_mice` | strategy as a weight vector; behavioral signatures | Fig 1B |
| 3 | `03_static_model` | logistic regression from scratch; weight recovery | Fig 1D |
| 4 | `04_model_evaluation` | ROC, AUC, cross-validation | Fig 2A |
| 5 | `05_evidence_and_ablation` | ablation, the strategy index (+ optional model evidence) | Fig 2D–F |
| 6 | `06_mixtures` | the cycle on blended strategies | Fig 2F |
| 7 | `07_dynamic_model` | the random-walk / dynamic logistic regression | Fig 1D |
| 8 | `08_engagement_optional` | strategy vs engagement (optional) | Fig 3 |

Notebooks 3–6 build the **fit → evaluate → ablate** cycle on fixed-strategy mice;
Notebook 7 extends it to a mouse whose strategy drifts.

## Structure

- `swcbehav/` — the backend. Generates synthetic mice (`task.py`, `generate.py`)
  and holds reference implementations of every technique (`design.py`,
  `models.py`) plus plotting (`plotting.py`). Each notebook imports its
  prerequisites from here, so **every notebook runs standalone**.
- `notebooks/` — paired files: `NN_name.ipynb` (student, with `# YOUR CODE HERE`
  blanks) and `NN_name_solutions.ipynb` (complete, pre-executed answer key).
- `build/` — the source of truth for the notebooks. Each `build_nbNN.py` emits
  its student/solutions pair; `python build/build_all.py` regenerates them all.
  Edit a builder and rebuild, or edit the `.ipynb` directly — whichever you
  prefer.

## Idealizations

The synthetic task is deliberately simplified for teaching (documented in
`swcbehav/task.py` and `generate.py`): the change schedule is open-loop (licking
doesn't delay changes), and a fresh licking bout may begin on any flash. These
keep the focus on the statistics without changing what the analyses teach.
