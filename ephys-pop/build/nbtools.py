"""Tiny helper to author paired student / solutions notebooks from one source.

Each notebook is described as a list of cells. A markdown cell is shared by both
versions; a code cell carries a ``solution`` body and an optional ``student``
body (the blanked version shown to students). If ``student`` is omitted, the
code is identical in both versions (e.g. imports, provided plotting calls).

Run a ``build_nbNN.py`` script to (re)generate its two .ipynb files. The .ipynb
files are the deliverable; these builders are the maintainable source of truth.

Mirrors the behaviour module's builder so the two modules stay consistent.
"""

from __future__ import annotations

import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

KERNEL_NAME = "swc-ephys-pop"
KERNEL_DISPLAY = "SWC Ephys-Pop (.venv)"
NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks")


def md(text: str) -> dict:
    return {"type": "md", "text": text.strip("\n")}


def code(solution: str, student: str | None = None) -> dict:
    return {"type": "code", "solution": solution.strip("\n"),
            "student": None if student is None else student.strip("\n")}


def _make_nb(cells, variant):
    nb = new_notebook()
    for c in cells:
        if c["type"] == "md":
            nb.cells.append(new_markdown_cell(c["text"]))
        else:
            body = c["solution"] if (variant == "solution" or c["student"] is None) \
                else c["student"]
            nb.cells.append(new_code_cell(body))
    nb.metadata["kernelspec"] = {"name": KERNEL_NAME, "display_name": KERNEL_DISPLAY,
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def build(slug: str, cells) -> tuple[str, str]:
    """Write ``<slug>.ipynb`` (student) and ``<slug>_solutions.ipynb``."""
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    student_path = os.path.join(NOTEBOOK_DIR, f"{slug}.ipynb")
    solution_path = os.path.join(NOTEBOOK_DIR, f"{slug}_solutions.ipynb")
    with open(student_path, "w") as f:
        nbf.write(_make_nb(cells, "student"), f)
    with open(solution_path, "w") as f:
        nbf.write(_make_nb(cells, "solution"), f)
    return student_path, solution_path
