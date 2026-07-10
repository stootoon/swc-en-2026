"""Render notebooks to a self-contained HTML preview for fast review.

Usage:
    python build/make_preview.py            # (re)render all notebooks + index
    python build/make_preview.py 03 07      # only these numbers (+ refresh index)

Output goes to behaviour/preview/. Open preview/index.html in a browser and hit
refresh after each change -- no need to reopen anything in the IDE. The solutions
pages carry baked-in plots; the student pages show the blanks.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
NB_DIR = os.path.join(ROOT, "notebooks")
OUT = os.path.join(ROOT, "preview")

NOTEBOOKS = [
    ("00_roadmap", "0. Roadmap"),
    ("01_task_and_design_matrix", "1. Task → design matrix"),
    ("02_meet_the_mice", "2. Meet the mice"),
    ("03_static_model", "3. Static model"),
    ("04_model_evaluation", "4. Evaluation: ROC / AUC / CV"),
    ("05_evidence_and_ablation", "5. Evidence & ablation"),
    ("06_mixtures", "6. Mixtures"),
    ("07_dynamic_model", "7. Dynamic model"),
    ("08_engagement_optional", "8. Engagement (optional)"),
    ("09_individual_differences", "9. Individual differences (Part 2)"),
    ("10_multiple_comparisons", "10. Multiple comparisons (Part 2)"),
    ("11_hierarchical_bootstrap", "11. Hierarchical bootstrap (Part 2)"),
]


def convert(ipynb):
    # --embed-images base64-inlines the paper panels so the preview HTML is
    # self-contained and never depends on relative asset paths.
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
                    "--embed-images", "--output-dir", OUT, ipynb],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_index():
    rows = []
    for slug, title in NOTEBOOKS:
        sol = f"{slug}_solutions.html"
        stu = f"{slug}.html"
        links = []
        if os.path.exists(os.path.join(OUT, sol)):
            links.append(f'<a href="{sol}">solutions</a>')
        if os.path.exists(os.path.join(OUT, stu)):
            links.append(f'<a class="muted" href="{stu}">student</a>')
        rows.append(f'<li><span>{title}</span>{" &middot; ".join(links)}</li>')
    html = f"""<!doctype html><meta charset="utf-8"><title>Behaviour module — preview</title>
<style>
 body{{font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem;color:#222}}
 h1{{font-size:1.4rem}} .sub{{color:#666;margin-bottom:2rem}}
 ul{{list-style:none;padding:0}} li{{display:flex;justify-content:space-between;gap:1rem;padding:.6rem 0;border-bottom:1px solid #eee}}
 li span{{font-weight:600}} a{{margin-left:.8rem;text-decoration:none;color:#2563eb}} a:hover{{text-decoration:underline}}
 a.muted{{color:#999}}
 @media(prefers-color-scheme:dark){{body{{background:#111;color:#ddd}}li{{border-color:#333}}a{{color:#6ea8fe}}}}
</style>
<h1>Behaviour module — notebook preview</h1>
<div class="sub">SWC ENC 2026. Refresh this page after each change.</div>
<ul>
{os.linesep.join(rows)}
</ul>"""
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)


def main():
    os.makedirs(OUT, exist_ok=True)
    nums = sys.argv[1:]
    slugs = [s for s, _ in NOTEBOOKS if not nums or s[:2] in nums]
    for slug in slugs:
        for ipynb in sorted(glob.glob(os.path.join(NB_DIR, f"{slug}*.ipynb"))):
            convert(ipynb)
    write_index()
    print(f"preview -> {os.path.join(OUT, 'index.html')}")


if __name__ == "__main__":
    main()
