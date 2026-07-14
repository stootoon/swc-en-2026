"""Render notebooks to a self-contained HTML preview for fast review.

Usage:
    python build/make_preview.py            # (re)render all notebooks + index
    python build/make_preview.py 03 07      # only these numbers (+ refresh index)

Output goes to behaviour/preview/. Open preview/index.html in a browser and hit
refresh after each change -- no need to reopen anything in the IDE. Each page gets a
nav bar (prev / index / next, plus a student<->solutions toggle) so you can read
straight through. The solutions pages carry baked-in plots; the student pages show
the blanks.
"""
import glob
import os
import re
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

NAV_CSS = """
<style>
.swc-nav{display:flex;justify-content:space-between;align-items:center;gap:1rem;
 font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 padding:.55rem 1.1rem;background:#f6f7f9;border-bottom:1px solid #e3e5e8}
.swc-nav.bottom{border-bottom:none;border-top:1px solid #e3e5e8;margin-top:2rem}
.swc-nav a{color:#2563eb;text-decoration:none;white-space:nowrap}
.swc-nav a:hover{text-decoration:underline}
.swc-nav .mid{color:#888;display:flex;gap:.9rem}
.swc-nav .spacer{flex:0 0 auto;color:#c3c6ca}
@media(prefers-color-scheme:dark){
 .swc-nav{background:#1b1d20;border-color:#33363a}
 .swc-nav.bottom{border-top-color:#33363a}
 .swc-nav a{color:#6ea8fe} .swc-nav .mid{color:#888}
}
</style>
"""


def _page(slug, solutions):
    return f"{slug}_solutions.html" if solutions else f"{slug}.html"


def _nav(idx, solutions, bottom=False):
    """Nav bar for the notebook at position idx (prev / index+toggle / next)."""
    prev_html = '<span class="spacer">&nbsp;</span>'
    next_html = '<span class="spacer">&nbsp;</span>'
    if idx > 0:
        slug, title = NOTEBOOKS[idx - 1]
        prev_html = f'<a href="{_page(slug, solutions)}">&larr; {title}</a>'
    if idx < len(NOTEBOOKS) - 1:
        slug, title = NOTEBOOKS[idx + 1]
        next_html = f'<a href="{_page(slug, solutions)}">{title} &rarr;</a>'

    slug = NOTEBOOKS[idx][0]
    other = _page(slug, not solutions)
    other_label = "student version" if solutions else "solutions"
    mid = (f'<a href="index.html">index</a>'
           f'<a href="{other}">{other_label}</a>')
    cls = "swc-nav bottom" if bottom else "swc-nav"
    return f'<div class="{cls}">{prev_html}<span class="mid">{mid}</span>{next_html}</div>'


def _inject_nav(path, idx, solutions):
    with open(path) as f:
        html = f.read()
    if 'class="swc-nav' in html:      # already injected
        return
    html = html.replace("</head>", NAV_CSS + "</head>", 1)
    m = re.search(r"<body[^>]*>", html)
    if m:
        html = html[: m.end()] + _nav(idx, solutions) + html[m.end():]
    html = html.replace("</body>", _nav(idx, solutions, bottom=True) + "</body>", 1)
    with open(path, "w") as f:
        f.write(html)


def convert(ipynb):
    # --embed-images inlines any images referenced from markdown cells.
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
                    "--embed-images", "--output-dir", OUT, ipynb],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_index():
    rows = []
    for slug, title in NOTEBOOKS:
        links = []
        if os.path.exists(os.path.join(OUT, _page(slug, True))):
            links.append(f'<a href="{_page(slug, True)}">solutions</a>')
        if os.path.exists(os.path.join(OUT, _page(slug, False))):
            links.append(f'<a class="muted" href="{_page(slug, False)}">student</a>')
        rows.append(f'<li><span>{title}</span>{"".join(links)}</li>')
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
<div class="sub">SWC ENC 2026. Refresh this page after each change. Each notebook links on to the next.</div>
<ul>
{os.linesep.join(rows)}
</ul>"""
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)


def main():
    os.makedirs(OUT, exist_ok=True)
    nums = sys.argv[1:]
    for idx, (slug, _) in enumerate(NOTEBOOKS):
        if nums and slug[:2] not in nums:
            continue
        for ipynb in sorted(glob.glob(os.path.join(NB_DIR, f"{slug}*.ipynb"))):
            convert(ipynb)
        for solutions in (True, False):
            page = os.path.join(OUT, _page(slug, solutions))
            if os.path.exists(page):
                _inject_nav(page, idx, solutions)
    write_index()
    print(f"preview -> {os.path.join(OUT, 'index.html')}")


if __name__ == "__main__":
    main()
