"""Render every module's notebooks to a browsable HTML site.

Used two ways:

    python tools/make_preview.py                 # all modules -> ./_site
    python tools/make_preview.py behaviour       # just one module
    python tools/make_preview.py --out _site     # choose the output dir

Locally, open `_site/index.html` and refresh after each change. In CI, the same
command builds the site that gets published to GitHub Pages.

Nothing here is module-specific: a "module" is any top-level directory containing a
`notebooks/` folder, and its notebooks are discovered and ordered by filename. Add
`ephys/notebooks/*.ipynb` and it shows up automatically.

The notebooks are expected to already carry their outputs (we execute the
`*_solutions.ipynb` copies when building them), so this only *renders* -- it never
executes a notebook.
"""
from __future__ import annotations

import argparse
import base64
import glob
import mimetypes
import os
import re
import subprocess
import sys

import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTIONS_SUFFIX = "_solutions"


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
# --------------------------------------------------------------------------- #
def find_modules():
    """Any top-level dir with a notebooks/ folder is a module."""
    out = []
    for name in sorted(os.listdir(ROOT)):
        if name.startswith((".", "_")):
            continue
        if os.path.isdir(os.path.join(ROOT, name, "notebooks")):
            out.append(name)
    return out


def notebook_title(path):
    """First H1 of the notebook, tidied: 'Notebook 3 — Static model' -> '3. Static model'."""
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception:
        return os.path.basename(path)
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            if line.startswith("# "):
                title = line[2:].strip().replace("*", "").replace("_", " ")
                m = re.match(r"Notebook\s+(\S+)\s*[—–-]\s*(.*)", title)
                return f"{m.group(1)}. {m.group(2)}".strip() if m else title
    return os.path.basename(path)


def module_title(module):
    readme = os.path.join(ROOT, module, "README.md")
    if os.path.exists(readme):
        with open(readme) as f:
            for line in f:
                if line.startswith("# "):
                    return line[2:].strip()
    return module.replace("-", " ").replace("_", " ").title()


def module_notebooks(module):
    """Ordered [(slug, title)] for a module, from its student notebooks."""
    paths = sorted(glob.glob(os.path.join(ROOT, module, "notebooks", "*.ipynb")))
    out = []
    for p in paths:
        slug = os.path.splitext(os.path.basename(p))[0]
        if slug.endswith(SOLUTIONS_SUFFIX):
            continue
        out.append((slug, notebook_title(p)))
    return out


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
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
 .swc-nav a{color:#6ea8fe}
}
</style>
"""

SITE_CSS = """
 body{font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  max-width:680px;margin:3rem auto;padding:0 1rem;color:#222}
 h1{font-size:1.5rem;margin-bottom:.2rem} h2{font-size:1.1rem;margin-top:2.2rem}
 .sub{color:#666;margin-bottom:1.5rem}
 ul{list-style:none;padding:0}
 li{display:flex;justify-content:space-between;gap:1rem;padding:.55rem 0;border-bottom:1px solid #eee}
 li span{font-weight:600}
 a{text-decoration:none;color:#2563eb} a:hover{text-decoration:underline}
 li a{margin-left:.8rem} a.muted{color:#999}
 @media(prefers-color-scheme:dark){body{background:#111;color:#ddd}li{border-color:#333}a{color:#6ea8fe}}
"""


def page(slug, solutions):
    return f"{slug}{SOLUTIONS_SUFFIX}.html" if solutions else f"{slug}.html"


def nav(nbs, idx, solutions, bottom=False):
    prev_html = next_html = '<span class="spacer">&nbsp;</span>'
    if idx > 0:
        s, t = nbs[idx - 1]
        prev_html = f'<a href="{page(s, solutions)}">&larr; {t}</a>'
    if idx < len(nbs) - 1:
        s, t = nbs[idx + 1]
        next_html = f'<a href="{page(s, solutions)}">{t} &rarr;</a>'
    slug = nbs[idx][0]
    other = page(slug, not solutions)
    label = "student version" if solutions else "solutions"
    mid = (f'<a href="index.html">module index</a>'
           f'<a href="{other}">{label}</a>'
           f'<a href="../index.html">all modules</a>')
    cls = "swc-nav bottom" if bottom else "swc-nav"
    return f'<div class="{cls}">{prev_html}<span class="mid">{mid}</span>{next_html}</div>'


IMG_SRC = re.compile(r'(<img[^>]*\ssrc=")([^"]+)(")')


def embed_local_images(html, base_dir):
    """Inline any locally-referenced image as a data: URI.

    Markdown cells reference the paper panels with a relative path
    (``../assets/paper/figX.png``) that only resolves from the notebook's own
    directory -- not from wherever we render to. nbconvert's --embed-images does
    not touch raw <img> tags, so we inline them here. This also makes every page
    self-contained, so it works from any location.
    """
    def repl(m):
        pre, src, post = m.groups()
        if src.startswith(("data:", "http://", "https://", "//")):
            return m.group(0)
        path = os.path.normpath(os.path.join(base_dir, src))
        if not os.path.isfile(path):
            print(f"    ! missing image: {src}")
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"{pre}data:{mime};base64,{data}{post}"

    return IMG_SRC.sub(repl, html)


def inject_nav(path, nbs, idx, solutions, nb_dir):
    with open(path) as f:
        html = f.read()
    if 'class="swc-nav' in html:
        return
    html = embed_local_images(html, nb_dir)
    html = html.replace("</head>", NAV_CSS + "</head>", 1)
    m = re.search(r"<body[^>]*>", html)
    if m:
        html = html[: m.end()] + nav(nbs, idx, solutions) + html[m.end():]
    html = html.replace("</body>", nav(nbs, idx, solutions, bottom=True) + "</body>", 1)
    with open(path, "w") as f:
        f.write(html)


def convert(ipynb, outdir):
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html",
                    "--embed-images", "--output-dir", outdir, ipynb],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_module_index(module, nbs, outdir):
    rows = []
    for slug, title in nbs:
        links = []
        if os.path.exists(os.path.join(outdir, page(slug, True))):
            links.append(f'<a href="{page(slug, True)}">solutions</a>')
        if os.path.exists(os.path.join(outdir, page(slug, False))):
            links.append(f'<a class="muted" href="{page(slug, False)}">student</a>')
        rows.append(f'<li><span>{title}</span><span>{"".join(links)}</span></li>')
    html = f"""<!doctype html><meta charset="utf-8"><title>{module_title(module)} — preview</title>
<style>{SITE_CSS}</style>
<h1>{module_title(module)}</h1>
<div class="sub"><a href="../index.html">&larr; all modules</a> · each notebook links on to the next</div>
<ul>
{os.linesep.join(rows)}
</ul>"""
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(html)


def write_site_index(modules, out):
    blocks = []
    for m in modules:
        nbs = module_notebooks(m)
        blocks.append(
            f'<li><span><a href="{m}/index.html">{module_title(m)}</a></span>'
            f'<span class="muted">{len(nbs)} notebooks</span></li>')
    html = f"""<!doctype html><meta charset="utf-8"><title>SWC ENC 2026 — course previews</title>
<style>{SITE_CSS}</style>
<h1>SWC ENC 2026</h1>
<div class="sub">Rendered previews of the course notebooks. To <em>run</em> them, clone the
<a href="https://github.com/stootoon/swc-en-2026">repository</a>.</div>
<h2>Modules</h2>
<ul>
{os.linesep.join(blocks)}
</ul>"""
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*", help="modules to build (default: all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
    args = ap.parse_args()

    modules = args.modules or find_modules()
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, ".nojekyll"), "w").close()

    for module in modules:
        nbs = module_notebooks(module)
        if not nbs:
            print(f"  (skipping {module}: no notebooks)")
            continue
        outdir = os.path.join(args.out, module)
        os.makedirs(outdir, exist_ok=True)
        for slug, _ in nbs:
            for ipynb in sorted(glob.glob(
                    os.path.join(ROOT, module, "notebooks", f"{slug}*.ipynb"))):
                convert(ipynb, outdir)
        nb_dir = os.path.join(ROOT, module, "notebooks")
        for idx, (slug, _) in enumerate(nbs):
            for solutions in (True, False):
                p = os.path.join(outdir, page(slug, solutions))
                if os.path.exists(p):
                    inject_nav(p, nbs, idx, solutions, nb_dir)
        write_module_index(module, nbs, outdir)
        print(f"  {module}: {len(nbs)} notebooks")

    write_site_index(find_modules(), args.out)
    print(f"site -> {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
