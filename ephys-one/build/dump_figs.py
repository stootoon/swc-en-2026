"""Dump the PNG outputs of an executed notebook to a directory: dump_figs.py NB out_dir"""
import base64, glob, json, os, sys
nb_id, out = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = glob.glob(os.path.join(HERE, "notebooks", f"{nb_id}_*_solutions.ipynb"))[0]
os.makedirs(out, exist_ok=True)
k = 0
for c in json.load(open(path))["cells"]:
    for o in c.get("outputs", []):
        if "data" in o and "image/png" in o["data"]:
            k += 1
            open(os.path.join(out, f"nb{nb_id}_fig{k:02d}.png"), "wb").write(base64.b64decode(o["data"]["image/png"]))
print(k, "figures ->", out)
