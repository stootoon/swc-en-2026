"""Report errors / figure counts / exercise counts for executed solutions notebooks."""
import glob, json, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in sorted(glob.glob(os.path.join(HERE, "notebooks", "*_solutions.ipynb"))):
    nb = json.load(open(path))
    errs, figs, ex = 0, 0, 0
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    errs += 1; print("   ERROR:", o.get("ename"), o.get("evalue")[:200])
                if "data" in o and "image/png" in o["data"]:
                    figs += 1
        else:
            ex += "".join(c["source"]).count("**Exercise")
    print(f"{os.path.basename(path):55s} errors={errs} figures={figs} exercises={ex}")
