"""End-to-end sanity check: run the whole picosort pipeline and score it.

Not shipped to students -- this is the harness used while building the module, to
confirm the reference implementations sort the synthetic data well before notebooks
are written around them.

    python build/validate_pipeline.py
"""
import numpy as np
import picosort as ps


def run(seed=0, verbose=True):
    rec = ps.make_recording(n_units=6, duration_s=20.0, seed=seed)
    gt = rec.ground_truth
    res = ps.run_picosort(rec)

    scores, overlap = ps.match_to_truth(res.spike_times, res.spike_labels,
                                        gt.spike_times, gt.spike_labels, rec.fs)
    s = ps.summary(scores)
    ev = ps.explained_variance(res.parts["waveforms"])
    if verbose:
        print(f"seed {seed}: "
              f"detected {len(res.det_times)} (kept {res.keep.sum()}), "
              f"{len(res.template_ids)} templates, {len(res.spike_times)} MP spikes | "
              f"PC1-3 var={np.round(ev[:3], 2)} | "
              f"recall={s['mean_recall']:.2f} precision={s['mean_precision']:.2f} "
              f"agreement={s['mean_agreement']:.2f}")
    return s


if __name__ == "__main__":
    results = [run(seed=s) for s in range(8)]
    print("\n=== across seeds ===")
    print("recall:   ", [round(r["mean_recall"], 2) for r in results])
    print("precision:", [round(r["mean_precision"], 2) for r in results])
    print("agreement:", [round(r["mean_agreement"], 2) for r in results])
