"""Compare the never-reached saddles with all the others, from a `connectivity_test.py --controls all` run.

Per (minimum, saddle) pair: whether the downhill test confirms it, the barrier from that minimum and from the
other end, the saddle's two lowest curvatures, the Li distance from minimum to saddle, how many saddles that
minimum has, and how often the model reached it (draws out of the archive's count per minimum). Pairs are
grouped as missed at every noise level, missed at only some, and reached; groups are summarised by mean,
standard deviation and range, and the reach rate is rank-correlated with each property. CPU only.

    python examples/LiC/experiments/connectivity/summarize_connectivity.py <run>/connectivity.json \
        $R/sigma0.15/source_study.npz $R/sigma0.30/source_study.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source_study"))
from coverage_curve import load  # noqa: E402

PROPERTIES = [("barrier_here", "barrier from this minimum (eV)"), ("barrier_other", "barrier from the other end (eV)"),
              ("lambda1", "lowest eigenvalue (eV/A^2)"), ("lambda2", "second eigenvalue (eV/A^2)"),
              ("distance", "minimum -> saddle Li distance (A)"), ("degree", "saddles connected to this minimum")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("connectivity")
    p.add_argument("archives", nargs="+")
    args = p.parse_args()
    result = json.loads(Path(args.connectivity).read_text())
    saddles = {int(s): r for s, r in result["saddles"].items()}
    both = {tuple(x) for x in result["pairs"]}
    some = {tuple(x) for x in result["ever_missed"]} - both

    data = np.load(args.archives[0], allow_pickle=True)
    config = json.loads(str(data["config_json"]))
    minima, edges, li = data["minima"], data["catalog_edges"], int(data["li_index"])
    frames = read(Path(config["study_dir"]) / "split/catalog.traj", index=":")
    runs = [load(path, "n12_uncond") for path in args.archives]
    degree = {m: int(np.sum(np.any(edges == m, axis=1))) for m in range(len(minima))}

    rows = []
    for s, r in saddles.items():
        found = [e["minimum"] for e in r["ends"]]
        for m in r["stored"]:
            here = [r["energy"] - e["energy"] for e in r["ends"] if e["minimum"] == m]
            other = [r["energy"] - e["energy"] for e in r["ends"] if e["minimum"] != m]
            li_s = frames[3 * s + 1].positions[li]
            reach = [np.sum((run["found"] == s) & (run["outcome"] == 0) & (run["minimum"] == m))
                     / np.sum(run["minimum"] == m) for run in runs]
            rows.append(dict(pair=(m, s), confirmed=m in found, barrier_here=here[0] if here else np.nan,
                             barrier_other=other[0] if other else np.nan, lambda1=r["eigenvalues"][0],
                             lambda2=r["eigenvalues"][1], degree=degree[m],
                             distance=float(find_mic(li_s - minima[m], frames[0].cell, frames[0].pbc)[1]),
                             reach=float(np.mean(reach)),
                             group="missed (all)" if (m, s) in both else "missed (some)" if (m, s) in some else "reached"))

    n_saddles, bad = len(saddles), [s for s, r in saddles.items()
                                    if sorted(e["minimum"] for e in r["ends"]) != r["stored"]]
    lines = ["Never-reached vs reached connected pairs (downhill test on every catalog saddle)", "=" * 104,
             f"Downhill test: {n_saddles - len(bad)}/{n_saddles} saddles relax to exactly their two stored minima"
             + (f"; failures: {bad}" if bad else "."),
             f"Pairs: {len(rows)}; missed at every noise level {len(both)}, at only some {len(some)}. "
             f"Reach = share of a minimum's draws that land on that saddle (mean over {len(runs)} archives).", "",
             "missed at every noise level, one row per pair:",
             "  (min, sad)  confirmed  barrier here  other end  lambda1  lambda2  distance  degree  reach"]
    for row in sorted((r for r in rows if r["group"] == "missed (all)"), key=lambda r: r["pair"]):
        lines.append(f"  {str(row['pair']):10s}  {str(row['confirmed']):9s}  {row['barrier_here']:12.3f}  "
                     f"{row['barrier_other']:9.3f}  {row['lambda1']:7.3f}  {row['lambda2']:7.3f}  "
                     f"{row['distance']:8.3f}  {row['degree']:6d}  {row['reach']:5.3f}")
    lines += ["", "group averages: mean +- sd [min, max]"]
    for key, label in PROPERTIES + [("reach", "reach rate")]:
        lines.append(f"  {label}")
        for group in ("missed (all)", "missed (some)", "reached"):
            v = np.array([r[key] for r in rows if r["group"] == group], float)
            v = v[np.isfinite(v)]
            if len(v):
                lines.append(f"    {group:14s} n={len(v):3d}  {v.mean():7.3f} +- {v.std():6.3f}  [{v.min():7.3f}, {v.max():7.3f}]")
    lines += ["", "rank correlation of reach rate with each property, all pairs (Spearman rho, p-value):"]
    reach = np.array([r["reach"] for r in rows])
    for key, label in PROPERTIES:
        v = np.array([r[key] for r in rows], float); ok = np.isfinite(v)
        rho, pval = spearmanr(v[ok], reach[ok])
        lines.append(f"  {label:38s} rho = {rho:+.3f}  (p = {pval:.1e})")
    text = "\n".join(lines) + "\n"
    out = Path(args.connectivity).with_name("connectivity_summary.txt")
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
