"""Catalog-connected coverage as a function of starts per minimum.

Reads one or more ``source_study.npz`` archives written by
``source_study.py compare --populations connected``.  For each requested
count k it keeps only the FIRST k noisy starts of every minimum, so the
8/32/128 numbers are nested subsets of one run and coverage can only grow
with k.  CPU only; no model or UMA calls.

"Connected" is the stored R/S/P catalog adjacency, exactly as in the
source-study report, NOT independent downhill verification.

The "uniform" column is a reference, not a model: the coverage expected if
each start independently hit one of its minimum's e connected saddles
uniformly at random, with the run's observed connected-hit rate p,
i.e. mean over minima of 1 - (1 - p/e)^k.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archives", nargs="+", help="source_study.npz files (one per noise level)")
    p.add_argument("--draws", default="8,32,128", help="Comma-separated starts per minimum")
    p.add_argument("--method", default="n12_uncond")
    p.add_argument("--out-dir", help="Where to write coverage_curve.txt/.pdf (default: next to the first archive)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load(path, method):
    data = np.load(path, allow_pickle=True)
    config = json.loads(str(data["config_json"]))
    use = (data["method"] == method) & (data["source_population"][data["source_index"]] == "connected")
    if not use.any():
        raise ValueError(f"no connected starts for {method} in {path}")
    rows = data["source_index"][use]
    minimum = data["source_minimum_index"][rows]
    # Starts are stored in generation order (minimum by minimum, draw by draw),
    # so a source's rank among its minimum's sources is its draw number.
    order = np.argsort(rows, kind="stable")
    rank = np.empty(len(rows), dtype=int)
    seen = {}
    for i in order:
        rank[i] = seen.get(minimum[i], 0); seen[minimum[i]] = rank[i]+1
    return dict(path=str(path), sigma=float(config["coverage_noise_sigma"]), minimum=minimum, rank=rank,
                found=data["nearest_true_index"][use], outcome=data["outcome"][use],
                physical=data["physical_success"][use], edges=data["catalog_edges"],
                max_draws=int(rank.max())+1)


def coverage(run, k):
    keep = run["rank"] < k
    minimum, found, outcome = run["minimum"][keep], run["found"][keep], run["outcome"][keep]
    hit = outcome == 0
    fractions, uniform, reached, total, connected_hits = [], [], 0, 0, 0
    missed = []
    for m in np.unique(minimum):
        expected = np.flatnonzero(np.any(run["edges"] == m, axis=1))
        mine = minimum == m
        observed = np.unique(found[mine & hit])
        matched = np.intersect1d(expected, observed)
        reached += len(matched); total += len(expected)
        fractions.append(len(matched)/len(expected))
        connected_hits += np.sum(mine & hit & np.isin(found, expected))
        missed += [(int(m), int(s)) for s in np.setdiff1d(expected, observed)]
    p = connected_hits/keep.sum()
    for m in np.unique(minimum):
        e = np.sum(np.any(run["edges"] == m, axis=1))
        uniform.append(1-(1-p/e)**k)
    return dict(k=k, n=int(keep.sum()), minima=len(fractions), coverage=100*np.mean(fractions),
                uniform=100*np.mean(uniform), reached=reached, total=total, connected_hit=100*p,
                true=100*np.mean(hit), false=100*np.mean(outcome == 1),
                physical=100*np.mean(run["physical"][keep]), missed=missed)


def main():
    args = parse_args()
    draws = sorted(map(int, args.draws.split(",")))
    runs = [load(path, args.method) for path in args.archives]
    out = Path(args.out_dir or Path(args.archives[0]).parent); out.mkdir(parents=True, exist_ok=True)
    lines = [f"connected-saddle coverage vs starts per minimum ({args.method}, EMA)", "="*112,
             "First k starts of each minimum (nested subsets of one run). Connectivity = stored R/S/P catalog adjacency.",
             "uniform% = expected coverage if each start hit a random connected saddle at the observed connected-hit rate.", "",
             "sigma(A)     k      N  minima  coverage%  uniform%  reached/total  connected_hit%  true%  false%  phys%",
             "-"*112]
    results = []
    for run in runs:
        for k in draws:
            if k > run["max_draws"]:
                lines.append(f"{run['sigma']:8.2f} {k:5d}  skipped: run has only {run['max_draws']} starts per minimum")
                continue
            r = coverage(run, k); results.append((run, r))
            lines.append(f"{run['sigma']:8.2f} {k:5d} {r['n']:6d} {r['minima']:7d} {r['coverage']:10.2f} {r['uniform']:9.2f}"
                         f" {r['reached']:7d}/{r['total']:<5d} {r['connected_hit']:15.2f} {r['true']:6.2f}"
                         f" {r['false']:7.2f} {r['physical']:6.2f}")
    for run in runs:
        k = min(run["max_draws"], max(draws)); r = coverage(run, k)
        lines += ["", f"Never-reached connected saddles at sigma={run['sigma']:.2f} A, k={k}: {len(r['missed'])}",
                  "  (minimum index, catalog saddle index)"]
        for first in range(0, len(r["missed"]), 8):
            lines.append("  " + "  ".join(f"({m:2d},{s:3d})" for m, s in r["missed"][first:first+8]))
    text = "\n".join(lines)+"\n"
    (out/"coverage_curve.txt").write_text(text); print(text, flush=True)
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for run in runs:
            ks = [k for k in range(1, run["max_draws"]+1)]
            curve = [coverage(run, k) for k in ks]
            line, = ax.plot(ks, [c["coverage"] for c in curve], label=f"model, sigma={run['sigma']:.2f} A")
            ax.plot(ks, [c["uniform"] for c in curve], ls="--", color=line.get_color(), alpha=.6,
                    label=f"uniform reference, sigma={run['sigma']:.2f} A")
        ax.set(xscale="log", xlabel="Starts per minimum", ylabel="Connected saddles found (%)", ylim=(0, 102),
               title=f"{args.method}: catalog-connected coverage")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out/"coverage_curve.pdf"); plt.close(fig)
        print(f"[plot] {out/'coverage_curve.pdf'}")


if __name__ == "__main__":
    main()
