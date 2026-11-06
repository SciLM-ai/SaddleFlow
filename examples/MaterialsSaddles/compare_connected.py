"""Compare eval_connected_uncond.py runs of several models on the same minima and seed.

    python examples/MaterialsSaddles/compare_connected.py fm2=<out dir> plain=<out dir> w2=<out dir>

The first run is the baseline. Endpoints are paired by (minimum, eval), which needs the same --dataset, --n_minima,
--n_evals and --seed in every run.

    verdicts    share of success, near success, stayed at the minimum and failure
    vs first    success against the baseline: endpoints only this run or only the baseline succeeded on (McNemar, exact)
    precision   endpoints both models refined to a connected first-order saddle within 1 eV: median distance from the
                endpoint to its saddle, and the Wilcoxon p of the paired difference
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

VERDICTS = ("success", "near success", "stayed at the minimum", "failure", "error")


def read_run(directory):
    with (Path(directory) / "results.csv").open() as f:
        return {(row["minimum"], row["eval"]): row for row in csv.DictReader(f)}


def found_saddle(row):
    """A connected first-order saddle within the barrier limit (connected is only set for those)."""
    return row["refined_index"] == "1" and row["connected"] == "True"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="name=<eval_connected_uncond.py output directory>; the first is the baseline")
    args = p.parse_args()

    runs = {name: read_run(directory) for name, directory in (run.split("=", 1) for run in args.runs)}
    first = next(iter(runs))

    print("VERDICTS")
    print(f"  {'run':10s} {'endpoints':>9s}  " + "  ".join(f"{v:>21s}" for v in VERDICTS))
    for name, rows in runs.items():
        counts = [sum(r["verdict"] == v for r in rows.values()) for v in VERDICTS]
        print(f"  {name:10s} {len(rows):9d}  " + "  ".join(f"{100 * c / len(rows):20.1f}%" for c in counts))

    print(f"\nAGAINST {first}, on the endpoints both have")
    print("  run        paired | success only here / only baseline   p | saddles both found  distance p50 here / baseline   p")
    for name, rows in list(runs.items())[1:]:
        shared = sorted(set(rows) & set(runs[first]))
        here = np.array([rows[k]["verdict"] == "success" for k in shared])
        base = np.array([runs[first][k]["verdict"] == "success" for k in shared])
        gained, lost = int(np.sum(here & ~base)), int(np.sum(base & ~here))
        p_success = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0

        both = [k for k in shared if found_saddle(rows[k]) and found_saddle(runs[first][k])]
        d_here = np.array([float(rows[k]["distance"]) for k in both])
        d_base = np.array([float(runs[first][k]["distance"]) for k in both])
        p_distance = wilcoxon(d_here, d_base).pvalue if np.any(d_here != d_base) else 1.0
        medians = f"{np.median(d_here):.3f} / {np.median(d_base):.3f}" if both else "-"

        print(f"  {name:10s} {len(shared):6d} | {gained:6d} / {lost:<6d} p {p_success:.3g} | "
              f"{len(both):6d}  {medians}  p {p_distance:.3g}")


if __name__ == "__main__":
    main()
