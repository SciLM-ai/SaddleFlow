"""
Score sample.py endpoints against the Al saddles. An endpoint within --tol (largest per-atom distance over the mobile atoms,
minimum image) of a dataset saddle hits that saddle (the nearest one); within --tol of the minimum it fell back; anything
else is elsewhere: a saddle that is not in the dataset (e.g. one of its symmetric copies at another site), or no saddle.
Telling those apart needs a saddle search from the endpoint (saddleflow.search.find_saddles runs Sella on proposals).
For --mode movers it also prints the recall: the saddles reached by at least one of their own draws.

    python examples/Al_adatom/score.py examples/Al_adatom/runs/fm2_ft/movers examples/Al_adatom/runs/fm2/movers
"""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="sample.py --out directories")
    p.add_argument("--tol", type=float, default=0.2)
    p.add_argument("--traj", default=str(here / "al_triplets.traj"))
    a = p.parse_args()

    frames = read(a.traj, ":")
    R = frames[0]
    mobile = np.ones(len(R), bool); mobile[R.constraints[0].index] = False
    ref = np.stack([R.positions] + [S.positions for S in frames[1::3]])[:, mobile]       # minimum, then the saddles
    cell = np.asarray(R.cell); inv = np.linalg.inv(cell)

    for run in a.runs:
        ends = read(Path(run) / "endpoints.traj", ":")
        hit, own, fell = [], set(), 0
        for e in ends:
            d = (e.positions[mobile] - ref) @ inv
            d = np.linalg.norm((d - np.round(d)) @ cell, axis=2).max(axis=1)              # largest atom distance per reference
            k = int(d.argmin())
            if d[k] < a.tol and k == 0:
                fell += 1
            elif d[k] < a.tol:
                hit.append(k - 1)
                if e.info.get("saddle") == k - 1:
                    own.add(k - 1)
        n, ns = len(ends), len(ref) - 1
        print(f"{run}: {n} endpoints ({ends[0].info.get('mode')}): dataset saddle {len(hit)} ({len(hit) / n:.0%}), "
              f"fell back {fell} ({fell / n:.0%}), elsewhere {n - len(hit) - fell} ({(n - len(hit) - fell) / n:.0%}); "
              f"distinct saddles reached {len(set(hit))}/{ns}")
        if ends[0].info.get("mode") == "movers":
            print(f"    recall: {len(own)}/{ns} saddles reached by at least one of their own draws")


if __name__ == "__main__":
    main()
