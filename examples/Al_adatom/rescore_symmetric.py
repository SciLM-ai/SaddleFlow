"""score.py with symmetry: does each endpoint sit on a dataset saddle, allowing relabelled atoms and the slab's symmetry?

    python examples/Al_adatom/rescore_symmetric.py <sample.py --out dir> ... --workers 32 --out <dir>

score.py compares atom by atom, so an endpoint on a rotated or mirrored copy of a dataset saddle, or on the same
structure with identical Al atoms swapped, counts as "elsewhere". Here each endpoint (mobile atoms) is compared with the
353 saddles and the minimum within 0.2 A: atom by atom; as a set of positions (relabelled); and as a set after each
in-plane symmetry operation about the adatom that maps the minimum onto itself (sella_check.Geometry). Geometry only,
no Sella: an endpoint that matches nothing may still be a saddle the dataset lacks (sella_check.py tells).

Writes <out>/<run name>.jsonl (one row per endpoint) and <out>/report.txt.
"""
import argparse
import json
import multiprocessing
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sella_check import SAME, Geometry      # noqa: E402

GEO = None


def init(traj):
    global GEO
    GEO = Geometry(traj)


def score(item):
    i, positions = item
    pos = np.asarray(positions)[GEO.mobile]
    if GEO.indexed_distance(pos, GEO.minimum.positions[GEO.mobile]) < SAME:
        return i, "minimum", None
    kind, k = GEO.compare(pos, GEO.ref, GEO.ref_z)
    return i, kind or "none", k


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="sample.py --out directories")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--traj", default=str(HERE / "al_triplets.traj"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["score.py with relabelled atoms and the slab's symmetry (within 0.2 A, mobile atoms; geometry only)", ""]
    with multiprocessing.get_context("spawn").Pool(a.workers, initializer=init, initargs=(a.traj,)) as pool:
        for run in a.runs:
            ends = read(Path(run) / "endpoints.traj", ":")
            res = sorted(pool.imap_unordered(score, [(i, e.positions.tolist()) for i, e in enumerate(ends)], chunksize=8))
            name = "_".join(Path(run).resolve().parts[-2:])
            with (out / f"{name}.jsonl").open("w") as fh:
                for (i, kind, k), e in zip(res, ends):
                    fh.write(json.dumps(dict(i=i, kind=kind, matched=k, own=e.info.get("saddle"), draw=e.info.get("draw"),
                                             centre=e.info.get("centre"))) + "\n")
            kinds = Counter(kind if not kind.startswith("symmetric") else "symmetric" for _, kind, _ in res)
            n = len(res)
            on = [k for _, kind, k in res if kind not in ("none", "minimum")]
            lines.append(f"{run}: {n} endpoints ({ends[0].info.get('mode')})")
            lines.append("  " + "   ".join(f"{k} {kinds.get(k, 0)} ({kinds.get(k, 0) / n:.1%})"
                                         for k in ("same atoms", "relabelled", "symmetric", "minimum", "none")))
            lines.append(f"  on a dataset saddle, any match: {len(on)} ({len(on) / n:.1%}); distinct dataset saddles "
                         f"matched {len(set(on))}/353 (atom by atom only: "
                         f"{len({k for _, kind, k in res if kind == 'same atoms'})})")
            print("\n".join(lines[-3:]), flush=True)
    (out / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
