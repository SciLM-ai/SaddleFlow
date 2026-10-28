"""Two CPU checks on the kinds of saddle each source holds (no new runs).

    distance  how far each saddle sits from its minimum (largest per-atom move, minimum image, overall shift removed)
              for: the mp20bat dataset saddles <= 1 eV (local train copy, which still holds the triplets the current
              release moved to tier2/, and the current release's main train file), the Tier B regions, and the
              24-minimum reference regions; the share beyond 1.5, 2.0 and 2.5 A (every Sella run here stops when an
              atom drifts 2.5 A from the minimum)
    reach     the 24 test minima: the stored dataset saddles <= 1 eV of every mp20bat triplet (test and train copies)
              whose reactant or product is that minimum (same atoms, within 0.1 A), and per method whether a landing
              came within 0.2 A of each (stage 1) and whether a converged Sella run ended within 0.15 A of it (stage 2)

    python examples/MaterialsSaddles/experiments/hint_finetune/saddle_reach_check.py --out-dir <evaluation directory>
"""
import argparse
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.db import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_hints as ev  # noqa: E402
from eval_hints import hl, read_rows  # noqa: E402

LOCAL = "/scratch/gqs7zr/saddleflow/MaterialsSaddles/mp20bat/{split}/mp20bat_{split}_*.aselmdb"
RELEASE = "/scratch/gqs7zr/saddleflow/practicality/hf/mp20bat/train/mp20bat_train_0000.aselmdb"
SAME, LAND, END, CUTOFF = 0.1, 0.2, 0.15, 1.0


def triplets(pattern):
    """[(numbers, cell, R, S, P, barrier from R, barrier from P)]"""
    out, trip = [], []
    for f in sorted(glob.glob(pattern)):
        with connect(f) as db:
            for row in db.select():
                trip.append(row)
                if len(trip) == 3:
                    R, S, P = (r.toatoms() for r in trip)
                    eR, eS, eP = (r.energy for r in trip)
                    out.append((R.numbers, R.cell.array, R.positions, S.positions, P.positions, eS - eR, eS - eP))
                    trip = []
    return out


def size(m, pos):
    return float(hl.saddle_distance(m, np.asarray(pos)[None], m.positions)[0])


def summary(name, d):
    d = np.asarray(d)
    return (f"  {name:44s} {len(d):7d} | {np.percentile(d, 10):5.2f} {np.median(d):5.2f} {np.percentile(d, 90):5.2f} | "
            + "  ".join(f"{100 * np.mean(d > x):4.0f}%" for x in (1.5, 2.0, 2.5)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    args = SimpleNamespace(old_dir=ev.OLD_DIR, limit=0)
    from ase import Atoms

    L = ["DISTANCE from the minimum to the saddle (largest per-atom move, A), saddles <= 1 eV above that minimum",
         "  source                                           count |   p10   p50   p90 |  >1.5  >2.0  >2.5"]
    for name, pattern in (("mp20bat train, local copy (main + flagged)", LOCAL.format(split="train")),
                          ("mp20bat train, current release (main)", RELEASE)):
        d = []
        for Z, cell, R, S, P, bR, bP in triplets(pattern):
            for E, b in ((R, bR), (P, bP)):
                if b <= CUTOFF:
                    d.append(size(Atoms(numbers=Z, positions=E, cell=cell, pbc=True), S))
        L.append(summary(name, d))
    tb = json.loads(Path(ev.LABELS).read_text())["minima"]
    d = [size(Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True), r["saddle"])
         for e in tb for r in e["regions"]]
    L.append(summary("Tier B regions", d))
    minima = {t: m for t, (m, _) in ev.test_minima(args).items()}
    ref = ev.reference(args, out, minima)
    d = [size(minima[t], r["pos"]) for t in minima for r in ref[t]["regions"]]
    L.append(summary("24-minimum reference regions", d))

    # Stored dataset saddles of the 24 test minima.
    stored = {t: [] for t in minima}
    for split in ("test", "train"):
        for Z, cell, R, S, P, bR, bP in triplets(LOCAL.format(split=split)):
            for t, m in minima.items():
                if len(Z) != len(m) or np.any(Z != m.numbers):
                    continue
                for E, b in ((R, bR), (P, bP)):
                    if b <= CUTOFF and size(m, E) < SAME:
                        if not stored[t] or hl.saddle_distance(m, np.array(stored[t]), S).min() >= SAME:
                            stored[t].append(S)
    n = sum(len(v) for v in stored.values())
    in_ref = sum(bool(ref[t]["regions"]) and hl.saddle_distance(m, np.array([r["pos"] for r in ref[t]["regions"]]), s).min() < END
                 for t, m in minima.items() for s in stored[t])
    far = [size(minima[t], s) for t in minima for s in stored[t]]
    L += ["", f"REACH: the 24 test minima hold {n} distinct stored dataset saddles <= {CUTOFF} eV, on "
          f"{sum(bool(v) for v in stored.values())} minima; {in_ref} of them are in the 24-minimum reference "
          f"(within {END} A); distance from the minimum p50 {np.median(far) if far else float('nan'):.2f} A",
          f"  stage 1: a landing within {LAND} A; stage 2: a converged Sella run ending within {END} A",
          "  method              stage 1   stage 2"]
    lands, runs = {}, {}
    for l in read_rows(out / "landings" / "*.json"):
        if l["key"] in minima:
            lands.setdefault(l["arm"], {}).setdefault(l["key"], []).append(l["landing"])
    for l in read_rows(Path(ev.OLD_DIR) / "landings" / "landings_*.json"):
        if l["tid"] in minima:
            lands.setdefault(ev.OLD_ARMS[l["arm"]], {}).setdefault(l["tid"], []).append(l["landing"])
    for r in read_rows(out / "runs" / "runs_*.json"):
        if r["key"] in minima and r.get("conv"):
            runs.setdefault(r["arm"], {}).setdefault(r["key"], []).append(r["pos"])
    for r in read_rows(Path(ev.OLD_DIR) / "runs" / "runs_*.json"):
        if r["tid"] in minima and r.get("conv") and r["arm"] in ev.OLD_ARMS:
            runs.setdefault(ev.OLD_ARMS[r["arm"]], {}).setdefault(r["tid"], []).append(r["pos"])

    def hits(points, radius):
        k = 0
        for t, m in minima.items():
            X = points.get(t)
            for s in stored[t]:
                k += bool(X) and hl.saddle_distance(m, np.array(X), s).min() <= radius
        return k

    for arm in ("ft_hint_1x8", "ft_hint", "ft_noise", "rel_hint", "rel_noise", "rel_mobile_hint", "rel_soft", "push", "softpush"):
        s1 = f"{hits(lands[arm], LAND):3d}/{n}" if arm in lands else "    -"
        s2 = f"{hits(runs[arm], END):3d}/{n}" if arm in runs else "    -"
        L.append(f"  {arm:18s} {s1:>8s}  {s2:>8s}")
    union = lambda arms: {t: [x for a_ in arms for x in runs.get(a_, {}).get(t, [])] for t in minima}
    L.append(f"  {'push + softpush':18s} {'':8s}  {hits(union(('push', 'softpush')), END):3d}/{n}")
    L.append(f"  {'any model arm':18s} {'':8s}  "
             f"{hits(union(('ft_hint_1x8', 'ft_hint', 'ft_noise', 'rel_hint', 'rel_noise', 'rel_mobile_hint', 'rel_soft')), END):3d}/{n}")
    text = "\n".join(L)
    print(text)
    (out / "saddle_reach_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
