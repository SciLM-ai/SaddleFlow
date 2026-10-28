"""Can a hint built from each saddle serve as an unambiguous training label, and how many hints cover the saddles?

Reads the verified saddle pools of hint_vs_noise.py (connected index-1 saddles, merged within 0.1 A, on the
minima that are true minima under UMA-S) and builds each saddle's canonical hint: the atom that moves most
from the minimum to the saddle (rigid shift removed) and that atom's direction of motion. No force calls.

    collisions  Two saddles of one minimum collide when their hints share the epicenter and their directions
                lie within an angle theta (the training jitter would make them the same input). For each theta:
                the share of saddles that collide with another, how far apart colliding saddles are (averaging
                two saddles 0.1 A apart does little harm, 1 A apart ruins the landing point), and whether the
                size of the epicenter's move would tell them apart. Also how well defined the epicenter is (the
                second-largest atomic move against the largest) and which elements lead the saddles.
    coverage    A saddle is reachable when some generated hint lies within theta of its canonical hint: same
                epicenter, direction within theta. Generators: K spread directions (Fibonacci sphere, randomly
                rotated per atom, averaged over 5 rotations) on every mobile-ion atom, or on every atom. Reported
                with the number of hints each generator needs.

    python examples/MaterialsSaddles/experiments/canonical_hints/canonical_hints.py --out-dir <hint_vs_noise dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hint_vs_noise"))
import hint_vs_noise as hvn  # noqa: E402
from hint_vs_noise import hs, mic, read_rows  # noqa: E402

THETAS = (10, 20, 30, 45)
KS = (8, 16, 32, 64, 128)
ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P", "C", "Si", "B", "Ge"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="hint_vs_noise.py output directory")
    p.add_argument("--cutoff", type=float, default=1.0, help="barrier cutoff, eV")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def pools(args, out):
    """{tid: list of saddle dicts} for the stable minima, and the minima."""
    plan, rows, minima = hvn.load_plan(args)
    eigs = {}
    for l in read_rows(out / "landings" / "landings_*.json"):
        eigs.setdefault(l["tid"], l["soft_eigs"][0])
    minima = {t: m for t, m in minima.items() if eigs.get(t, -1) >= -0.05}
    points = {}
    for c in read_rows(out / "verify" / "verify_*.json"):
        if c["tid"] not in minima:
            continue
        P = points.setdefault(c["tid"], [])
        if not P or hvn.dist(minima[c["tid"]], np.array([p["pos"] for p in P]), np.array(c["pos"])).min() >= hs.MERGE:
            P.append(c)
    saddles = {t: [p for p in P if p.get("nneg") == 1 and p.get("connected") and p["barrier"] <= args.cutoff]
               for t, P in points.items()}
    return {t: s for t, s in saddles.items() if s}, minima


def canonical(m, saddle):
    d = mic(m, np.array(saddle["pos"]) - m.positions)
    d -= d.mean(0)
    size = np.linalg.norm(d, axis=1)
    order = np.argsort(size)[::-1]
    e = int(order[0])
    return dict(epicenter=e, element=m.get_chemical_symbols()[e], direction=d[e] / size[e], size=float(size[e]),
                second=float(size[order[1]] / size[e]), barrier=saddle["barrier"], pos=saddle["pos"])


def angle(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def main():
    args = parse_args()
    out = Path(args.out_dir)
    saddles, minima = pools(args, out)
    hints = {t: [canonical(minima[t], s) for s in S] for t, S in saddles.items()}
    n = sum(len(h) for h in hints.values())
    L = [f"Canonical hints from {n} connected index-1 saddles (barrier <= {args.cutoff} eV) on {len(hints)} stable minima",
         "=" * 104, ""]

    # Epicenters.
    allh = [h for H in hints.values() for h in H]
    kind = lambda el: "mobile ion" if el in hs.MOBILE else ("anion / framework" if el in ANIONS else "other cation")
    L += ["1. the epicenter (the atom that moves most)"]
    for k in ("mobile ion", "other cation", "anion / framework"):
        L.append(f"   led by a {k:18s} {100 * np.mean([kind(h['element']) == k for h in allh]):5.1f}%")
    sec = np.array([h["second"] for h in allh])
    L.append(f"   second-largest move / largest move: p50 {np.median(sec):.2f}; above 0.8 (epicenter ambiguous) "
             f"{100 * np.mean(sec > 0.8):.0f}%, above 0.9 {100 * np.mean(sec > 0.9):.0f}%")
    L.append(f"   epicenter's move p10 / p50 / p90: {np.percentile([h['size'] for h in allh], 10):.2f} / "
             f"{np.median([h['size'] for h in allh]):.2f} / {np.percentile([h['size'] for h in allh], 90):.2f} A")

    # Collisions.
    L += ["", "2. collisions: two saddles of one minimum with the same epicenter and directions within theta",
          "   theta   saddles colliding   per-minimum mean   colliding pairs   pair distance p50 / share > 0.3 A   "
          "pairs the move size separates (> 30%)"]
    for th in THETAS:
        hit, pair_d, sep, per_min = [], [], [], []
        for t, H in hints.items():
            m = minima[t]
            c = np.zeros(len(H), bool)
            for i in range(len(H)):
                for j in range(i + 1, len(H)):
                    if H[i]["epicenter"] == H[j]["epicenter"] and angle(H[i]["direction"], H[j]["direction"]) < th:
                        c[i] = c[j] = True
                        pair_d.append(hvn.dist(m, np.array([H[i]["pos"]]), np.array(H[j]["pos"]))[0])
                        a, b = sorted((H[i]["size"], H[j]["size"]))
                        sep.append(b / a > 1.3)
            hit += c.tolist(); per_min.append(c.mean())
        pair_d = np.array(pair_d)
        L.append(f"   {th:4d}    {100 * np.mean(hit):10.1f}%       {100 * np.mean(per_min):10.1f}%     {len(pair_d):10d}       "
                 + (f"{np.median(pair_d):.2f} A / {100 * np.mean(pair_d > 0.3):.0f}%" if len(pair_d) else "-")
                 + f"{'':19s}" + (f"{100 * np.mean(sep):.0f}%" if sep else "-"))

    # Coverage by generated hints.
    L += ["", "3. coverage: share of saddles with a generated hint within theta of their canonical hint "
              "(per-minimum mean; hints per minimum in brackets)"]
    rng = np.random.default_rng(0)
    for label, chooser in (("every mobile-ion atom", lambda m: [i for i, s in enumerate(m.get_chemical_symbols()) if s in hs.MOBILE]),
                           ("every atom", lambda m: list(range(len(m))))):
        L.append(f"   pushing {label}:  (upper bound, any direction: "
                 f"{100 * np.mean([np.mean([h['epicenter'] in set(chooser(minima[t])) for h in H]) for t, H in hints.items()]):.0f}%)")
        L.append("   theta  " + "".join(f"{f'K = {k}':>16s}" for k in KS))
        for th in THETAS:
            cells = []
            for k in KS:
                share, count = [], []
                for t, H in hints.items():
                    atoms = set(chooser(minima[t]))
                    count.append(len(atoms) * k)
                    vals = []
                    for _ in range(5):
                        dirs = {a: hs.fibonacci(k, rng) for a in {h["epicenter"] for h in H} if a in atoms}
                        vals.append(np.mean([h["epicenter"] in dirs and np.degrees(np.arccos(np.clip(
                            (dirs[h["epicenter"]] @ h["direction"]).max(), -1, 1))) <= th for h in H]))
                    share.append(np.mean(vals))
                cells.append(f"{100 * np.mean(share):5.1f}% [{np.median(count):5.0f}]")
            L.append(f"   {th:4d}   " + "".join(f"{c:>16s}" for c in cells))

    # Per minimum.
    L += ["", "4. per minimum: saddles, colliding at 20 degrees, epicenters used, led by a mobile ion",
          "    tid  formula               saddles  colliding@20  epicenters  mobile-led"]
    for t, H in hints.items():
        c = sum(any(H[i]["epicenter"] == H[j]["epicenter"] and angle(H[i]["direction"], H[j]["direction"]) < 20
                    for j in range(len(H)) if j != i) for i in range(len(H)))
        L.append(f"   {t:4d}  {minima[t].get_chemical_formula():20s} {len(H):7d}  {c:12d}  {len({h['epicenter'] for h in H}):10d}  "
                 f"{sum(h['element'] in hs.MOBILE for h in H):10d}")
    text = "\n".join(L) + "\n"
    (out / "canonical_hints.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
