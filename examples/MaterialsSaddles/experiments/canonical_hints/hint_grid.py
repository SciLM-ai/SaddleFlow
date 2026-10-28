"""Single-atom hints with several labels per saddle: coverage, cost and crowding on the hint grid used at inference.

Labels. Every atom whose move from the minimum to a saddle (rigid shift removed) is at least `thr` times the
largest move and at least 0.2 A gives the saddle one label: (that atom, its direction). The largest mover always
gives one. A training example is (label -> saddle).

Inference hints. A fixed grid of K directions per atom (Fibonacci sphere, randomly rotated per atom), on every
atom or on the mobile ions only. A hint (atom a, direction g) points to a saddle when one of the saddle's labels
is on atom a with a direction within theta of g: the hint's cone, the set a model trained with jitter of about
theta would treat as one input.

For each setting (theta x thr x atom set x K), per minimum, then the median and worst minima across minima:
    coverage        share of the minimum's saddles that some hint of the grid points to
    hints to 90%    hints taken in random order (as at inference, which cannot know the good ones) until 90%
                    of the minimum's saddles are pointed to; the mean over 5 orders; "-" if the grid never gets there
    crowding        for each covered saddle, its least crowded cone: the fewest saddles sharing any hint that
                    points to it, counting only competitors more than 0.3 A away (closer ones are likely the
                    same saddle loosely converged). Share of covered saddles with that number <= 3, and <= 10
Cd3Li13 (153 saddles) is reported separately and left out of the medians. Every number is averaged over 3
rotations of the grid. Coverage is also split by which starts found each saddle (single-atom pushes, soft-mode
starts only, or noise starts only), since saddles that no push finds are missing from the pool altogether.

    python examples/MaterialsSaddles/experiments/canonical_hints/hint_grid.py --out-dir <hint_vs_noise dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canonical_hints as ch  # noqa: E402
from canonical_hints import hs, hvn, mic  # noqa: E402

THETAS, THRESHOLDS, KS = (20, 30), (0.7, 0.5), (8, 16, 32)
MIN_MOVE, APART, ROTATIONS, ORDERS = 0.2, 0.3, 3, 5
SEPARATE = 167          # Cd3Li13


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="hint_vs_noise.py output directory")
    p.add_argument("--cutoff", type=float, default=1.0, help="barrier cutoff, eV")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def displacements(m, saddles):
    out = []
    for s in saddles:
        d = mic(m, np.array(s["pos"]) - m.positions)
        out.append(d - d.mean(0))
    return out


def labels(D, thr):
    """[(saddle index, atom, unit direction)]: every atom moving >= thr x the largest and >= MIN_MOVE."""
    out = []
    for k, d in enumerate(D):
        size = np.linalg.norm(d, axis=1)
        top = int(size.argmax())
        for a in np.flatnonzero((size >= thr * size[top]) & (size >= MIN_MOVE)).tolist() or [top]:
            out.append((k, a, d[a] / size[a]))
    return out


def evaluate(m, D, apart, lab, atoms, K, theta, rng):
    """(coverage, hints to 90% or nan, least crowded cone per covered saddle) for one grid rotation."""
    n, cos = len(D), np.cos(np.radians(theta))
    grid = {a: hs.fibonacci(K, rng) for a in atoms}
    hint_atoms = [a for a in atoms for _ in range(K)]
    P = np.zeros((len(hint_atoms), n), bool)           # hint -> saddles it points to
    offset = {a: i * K for i, a in enumerate(atoms)}
    for k, a, u in lab:
        if a in grid:
            P[offset[a] + np.flatnonzero(grid[a] @ u >= cos), k] = True
    covered = P.any(0)
    crowd = np.full(n, np.inf)
    for h in np.flatnonzero(P.any(1)):
        S = np.flatnonzero(P[h])
        for s in S:
            crowd[s] = min(crowd[s], 1 + int(np.sum(apart[s, S])))
    need = []
    for _ in range(ORDERS):
        seen = np.cumsum(P[rng.permutation(len(P))], axis=0) > 0
        reached = np.flatnonzero(seen.sum(1) >= 0.9 * n)
        need.append(reached[0] + 1 if len(reached) else np.nan)
    return covered.mean(), np.nanmean(need) if not np.all(np.isnan(need)) else np.nan, crowd[covered], covered


def main():
    args = parse_args()
    out = Path(args.out_dir)
    saddles, minima = ch.pools(args, out)
    data = {}
    for t, S in saddles.items():
        m = minima[t]
        D = displacements(m, S)
        P = np.array([s["pos"] for s in S])
        apart = np.array([hvn.dist(m, P, p) > APART for p in P])
        np.fill_diagonal(apart, False)
        arms = [{i.split(":")[0] for i in s["members"]} for s in S]
        finder = ["push" if a & {"hint", "push"} else "soft only" if a <= {"soft", "softpush"} else
                  "noise only" if a <= {"noise", "null"} else "mixed" for a in arms]
        data[t] = dict(m=m, D=D, apart=apart, finder=np.array(finder),
                       mobile=[i for i, s in enumerate(m.get_chemical_symbols()) if s in hs.MOBILE])
    n_all = sum(len(d["D"]) for d in data.values())
    L = [f"Single-atom hints with several labels per saddle: {n_all} connected saddles (<= {args.cutoff} eV) on "
         f"{len(data)} stable minima; medians over the {len(data) - (SEPARATE in data)} minima other than "
         f"Cd3Li13 ({SEPARATE}), which is shown separately", "=" * 118, ""]

    for thr in THRESHOLDS:
        per = [len(labels(d["D"], thr)) / len(d["D"]) for d in data.values()]
        mob = np.mean([np.mean([any(a in d["mobile"] for k2, a, _ in labels(d["D"], thr) if k2 == k)
                                for k in range(len(d["D"]))]) for d in data.values()])
        L.append(f"threshold {thr}x (and >= {MIN_MOVE} A): {np.mean(per):.1f} labels per saddle (per-minimum mean); "
                 f"saddles with a mobile ion among their labelled atoms {100 * mob:.0f}% (the ceiling for mobile-ion hints)")
    L.append("")

    results = {}
    rng = np.random.default_rng(0)
    for atoms_name in ("every atom", "mobile ions"):
        L += [f"hints on {atoms_name}",
              "  theta  thr    K   hints  | coverage: median  worst 3            | hints to 90%: median  (minima reaching) "
              "| least crowded cone <= 3 / <= 10 | Cd3Li13: coverage, <= 3"]
        for theta in THETAS:
            for thr in THRESHOLDS:
                for K in KS:
                    rows = {}
                    for t, d in data.items():
                        atoms = list(range(len(d["m"]))) if atoms_name == "every atom" else d["mobile"]
                        if not atoms:
                            rows[t] = (0.0, np.nan, np.array([]), np.zeros(len(d["D"]), bool), 0)
                            continue
                        lab = labels(d["D"], thr)
                        runs = [evaluate(d["m"], d["D"], d["apart"], lab, atoms, K, theta, rng) for _ in range(ROTATIONS)]
                        rows[t] = (np.mean([r[0] for r in runs]), np.nanmean([r[1] for r in runs])
                                   if not all(np.isnan(r[1]) for r in runs) else np.nan,
                                   np.concatenate([r[2] for r in runs]), runs[0][3], len(atoms) * K)
                    results[(atoms_name, theta, thr, K)] = rows
                    main_t = [t for t in rows if t != SEPARATE]
                    cov = np.array([rows[t][0] for t in main_t])
                    worst = sorted(main_t, key=lambda t: rows[t][0])[:3]
                    need = np.array([rows[t][1] for t in main_t])
                    crowd = np.concatenate([rows[t][2] for t in main_t])
                    hints = np.median([rows[t][4] for t in main_t])
                    sep = rows.get(SEPARATE)
                    L.append(f"  {theta:4d}  {thr:3.1f}  {K:3d}  {hints:5.0f}  |  {100 * np.median(cov):5.1f}%   "
                             + " ".join(f"{t}:{100 * rows[t][0]:3.0f}%" for t in worst).ljust(22)
                             + f" |  {np.nanmedian(need) if np.any(~np.isnan(need)) else float('nan'):6.0f}       "
                             f"({np.sum(~np.isnan(need)):2d} of {len(main_t)})       "
                             f"|   {100 * np.mean(crowd <= 3):5.1f}% / {100 * np.mean(crowd <= 10):5.1f}%"
                             + (f"       | {100 * sep[0]:4.0f}%, {100 * np.mean(sep[2] <= 3):3.0f}%" if sep else ""))
        L.append("")

    # Coverage by which starts found the saddle, at the middle setting.
    for key in (("every atom", 30, 0.5, 16), ("every atom", 20, 0.7, 32)):
        rows = results[key]
        L.append(f"coverage by which starts found the saddle (hints on {key[0]}, theta {key[1]}, thr {key[2]}, K {key[3]}; "
                 f"pooled over minima other than Cd3Li13):")
        for f in ("push", "mixed", "soft only", "noise only"):
            cov = [c for t, r in rows.items() if t != SEPARATE for c, fi in zip(r[3], data[t]["finder"]) if fi == f]
            if cov:
                L.append(f"   {f:11s} {len(cov):4d} saddles, covered {100 * np.mean(cov):5.1f}%")
        L.append("")

    # Per minimum at two settings.
    keys = (("every atom", 30, 0.5, 16), ("every atom", 20, 0.7, 32))
    L += ["per minimum (hints on every atom): coverage, hints to 90%, least crowded cone <= 3",
          "   tid  formula               saddles  atoms  | theta 30, thr 0.5, K 16       | theta 20, thr 0.7, K 32"]
    for t, d in sorted(data.items(), key=lambda x: -len(x[1]["D"])):
        cells = []
        for key in keys:
            c, need, crowd, _, _ = results[key][t]
            cells.append(f"{100 * c:4.0f}%  {need if not np.isnan(need) else float('nan'):6.0f}  "
                         f"{100 * np.mean(crowd <= 3) if len(crowd) else float('nan'):4.0f}%")
        L.append(f"  {t:4d}  {d['m'].get_chemical_formula():20s} {len(d['D']):7d}  {len(d['m']):5d}  |  {cells[0]:26s} |  {cells[1]}")
    text = "\n".join(L) + "\n"
    (out / "hint_grid.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
