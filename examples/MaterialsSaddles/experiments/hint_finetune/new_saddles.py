"""Are the model-only saddles really new? One check on a finished evaluation (eval_hints.py), no new runs.

For each fine-tuned arm, take the regions it reaches that the Sella-only arms (push, softpush) do not (the report's
"vs Sella only" gain) and keep those that pass two filters, in order:
    distance  more than 0.3 A (largest per-atom move, minimum image, overall shift removed) from every saddle of every
              region the Sella-only arms reach, so the 0.15 A merge threshold cannot have split them off
    symmetry  not a symmetry copy of a Sella-only region: ase SymmetryEquivalenceCheck on the region representatives
              (same cell, so no primitive reduction or volume scaling), every atom matched within 0.15 A
A sanity line checks that the symmetry test recognises copies made with the minimum's own space-group operations.
Lists the five survivors of ft_hint_1x8 farthest from any Sella-only saddle, and writes each to a .traj
(the minimum, the survivor, the nearest Sella-only saddle) for viewing.

    python examples/MaterialsSaddles/experiments/hint_finetune/new_saddles.py --out-dir <evaluation directory>
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.io import write
from ase.utils.structure_comparator import SymmetryEquivalenceCheck

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_hints as ev  # noqa: E402
from eval_hints import endpoint_region, hl, read_rows  # noqa: E402

ARMS = ("ft_hint", "ft_hint_1x8", "ft_noise")
FAR, SYM_TOL = 0.3, 0.15


def structure(m, pos):
    a = m.copy(); a.set_constraint(); a.positions = np.asarray(pos)
    return a


def same_up_to_symmetry(m, pos, others):
    """Is the structure at pos a symmetry copy of any of the structures in others (same cell)?"""
    if not others:
        return False
    stol = SYM_TOL / (m.get_volume() / len(m)) ** (1 / 3)
    comp = SymmetryEquivalenceCheck(stol=stol, scale_volume=False, to_primitive=False)
    return comp.compare(structure(m, pos), [structure(m, p) for p in others])


def symmetry_sanity(minima, ref, sella_only):
    """(recognised, tried): copies of each minimum's first Sella-only region made with the minimum's space-group
    operations that move it more than FAR."""
    import spglib
    hit = tried = 0
    for t, m in minima.items():
        if not sella_only[t]:
            continue
        pos = ref[t]["regions"][min(sella_only[t])]["pos"]
        ops = spglib.get_symmetry((m.cell.array, m.get_scaled_positions(), m.numbers), symprec=1e-2)
        frac = np.linalg.solve(m.cell.array.T, pos.T).T
        for R, tr in zip(ops["rotations"], ops["translations"]):
            copy = (frac @ R.T + tr) @ m.cell.array
            if hl.saddle_distance(m, np.array([pos]), copy)[0] > FAR:
                tried += 1; hit += same_up_to_symmetry(m, copy, [pos])
                break
    return hit, tried


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--old-dir", default=ev.OLD_DIR)
    a = p.parse_args()
    out = Path(a.out_dir)
    args = SimpleNamespace(old_dir=a.old_dir, limit=0)
    minima = {t: m for t, (m, _) in ev.test_minima(args).items()}
    ref = ev.reference(args, out, minima)

    def reached(rows, t):
        return {endpoint_region(minima[t], ref[t], r["pos"]) for r in rows if r["key"] == t and r.get("conv")} - {-1}

    sella_rows = [dict(r, key=r["tid"]) for r in read_rows(Path(a.old_dir) / "runs" / "runs_*.json")
                  if r["tid"] in minima and r["arm"] in ("push", "softpush")]
    sella_only = {t: reached(sella_rows, t) for t in minima}
    runs = {}
    for r in read_rows(out / "runs" / "runs_*.json"):
        if r["arm"] in ARMS and r["key"] in minima:
            runs.setdefault(r["arm"], []).append(r)

    L = [f"Model-only saddles that survive a distance and a symmetry filter: {out}", "=" * 110, "",
         f"  model-only: regions an arm reaches that push and softpush (128 + 128 Sella runs per minimum) do not",
         f"  far: also more than {FAR} A from every saddle of every Sella-only region",
         f"  far, not symmetric: also not a symmetry copy of a Sella-only region (every atom within {SYM_TOL} A)",
         f"  per minimum: mean over the {len(minima)} test minima; minima: how many have at least one", "",
         "  arm             model-only   far   far, not symmetric | minima: model-only  far  far, not symmetric"]
    survivors = []
    for arm in ARMS:
        n0, n1, n2 = [], [], []
        for t, m in minima.items():
            mine = reached(runs.get(arm, []), t) - sella_only[t]
            S_sella = ref[t]["P"][np.isin(ref[t]["region"], list(sella_only[t]))]
            sella_reps = [ref[t]["regions"][k]["pos"] for k in sorted(sella_only[t])]
            far, kept = [], []
            for k in sorted(mine):
                members = ref[t]["P"][ref[t]["region"] == k]
                d = min(hl.saddle_distance(m, S_sella, q).min() for q in members) if len(S_sella) else np.inf
                if d > FAR:
                    far.append(k)
                    if not same_up_to_symmetry(m, ref[t]["regions"][k]["pos"], sella_reps):
                        kept.append(k)
                        if arm == "ft_hint_1x8":
                            survivors.append((d, t, k))
            n0.append(len(mine)); n1.append(len(far)); n2.append(len(kept))
        n0, n1, n2 = map(np.array, (n0, n1, n2))
        L.append(f"  {arm:14s} {n0.mean():10.1f} {n1.mean():5.1f} {n2.mean():20.1f} | "
                 f"{(n0 > 0).sum():18d} {(n1 > 0).sum():4d} {(n2 > 0).sum():19d}")

    hit, tried = symmetry_sanity(minima, ref, sella_only)
    L += ["", f"  sanity: the symmetry test recognised {hit} of {tried} copies made with the minimum's own space-group "
          f"operations (one per minimum that has one moving the saddle more than {FAR} A)"]

    L += ["", "  ft_hint_1x8: the five survivors farthest from any Sella-only saddle "
          "(traj: the minimum, the survivor, the nearest Sella-only saddle)",
          "  minimum  formula        barrier | nearest Sella-only: dist  barrier | moving atoms (A)   traj"]
    tdir = out / "new_saddles"; tdir.mkdir(exist_ok=True)
    for d, t, k in sorted(survivors, reverse=True)[:5]:
        m, reg = minima[t], ref[t]["regions"][k]
        reps = sorted(sella_only[t])
        dist = hl.saddle_distance(m, np.array([ref[t]["regions"][j]["pos"] for j in reps]), reg["pos"])
        near = ref[t]["regions"][reps[int(dist.argmin())]]
        moves = ", ".join(f"{m[i].symbol}{i} {s:.2f}" for i, _, s in hl.make_hints(m, reg["pos"])[:3])
        path = tdir / f"survivor_{t}_{k}.traj"
        write(path, [structure(m, m.positions), structure(m, reg["pos"]), structure(m, near["pos"])])
        L.append(f"  {t:7d}  {m.get_chemical_formula():12s} {reg['barrier']:7.3f} | {d:22.2f} {near['barrier']:8.3f} | "
                 f"{moves:18s} {path.name}")

    text = "\n".join(L)
    print(text)
    (out / "new_saddles.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
