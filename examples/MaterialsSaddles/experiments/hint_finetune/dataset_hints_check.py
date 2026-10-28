"""Before the dataset-hint fine-tunes (arms A and B): three CPU checks on mp20bat and the Tier B labels.

    copies    symmetry copies of the stored saddles: for a seeded sample of (minimum, saddle) pairs from the mp20bat
              train triplets (reactant and product sides, barrier 0.05-1 eV), apply every operation of the minimum's
              space group (spglib on the minimum, translations of a supercell included) to the minimum-to-saddle
              displacement, map the atoms back to the minimum's order, and count the distinct copies (0.1 A)
    tierb     Tier B regions per minimum, and per minimum the classes left after merging regions that are symmetry
              copies of one another (a copy within 0.15 A)
    leak      Tier B minima in the dataset: per minimum, the stored saddles of every train triplet whose reactant or
              product is that minimum (same atoms, within 0.1 A); how many Tier B regions they match (within 0.15 A,
              directly or as a symmetry copy), and how many held-out regions of the steering test (barrier >= 0.05 eV;
              the clean subset as in eval_hints.py) the pilot's replay half could have seen

    python examples/MaterialsSaddles/experiments/hint_finetune/dataset_hints_check.py
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.db import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hint_labels as hl  # noqa: E402

TRAIN_SHARDS = "/scratch/gqs7zr/saddleflow/MaterialsSaddles/mp20bat/train/mp20bat_train_*.aselmdb"
LABELS = "/scratch/gqs7zr/saddleflow/hint_finetune/tierb/labels.json"
SAME, REGION, LOW, CUTOFF, TILT = 0.1, 0.15, 0.05, 1.0, 30.0


def triplets():
    """[(element set, numbers, cell, R, S, P, eR, eS, eP)] in file order (Tier B's triplet index)."""
    out, trip = [], []
    for f in sorted(glob.glob(TRAIN_SHARDS)):
        with connect(f) as db:
            for row in db.select():
                trip.append(row)
                if len(trip) == 3:
                    R, S, P = (r.toatoms() for r in trip)
                    out.append((tuple(sorted(set(R.get_chemical_symbols()))), R.numbers, R.cell.array, R.positions,
                                S.positions, P.positions, trip[0].energy, trip[1].energy, trip[2].energy))
                    trip = []
    return out


def operations(m, symprec):
    """[(permutation, rotation for row-vector displacements)] of the minimum's symmetry operations."""
    cell, frac, Z = m.cell.array, m.get_scaled_positions(), m.numbers
    sym = spglib.get_symmetry((cell, frac, Z), symprec=symprec)
    if sym is None:
        return []
    tol, ops = 2 * symprec + 0.02, []
    for R, t in zip(sym["rotations"], sym["translations"]):
        d = (frac @ R.T + t)[:, None, :] - frac[None, :, :]
        d -= np.round(d)
        dc = np.linalg.norm(d @ cell, axis=-1)
        dc[Z[:, None] != Z[None, :]] = np.inf
        perm = dc.argmin(1)
        if dc[np.arange(len(Z)), perm].max() > tol or len(set(perm.tolist())) < len(Z):
            continue
        ops.append((perm, np.linalg.inv(cell) @ R.T @ cell))
    return ops


def copies(m, ops, saddle):
    """Distinct symmetry copies of the saddle, as minimum-to-saddle displacements (N, 3), overall shift removed."""
    D, out = hl.displacement(m, saddle), []
    for perm, rot in ops:
        C = np.zeros_like(D); C[perm] = D @ rot
        C -= C.mean(0)
        if not out or np.linalg.norm(np.array(out) - C[None], axis=-1).max(-1).min() >= SAME:
            out.append(C)
    return out or [D]


def close(m, copies_k, saddle, radius):
    D = hl.displacement(m, saddle)
    return np.linalg.norm(np.array(copies_k) - D[None], axis=-1).max(-1).min() < radius


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-pairs", type=int, default=4000)
    p.add_argument("--symprec", type=float, nargs="*", default=[0.01, 0.1])
    a = p.parse_args()
    T = triplets()
    print(f"mp20bat train: {len(T)} triplets")

    # copies per stored saddle
    pairs = []
    for i, (es, Z, cell, R, S, P, eR, eS, eP) in enumerate(T):
        for side, E, e in (("R", R, eR), ("P", P, eP)):
            if LOW <= eS - e <= CUTOFF:
                pairs.append((i, side))
    pick = np.random.default_rng(0).choice(len(pairs), min(a.n_pairs, len(pairs)), replace=False)
    print(f"\nCOPIES: {len(pairs)} (minimum, saddle) pairs with a barrier of {LOW}-{CUTOFF} eV; {len(pick)} sampled")
    print("  symprec  operations p50 mean | copies per saddle mean p50 | share with >= 2   >= 4   >= 8")
    for sp in a.symprec:
        n_ops, n_cop, cache = [], [], {}
        for j in pick:
            i, side = pairs[j]
            es, Z, cell, R, S, P, *_ = T[i]
            m = Atoms(numbers=Z, positions=R if side == "R" else P, cell=cell, pbc=True)
            if (i, side) not in cache:
                cache[(i, side)] = operations(m, sp)
            ops = cache[(i, side)]
            n_ops.append(len(ops)); n_cop.append(len(copies(m, ops, S)))
        n_ops, n_cop = np.array(n_ops), np.array(n_cop)
        print(f"  {sp:7.2f} {np.median(n_ops):14.0f} {n_ops.mean():5.1f} | {n_cop.mean():20.2f} {np.median(n_cop):4.0f} | "
              f"{100 * np.mean(n_cop >= 2):14.0f}% {100 * np.mean(n_cop >= 4):5.0f}% {100 * np.mean(n_cop >= 8):5.0f}%")

    # Tier B up to symmetry, and the leak
    minima = json.loads(Path(LABELS).read_text())["minima"]
    by_set = {}
    for i, t in enumerate(T):
        by_set.setdefault(t[0], []).append(i)
    cos = np.cos(np.radians(TILT))
    for sp in a.symprec:
        regs, classes, regs_hi, classes_hi = [], [], [], []
        stored_n, matched, matched_sym, n_reg = [], 0, 0, 0
        held = dict(all=[0, 0, 0], clean=[0, 0, 0])      # regions, matched directly, matched as a copy
        for e in minima:
            m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
            ops = operations(m, sp)
            R_ = e["regions"]
            cop = [copies(m, ops, np.array(r["saddle"])) for r in R_]
            for keep, (rl, cl) in ((lambda r: True, (regs, classes)), (lambda r: r["barrier"] >= LOW, (regs_hi, classes_hi))):
                idx = [k for k, r in enumerate(R_) if keep(r)]
                rep = []
                for k in idx:
                    if not any(close(m, cop[j], np.array(R_[k]["saddle"]), REGION) for j in rep):
                        rep.append(k)
                rl.append(len(idx)); cl.append(len(rep))
            # stored saddles of the dataset triplets touching this minimum
            es = tuple(sorted(set(m.get_chemical_symbols())))
            stored = []
            for i in by_set.get(es, []):
                _, Z, cell, R, S, P, eR, eS, eP = T[i]
                if len(Z) != len(m) or np.any(Z != m.numbers):
                    continue
                for E, eE in ((R, eR), (P, eP)):
                    if hl.saddle_distance(m, E[None], m.positions)[0] < SAME and eS - eE <= CUTOFF:
                        stored.append(S)
            stored_n.append(len(stored))
            scop = [copies(m, ops, s) for s in stored]
            train = [r for r in R_ if not r["held_out"]]
            for r in R_:
                S_r = np.array(r["saddle"])
                direct = any(hl.saddle_distance(m, s[None], S_r)[0] < REGION for s in stored)
                sym = direct or any(close(m, c, S_r, REGION) for c in scop)
                n_reg += 1; matched += direct; matched_sym += sym
                if r["held_out"] and r["barrier"] >= LOW:
                    clean = not any(any(g["atom"] == h["atom"] and np.dot(g["direction"], h["direction"]) >= cos
                                        for g in t["hints"] for h in r["hints"])
                                    and hl.saddle_distance(m, np.array([t["saddle"]]), S_r)[0] > 0.3 for t in train)
                    for sub in ("all", "clean") if clean else ("all",):
                        held[sub][0] += 1; held[sub][1] += direct; held[sub][2] += sym
        print(f"\nTIER B (symprec {sp}): {len(minima)} minima")
        print(f"  regions per minimum {np.mean(regs):.1f}, distinct up to symmetry {np.mean(classes):.1f}; "
              f"barrier >= {LOW} eV: {np.mean(regs_hi):.1f} -> {np.mean(classes_hi):.1f}")
        print(f"  LEAK: stored dataset saddles <= {CUTOFF} eV per Tier B minimum: mean {np.mean(stored_n):.1f}, "
              f"p50 {np.median(stored_n):.0f}; Tier B regions they match: {matched} of {n_reg} directly, "
              f"{matched_sym} counting symmetry copies")
        for sub in ("all", "clean"):
            n, d, s = held[sub]
            print(f"  held-out steering regions ({sub}): {n}; a stored saddle matches {d} directly, {s} counting copies")


if __name__ == "__main__":
    main()
