"""
Build a SMALL, exactly-C6-symmetric Li-on-graphene cell for fast iteration.

The stock `one_saddle.traj` uses a 112-C rectangular supercell (17.11 x 17.28 Å).
Rectangular supercells of graphene do NOT carry exact C6 symmetry about a hollow
site (the periodic images break it), and 113 atoms is far more than the physics
of a single Li hop needs.

This builds an n x n HEXAGONAL supercell (lattice vectors at 60°, so the point
symmetry about a hexagon centre is exactly C6 under PBC) with the SAME local
geometry as the original:

    a (graphene)      2.4692 Å        (C-C = 1.4256 Å)
    Li height (R)     1.7188 Å        above the C plane, at a hollow site
    Li height (S)     1.9758 Å        above the plane, at the bridge midpoint
    R -> S            a/2 = 1.2346 Å  in-plane
    R -> P            a   = 2.4692 Å  in-plane (adjacent hollow)
    C sheet           FixAtoms, vacuum 10 Å each side (cell z = 20 Å)

Outputs (per n): `small_n<N>_one_saddle.traj` (R,S,P) and
`small_n<N>_six_saddles.traj` (the full C6 orbit, 6 triplets), plus a
verification report: exact-C6 check, local-environment match, image distances.

    python make_small_cell.py --n 4 --n 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory

A_GRAPHENE = 2.4692      # Å, measured from the stock cell
Z_PLANE = 10.0           # C sheet height
Z_CELL = 20.0            # vacuum 10 Å either side
H_HOLLOW = 1.7188        # Li height at the hollow (reactant/product)
H_BRIDGE = 1.9758        # Li height at the bridge (saddle)
HERE = Path(__file__).resolve().parent


def build_sheet(n: int, a: float = A_GRAPHENE):
    """n x n hexagonal supercell of graphene; hexagon centre at the origin."""
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a * 0.5, a * math.sqrt(3) / 2, 0.0])
    basis_frac = [(1 / 3, 1 / 3), (2 / 3, 2 / 3)]     # honeycomb; hollow at (0,0)
    pos = []
    for i in range(n):
        for j in range(n):
            for (fx, fy) in basis_frac:
                p = (i + fx) * a1 + (j + fy) * a2
                pos.append([p[0], p[1], Z_PLANE])
    cell = np.array([n * a1, n * a2, [0.0, 0.0, Z_CELL]])
    return np.array(pos), cell


def mic(v, cell):
    frac = v @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def make_atoms(C, li, cell):
    at = Atoms(["C"] * len(C) + ["Li"], positions=np.vstack([C, li]), cell=cell, pbc=True)
    at.set_constraint(FixAtoms(indices=list(range(len(C)))))
    at.info.update(task_name="omat", charge=0, spin=0)
    return at


def verify(C, cell, li_r, n, a):
    """Exact-C6 check + local environment + image distances."""
    ok = True
    # 1. C6 about the Li site: rotate the sheet 60° and match under PBC.
    for k in (1, 2, 3):
        ang = math.radians(60.0 * k)
        rot = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1.0]])
        Cr = (C - li_r) @ rot.T + li_r
        d = np.linalg.norm(mic(Cr[:, None, :] - C[None, :, :], cell), axis=-1)
        worst = d.min(axis=1).max()
        print(f"    C6 rotation {60*k:3.0f}°: max mismatch to a lattice image = {worst:.2e} Å")
        ok &= worst < 1e-8
    # 2. Local environment: 6 nearest C to the hollow site (IN-PLANE, as measured
    #    on the stock cell; the 3D distance additionally carries the Li height).
    dv = mic(C - li_r, cell)
    dr = np.sort(np.linalg.norm(dv[:, :2], axis=1))[:8]
    d3 = np.sort(np.linalg.norm(dv, axis=1))[:8]
    print(f"    6 nearest C to Li_R (in-plane): {np.round(dr[:6], 4)}  (stock: 1.4244-1.4268)")
    print(f"    6 nearest C to Li_R (3D):       {np.round(d3[:6], 4)}  (stock: 2.2331)")
    ok &= abs(dr[:6].mean() - 1.4256) < 5e-3
    # 3. Periodic images of the Li.
    print(f"    Li-Li image distance: {n * a:.3f} Å   |   C atoms: {len(C)}   cell: {n*a:.2f} Å hexagonal")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[4, 5])
    args = p.parse_args()

    for n in args.n:
        print(f"\n=== n = {n} ===")
        C, cell = build_sheet(n)
        li_r = np.array([0.0, 0.0, Z_PLANE + H_HOLLOW])          # hollow (origin)
        # 6 hollow-hollow directions (adjacent hexagon centres, |d| = a)
        dirs = [np.array([math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k)), 0.0])
                for k in range(6)]

        if not verify(C, cell, li_r, n, A_GRAPHENE):
            print("    !! verification FAILED — not writing")
            continue

        # bridge check on direction 0
        li_s0 = li_r + dirs[0] * (A_GRAPHENE / 2) + np.array([0, 0, H_BRIDGE - H_HOLLOW])
        dsv = mic(C - li_s0, cell)
        ds = np.sort(np.linalg.norm(dsv[:, :2], axis=1))[:4]
        print(f"    2 nearest C to Li_S (in-plane): {np.round(ds[:2], 4)}  (stock: 0.7122/0.7134)")

        # --- one_saddle.traj: R, S, P along direction 0
        frames = []
        li_p0 = li_r + dirs[0] * A_GRAPHENE
        frames += [make_atoms(C, li_r, cell), make_atoms(C, li_s0, cell), make_atoms(C, li_p0, cell)]
        out1 = HERE / f"small_n{n}_one_saddle.traj"
        tr = Trajectory(str(out1), "w")
        for f in frames:
            tr.write(f)
        tr.close()

        # --- six_saddles.traj: the full C6 orbit
        out6 = HERE / f"small_n{n}_six_saddles.traj"
        tr = Trajectory(str(out6), "w")
        for u in dirs:
            li_s = li_r + u * (A_GRAPHENE / 2) + np.array([0, 0, H_BRIDGE - H_HOLLOW])
            li_p = li_r + u * A_GRAPHENE
            for f in (make_atoms(C, li_r, cell), make_atoms(C, li_s, cell), make_atoms(C, li_p, cell)):
                tr.write(f)
        tr.close()
        print(f"    wrote {out1.name} (3 frames) and {out6.name} (18 frames), {len(C)+1} atoms each")


if __name__ == "__main__":
    main()
