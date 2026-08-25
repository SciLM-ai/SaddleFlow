"""Nearest-saddle oracle: run Sella from each candidate geometry and report where it lands.

Why this exists
---------------
Scoring a predicted transition state by RMSD to a *stored* label conflates two
different errors: how far the prediction is from a saddle, and which saddle the
label happens to be. Running a saddle optimiser from the prediction and measuring
how far it had to move separates them — and answers the question the model is
actually trained on ("is there a saddle near here?").

Sella, not Dimer. Measured on identical MP20Bat structures, the ASE Dimer walked
to a *different* saddle in ~8% of cases, which silently inflated every tail metric
(see CLAUDE.md, "Evaluation"). Sella: 44 vs 384 force calls, 100% vs 77% converged,
0% vs 8% wandering, 93% verified index-1. Started from an already-converged saddle
it takes 0 steps and moves 0.0000 A.

Input is an ASE trajectory of candidate geometries (see dump_predictions.py), each
frame carrying `info['tid']` and `info['src']`. Output is JSON, one record per frame:
converged flag, final fmax, force calls, rmsd/maxd moved, and the final positions.
With --check-index it also returns the lowest `nev` Hessian eigenvalues by
finite-difference Lanczos, so the result can be filtered to genuine index-1 saddles.

Requires the `eval` extra:  CC=gcc CXX=g++ pip install 'saddleflow[eval]'
(the default NVIDIA `nvc` rejects Sella's -fno-strict-overflow).

Shard over N GPUs with --shard/--nshards; each shard writes its own JSON.
"""
import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from ase import Atoms
from ase.io import Trajectory
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from scipy.sparse.linalg import LinearOperator, eigsh
from sella import Sella


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="ASE trajectory of candidate geometries.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--fmax", type=float, default=0.01, help="Convergence threshold (eV/A).")
    p.add_argument("--steps", type=int, default=300, help="Max Sella steps.")
    p.add_argument("--delta0", type=float, default=0.1, help="Initial trust radius (A).")
    p.add_argument("--model", default="uma-s-1p2", help="fairchem predict unit.")
    p.add_argument("--task-name", default="omat", help="UMA MoE routing task.")
    p.add_argument("--nev", type=int, default=4, help="Eigenvalues for --check-index.")
    p.add_argument("--check-index", action="store_true",
                   help="Finite-difference Lanczos check that the result is a true "
                        "first-order saddle. Adds ~2 force calls per Lanczos matvec.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def mic(x, y, cell):
    """Minimum-image displacement x - y under periodic boundary conditions."""
    d = x - y
    frac = np.linalg.solve(np.array(cell).T, d.T).T
    frac -= np.round(frac)
    return frac @ np.array(cell)


def hessian_index(atoms, nev, eps=2e-3, tol=1e-2):
    """Lowest `nev` Hessian eigenvalues via finite-difference Lanczos.

    Never forms the Hessian: eigsh only needs H@v, and H@v is one central
    difference of forces along v (2 force calls per matvec).
    """
    n = len(atoms)
    x_center = atoms.get_positions().copy()

    def hv(v):
        v = v.reshape(n, 3)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return np.zeros(3 * n)
        u = v / norm
        atoms.set_positions(x_center + eps * u)
        f_plus = atoms.get_forces()
        atoms.set_positions(x_center - eps * u)
        f_minus = atoms.get_forces()
        atoms.set_positions(x_center)
        return (-(f_plus - f_minus) / (2 * eps) * norm).ravel()

    op = LinearOperator((3 * n, 3 * n), matvec=hv, dtype=float)
    w = eigsh(op, k=nev, which="SA", return_eigenvectors=False, maxiter=300, tol=1e-3)
    eigs = sorted(float(x) for x in w)
    return eigs, int(sum(1 for x in eigs if x < -tol))


def main():
    args = parse_args()
    predict_unit = pretrained_mlip.get_predict_unit(args.model, device="cuda")

    frames = list(Trajectory(args.src))[args.shard::args.nshards]
    rows = []
    for i, frame in enumerate(frames):
        atoms = Atoms(positions=frame.get_positions(), numbers=frame.get_atomic_numbers(),
                      cell=frame.get_cell(), pbc=True)
        atoms.calc = FAIRChemCalculator(predict_unit, task_name=args.task_name)
        x_start = atoms.get_positions().copy()
        record = {"tid": int(frame.info.get("tid", -1)), "src": str(frame.info.get("src", "?"))}
        try:
            dyn = Sella(atoms, order=1, delta0=args.delta0, internal=False, logfile=None)
            dyn.run(fmax=args.fmax, steps=args.steps)
            forces = atoms.get_forces()
            fmax = float(np.linalg.norm(forces, axis=1).max())
            moved = mic(atoms.get_positions(), x_start, frame.get_cell())
            eigs, nneg = None, None
            if args.check_index:
                try:
                    eigs, nneg = hessian_index(atoms, args.nev)
                except Exception:
                    pass
            record.update(conv=bool(fmax < args.fmax), fmax=fmax,
                          nfc=int(dyn.get_number_of_steps()),
                          rmsd=float(np.sqrt((moved ** 2).sum(1).mean())),
                          maxd=float(np.sqrt((moved ** 2).sum(1)).max()),
                          eigs=eigs, nneg=nneg, pos=atoms.get_positions().tolist())
        except Exception as exc:  # a single bad system must not lose the shard
            record.update(conv=False, fmax=float("nan"), nfc=-1, rmsd=float("nan"),
                          maxd=float("nan"), err=str(exc)[:200])
        rows.append(record)
        # Rewrite each iteration so a killed shard still yields partial results.
        with open(args.out, "w") as fh:
            json.dump(rows, fh)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(frames)}", flush=True)
    print(f"wrote {args.out} ({len(rows)} records)")


if __name__ == "__main__":
    main()
