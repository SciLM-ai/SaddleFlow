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
converged flag, final fmax, force calls, rmsd/maxd moved, and the final positions, plus
`fmax_start`, the force at the candidate before Sella moves it (one extra call, not in `nfc`).
With --check-index it also returns the lowest `nev` Hessian eigenvalues by
finite-difference Lanczos, so the result can be filtered to genuine index-1 saddles.
The check is skipped when Sella did not converge (the case is a failure either way).

Two optional force-call caps (off by default, which is the paper protocol) bound the cost of a
single hard case: --max-force-calls stops Sella (status "query_budget", counted as not converged)
and --index-max-force-calls stops the index check (nneg None, index_status "budget").

Requires the `eval` extra:  CC=gcc CXX=g++ pip install 'saddleflow[eval]'
(the default NVIDIA `nvc` rejects Sella's -fno-strict-overflow).

Shard over N GPUs with --shard/--nshards; each shard writes its own JSON.
"""
import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from ase.io import Trajectory
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from scipy.sparse.linalg import ArpackNoConvergence

from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, QueryBudgetExceeded, run_sella


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
    p.add_argument("--max-force-calls", type=int, default=None,
                   help="Cap on Sella's force calls per case (default or 0: none, only --steps).")
    p.add_argument("--index-max-force-calls", type=int, default=None,
                   help="Cap on the index check's force calls per case (default or 0: none).")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--skip-done", nargs="*", default=[],
                   help="Output JSONs of an earlier run; frames whose tid they already hold are skipped "
                        "(finishes a run that hit its time limit).")
    return p.parse_args()


def mic(x, y, cell, pbc):
    """Minimum-image displacement x - y under periodic boundary conditions."""
    from ase.geometry import find_mic
    d = x - y
    return find_mic(d, cell, pbc)[0] if np.any(pbc) else d


def hessian_index(atoms, nev, eps=2e-3, tol=1e-2):
    """Lowest `nev` FREE-space Hessian eigenvalues.

    Small spaces use dense central differences. Larger spaces use Lanczos:
    H@v is one central difference of free forces (2 PES queries per matvec).
    """
    pes = PESBackend(atoms, atoms.calc)
    estimator = MinimumModeEstimator(eps=eps, lanczos_maxiter=300)
    w, _, _ = estimator.lowest_modes(pes, pes.dofs.pack(atoms.positions), n_modes=nev)
    eigs = [float(x) for x in w]
    return eigs, int(sum(x < -tol for x in eigs))


def main():
    args = parse_args()
    predict_unit = pretrained_mlip.get_predict_unit(args.model, device="cuda")

    # Shard first, then skip: a frame's shard depends only on its position in --src, so array tasks
    # that start at different times (and see different --skip-done files) never split the work
    # differently, which would run some frames twice and leave others out.
    done = {r["tid"] for path in args.skip_done for r in json.load(open(path))}
    frames = list(Trajectory(args.src))[args.shard::args.nshards]
    frames = [f for f in frames if int(f.info.get("tid", -1)) not in done]
    rows = []
    for i, frame in enumerate(frames):
        # ASE copy preserves FixAtoms, actual PBC and charge/spin metadata.
        atoms = frame.copy()
        calculator = FAIRChemCalculator(predict_unit, task_name=args.task_name)
        pes = PESBackend(atoms, calculator)
        x_start = atoms.positions.copy()
        record = {"tid": int(frame.info.get("tid", -1)), "src": str(frame.info.get("src", "?")),
                  "fmax_start": float("nan")}
        try:
            # The force AT the candidate, before Sella moves it -- what a one-call filter
            # would see. A separate calculator keeps this call out of the search's nfc count.
            start = atoms.copy()
            start.calc = FAIRChemCalculator(predict_unit, task_name=args.task_name)
            record["fmax_start"] = float(np.linalg.norm(start.get_forces(), axis=1).max())
            result = run_sella(pes, pes.dofs.pack(x_start), fmax_tol=args.fmax,
                               max_steps=args.steps, max_force_calls=args.max_force_calls or 1_000_000,
                               delta0=args.delta0)
            # Verification is distinct from search cost, and uses FREE forces.
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(result["final"]))
            moved = mic(result["positions"], x_start, frame.cell, frame.pbc)
            eigs, nneg, index_status = None, None, "skipped"
            verification_calls = 0
            if args.check_index and fmax < args.fmax:
                before = pes.force_calls
                if args.index_max_force_calls:
                    pes.max_force_calls = before + args.index_max_force_calls
                try:
                    estimator = MinimumModeEstimator(lanczos_maxiter=300)
                    values, _, _ = estimator.lowest_modes(pes, result["final"], n_modes=args.nev)
                    eigs = values.tolist()
                    nneg = int(np.sum(values < -0.01))
                    index_status = "ok"
                except QueryBudgetExceeded:
                    index_status = "budget"
                except ArpackNoConvergence:
                    index_status = "no_convergence"
                pes.max_force_calls = None
                verification_calls = pes.force_calls - before
            record.update(conv=bool(fmax < args.fmax), fmax=fmax, index_status=index_status,
                          nfc=result["force_calls"], nsteps=result["steps"],
                          verification_nfc=verification_calls, status=result["status"],
                          free_atoms=pes.dofs.n_free,
                          rmsd=float(np.sqrt((moved ** 2).sum(1).mean())),
                          maxd=float(np.sqrt((moved ** 2).sum(1)).max()),
                          eigs=eigs, nneg=nneg, pos=result["positions"].tolist())
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
