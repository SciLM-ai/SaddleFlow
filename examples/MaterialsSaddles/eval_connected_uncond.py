"""Does an unconditional SaddleFlow model reach connected first-order saddles from a dataset's minima?

    python examples/MaterialsSaddles/eval_connected_uncond.py --model fm1 --dataset <test dir> --n_minima 30 --n_evals 5

--model is fm1, fm2 or a release directory (config.json, model.safetensors). --dataset is a directory of .aselmdb shards
or a .traj of [R, S, P] triplets. Minima are the dataset's unique endpoints, picked at random with the seed.

Each minimum gets --n_evals noised starts, rolled out with the model's defaults (5 passes x 4 steps, 0.3 A noise). FM2's
flag marks every mobile atom. Then, for each SaddleFlow endpoint, on UMA-S-1.2:

    1. Hessian at the endpoint: its index k is the number of eigenvalues below -0.01 eV/A^2.
    2. Sella refines the endpoint at order k (order 1 for k = 0) to the refined point. --eta is the finite-difference
       step of Sella's Hessian-vector products: 5e-3 A, not Sella's 1e-4, which amplifies the noise of GPU forces.
    3. Hessian at the refined point: its index, the distance from the endpoint, and the barrier from the minimum.
    4. If the refined point has index 1 and a barrier of at most 1 eV: nudge it both ways along the negative mode and
       relax with FIRE. It is connected if at least one side comes within 0.2 A of the dataset minimum.

Success: index 1 at the endpoint, connected first-order saddle within 0.2 A, barrier at most 1 eV. Near success: the same
from index 0. Stayed at the minimum: the endpoint is within 0.2 A of the minimum. Anything else is a failure. Distance is
the largest per-atom move, rigid shift removed for cells with no fixed atoms.

Writes to <out-dir>:
    report.txt      the summary
    results.csv     one row per endpoint: the scalars (pandas.read_csv)
    results.json    the same rows with positions, forces, eigenvalues and each relaxation (one endpoint per line)
    minima.traj, endpoints.json, results.jsonl      what a rerun reuses
"""
import argparse
import csv
import glob
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.optimize import FIRE

from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, run_sella
from saddleflow.classical_saddle_search.pes_backend import make_uma_calculator
from saddleflow.data import MaterialsSaddlesDataset, TrajTripletDataset

REVISIONS = {"fm1": "main", "fm2": "fm2"}
UMA = "uma-s-1p2"

DUPLICATE = 0.05        # A: two minima closer than this are the same minimum
NEGATIVE = -0.01        # eV/A^2: eigenvalues below this count toward the index
SELLA_FMAX = 0.01       # eV/A
SELLA_STEPS = 300
NEAR = 0.2              # A: a saddle this close to the endpoint counts as reached
RETURNED = 0.2          # A: a relaxation this close to the minimum has returned to it
RELAX_FMAX = 0.005      # eV/A
RELAX_STEPS = 1000
MAX_BARRIER = 1.0       # eV: saddles above this do not count
STAYED = 0.2            # A: an endpoint this close to the minimum never left it

CALCULATORS = {}        # one UMA calculator per task, per worker

CSV_COLUMNS = ["minimum", "eval", "formula", "n_atoms", "task",
               "endpoint_index", "endpoint_fmax", "endpoint_to_minimum",
               "status", "sella_calls", "refined_index", "refined_fmax", "distance", "barrier", "connected",
               "outcome", "verdict", "error"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="fm1, fm2 or a release directory")
    p.add_argument("--dataset", required=True, help="directory of .aselmdb shards, or a .traj of triplets")
    p.add_argument("--n_minima", type=int, default=0, help="minima to evaluate, picked with the seed (0 = all)")
    p.add_argument("--n_evals", type=int, default=1, help="SaddleFlow rollouts per minimum")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eta", type=float, default=5e-3, help="finite-difference step (A) inside Sella")
    p.add_argument("--workers", type=int, default=8, help="processes running Hessians, Sella and relaxations")
    p.add_argument("--out-dir", help="default eval_connected/<model>_<dataset>_<n_minima>x<n_evals>_seed<seed>_eta<eta>")
    return p.parse_args()


# ---------------------------------------------------------------- minima
def read_dataset(path):
    if Path(path).is_dir():
        return MaterialsSaddlesDataset(sorted(glob.glob(f"{path}/*.aselmdb")))
    return TrajTripletDataset([path])


def record_to_atoms(record):
    atoms = Atoms(numbers=record["Z"].numpy(), positions=record["start_pos"].double().numpy(),
                  cell=record["cell"].double().numpy(), pbc=True)
    fixed = record["fixed"].numpy().astype(bool)
    if fixed.any():
        atoms.set_constraint(FixAtoms(mask=fixed))
    atoms.info.update(task_name=record["task_name"], charge=int(record["charge"]), spin=int(record["spin"]))
    return atoms


def same_structure(a, b):
    if not np.array_equal(a.numbers, b.numbers):
        return False
    return find_mic(a.positions - b.positions, a.cell, a.pbc)[1].max() < DUPLICATE


def pick_minima(dataset, n_minima, seed):
    """Unique minima in a seeded random order, reading only as many records as it takes."""
    seen = defaultdict(list)
    minima = []
    for i in np.random.default_rng(seed).permutation(len(dataset)):
        atoms = record_to_atoms(dataset[int(i)])
        key = (atoms.get_chemical_formula(), tuple(np.round(atoms.cell.cellpar(), 3)))
        if any(same_structure(atoms, other) for other in seen[key]):
            continue
        seen[key].append(atoms)
        minima.append(atoms)
        if len(minima) == n_minima:
            break
    return minima


# ---------------------------------------------------------------- SaddleFlow
def load_model(name):
    """fm1, fm2 or a release directory."""
    from saddleflow import SaddleFlow

    if name in REVISIONS:
        return SaddleFlow.from_pretrained(revision=REVISIONS[name])
    return SaddleFlow.from_pretrained(name)


def roll_out(args, minima):
    """{"minimum", "eval", "positions"} for every rollout."""
    import torch

    model = load_model(args.model)
    proposals = model.propose(minima, n=args.n_evals, seed=args.seed)
    del model
    torch.cuda.empty_cache()

    return [dict(minimum=i, eval=j, positions=atoms.positions.tolist())
            for i, draws in enumerate(proposals) for j, atoms in enumerate(draws)]


# ---------------------------------------------------------------- checks on one endpoint
def use_one_gpu():
    """Spread the workers over the visible GPUs (before CUDA starts in this process)."""
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    worker = multiprocessing.current_process()._identity[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[worker % len(gpus)]


def calculator(task):
    if task not in CALCULATORS:
        CALCULATORS[task] = make_uma_calculator(UMA, "cuda", task)[0]
    return CALCULATORS[task]


def distance(a, b):
    """Largest per-atom distance between two structures, rigid shift removed when no atom is fixed."""
    d = find_mic(a.positions - b.positions, a.cell, a.pbc)[0]
    if not a.constraints:
        d = d - d.mean(0)
    return float(np.linalg.norm(d, axis=1).max())


def hessian_index(pes, x):
    """(index, eigenvalues, eigenvectors) of the full Hessian over the free coordinates."""
    estimator = MinimumModeEstimator(backend="dense")
    values, vectors, _ = estimator.lowest_modes(pes, x, n_modes=pes.dofs.dimension)
    return int(np.sum(values < NEGATIVE)), values, vectors


def all_forces(pes, x):
    """Forces (N, 3), zero on fixed atoms."""
    forces = np.zeros((len(pes.atoms), 3))
    forces[pes.dofs.free] = pes.forces(x).reshape(-1, 3)
    return forces


def rounded(array):
    return np.round(array, 5).tolist()


def relax_side(start, minimum, calc):
    """Relax `start` until it comes within RETURNED of the minimum or converges.

    FIRE, not LBFGS: from next to a saddle, LBFGS often steps back onto the saddle instead of going downhill.
    """
    atoms = start.copy()
    atoms.calc = calc
    optimizer = FIRE(atoms, logfile=None)
    returned = False
    for _ in optimizer.irun(fmax=RELAX_FMAX, steps=RELAX_STEPS):
        if distance(minimum, atoms) < RETURNED:
            returned = True
            break
    return dict(returned=returned, steps=optimizer.nsteps, end_to_minimum=round(distance(minimum, atoms), 5))


def relax_sides(pes, x, eigenvalue, mode, minimum, calc):
    """Nudge the saddle each way along its negative mode and relax, stopping at the first side that returns."""
    size = float(np.clip(0.1 / abs(eigenvalue), 0.1, 0.3))       # force of about 0.1 eV/A, on the most-moving atom
    nudge = mode.reshape(-1, 3)
    nudge = (size * nudge / np.linalg.norm(nudge, axis=1).max()).ravel()

    sides = []
    for sign in (1, -1):
        start = pes.dofs.atoms_at(x + sign * nudge)
        sides.append(dict(relax_side(start, minimum, calc), sign=sign))
        if sides[-1]["returned"]:
            break
    return sides


def evaluate(job):
    """Run the checks on one SaddleFlow endpoint; returns one row of results."""
    minimum, positions = job["atoms"], np.array(job["positions"])
    row = dict(minimum=job["minimum"], eval=job["eval"], formula=minimum.get_chemical_formula(), n_atoms=len(minimum),
               task=minimum.info["task_name"], endpoint_positions=rounded(positions))
    try:
        calc = calculator(minimum.info["task_name"])
        endpoint = minimum.copy()
        endpoint.set_positions(positions, apply_constraint=False)
        pes = PESBackend(endpoint, calc)
        x = pes.dofs.pack(positions)

        # 1. the SaddleFlow endpoint
        energy_of_minimum = pes.energy(pes.dofs.pack(minimum.positions))
        row["endpoint_to_minimum"] = distance(minimum, endpoint)
        row["endpoint_fmax"] = pes.dofs.force_metrics(pes.forces(x))[1]
        row["endpoint_forces"] = rounded(all_forces(pes, x))
        row["endpoint_index"], values, _ = hessian_index(pes, x)
        row["endpoint_eigenvalues"] = rounded(values)

        # 2. refine at that order
        order = max(1, row["endpoint_index"])
        before = pes.force_calls
        result = run_sella(pes, x, fmax_tol=SELLA_FMAX, max_steps=SELLA_STEPS, max_force_calls=1_000_000,
                           delta0=0.1, order=order, eta=job["eta"])
        pes.max_force_calls = None
        row["status"] = result["status"]
        row["sella_calls"] = result["force_calls"] - before
        refined = result["final"]
        row["refined_positions"] = rounded(pes.dofs.expand(refined))
        row["refined_forces"] = rounded(all_forces(pes, refined))
        row["refined_fmax"] = pes.dofs.force_metrics(pes.forces(refined))[1]
        row["distance"] = distance(endpoint, pes.dofs.atoms_at(refined))
        if result["status"] != "converged" or row["refined_fmax"] > SELLA_FMAX:
            return row

        # 3. the refined point
        index, values, vectors = hessian_index(pes, refined)
        row["refined_index"] = index
        row["refined_eigenvalues"] = rounded(values)
        row["barrier"] = pes.energy(refined) - energy_of_minimum

        # 4. connected to the minimum
        if index == 1 and row["barrier"] <= MAX_BARRIER:
            row["relaxations"] = relax_sides(pes, refined, values[0], vectors[:, 0], minimum, calc)
            row["connected"] = any(side["returned"] for side in row["relaxations"])
    except Exception as exc:                # one bad structure must not lose the run
        row["error"] = str(exc)[:200]
    return row


# ---------------------------------------------------------------- report
def outcome(row):
    """What refinement found, in words."""
    start, end = row["endpoint_index"], row.get("refined_index")
    if row["endpoint_to_minimum"] < STAYED:
        return "stayed at the minimum"
    if end is None:
        return "Sella did not converge"
    near = row["distance"] <= NEAR

    if start >= 2:
        if end != start:
            return "refined to a different index"
        return f"same-order saddle within {NEAR} A" if near else f"same-order saddle more than {NEAR} A away"

    if end == 0:
        return "refined to a minimum"
    if end >= 2:
        return "refined to a higher-order saddle"
    if row["barrier"] > MAX_BARRIER:
        return f"first-order saddle above {MAX_BARRIER} eV"
    if not near:
        return f"first-order saddle more than {NEAR} A away"
    return f"first-order saddle within {NEAR} A, " + ("connected" if row["connected"] else "not connected")


def verdict(row):
    if row["endpoint_to_minimum"] < STAYED:
        return "stayed at the minimum"
    reached = row.get("refined_index") == 1 and row["distance"] <= NEAR and row.get("connected")
    if reached and row["endpoint_index"] == 1:
        return "success"
    if reached and row["endpoint_index"] == 0:
        return "near success"
    return "failure"


def report(args, minima, rows):
    n = len(rows)
    errors = [r for r in rows if "endpoint_index" not in r or "error" in r]
    rows = [r for r in rows if r not in errors]
    groups = {"0 (positive curvature)": [r for r in rows if r["endpoint_index"] == 0],
              "1": [r for r in rows if r["endpoint_index"] == 1],
              "2": [r for r in rows if r["endpoint_index"] == 2],
              ">= 3": [r for r in rows if r["endpoint_index"] >= 3]}

    lines = [f"Connected-saddle evaluation: {args.model} on {args.dataset}", "=" * 100, "",
             f"{len(minima)} minima x {args.n_evals} rollouts = {n} SaddleFlow endpoints; {len(errors)} errors", "",
             "HESSIAN INDEX AT THE SADDLEFLOW ENDPOINT",
             "  index                       count   share   fmax p50 (eV/A)"]
    for name, group in groups.items():
        if group:
            fmax = np.median([r["endpoint_fmax"] for r in group])
            lines.append(f"  {name:26s} {len(group):6d}  {100 * len(group) / n:5.1f}%  {fmax:8.3f}")

    lines += ["", "WHAT REFINEMENT FOUND", "  outcome                                                count   share   distance p50 / p90 (A)"]
    for title, group in (("index 1 at the endpoint (Sella order 1)", groups["1"]),
                         ("index 0 at the endpoint (Sella order 1)", groups["0 (positive curvature)"]),
                         ("index >= 2 at the endpoint (Sella at that order)", groups["2"] + groups[">= 3"])):
        if not group:
            continue
        lines.append(f"  {title}")
        for name in sorted({outcome(r) for r in group}):
            mine = [r for r in group if outcome(r) == name]
            d = [r["distance"] for r in mine if "distance" in r]
            where = f"{np.median(d):.3f} / {np.percentile(d, 90):.3f}" if d else "-"
            lines.append(f"    {name:50s} {len(mine):6d}  {100 * len(mine) / len(group):5.1f}%   {where}")

    lines += ["", "VERDICT"]
    for name in ("success", "near success", "stayed at the minimum", "failure"):
        count = sum(verdict(r) == name for r in rows)
        lines.append(f"  {name:22s} {count:6d}  {100 * count / n:5.1f}%")
    lines.append(f"  {'error':22s} {len(errors):6d}  {100 * len(errors) / n:5.1f}%")
    return "\n".join(lines) + "\n"


def save_results(out, rows):
    """results.json (everything, one endpoint per line) and results.csv (the scalars)."""
    rows = sorted(rows, key=lambda r: (r["minimum"], r["eval"]))
    for row in rows:
        failed = "endpoint_index" not in row or "error" in row
        row["outcome"] = "error" if failed else outcome(row)
        row["verdict"] = "error" if failed else verdict(row)

    (out / "results.json").write_text("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n")

    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    name = f"{Path(args.model).name}_{dataset.parent.name}_{dataset.stem}_{args.n_minima}x{args.n_evals}_seed{args.seed}_eta{args.eta:g}"
    out = Path(args.out_dir or f"eval_connected/{name}")
    out.mkdir(parents=True, exist_ok=True)

    # the minima
    minima_path = out / "minima.traj"
    if not minima_path.exists():
        write(minima_path, pick_minima(read_dataset(args.dataset), args.n_minima, args.seed))
    minima = read(minima_path, index=":")
    print(f"{len(minima)} minima", flush=True)

    # the SaddleFlow endpoints
    endpoints_path = out / "endpoints.json"
    if not endpoints_path.exists():
        endpoints_path.write_text(json.dumps(roll_out(args, minima)))
    endpoints = json.loads(endpoints_path.read_text())
    print(f"{len(endpoints)} SaddleFlow endpoints", flush=True)

    # the checks, skipping endpoints already done
    results_path = out / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    done = {(r["minimum"], r["eval"]) for r in rows}
    jobs = [dict(e, atoms=minima[e["minimum"]], eta=args.eta) for e in endpoints if (e["minimum"], e["eval"]) not in done]

    with multiprocessing.get_context("spawn").Pool(args.workers, initializer=use_one_gpu) as pool, results_path.open("a") as f:
        for row in pool.imap_unordered(evaluate, jobs):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  {len(rows)}/{len(endpoints)}", flush=True)

    save_results(out, rows)
    text = report(args, minima, rows)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
