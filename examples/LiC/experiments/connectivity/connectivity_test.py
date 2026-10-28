"""Check by relaxation whether the never-reached (minimum, saddle) pairs are really connected.

The coverage study calls a saddle "connected" to a minimum only because a stored [R, S, P] triplet says so.
This test checks that downhill: for every pair that no start reached (at the largest draw count, in every
archive given), it finds the saddle's negative curvature mode with UMA, pushes the saddle 0.05 A each way
along it, relaxes both copies to a minimum, and matches the Li position to the study's minima. The pair is
confirmed if one of the two downhill ends is that minimum.

Controls: the same test on --n-controls ordinary saddles (chosen at random with --seed among saddles reached
from both stored ends in every archive). Each control should come back with its two stored ends; if the
controls fail too, the test is at fault rather than the catalog.

    R=/scratch/gqs7zr/saddleflow/lic_source_study_seed42/coverage_curve
    python examples/LiC/experiments/connectivity/connectivity_test.py \
        $R/sigma0.15/source_study.npz $R/sigma0.30/source_study.npz --out-dir $R/connectivity

--controls picks the control saddles by hand instead; --controls all checks every catalog saddle.
summarize_connectivity.py groups and averages the result. Writes connectivity.txt and connectivity.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source_study"))
from coverage_curve import coverage, load  # noqa: E402

from saddleflow.utils.relaxation import relax, uma_calculator  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archives", nargs="+", help="source_study.npz files from the coverage runs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-controls", type=int, default=5, help="Ordinary saddles checked as controls")
    p.add_argument("--controls", default="", help="Control saddle indices, comma-separated, or 'all' (overrides --n-controls)")
    p.add_argument("--seed", type=int, default=0, help="Seeds the choice of control saddles")
    p.add_argument("--push", type=float, default=0.05, help="A along the negative mode, each way")
    p.add_argument("--fmax", type=float, default=0.005, help="Relaxation threshold, eV/A (the catalog's)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--hessian-step", type=float, default=0.01, help="A, central finite differences")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--task-name", default="omat")
    p.add_argument("--device", default=None)
    return p.parse_args()


def missed_pairs(archives):
    """Pairs no start reached at the largest draw count: (in every archive, in any archive)."""
    sets = []
    for path in archives:
        run = load(path, "n12_uncond")
        sets.append(set(coverage(run, run["max_draws"])["missed"]))
    return sorted(set.intersection(*sets)), set.union(*sets)


def lowest_mode(atoms, free, step):
    """Lowest Hessian eigenpair over the free atoms, by central differences of the forces."""
    def forces(x):
        a = atoms.copy(); a.calc = atoms.calc; a.positions[free] = x
        return a.get_forces()[free].ravel()
    x0, n = atoms.positions[free].copy(), 3 * len(free)
    hessian = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = step
        hessian[i] = -(forces(x0 + e.reshape(-1, 3)) - forces(x0 - e.reshape(-1, 3))) / (2 * step)
    w, v = np.linalg.eigh((hessian + hessian.T) / 2)
    return w, v[:, 0].reshape(-1, 3)


def main():
    args = parse_args()
    data = np.load(args.archives[0], allow_pickle=True)
    config = json.loads(str(data["config_json"]))
    minima, edges, li = data["minima"], data["catalog_edges"], int(data["li_index"])
    tol = float(config["minimum_tol"])
    frames = read(Path(config["study_dir"]) / "split/catalog.traj", index=":")
    pairs, ever_missed = missed_pairs(args.archives)
    missed_saddles = {s for _, s in pairs}
    if args.controls == "all":
        controls = [s for s in range(len(edges)) if s not in missed_saddles]
    elif args.controls:
        controls = sorted(int(s) for s in args.controls.split(","))
    else:
        ordinary = [s for s in range(len(edges)) if s not in {s for _, s in ever_missed}]
        controls = sorted(np.random.default_rng(args.seed).choice(ordinary, args.n_controls, replace=False).tolist())
    saddles = sorted(missed_saddles) + controls
    calc = uma_calculator(args.model, task_name=args.task_name, device=args.device)

    results = {}
    for s in saddles:
        saddle = frames[3 * s + 1].copy()
        saddle.calc = calc
        frozen = [i for c in saddle.constraints for i in c.get_indices()]
        free = np.setdiff1d(np.arange(len(saddle)), frozen)
        eigenvalues, mode = lowest_mode(saddle, free, args.hessian_step)
        mode /= np.linalg.norm(mode)
        ends = []
        for sign in (+1, -1):
            start = saddle.copy()
            start.positions[free] += sign * args.push * mode
            r = relax(start, calc, fmax=args.fmax, steps=args.steps)
            distance = find_mic(minima - r.atoms.positions[li], saddle.cell, saddle.pbc)[1]
            k = int(np.argmin(distance))
            ends.append(dict(minimum=k if distance[k] < tol else -1, nearest=k, distance=float(distance[k]),
                             converged=r.converged, fmax=r.fmax, steps=r.steps, energy=r.energy))
        energy = float(saddle.get_potential_energy())
        results[s] = dict(stored=sorted(map(int, edges[s])), eigenvalues=eigenvalues.tolist(), energy=energy,
                          saddle_fmax=float(np.linalg.norm(saddle.get_forces(), axis=1).max()), ends=ends)
        print(f"saddle {s}: stored {results[s]['stored']}, downhill {[e['minimum'] for e in ends]}", flush=True)

    lines = ["Downhill connectivity of never-reached catalog pairs (n12_uncond coverage, all archives)", "=" * 100,
             f"{args.model}/{args.task_name}; push {args.push} A each way along the lowest mode; FIRE to "
             f"fmax {args.fmax}; an end matches a minimum within {tol} A (-1 = none of the {len(minima)}).", "",
             "saddle  stored ends   downhill ends   end distances (A)   barriers from each end (eV)   eigenvalues (eV/A^2)"]
    for s, r in results.items():
        if controls and s == controls[0]:
            lines.append("controls (ordinary saddles):")
        found = [e["minimum"] for e in r["ends"]]
        barriers = [r["energy"] - e["energy"] for e in r["ends"]]
        flag = "" if all(e["converged"] for e in r["ends"]) else "  (a relaxation did not converge)"
        distances = ", ".join("%.3f" % e["distance"] for e in r["ends"])
        barrier_text = ", ".join("%.3f" % b for b in barriers)
        lines.append(f"{s:6d}  {str(r['stored']):12s}  {str(found):14s}  {distances:18s}  {barrier_text:28s}  "
                     f"{np.round(r['eigenvalues'], 3)}{flag}")
    lines += ["", "missed pair (minimum, saddle) -> verdict"]
    for m, s in pairs:
        verdict = "CONFIRMED connected" if m in [e["minimum"] for e in results[s]["ends"]] else "NOT downhill-connected"
        lines.append(f"  ({m:2d}, {s:3d})  {verdict}")
    lines += ["", "control saddle -> both stored ends found downhill?"]
    for s in controls:
        ends = sorted(e["minimum"] for e in results[s]["ends"])
        lines.append(f"  {s:3d}  stored {results[s]['stored']}  downhill {ends}  "
                     f"{'yes' if ends == results[s]['stored'] else 'NO'}")
    text = "\n".join(lines) + "\n"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "connectivity.txt").write_text(text)
    (out / "connectivity.json").write_text(json.dumps(dict(pairs=pairs, ever_missed=sorted(ever_missed), controls=controls, saddles=results, args=vars(args)), indent=1))
    print(text)


if __name__ == "__main__":
    main()
