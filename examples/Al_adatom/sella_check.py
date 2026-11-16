"""Are the Al endpoints that miss every dataset saddle real saddles, and are any of them saddles the dataset lacks?

    python examples/Al_adatom/sella_check.py --runs plain=<sample.py ball dir> conv_layer_separation=<dir> \\
        --n 300 --out <dir>

The same --n draws (picked with --seed from the runs' common draws, identical starts in every run since sample.py used
one seed) are checked for every run. For each endpoint, on UMA-S-1.2 (task oc20), as eval_connected_uncond.py does:

    1. Sella, first order, fmax 0.01 eV/A, at most 300 steps, eta 5e-3 A.
    2. If it converged: the three lowest Hessian eigenvalues by LOBPCG on finite-difference Hessian-vector products (a
       dense Hessian of 771 free coordinates is too slow); index = eigenvalues below -0.01 eV/A^2. The barrier from the
       minimum.
    3. If index 1: nudged both ways along the negative mode and relaxed with FIRE; connected if one side returns within
       0.2 A of the minimum.
    4. The saddle against the dataset's 353, three ways, within 0.2 A (largest per-atom distance, mobile atoms):
         same atoms    atom by atom, as score.py
         relabelled    as a set of positions (the Al atoms are identical: an exchange event can return the same
                       structure with atoms swapped)
         symmetric     as a set, after any in-plane symmetry operation about the adatom that maps the minimum onto
                       itself (checked, not assumed)
       and otherwise a new saddle. New saddles are then grouped (the same test, between each other) into distinct ones,
       per run and across runs.

Writes <out>/results.jsonl (one row per endpoint and run; rerunning skips rows already there), <out>/report.txt and
<out>/new_saddles.traj (one structure per distinct new, connected, first-order saddle, with the runs that found it).
"""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "MaterialsSaddles"))
from eval_connected_uncond import calculator, distance, relax_sides, use_one_gpu   # noqa: E402

from saddleflow.classical_saddle_search import PESBackend, run_sella                # noqa: E402
from saddleflow.classical_saddle_search.verify import lowest_modes_block            # noqa: E402

TASK = "oc20"
SELLA_FMAX, SELLA_STEPS, ETA = 0.01, 300, 5e-3
NEGATIVE = -0.01        # eV/A^2
SAME = 0.2              # A
SYMMETRY_TOL = 0.05     # A: an operation is a symmetry if it maps the minimum onto itself this closely


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="name=<sample.py --out directory>")
    p.add_argument("--n", type=int, default=300, help="draws checked, the same ones in every run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--traj", default=str(HERE / "al_triplets.traj"))
    p.add_argument("--out", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- structure comparison
class Geometry:
    """The dataset, the mobile atoms, and the in-plane symmetry operations about the adatom."""

    def __init__(self, traj):
        frames = read(traj, ":")
        self.minimum = frames[0]
        self.saddles = frames[1::3]
        self.mobile = np.ones(len(self.minimum), bool)
        self.mobile[self.minimum.constraints[0].index] = False
        self.cell = np.asarray(self.minimum.cell)
        self.inv = np.linalg.inv(self.cell)
        self.centre = self.minimum.positions[int(np.argmax(self.minimum.positions[:, 2]))]
        candidates = {"identity": [[1, 0], [0, 1]], "rotation 90": [[0, -1], [1, 0]], "rotation 180": [[-1, 0], [0, -1]],
                      "rotation 270": [[0, 1], [-1, 0]], "mirror x": [[-1, 0], [0, 1]], "mirror y": [[1, 0], [0, -1]],
                      "mirror diagonal": [[0, 1], [1, 0]], "mirror antidiagonal": [[0, -1], [-1, 0]]}
        m = self.minimum.positions[self.mobile]
        self.ops = {}
        for name, M in candidates.items():
            if self.set_distance(self.apply(m, np.array(M, float)), m) < SYMMETRY_TOL:
                self.ops[name] = np.array(M, float)
        self.ref = [S.positions[self.mobile] for S in self.saddles]
        self.ref_z = [np.sort(r[:, 2]) for r in self.ref]

    def apply(self, pos, M):
        out = pos.copy()
        out[:, :2] = self.centre[:2] + (pos[:, :2] - self.centre[:2]) @ M.T
        return out

    def mic(self, d):
        f = d @ self.inv
        return (f - np.round(f)) @ self.cell

    def indexed_distance(self, a, b):
        return float(np.linalg.norm(self.mic(a - b), axis=-1).max())

    def set_distance(self, a, b):
        """Largest distance when each atom of a is paired with its nearest atom of b; inf unless that pairing is one
        to one (atoms are ~2.8 A apart, so within SAME the nearest atom is the only candidate)."""
        d = np.linalg.norm(self.mic(a[:, None, :] - b[None, :, :]), axis=-1)
        j = d.argmin(1)
        if len(set(j.tolist())) != len(j):
            return float("inf")
        return float(d[np.arange(len(a)), j].max())

    def compare(self, pos, refs, refs_z):
        """(kind, index) of the first match of pos (mobile atoms) among refs: same atoms, relabelled, symmetric."""
        for k, r in enumerate(refs):
            if self.indexed_distance(pos, r) < SAME:
                return "same atoms", k
        z = np.sort(pos[:, 2])
        close = [k for k, rz in enumerate(refs_z) if np.abs(z - rz).max() < SAME]   # sorted z: a necessary condition
        for k in close:
            if self.set_distance(pos, refs[k]) < SAME:
                return "relabelled", k
        for name, M in self.ops.items():
            if name == "identity":
                continue
            moved = self.apply(pos, M)
            for k in close:
                if self.set_distance(moved, refs[k]) < SAME:
                    return f"symmetric ({name})", k
        return None, None


# ---------------------------------------------------------------- one endpoint
def check(job):
    minimum, positions = job["minimum"], np.array(job["positions"])
    row = dict(run=job["run"], draw=job["draw"], centre=job["centre"])
    try:
        calc = calculator(TASK)
        endpoint = minimum.copy()
        endpoint.set_positions(positions, apply_constraint=False)
        pes = PESBackend(endpoint, calc)
        x = pes.dofs.pack(positions)
        e_min = pes.energy(pes.dofs.pack(minimum.positions))
        row["endpoint_fmax"] = pes.dofs.force_metrics(pes.forces(x))[1]
        before = pes.force_calls
        result = run_sella(pes, x, fmax_tol=SELLA_FMAX, max_steps=SELLA_STEPS, max_force_calls=1_000_000,
                           delta0=0.1, order=1, eta=ETA)
        pes.max_force_calls = None
        refined = result["final"]
        row["status"] = result["status"]
        row["sella_calls"] = result["force_calls"] - before
        row["refined_fmax"] = pes.dofs.force_metrics(pes.forces(refined))[1]
        row["sella_distance"] = distance(endpoint, pes.dofs.atoms_at(refined))
        row["refined_positions"] = np.round(pes.dofs.expand(refined), 5).tolist()
        if result["status"] != "converged" or row["refined_fmax"] > SELLA_FMAX:
            return row
        values, vectors, calls = lowest_modes_block(pes, refined, k=3)
        row["eigenvalues"] = [float(v) for v in values]
        row["mode_calls"] = int(calls)
        row["index"] = int(np.sum(values < NEGATIVE))
        row["barrier"] = float(pes.energy(refined) - e_min)
        if row["index"] == 1:
            sides = relax_sides(pes, refined, values[0], vectors[:, 0], minimum, calc)
            row["connected"] = any(s["returned"] for s in sides)
            row["relaxations"] = sides
    except Exception as exc:                 # one bad structure must not lose the run
        row["error"] = str(exc)[:200]
    return row


# ---------------------------------------------------------------- report
def classify(geo, rows):
    for r in rows:
        r["match"] = r["match_saddle"] = None
        if r.get("index") == 1:
            pos = np.array(r["refined_positions"])[geo.mobile]
            kind, k = geo.compare(pos, geo.ref, geo.ref_z)
            r["match"], r["match_saddle"] = kind or "new", k
    new = [r for r in rows if r["match"] == "new"]
    groups, gpos, gz = [], [], []
    for r in new:                            # distinct new saddles, by the same test
        pos = np.array(r["refined_positions"])[geo.mobile]
        kind, k = geo.compare(pos, gpos, gz)
        if kind is None:
            groups.append([r])
            gpos.append(pos)
            gz.append(np.sort(pos[:, 2]))
        else:
            groups[k].append(r)
    for g, members in enumerate(groups):
        for r in members:
            r["new_group"] = g
    return groups


def report(args, geo, rows, groups):
    runs = [s.split("=", 1)[0] for s in args.runs]
    L = ["Sella from SaddleFlow endpoints on Al(100) + adatom (ball mode): real saddles, and saddles the dataset lacks",
         f"{args.n} draws per run, the same draws in every run (seed {args.seed}); UMA-S-1.2 {TASK}; Sella first order, "
         f"fmax {SELLA_FMAX}, {SELLA_STEPS} steps; index from the 3 lowest eigenvalues (LOBPCG)",
         f"symmetry operations about the adatom that map the minimum onto itself: {', '.join(geo.ops)}",
         "=" * 110, ""]
    kinds = ["same atoms", "relabelled"] + [f"symmetric ({n})" for n in geo.ops if n != "identity"] + ["new"]
    for run in runs:
        rs = [r for r in rows if r["run"] == run]
        conv = [r for r in rs if "index" in r]
        first = [r for r in conv if r["index"] == 1]
        L += [f"[{run}]  {len(rs)} endpoints",
              f"  Sella converged {len(conv)};  first-order (index 1) {len(first)};  higher-order "
              f"{sum(r['index'] >= 2 for r in conv)};  index 0 {sum(r['index'] == 0 for r in conv)};  "
              f"not converged {sum('index' not in r and 'error' not in r for r in rs)};"
              f"  errors {sum('error' in r for r in rs)}",
              "  first-order saddles, by match to the 353 dataset saddles:"]
        for kind in kinds:
            ks = [r for r in first if r["match"] == kind]
            if not ks:
                continue
            con = [r for r in ks if r.get("connected")]
            bar = [r["barrier"] for r in con]
            L.append(f"    {kind:30s} {len(ks):4d}   connected to the minimum {len(con):4d}"
                     + (f"   barrier p50 {np.median(bar):.2f} eV, range {min(bar):.2f}-{max(bar):.2f}" if bar else ""))
        mine = [g for g in groups if any(r["run"] == run for r in g)]
        mine_c = [g for g in mine if any(r["run"] == run and r.get("connected") for r in g)]
        L += [f"  distinct new first-order saddles: {len(mine)}, connected to the minimum {len(mine_c)}",
              f"  Sella's move from the endpoint (A), first-order saddles: p50 "
              f"{np.median([r['sella_distance'] for r in first]):.3f}" if first else "", ""]
    if len(runs) > 1:
        L.append("Distinct new connected first-order saddles found by:")
        conn = [g for g in groups if any(r.get("connected") for r in g)]
        for combo in sorted({tuple(sorted({r["run"] for r in g if r.get("connected")})) for g in conn}):
            L.append(f"  {' + '.join(combo):45s} {sum(tuple(sorted({r['run'] for r in g if r.get('connected')})) == combo for g in conn)}")
        L.append("")
        a, b = runs[0], runs[1]
        by = {(r["run"], r["draw"]): r for r in rows}
        draws = sorted({r["draw"] for r in rows})
        good = lambda r: r is not None and r.get("index") == 1 and r.get("connected")
        ga = [good(by.get((a, d))) for d in draws]
        gb = [good(by.get((b, d))) for d in draws]
        L.append(f"Paired by draw, a connected first-order saddle: both {sum(x and y for x, y in zip(ga, gb))}, only {a} "
                 f"{sum(x and not y for x, y in zip(ga, gb))}, only {b} {sum(y and not x for x, y in zip(ga, gb))}")
    return "\n".join(L) + "\n"


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geo = Geometry(args.traj)
    print(f"[check] symmetry operations kept: {list(geo.ops)}", flush=True)

    ends, centres = {}, None
    for spec in args.runs:
        name, path = spec.split("=", 1)
        es = read(Path(path) / "endpoints.traj", ":")
        ends[name] = {int(e.info["draw"]): e for e in es}
        c = {d: int(e.info["centre"]) for d, e in ends[name].items()}
        if centres is not None and c != centres:
            raise SystemExit(f"{name}: different ball centres from the first run: not the same starts")
        centres = c
    draws = sorted(np.random.default_rng(args.seed).choice(sorted(centres), args.n, replace=False).tolist())

    path = out / "results.jsonl"
    done = set()
    if path.exists():
        done = {(r["run"], r["draw"]) for r in map(json.loads, path.read_text().splitlines())}
    jobs = [dict(run=name, draw=d, centre=centres[d], minimum=geo.minimum, positions=ends[name][d].positions.tolist())
            for d in draws for name in ends if (name, d) not in done]
    print(f"[check] {len(jobs)} endpoints to check ({len(done)} already done)", flush=True)
    with multiprocessing.get_context("spawn").Pool(args.workers, initializer=use_one_gpu) as pool, path.open("a") as fh:
        for i, row in enumerate(pool.imap_unordered(check, jobs), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"  {i}/{len(jobs)}  {row['run']} draw {row['draw']}: {row.get('status')} index {row.get('index')} "
                  f"connected {row.get('connected')}", flush=True)

    rows = [r for r in map(json.loads, path.read_text().splitlines()) if r["draw"] in set(draws)]
    groups = classify(geo, rows)
    text = report(args, geo, rows, groups)
    (out / "report.txt").write_text(text)
    structures = []
    for g, members in enumerate(groups):
        con = [r for r in members if r.get("connected")]
        if not con:
            continue
        s = geo.minimum.copy()
        s.set_positions(np.array(con[0]["refined_positions"]), apply_constraint=False)
        s.info = dict(new_group=g, runs=",".join(sorted({r["run"] for r in con})), barrier=con[0]["barrier"],
                      draws=",".join(f"{r['run']}:{r['draw']}" for r in con))
        structures.append(s)
    if structures:
        write(out / "new_saddles.traj", structures)
    with (out / "classified.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k not in ("refined_positions", "relaxations")}) + "\n")
    print(text)


if __name__ == "__main__":
    main()
