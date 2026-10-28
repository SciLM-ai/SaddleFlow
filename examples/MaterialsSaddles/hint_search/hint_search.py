"""Classical hint search on mp20bat minima, and the released model scored against it.

For each selected minimum (from a failure-mode run directory), the search pushes an epicenter atom along
32 directions and runs Sella. Epicenters are one atom per symmetry class (spglib) among the mobile ions;
on the first few minima every symmetry class is also searched, as a check. The released model is scored on
the same minima: 128 draws, plus the same draws inverted (r -> -r) and mapped back, each refined by Sella.
Every converged point is merged into distinct saddles per minimum, and each distinct saddle is checked once:
index 1 and connected (one side of a +-0.05 A push along the negative mode relaxes, FIRE to 0.005 eV/A, back to the minimum).

    select    choose the minima, epicenters and directions; write plan.json (CPU)
    model     released-model rollouts, normal and inverted (GPU, one job)
    search    Sella from every search start and every model landing point (GPU; --shard/--nshards)
    verify    merge into distinct saddles; index, connectivity, and splitting of higher-order points (GPU; sharded)
    analyze   coverage, discovery curves, Good-Turing, cost, repeatability, symmetry counts (CPU)

    python examples/MaterialsSaddles/hint_search/hint_search.py <mode> --run-dir <failure-mode run> --out-dir <dir>

Search arms (one start per minimum x epicenter x direction): nudge0.3 and nudge0.6 move the epicenter that
far along the direction; climb runs climb_along it, then Sella. Atoms within 5 A of the epicenter get a fixed
0.02 A jitter (0.05 A strains stiff oxides to ~3 eV/A), seeded per (minimum, epicenter, direction) so every arm and every repeat sees the same start.
Sella: UMA-S-1.2 omat, fmax 0.01, <= 300 steps, delta0 0.1, <= 600 force calls; the index check <= 300.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "failure_mode_analysis"))
from analyze_failure_modes import category, load, max_mic_untranslated  # noqa: E402

MOBILE = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Zn"}
SEARCH_ARMS = ("nudge0.3", "nudge0.6", "climb")
MODEL_ARMS = ("model", "model_inv")
FMAX, SELLA_STEPS, SELLA_CALLS, INDEX_CALLS = 0.01, 300, 600, 300
MERGE, RETURN = 0.1, 0.2          # A: same saddle; a relaxation "returns" to the minimum


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["select", "model", "search", "verify", "analyze"])
    p.add_argument("--run-dir", required=True, help="failure-mode run (make_proposals.py + sella_eval.py)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-random", type=int, default=10, help="random minima")
    p.add_argument("--n-higher-order", type=int, default=3, help="minima behind higher-order landings")
    p.add_argument("--max-atoms", type=int, default=80)
    p.add_argument("--limit", type=int, default=0, help="use only the first N selected minima (cost check)")
    p.add_argument("--all-atom-minima", type=int, default=3, help="minima that also search every symmetry class")
    p.add_argument("--max-epicenters", type=int, default=8)
    p.add_argument("--directions", type=int, default=32)
    p.add_argument("--draws", type=int, default=128)
    p.add_argument("--repeats", type=int, default=40, help="repeated identical starts per search arm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------- shared helpers
def at(atoms, positions):
    b = atoms.copy(); b.positions = np.asarray(positions, float)
    return b


def same(atoms, a_pos, b_pos, tol=MERGE):
    return max_mic_untranslated(at(atoms, a_pos), at(atoms, b_pos)) < tol


def spg_cell(atoms):
    return atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers


def fibonacci(n, rng):
    """n directions spread evenly on the sphere, randomly rotated."""
    from scipy.spatial.transform import Rotation
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r, phi = np.sqrt(1 - z * z), np.pi * (1 + 5 ** 0.5) * k
    return Rotation.random(random_state=rng).apply(np.c_[r * np.cos(phi), r * np.sin(phi), z])


def plan_minima(args):
    plan = json.loads((Path(args.out_dir) / "plan.json").read_text())
    minima = read(Path(args.run_dir) / "minima.traj", index=":")
    by_tid = {int(m.info["tid"]): m for m in minima}
    rows = plan["minima"][:args.limit] if args.limit else plan["minima"]
    return plan, rows, {r["tid"]: by_tid[r["tid"]] for r in rows}


def calculator(device):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    return FAIRChemCalculator(pretrained_mlip.get_predict_unit("uma-s-1p2", device=device), task_name="omat")


def write_rows(path, rows):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rows)); tmp.replace(path)   # never half-written


def read_rows(pattern):
    return [r for p in sorted(glob.glob(str(pattern))) for r in json.loads(Path(p).read_text())]


# ---------------------------------------------------------------- select
def select(args, out):
    import spglib
    selection, proposals, minima, _, records, _, _, _ = load(args.run_dir)
    rng = np.random.default_rng(args.seed)
    small = [k for k, m in enumerate(minima) if len(m) <= args.max_atoms and records[k]]
    higher = [k for k in small if category(records[k], 0.2) == "higher_order"
              and max_mic_untranslated(proposals[k], minima[k]) >= 0.05]
    pick_h = list(rng.choice(higher, min(args.n_higher_order, len(higher)), replace=False))
    rest = [k for k in small if k not in pick_h]
    pick = list(rng.choice(rest, args.n_random, replace=False)) + pick_h
    rows, specs = [], []
    for order, k in enumerate(pick):
        m = minima[k]
        symbols = np.array(m.get_chemical_symbols())
        classes = spglib.get_symmetry_dataset(spg_cell(m), symprec=0.01).equivalent_atoms
        firsts = sorted({int(np.flatnonzero(classes == c)[0]) for c in set(classes)})
        mobile = [i for i in firsts if symbols[i] in MOBILE]
        if len(mobile) > args.max_epicenters:
            mobile = sorted(rng.choice(mobile, args.max_epicenters, replace=False).tolist())
        others = [i for i in firsts if i not in mobile] if order < args.all_atom_minima else []
        if len(others) > args.max_epicenters:
            others = sorted(rng.choice(others, args.max_epicenters, replace=False).tolist())
        dirs = fibonacci(args.directions, rng)
        tid = int(m.info["tid"])
        rows.append(dict(tid=tid, formula=m.get_chemical_formula(), natoms=len(m), mobile=mobile, others=others,
                         category=category(records[k], 0.2), directions=dirs.tolist()))
        for e in mobile:
            for j in range(args.directions):
                specs += [dict(tid=tid, epicenter=int(e), element=symbols[e], direction=j, arm=a, repeat=0)
                          for a in SEARCH_ARMS]
        for e in others:          # the every-class check: one arm only
            specs += [dict(tid=tid, epicenter=int(e), element=symbols[e], direction=j, arm="nudge0.6", repeat=0,
                           all_atoms=True) for j in range(args.directions)]
    for arm in SEARCH_ARMS:
        first = [s for s in specs if s["arm"] == arm and not s.get("all_atoms")]
        specs += [dict(s, repeat=1) for s in (first[i] for i in rng.choice(len(first), min(args.repeats, len(first)),
                                                                          replace=False))]
    plan = dict(run_dir=args.run_dir, model=selection["settings"]["model"], sigma=selection["settings"]["inference"]["sigma"],
                minima=rows, specs=specs, settings=vars(args))
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(rows)} minima ({len(pick_h)} higher-order), {len(specs)} search runs; "
          f"epicenters per minimum {[len(r['mobile']) for r in rows]}, every-class check on {args.all_atom_minima}")


# ---------------------------------------------------------------- model rollouts
def model(args, out):
    import torch
    from saddleflow import SaddleFlow
    from saddleflow.data.transforms import transform_structure
    torch.backends.cuda.matmul.allow_tf32 = False
    plan, rows, minima = plan_minima(args)
    sf = SaddleFlow.from_pretrained(plan["model"], device=args.device)
    result = []
    for r in rows:
        m = minima[r["tid"]]
        gen = torch.Generator().manual_seed(args.seed * 100003 + r["tid"])
        starts = []
        for _ in range(args.draws):
            s = m.copy()
            s.set_positions(m.positions + plan["sigma"] * torch.randn(m.positions.shape, generator=gen).numpy(),
                            apply_constraint=False)
            starts.append(s)
        normal = sf.propose(starts, n=1, sigma=0.0, task_name="omat")
        inverted = sf.propose([transform_structure(s, -np.eye(3)) for s in starts], n=1, sigma=0.0, task_name="omat")
        for j in range(args.draws):
            result.append(dict(tid=r["tid"], draw=j, arm="model", pos=np.round(normal[j][0].positions, 5).tolist()))
            result.append(dict(tid=r["tid"], draw=j, arm="model_inv", pos=np.round(-inverted[j][0].positions, 5).tolist()))
        print(f"  minimum {r['tid']}: {args.draws} draws x 2", flush=True)
    write_rows(out / "model_landings.json", result)


# ---------------------------------------------------------------- search (Sella from every start)
def search_start(m, spec, plan_row):
    """The jittered start and the full-structure direction of a search spec."""
    rng = np.random.default_rng([spec["tid"], spec["epicenter"], spec["direction"]])
    e = spec["epicenter"]
    near = find_mic(m.positions - m.positions[e], m.cell, m.pbc)[1] < 5.0
    near[e] = False
    pos = m.positions.copy()
    pos[near] += rng.normal(0, 0.02, (near.sum(), 3))
    d = np.zeros_like(pos); d[e] = plan_row["directions"][spec["direction"]]
    return pos, d


def search(args, out):
    from saddleflow.classical_saddle_search import PESBackend, climb_along, run_sella
    plan, rows, minima = plan_minima(args)
    tids = set(minima)
    plan_rows = {r["tid"]: r for r in rows}
    jobs = [dict(s, id=f"{s['arm']}:{s['tid']}:{s['epicenter']}:{s['direction']}:{s['repeat']}")
            for s in plan["specs"] if s["tid"] in tids]
    models = out / "model_landings.json"
    if models.exists():
        jobs += [dict(l, id=f"{l['arm']}:{l['tid']}:{l['draw']}", repeat=0)
                 for l in json.loads(models.read_text()) if l["tid"] in tids]
    mine = jobs[args.shard::args.nshards]
    path = out / "search" / f"runs_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    # Files are replaced atomically, so reading every shard is safe; an earlier run with another --limit
    # split the work differently, and anything it finished is skipped here.
    done = {r["id"] for r in read_rows(path.parent / "runs_*.json")}
    calc = calculator(args.device)
    for job in mine:
        if job["id"] in done:
            continue
        m = minima[job["tid"]]
        pes = PESBackend(m.copy(), calc)
        row = {k: job[k] for k in job if k != "pos"}
        try:
            if job["arm"] in MODEL_ARMS:
                x0 = pes.dofs.pack(np.array(job["pos"]))
            else:
                pos, d = search_start(m, job, plan_rows[job["tid"]])
                x0, dvec = pes.dofs.pack(pos), pes.dofs.pack(d)
                if job["arm"] == "climb":
                    c = climb_along(pes, x0, dvec)
                    row.update(climb_status=c["status"], climb_calls=c["force_calls"], climb_steps=c["steps"],
                               perp_force_rms=c["perp_force_rms"])
                    if c["status"] != "left_bowl":
                        row.update(status="climb_" + c["status"]); done_rows.append(row)
                        write_rows(path, done_rows); continue
                    x0 = c["x"]
                else:
                    x0 = x0 + float(job["arm"][len("nudge"):]) * dvec
            before = pes.force_calls
            res = run_sella(pes, x0, fmax_tol=FMAX, max_steps=SELLA_STEPS, max_force_calls=before + SELLA_CALLS, delta0=0.1)
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            row.update(status=res["status"], sella_calls=pes.force_calls - before, sella_steps=res["steps"],
                       fmax=fmax, conv=bool(fmax <= FMAX), pos=np.round(res["positions"], 5).tolist())
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"shard {args.shard}: {len(done_rows)} runs")


# ---------------------------------------------------------------- verify distinct saddles
def clusters(args, out, minima):
    """Distinct converged points per minimum, merged within MERGE A (deterministic, so every shard agrees)."""
    runs = sorted(read_rows(out / "search" / "runs_*.json"), key=lambda r: r["id"])
    groups = []
    for tid, m in minima.items():
        reps = []
        for r in runs:
            if r["tid"] != tid or not r.get("conv"):
                continue
            for c in reps:
                if same(m, c["pos"], r["pos"]):
                    c["members"].append(r["id"]); break
            else:
                reps.append(dict(tid=tid, pos=r["pos"], members=[r["id"]]))
        groups += reps
    return groups


def check_point(pes, m, x, estimator_cls, calc):
    """Index and connectivity of one converged point: dict with nneg, eigs, connected, barrier.

    Only the signs of the two lowest eigenvalues matter, so Lanczos runs at a loose 1e-2 tolerance with a
    20-vector subspace."""
    from scipy.sparse.linalg import ArpackNoConvergence
    from saddleflow.classical_saddle_search import QueryBudgetExceeded
    from saddleflow.utils.relaxation import relax
    before = pes.force_calls
    pes.max_force_calls = before + INDEX_CALLS
    try:
        # Full FD Hessian when it fits in the budget (2 calls per degree of freedom), else Lanczos.
        estimator = estimator_cls(dense_max_dimension=INDEX_CALLS // 2, lanczos_maxiter=300, lanczos_tol=1e-2, ncv=20)
        values, vectors, _ = estimator.lowest_modes(pes, x, n_modes=2)
    except (QueryBudgetExceeded, ArpackNoConvergence):
        pes.max_force_calls = None
        return dict(nneg=None, index="unknown", index_calls=pes.force_calls - before)
    pes.max_force_calls = None
    out = dict(nneg=int(np.sum(values < -0.01)), eigs=values.tolist(), index_calls=pes.force_calls - before,
               modes=vectors)
    if out["nneg"] == 1:
        v = vectors[:, 0].reshape(-1, 3)
        v = 0.05 * v / np.linalg.norm(v, axis=1).max()
        energy = pes.energy(x)
        sides = []
        for sign in (1, -1):
            start = at(m, pes.dofs.expand(x + sign * v.ravel()))
            # Tight threshold: 0.05 A off a saddle the force is only ~0.03 eV/A, so a loose one stops at once.
            r = relax(start, calc, fmax=0.005, steps=1000)
            sides.append(dict(back=float(max_mic_untranslated(m, r.atoms)), barrier=energy - r.energy, steps=r.steps))
        k = int(np.argmin([s["back"] for s in sides]))
        out.update(connected=sides[k]["back"] < RETURN, barrier=sides[k]["barrier"], sides=sides)
    return out


def verify(args, out):
    from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, run_sella
    plan, rows, minima = plan_minima(args)
    groups = clusters(args, out, minima)
    path = out / "verify" / f"verify_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["tid"], r["members"][0]) for r in read_rows(path.parent / "verify_*.json")}
    calc = calculator(args.device)
    for g in groups[args.shard::args.nshards]:
        if (g["tid"], g["members"][0]) in done:
            continue
        m = minima[g["tid"]]
        pes = PESBackend(m.copy(), calc)
        x = pes.dofs.pack(np.array(g["pos"]))
        res = check_point(pes, m, x, MinimumModeEstimator, calc)
        modes = res.pop("modes", None)
        row = dict(g, **res)
        # Split a higher-order point found by the model along its two lowest modes.
        if res.get("nneg") is not None and res["nneg"] >= 2 and any(i.split(":")[0] in MODEL_ARMS for i in g["members"]):
            splits = []
            for k in range(2):
                v = modes[:, k].reshape(-1, 3); v = 0.1 * v / np.linalg.norm(v, axis=1).max()
                for sign in (1, -1):
                    before = pes.force_calls
                    s = run_sella(pes, x + sign * v.ravel(), fmax_tol=FMAX, max_steps=SELLA_STEPS,
                                  max_force_calls=before + SELLA_CALLS, delta0=0.1)
                    pes.max_force_calls = None
                    _, fmax = pes.dofs.force_metrics(pes.forces(s["final"]))
                    entry = dict(mode=k, sign=sign, conv=bool(fmax <= FMAX), calls=pes.force_calls - before,
                                 pos=np.round(s["positions"], 5).tolist())
                    if entry["conv"]:
                        c = check_point(pes, m, s["final"], MinimumModeEstimator, calc); c.pop("modes", None)
                        entry.update(c)
                    splits.append(entry)
            row["splits"] = splits
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"shard {args.shard}: {len(done_rows)} distinct points checked")


# ---------------------------------------------------------------- analyze
def symmetry_orbits(m, points):
    """Group points (positions arrays) into orbits under the minimum's space group (spglib, 0.01)."""
    import spglib
    sym = spglib.get_symmetry(spg_cell(m), symprec=0.01)
    frac0 = m.get_scaled_positions()
    perms = []
    for R, t in zip(sym["rotations"], sym["translations"]):
        mapped = frac0 @ R.T + t
        d = mapped[:, None, :] - frac0[None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ m.cell.array, axis=2) + 1e3 * (m.numbers[:, None] != m.numbers[None, :])
        perms.append((R, t, dist.argmin(1)))
    label = list(range(len(points)))
    for a in range(len(points)):
        if label[a] != a:
            continue
        fa = at(m, points[a]).get_scaled_positions(wrap=False)
        for R, t, perm in perms:
            img = np.empty_like(fa); img[perm] = fa @ R.T + t
            pos = img @ m.cell.array
            for b in range(a + 1, len(points)):
                if label[b] == b and same(m, pos, points[b]):
                    label[b] = a
    return len(set(label))


def analyze(args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan, rows, minima = plan_minima(args)
    runs = {r["id"]: r for r in read_rows(out / "search" / "runs_*.json")}
    checked = list({(c["tid"], c["members"][0]): c for c in read_rows(out / "verify" / "verify_*.json")}.values())
    rng = np.random.default_rng(0)
    arm_of = lambda rid: "all_atoms" if runs[rid].get("all_atoms") else rid.split(":")[0]
    lines = ["mp20bat hint search vs the released model", "=" * 104,
             f"{len(rows)} minima, {len(runs)} runs, {len(checked)} distinct converged points checked. Sella fmax {FMAX}, "
             f"<= {SELLA_STEPS} steps, <= {SELLA_CALLS} calls; index <= {INDEX_CALLS} calls; merge {MERGE} A; connected = "
             f"one side of the negative mode relaxes to within {RETURN} A of the minimum.", ""]

    # Outcomes and cost per arm (first runs).
    arms = list(SEARCH_ARMS) + ["all_atoms"] + list(MODEL_ARMS)
    lines += ["run outcomes and cost per arm (first runs):",
              "arm          runs  converged%  budget%  max_steps%  climb_fail%  error%   calls/run (climb+sella)"]
    for arm in arms:
        rs = [r for rid, r in runs.items() if arm_of(rid) == arm and r.get("repeat", 0) == 0]
        if not rs:
            continue
        st = [r.get("status", "") for r in rs]
        calls = np.mean([r.get("climb_calls", 0) + r.get("sella_calls", 0) for r in rs])
        lines.append(f"{arm:11s} {len(rs):5d}  {100 * np.mean([r.get('conv', False) for r in rs]):9.1f}  "
                     f"{100 * np.mean([s == 'query_budget' for s in st]):7.1f}  {100 * np.mean([s == 'max_steps' for s in st]):10.1f}  "
                     f"{100 * np.mean([s.startswith('climb_') for s in st]):11.1f}  {100 * np.mean([s == 'error' for s in st]):6.1f}   {calls:8.1f}")

    # Distinct saddles: connected index-1, plus those from splitting higher-order points.
    found = {}      # tid -> list of (pos, set of arms, source)
    for c in checked:
        m = minima[c["tid"]]
        entries = found.setdefault(c["tid"], [])
        if c.get("nneg") == 1 and c.get("connected"):
            entries.append(dict(pos=c["pos"], arms={arm_of(i) for i in c["members"] if runs[i].get("repeat", 0) == 0},
                                members=[i for i in c["members"] if runs[i].get("repeat", 0) == 0]))
        for s in c.get("splits", []):
            if s.get("nneg") == 1 and s.get("connected"):
                hit = next((e for e in entries if same(m, e["pos"], s["pos"])), None)
                if hit:
                    hit["arms"].add("split")
                else:
                    entries.append(dict(pos=s["pos"], arms={"split"}, members=[]))
    counts = {k: sum(1 for c in checked if c.get(k[0]) == k[1]) for k in [("nneg", 0), ("nneg", 1), ("index", "unknown")]}
    lines += ["", f"distinct converged points: {len(checked)}; index 0: {counts[('nneg', 0)]}, index 1: {counts[('nneg', 1)]}, "
              f"index >= 2: {sum(1 for c in checked if (c.get('nneg') or 0) >= 2)}, index unknown (budget): {counts[('index', 'unknown')]}; "
              f"index-1 not connected: {sum(1 for c in checked if c.get('nneg') == 1 and not c.get('connected'))}"]

    # Coverage of the union of connected saddles, per method.
    def draws_below(e, arm, k):
        return any(i.startswith(arm + ":") and int(i.split(":")[2]) < k for i in e["members"])
    methods = {"nudge0.3": lambda e: "nudge0.3" in e["arms"], "nudge0.6": lambda e: "nudge0.6" in e["arms"],
               "climb": lambda e: "climb" in e["arms"], "search (3 arms)": lambda e: bool(e["arms"] & set(SEARCH_ARMS)),
               "model, 32 draws": lambda e: draws_below(e, "model", 32),
               "model, 128 draws": lambda e: draws_below(e, "model", 128),
               "inverted, 128 draws": lambda e: draws_below(e, "model_inv", 128),
               "model + inverted, 64 + 64": lambda e: draws_below(e, "model", 64) or draws_below(e, "model_inv", 64),
               "model + inverted, 128 + 128": lambda e: draws_below(e, "model", 128) or draws_below(e, "model_inv", 128),
               "split of higher-order": lambda e: "split" in e["arms"]}
    total = sum(len(v) for v in found.values())
    lines += ["", f"coverage of the {total} distinct connected saddles found by any method (union):",
              "method                         found   share%   only this method"]
    for name, test in methods.items():
        n = sum(test(e) for v in found.values() for e in v)
        lines.append(f"{name:30s} {n:6d}  {100 * n / max(total, 1):7.1f}")
    model_any = lambda e: bool(e["arms"] & set(MODEL_ARMS))
    search_any = lambda e: bool(e["arms"] & set(SEARCH_ARMS))
    lines += [f"  found by the model (either) but by no search arm: {sum(model_any(e) and not search_any(e) for v in found.values() for e in v)}",
              f"  found by the search but never by the model:       {sum(search_any(e) and not model_any(e) for v in found.values() for e in v)}",
              f"  found only by the inverted runs (not the normal):  {sum(draws_below(e, 'model_inv', 10**6) and not draws_below(e, 'model', 10**6) for v in found.values() for e in v)}"]
    extra = sum(1 for v in found.values() for e in v if e["arms"] == {"all_atoms"})
    lines.append(f"  found only by non-mobile epicenters (every-class check, {sum(1 for r in rows if r['others'])} minima): {extra}")

    # Per minimum: saddles, symmetry orbits, discovery saturation, Good-Turing.
    lines += ["", "per minimum:",
              "  tid  formula            atoms  cat            saddles  up to sym  search runs  Good-Turing  model128  search"]
    fig, axes = plt.subplots(int(np.ceil(len(rows) / 4)), 4, figsize=(14, 3 * np.ceil(len(rows) / 4)), squeeze=False)
    for ax, r in zip(axes.flat, rows):
        tid, m = r["tid"], minima[r["tid"]]
        es = found.get(tid, [])
        orbit = symmetry_orbits(m, [e["pos"] for e in es]) if es else 0
        owner = {i: n for n, e in enumerate(es) for i in e["members"]}
        order = [rid for rid in runs if rid.split(":")[0] in SEARCH_ARMS and runs[rid]["tid"] == tid
                 and runs[rid].get("repeat", 0) == 0]
        order = [order[i] for i in rng.permutation(len(order))]
        seen, curve, hits = set(), [], []
        for rid in order:
            if rid in owner:
                seen.add(owner[rid]); hits.append(owner[rid])
            curve.append(len(seen))
        singletons = sum(1 for n in set(hits) if hits.count(n) == 1)
        gt = singletons / max(len(order), 1)
        ax.plot(curve); ax.set_title(f"{tid} {r['formula']}", fontsize=8); ax.set_xlabel("searches", fontsize=7)
        ax.set_ylabel("distinct connected saddles", fontsize=7)
        lines.append(f"  {tid:4d} {r['formula']:18s} {r['natoms']:5d}  {r['category']:13s}  {len(es):7d}  {orbit:9d}  "
                     f"{len(order):11d}  {gt:11.3f}  {sum(draws_below(e, 'model', 128) for e in es):8d}  {sum(search_any(e) for e in es):6d}")
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "discovery_curves.pdf"); plt.close(fig)
    lines.append("  Good-Turing: share of search runs that found a saddle seen only once, i.e. the estimated chance the next "
                 "search finds a new saddle (for these directions, not overall completeness).")

    # Cost per distinct connected saddle.
    lines += ["", "force calls per distinct connected saddle found (search: climb + Sella; model: Sella only, plus 20 network "
              "calls per draw):"]
    for arm in list(SEARCH_ARMS) + list(MODEL_ARMS):
        rs = [r for rid, r in runs.items() if arm_of(rid) == arm and r.get("repeat", 0) == 0]
        n = sum(arm in e["arms"] for v in found.values() for e in v)
        spent = sum(r.get("climb_calls", 0) + r.get("sella_calls", 0) for r in rs)
        lines.append(f"  {arm:10s} {spent:9d} calls / {n:4d} saddles = {spent / max(n, 1):8.1f}")
    climbs = [r for r in runs.values() if "perp_force_rms" in r]
    if climbs:
        lines.append(f"  climb: median {np.median([r['climb_steps'] for r in climbs]):.0f} steps, "
                     f"median perpendicular force at handover {np.median([r['perp_force_rms'] for r in climbs]):.3f} eV/A")

    # Repeatability: identical starts.
    owner_all = {i: (c["tid"], n) for n, c in enumerate(checked) for i in c["members"]}
    lines += ["", "Sella repeatability (identical starts, repeat vs first run):"]
    for arm in SEARCH_ARMS:
        reps = [r for rid, r in runs.items() if arm_of(rid) == arm and r.get("repeat") == 1]
        flips = [owner_all.get(r["id"]) != owner_all.get(r["id"][:-1] + "0") for r in reps]
        if flips:
            lines.append(f"  {arm:10s} {len(flips):3d} repeats, {100 * np.mean(flips):.1f}% end on a different point")

    # Higher-order splitting.
    split_src = [c for c in checked if "splits" in c]
    if split_src:
        good = sum(any(s.get("nneg") == 1 and s.get("connected") for s in c["splits"]) for c in split_src)
        lines += ["", f"higher-order points found by the model: {len(split_src)}; split into at least one connected index-1 "
                      f"saddle: {good}; saddles found only by splitting: {sum(e['arms'] == {'split'} for v in found.values() for e in v)}"]
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"select": select, "model": model, "search": search, "verify": verify, "analyze": analyze}[args.mode](args, out)


if __name__ == "__main__":
    main()
