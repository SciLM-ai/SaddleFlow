"""Noise, hint and soft-mode starts for the released model, on 30 diverse mp20bat test minima.

The two-minimum hint search found that the released SaddleFlow model, started from hints instead of noise,
reaches many more of the connected saddles around a minimum, but lands further from them. This experiment
repeats that comparison on 30 minima with matched budgets, adds soft-mode starts (a random thermal-like
displacement along the minimum's vibration modes, closer to the collective displacements the model was
trained from), runs every structured start also without the model, and tests whether a landing point near
a saddle means more than a random but physically sensible displacement of the same size would.

Minima: the two minima of the hint search (tids 486 and 18), plus 28 test-split minima of 8 to 40 atoms
from the failure-mode run, one per element set, picked by farthest-point sampling on composition, mobile
ion and anion family. Each minimum belongs to one of 4 groups (one GPU job each), balanced by size.

Arms per minimum (Sella: UMA-S-1.2 omat, fmax 0.01, <= 300 steps, <= 600 calls, delta0 0.1, stopped when the
structure drifts 2.5 A from the minimum or rises above both the minimum + 1.5 eV and its own start; a raw
start can begin above the 1.5 eV cap, and may then relax downhill but not climb):
    noise     256 starts: the minimum plus 0.3 A Gaussian noise on every atom; model rollout (5 x 4); Sella
    hint      256 starts: one symmetry-distinct mobile atom (<= 8 epicenters) pushed 0.6 A along a spread
              direction (Fibonacci sphere, randomly rotated per epicenter), plus 0.05 A noise on every atom;
              model rollout; Sella
    soft      256 starts: a random soft-mode displacement (soft_modes.py: full Hessian at the minimum, modes
              weighted by 1/sqrt(eigenvalue)) scaled so the most-moving atom moves 0.6 A, plus 0.05 A noise;
              model rollout; Sella
    push      128 of the hint starts, Sella directly from the start (no model)
    softpush  128 of the soft starts, Sella directly from the start (no model)
    null      96 model landing points (32 per model arm) replaced by a random soft-mode displacement of the
              same size (largest atomic displacement from the minimum); Sella from each
Every landing point and its null copy also get one energy and force evaluation.

Cost: every Sella run records its force calls and wall time; each minimum records the Hessian's force calls
and wall time, the rollout time per start and the time of one force call (both measured with the GPU to
itself, so the rollout can also be expressed in force-call equivalents). Sella runs share the GPU with the
other workers of the job, so their wall times are comparable between arms, not with a run alone on a GPU.
The report gives distinct connected saddles found within force-call and wall-time budgets, and the cost to
reach the first 1, 5 and 10, with the soft arms charged their Hessian and the model arms their rollouts.

Verification (per minimum, every converged point merged within 0.1 A): energy above the minimum (above
1.5 eV out of scope), full central-difference Hessian (index = eigenvalues below -0.01), and for index 1
connectivity (push along the negative mode, LBFGS to 0.005 eV/A, back within 0.2 A of the minimum).

    select   choose the minima, epicenters and directions; write plan.json (CPU)
    rollout  the Hessian at each minimum, model rollouts, energy and force at every landing point and its
             null copy (GPU; --group)
    sella    Sella for every arm (GPU; --group, --shard/--nshards within the group)
    verify   distinct points of the group's minima (GPU; --group, --shard/--nshards)
    analyze  report.txt (CPU)

    python examples/MaterialsSaddles/experiments/hint_vs_noise/hint_vs_noise.py <mode> --out-dir <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "hint_search"))
import hint_search as hs  # noqa: E402
import mode_search as ms  # noqa: E402
from hint_search import read_rows, write_rows  # noqa: E402
from soft_modes import soft_displacement, soft_modes  # noqa: E402

RUN_DIR = "/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_21421000"
OLD_DIR = "/scratch/gqs7zr/saddleflow/mp20bat_hint_search"      # the two-minimum hint search
KEEP = (486, 18)
N_STARTS = dict(noise=256, hint=256, soft=256, push=128, softpush=128, null=96)
NOISE_SIGMA, HINT_PUSH, HINT_NOISE, SOFT_SIZE, SOFT_NOISE = 0.3, 0.6, 0.05, 0.6, 0.05
MODEL_ARMS = ("noise", "hint", "soft")
ARMS = MODEL_ARMS + ("push", "softpush", "null")
WITHOUT_MODEL = {"push": "hint", "softpush": "soft"}     # control arm -> the model arm with the same starts
ANIONS = (("halide", {"F", "Cl", "Br", "I"}), ("phosphate", {"P"}), ("sulfur", {"S", "Se", "Te"}),
          ("nitrogen", {"N"}), ("carbon", {"C"}), ("silicon/boron", {"Si", "B", "Ge"}), ("oxide", {"O"}))
CUTS = (0.3, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["select", "rollout", "sella", "verify", "analyze"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-dir", default=RUN_DIR, help="failure-mode run whose minima.traj holds the test minima")
    p.add_argument("--n-minima", type=int, default=30)
    p.add_argument("--max-atoms", type=int, default=40)
    p.add_argument("--min-atoms", type=int, default=8, help="smaller cells have too few distinct moves to compare arms")
    p.add_argument("--max-epicenters", type=int, default=8)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--group", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="use only the first N minima of the group (smoke tests)")
    p.add_argument("--scale", type=float, default=1.0, help="scale every arm's start count (smoke tests)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stable-only", action="store_true",
                   help="analyze: leave out minima with a Hessian eigenvalue below -0.05 eV/A^2 (not minima under UMA-S)")
    return p.parse_args()


# ---------------------------------------------------------------- shared helpers
def mic(m, d):
    A = m.cell.array
    f = d @ np.linalg.inv(A)
    return (f - np.round(f)) @ A


def dist(m, P, q):
    """Max per-atom minimum-image distance from each of P (k, N, 3) to q (N, 3), rigid shift removed
    (vectorised hint_search.same; the minima have no fixed atoms)."""
    d = mic(m, np.asarray(P) - np.asarray(q)[None])
    d -= d.mean(-2, keepdims=True)
    return np.linalg.norm(d, axis=-1).max(-1)


def counts(args):
    return {a: max(len(MODEL_ARMS), int(round(n * args.scale))) for a, n in N_STARTS.items()}


def load_plan(args, group=None):
    plan = json.loads((Path(args.out_dir) / "plan.json").read_text())
    by_tid = {int(m.info["tid"]): m for m in read(Path(plan["run_dir"]) / "minima.traj", index=":")}
    rows = [r for r in plan["minima"] if group is None or r["group"] == group]
    if args.limit:
        rows = rows[:args.limit]
    return plan, rows, {r["tid"]: by_tid[r["tid"]] for r in rows}


def anion_family(symbols):
    s = set(symbols)
    return next((name for name, els in ANIONS if s & els), "other")


def size_from(m, pos):
    """Largest atomic displacement from the minimum, rigid shift removed."""
    return float(dist(m, np.asarray(pos)[None], m.positions)[0])


# ---------------------------------------------------------------- select
def select(args, out):
    import spglib
    run = Path(args.run_dir)
    minima = read(run / "minima.traj", index=":")
    by_tid = {int(m.info["tid"]): m for m in minima}
    rng = np.random.default_rng(args.seed)
    # One candidate per element set (the first in a seeded shuffle), with a mobile ion, small enough.
    cands, sets = {}, {tuple(sorted(set(by_tid[t].get_chemical_symbols()))) for t in KEEP}
    for k in rng.permutation(len(minima)):
        m = minima[k]
        s = tuple(sorted(set(m.get_chemical_symbols())))
        if args.min_atoms <= len(m) <= args.max_atoms and set(s) & hs.MOBILE and s not in sets and s not in cands:
            cands[s] = int(m.info["tid"])
    elements = sorted({e for t in list(cands.values()) + list(KEEP) for e in by_tid[t].get_chemical_symbols()})
    mobiles, families = sorted(hs.MOBILE), [n for n, _ in ANIONS] + ["other"]

    def feature(t):
        sym = by_tid[t].get_chemical_symbols()
        comp = np.array([sym.count(e) for e in elements], float) / len(sym)
        mob = np.array([e in sym for e in mobiles], float); mob /= max(mob.sum(), 1)
        fam = np.array([anion_family(sym) == f for f in families], float)
        return np.concatenate([comp, mob, fam, [len(sym) / args.max_atoms]])

    pool = list(cands.values())
    F = {t: feature(t) for t in pool + list(KEEP)}
    picked = list(KEEP)
    while len(picked) < args.n_minima and pool:
        gaps = [min(np.linalg.norm(F[t] - F[p]) for p in picked) for t in pool]
        picked.append(pool.pop(int(np.argmax(gaps))))

    rows = []
    for tid in picked:
        m = by_tid[tid]
        symbols = np.array(m.get_chemical_symbols())
        sym = spglib.get_symmetry_dataset(hs.spg_cell(m), symprec=0.01)
        classes = sym.equivalent_atoms
        firsts = sorted({int(np.flatnonzero(classes == c)[0]) for c in set(classes)})
        mobile = [i for i in firsts if symbols[i] in hs.MOBILE]
        if len(mobile) > args.max_epicenters:
            mobile = sorted(rng.choice(mobile, args.max_epicenters, replace=False).tolist())
        n_hint = N_STARTS["hint"]
        per = [n_hint // len(mobile) + (k < n_hint % len(mobile)) for k in range(len(mobile))]
        dirs = {str(e): hs.fibonacci(n, rng).tolist() for e, n in zip(mobile, per)}
        rows.append(dict(tid=tid, formula=m.get_chemical_formula(), natoms=len(m), spacegroup=sym.international,
                         mobile_ions=sorted(set(symbols) & hs.MOBILE), anion=anion_family(symbols),
                         epicenters=mobile, directions=dirs, kept=tid in KEEP))
    # Groups balanced by cost (force calls scale roughly with atoms^1.5): largest first, to the lightest group.
    load = [0.0] * args.groups
    for r in sorted(rows, key=lambda r: -r["natoms"]):
        g = int(np.argmin(load)); r["group"] = g; load[g] += r["natoms"] ** 1.5
    plan = dict(run_dir=str(run), model="SciLM/SaddleFlow", minima=rows, settings=vars(args))
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(rows)} minima from {len(cands) + len(KEEP)} element sets; groups (relative load) "
          f"{[round(x / max(load), 2) for x in load]}")
    for r in rows:
        print(f"  {r['tid']:4d} g{r['group']} {r['formula']:20s} {r['natoms']:3d} atoms  {r['spacegroup']:10s} "
              f"{','.join(r['mobile_ions']):8s} {r['anion']:12s} {len(r['epicenters'])} epicenters")


# ---------------------------------------------------------------- rollout
def starts_for(args, r, m, values, modes):
    """(noise starts, hint starts with their epicenter and direction, soft starts), seeded per minimum."""
    n = counts(args)
    noise = []
    for j in range(n["noise"]):
        g = np.random.default_rng([r["tid"], 1, j])
        noise.append(m.positions + NOISE_SIGMA * g.normal(size=m.positions.shape))
    hint = []
    for e in r["epicenters"]:
        for k, d in enumerate(r["directions"][str(e)]):
            g = np.random.default_rng([r["tid"], 2, e, k])
            pos = m.positions + HINT_NOISE * g.normal(size=m.positions.shape)
            pos[e] += HINT_PUSH * np.asarray(d)
            hint.append(dict(pos=pos, epicenter=int(e), direction=k, vector=d))
    # With --scale, a seeded subset that still spans every epicenter.
    if len(hint) > n["hint"]:
        keep = np.sort(np.random.default_rng([r["tid"], 3]).choice(len(hint), n["hint"], replace=False))
        hint = [hint[i] for i in keep]
    soft = []
    for j in range(n["soft"]):
        g = np.random.default_rng([r["tid"], 6, j])
        soft.append(m.positions + soft_displacement(values, modes, g, SOFT_SIZE) + SOFT_NOISE * g.normal(size=m.positions.shape))
    return noise, hint, soft


def energy_force(m, pos, calc):
    """(energy, largest atomic force) at pos."""
    a = m.copy(); a.set_positions(pos, apply_constraint=False); a.calc = calc
    return float(a.get_potential_energy()), float(np.linalg.norm(a.get_forces(), axis=1).max())


def rollout(args, out):
    import torch
    from saddleflow import SaddleFlow
    from saddleflow.classical_saddle_search import PESBackend
    torch.backends.cuda.matmul.allow_tf32 = False
    plan, rows, minima = load_plan(args, args.group)
    path = out / "landings" / f"landings_{args.group}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["tid"] for r in done_rows}
    sf = SaddleFlow.from_pretrained(plan["model"], device=args.device)
    calc = hs.calculator(args.device)
    for r in rows:
        if r["tid"] in done:
            continue
        m = minima[r["tid"]]
        pes = PESBackend(m.copy(), calc)
        t0 = time.perf_counter()
        values, modes, hess_calls = soft_modes(pes, pes.dofs.pack(m.positions))
        hess_seconds = time.perf_counter() - t0
        noise, hint, soft = starts_for(args, r, m, values, modes)
        structures = []
        for pos in noise + [h["pos"] for h in hint] + soft:
            s = m.copy(); s.set_positions(pos, apply_constraint=False); structures.append(s)
        t0 = time.perf_counter()
        props = sf.propose(structures, n=1, sigma=0.0, task_name="omat", batch_size=args.batch_size)
        rollout_seconds = (time.perf_counter() - t0) / len(structures)
        e_min, f_min = energy_force(m, m.positions, calc)
        t0 = time.perf_counter()
        for _ in range(10):
            energy_force(m, m.positions + 1e-4 * np.random.default_rng(0).normal(size=m.positions.shape), calc)
        force_seconds = (time.perf_counter() - t0) / 10
        bounds = {"noise": 0, "hint": len(noise), "soft": len(noise) + len(hint)}
        for j, p in enumerate(props):
            kind = [k for k, b in bounds.items() if j >= b][-1]
            idx = j - bounds[kind]
            land = p[0].positions
            size = size_from(m, land)
            null = m.positions + soft_displacement(values, modes, np.random.default_rng([r["tid"], 4, j]), size)
            e_l, f_l = energy_force(m, land, calc)
            e_n, f_n = energy_force(m, null, calc)
            row = dict(id=f"{kind}:{r['tid']}:{idx}", arm=kind, tid=r["tid"], idx=idx, f_min=f_min, size=size,
                       e_landing=e_l - e_min, f_landing=f_l, e_null=e_n - e_min, f_null=f_n, hessian_calls=hess_calls, hessian_seconds=hess_seconds,
                       rollout_seconds=rollout_seconds, force_seconds=force_seconds,
                       soft_eigs=np.round(values[:6], 4).tolist(),
                       landing=np.round(land, 5).tolist(), null=np.round(null, 5).tolist())
            if kind == "hint":
                h = hint[idx]
                row.update(epicenter=h["epicenter"], direction=h["direction"], vector=h["vector"],
                           start=np.round(h["pos"], 5).tolist())
            elif kind == "soft":
                row.update(start=np.round(soft[idx], 5).tolist())
            done_rows.append(row)
        write_rows(path, done_rows)
        print(f"  minimum {r['tid']}: {len(noise)} noise + {len(hint)} hint + {len(soft)} soft rollouts "
              f"({rollout_seconds:.2f} s per start, one force call {force_seconds * 1000:.0f} ms); Hessian {hess_calls} calls "
              f"in {hess_seconds:.0f} s, softest modes {np.round(values[:3], 3).tolist()} eV/A^2", flush=True)
    print(f"group {args.group}: {len(done_rows)} landing points")


# ---------------------------------------------------------------- sella
def sella_jobs(args, rows, landings):
    """Every Sella run of the group, in a fixed order: (id, arm, tid, start positions, extra fields)."""
    n = counts(args)
    jobs = []
    for r in rows:
        mine = {a: [l for l in landings if l["tid"] == r["tid"] and l["arm"] == a] for a in MODEL_ARMS}
        for a in MODEL_ARMS:
            for l in mine[a]:
                extra = {k: l[k] for k in ("epicenter", "direction") if k in l}
                jobs.append((l["id"], a, r["tid"], l["landing"], dict(extra, idx=l["idx"])))
        g = np.random.default_rng([r["tid"], 5])
        for control, a in WITHOUT_MODEL.items():
            src = mine[a]
            for i in np.sort(g.choice(len(src), min(n[control], len(src)), replace=False)):
                l = src[i]
                extra = {k: l[k] for k in ("epicenter", "direction") if k in l}
                jobs.append((f"{control}:{r['tid']}:{l['idx']}", control, r["tid"], l["start"], dict(extra, idx=l["idx"])))
        per = n["null"] // len(MODEL_ARMS)
        for a in MODEL_ARMS:
            src = mine[a]
            for i in np.sort(g.choice(len(src), min(per, len(src)), replace=False)):
                l = src[i]
                jobs.append((f"null:{r['tid']}:{a}{l['idx']}", "null", r["tid"], l["null"], dict(idx=l["idx"], source=a)))
    return jobs


def sella(args, out):
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    plan, rows, minima = load_plan(args, args.group)
    landings = read_rows(out / "landings" / f"landings_{args.group}.json")
    path = out / "runs" / f"runs_{args.group}_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in read_rows(path.parent / f"runs_{args.group}_*.json")}
    calc = hs.calculator(args.device)
    e_min = {}
    for rid, arm, tid, start, extra in sella_jobs(args, rows, landings)[args.shard::args.nshards]:
        if rid in done:
            continue
        m = minima[tid]
        pes = PESBackend(m.copy(), calc)
        if tid not in e_min:
            e_min[tid] = pes.energy(pes.dofs.pack(m.positions))
        row = dict(id=rid, arm=arm, tid=tid, **extra)
        try:
            x0 = pes.dofs.pack(np.array(start))
            e_start = pes.energy(x0)
            before, t0 = pes.force_calls, time.perf_counter()
            res = run_sella(pes, x0, fmax_tol=hs.FMAX, max_steps=hs.SELLA_STEPS, max_force_calls=before + hs.SELLA_CALLS,
                            delta0=0.1, stop=ms.stop_rule(m, e_min[tid], pes.dofs, e_start))
            row["seconds"] = time.perf_counter() - t0
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            final = np.asarray(res["positions"])
            row.update(start_rise=e_start - e_min[tid], status=res["status"], sella_calls=pes.force_calls - before, sella_steps=res["steps"], fmax=fmax,
                       conv=bool(fmax <= hs.FMAX and res["status"] == "converged"),
                       moved=float(dist(m, final[None], np.array(start))[0]), pos=np.round(final, 5).tolist())
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"group {args.group} shard {args.shard}: {len(done_rows)} runs")


# ---------------------------------------------------------------- verify
def distinct_points(runs, minima):
    """Distinct converged points per minimum, merged within hs.MERGE A, in a fixed order (every shard agrees)."""
    groups = []
    for tid, m in minima.items():
        reps, P = [], []
        for r in sorted((r for r in runs if r["tid"] == tid and r.get("conv")), key=lambda r: r["id"]):
            if P:
                d = dist(m, np.array(P), np.array(r["pos"]))
                k = int(d.argmin())
                if d[k] < hs.MERGE:
                    reps[k]["members"].append(r["id"]); continue
            reps.append(dict(tid=tid, pos=r["pos"], members=[r["id"]])); P.append(r["pos"])
        groups += reps
    return groups


def check(pes, m, x, e_min, calc):
    """mode_search.verify's check of one point: barrier, full-Hessian index, connectivity."""
    from saddleflow.classical_saddle_search import MinimumModeEstimator
    row = dict(barrier=pes.energy(x) - e_min)
    if row["barrier"] > ms.MAX_RISE:
        return dict(row, status="above_cutoff")
    before = pes.force_calls
    values, vectors, _ = MinimumModeEstimator(backend="dense", eps=0.002).lowest_modes(pes, x, n_modes=4)
    row.update(eigs=values.tolist(), nneg=int(np.sum(values < -0.01)), index_calls=pes.force_calls - before)
    if row["nneg"] == 1:
        size = float(np.clip(0.1 / abs(values[0]), 0.1, 0.3))
        v = vectors[:, 0].reshape(-1, 3); v = size * v / np.linalg.norm(v, axis=1).max()
        toward = np.dot(v.ravel(), pes.dofs.pack(m.positions) - x)
        sides = []
        for sign in ((1, -1) if toward >= 0 else (-1, 1)):
            ok, steps, d = ms.returns_to(m, hs.at(m, pes.dofs.expand(x + sign * v.ravel())), calc, e_min)
            sides.append(dict(returned=bool(ok), steps=steps, distance=float(d)))
            if ok:
                break
        row.update(connected=any(s["returned"] for s in sides), sides=sides, relax_calls=sum(s["steps"] for s in sides))
    return dict(row, status="checked")


def verify(args, out):
    from saddleflow.classical_saddle_search import PESBackend
    plan, rows, minima = load_plan(args, args.group)
    runs = read_rows(out / "runs" / f"runs_{args.group}_*.json")
    groups = distinct_points(runs, minima)
    path = out / "verify" / f"verify_{args.group}_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["tid"], r["members"][0]) for r in read_rows(path.parent / f"verify_{args.group}_*.json")}
    calc = hs.calculator(args.device)
    e_min = {}
    for g in groups[args.shard::args.nshards]:
        if (g["tid"], g["members"][0]) in done:
            continue
        m = minima[g["tid"]]
        pes = PESBackend(m.copy(), calc)
        if g["tid"] not in e_min:
            e_min[g["tid"]] = pes.energy(pes.dofs.pack(m.positions))
        try:
            row = dict(g, **check(pes, m, pes.dofs.pack(np.array(g["pos"])), e_min[g["tid"]], calc))
        except Exception as exc:
            row = dict(g, status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"group {args.group} shard {args.shard}: {len(done_rows)} distinct points checked")


# ---------------------------------------------------------------- analyze
def pvalue(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    return wilcoxon(d).pvalue if np.count_nonzero(d) >= 2 else float("nan")


def analyze(args, out):
    plan, rows, minima = load_plan(args)
    landings = {l["id"]: l for l in read_rows(out / "landings" / "landings_*.json")}
    unstable = sorted({l["tid"] for l in landings.values() if l["soft_eigs"][0] < -0.05})
    if args.stable_only:
        rows = [r for r in rows if r["tid"] not in unstable]
        minima = {t: m for t, m in minima.items() if t not in unstable}
        landings = {k: l for k, l in landings.items() if l["tid"] in minima}
    runs = {r["id"]: r for r in read_rows(out / "runs" / "runs_*.json") if r["tid"] in minima}
    # Distinct points again over all groups (a minimum belongs to one group, so this matches verify), and
    # every converged run assigned to its verified point by position.
    points = []
    for c in read_rows(out / "verify" / "verify_*.json"):
        m = minima.get(c["tid"])
        if m is None:
            continue
        P = [p["pos"] for p in points if p["tid"] == c["tid"]]
        if not P or dist(m, np.array(P), np.array(c["pos"])).min() >= hs.MERGE:
            points.append(dict(c, members=[]))
    by_tid = {}
    for k, p in enumerate(points):
        by_tid.setdefault(p["tid"], []).append(k)
    owner = {}
    for rid, r in runs.items():
        if r.get("conv") and r["tid"] in by_tid:
            ks = by_tid[r["tid"]]
            d = dist(minima[r["tid"]], np.array([points[k]["pos"] for k in ks]), np.array(r["pos"]))
            if d.min() < hs.MERGE:
                owner[rid] = ks[int(d.argmin())]; points[owner[rid]]["members"].append(rid)
    saddle = lambda k, cut: points[k].get("nneg") == 1 and points[k].get("connected") and points[k]["barrier"] <= cut
    hit = lambda rid, cut=1.0: rid in owner and saddle(owner[rid], cut)
    tids = [r["tid"] for r in rows if r["tid"] in by_tid]
    n_runs = {a: sum(r["arm"] == a for r in runs.values()) for a in ARMS}
    hess = [l["hessian_calls"] for l in landings.values() if l["idx"] == 0 and l["arm"] == "noise"]

    L = [f"Noise, hint and soft-mode starts for the released model: {len(tids)} of {len(rows)} mp20bat minima finished",
         "=" * 110,
         f"runs: {n_runs}; distinct converged points {len(points)}; connected index-1 saddles <= 1 eV: "
         f"{sum(saddle(k, 1.0) for k in range(len(points)))}. Coverage is counted against the union of all six arms. "
         f"Hessian for the soft modes: {np.mean(hess) if hess else 0:.0f} force calls per minimum.", ""]

    # 1. Outcomes per arm.
    L += ["1. run outcomes per arm", "arm        runs  converged%  energy-stop%  drift%  budget%  error%  Sella calls p50 / mean"]
    for a in ARMS:
        rs = [r for r in runs.values() if r["arm"] == a]
        if not rs:
            continue
        st = [r.get("status", "") for r in rs]; c = np.array([r.get("sella_calls", 0) for r in rs])
        L.append(f"{a:9s} {len(rs):6d}  {100 * np.mean([r.get('conv', False) for r in rs]):9.1f}  "
                 f"{100 * np.mean([s == 'energy' for s in st]):11.1f}  {100 * np.mean([s == 'drift' for s in st]):5.1f}  "
                 f"{100 * np.mean([s == 'query_budget' for s in st]):6.1f}  {100 * np.mean([s == 'error' for s in st]):5.1f}  "
                 f"{np.median(c):8.0f} / {c.mean():5.0f}")

    # 2. Coverage per minimum, and paired across minima.
    def found(tid, arm, cut, ids=None):
        return {owner[rid] for rid, r in runs.items() if r["tid"] == tid and r["arm"] == arm and hit(rid, cut)
                and (ids is None or rid in ids)}
    L += ["", "2. connected saddles per minimum (noise, hint and soft: 256 starts each; push and softpush: 128)",
          "  tid  formula              atoms  anion         <=0.3: union noise hint soft   <=1: union noise hint soft"
          "  push softpush  only: noise hint soft"]
    cov = {cut: {a: [] for a in ("union",) + MODEL_ARMS} for cut in CUTS}
    for r in rows:
        tid = r["tid"]
        if tid not in by_tid:
            continue
        f = {cut: {a: found(tid, a, cut) for a in ARMS} for cut in CUTS}
        u = {cut: {k for k in by_tid[tid] if saddle(k, cut)} for cut in CUTS}
        for cut in CUTS:
            cov[cut]["union"].append(len(u[cut]))
            for a in MODEL_ARMS:
                cov[cut][a].append(len(f[cut][a]))
        only = [len(f[1.0][a] - set().union(*[f[1.0][b] for b in MODEL_ARMS if b != a])) for a in MODEL_ARMS]
        L.append(f"  {tid:4d} {r['formula']:20s} {r['natoms']:5d}  {r['anion']:12s}  "
                 f"{len(u[0.3]):11d} {len(f[0.3]['noise']):5d} {len(f[0.3]['hint']):4d} {len(f[0.3]['soft']):4d}   "
                 f"{len(u[1.0]):9d} {len(f[1.0]['noise']):5d} {len(f[1.0]['hint']):4d} {len(f[1.0]['soft']):4d}  "
                 f"{len(f[1.0]['push']):4d} {len(f[1.0]['softpush']):8d}  {only[0]:11d} {only[1]:4d} {only[2]:4d}")
    for cut in CUTS:
        c = {a: np.array(v, float) for a, v in cov[cut].items()}
        if not len(c["union"]):
            continue
        shares = ", ".join(f"{a} {100 * np.mean(c[a] / np.maximum(c['union'], 1)):.1f}%" for a in MODEL_ARMS)
        L.append(f"  <= {cut} eV: mean share of the union per minimum: {shares}")
        for a, b in (("hint", "noise"), ("soft", "noise"), ("soft", "hint")):
            d = c[a] - c[b]
            L.append(f"     {a} - {b}: more on {np.sum(d > 0)}, fewer on {np.sum(d < 0)}, equal on {np.sum(d == 0)} minima; "
                     f"median difference {np.median(d):+.1f}; Wilcoxon p = {pvalue(d):.2g}")

    # 3. Discovery against starts.
    L += ["", "3. distinct connected saddles (<= 1 eV) against starts, mean over minima (starts in random order)"]
    marks = [8, 16, 32, 64, 128, 256]
    rng = np.random.default_rng(0)
    for a in MODEL_ARMS:
        curves = []
        for tid in tids:
            ids = [rid for rid, r in runs.items() if r["tid"] == tid and r["arm"] == a]
            ids = [ids[i] for i in rng.permutation(len(ids))]
            curves.append([len({owner[i] for i in ids[:n] if hit(i)}) for n in marks])
        if curves:
            L.append(f"  {a:6s} " + "  ".join(f"{n}: {v:5.1f}" for n, v in zip(marks, np.mean(curves, 0))))

    # 4. Landing accuracy and cost.
    L += ["", "4. how far Sella moved the start (the landing point for model arms), and what it cost (converged runs)",
          "arm        runs  moved p50 / p90 (A)  within 0.1 A  Sella calls p50  calls per connected saddle (<= 1 eV)"]
    per_min = {}
    for a in ARMS:
        rs = [r for r in runs.values() if r["arm"] == a and r.get("conv")]
        if not rs:
            continue
        mv = np.array([r["moved"] for r in rs]); c = np.array([r["sella_calls"] for r in rs])
        spent = sum(r.get("sella_calls", 0) for r in runs.values() if r["arm"] == a)
        n = sum(len(found(t, a, 1.0)) for t in tids)
        L.append(f"{a:9s} {len(rs):6d}  {np.median(mv):6.2f} / {np.percentile(mv, 90):5.2f}       {100 * np.mean(mv < 0.1):9.0f}%  "
                 f"{np.median(c):14.0f}  {spent / max(n, 1):10.0f}")
        per_min[a] = {t: np.median([r["moved"] for r in rs if r["tid"] == t]) for t in tids if any(r["tid"] == t for r in rs)}
    for a, b in (("hint", "noise"), ("soft", "noise"), ("soft", "hint")):
        common = [t for t in tids if t in per_min.get(a, {}) and t in per_min.get(b, {})]
        if common:
            d = np.array([per_min[a][t] - per_min[b][t] for t in common])
            L.append(f"  per minimum, median distance moved, {a} - {b}: larger on {np.sum(d > 0)} of {len(d)} minima "
                     f"(median difference {np.median(d):+.2f} A; Wilcoxon p = {pvalue(d):.2g})")
    L.append("  network calls (not counted above): 20 per noise, hint or soft start")

    # 4b. Cost to find distinct connected saddles, with every overhead charged.
    per_tid = {}
    for l in landings.values():
        per_tid.setdefault(l["tid"], l)
    over = {t: per_tid[t] for t in tids if t in per_tid}
    budgets_calls, budgets_s, firsts = [2000, 5000, 10000, 20000, 40000], [300, 900, 1800, 3600, 7200], [1, 5, 10]
    L += ["", "4b. cost to find distinct connected saddles (<= 1 eV), runs in random order, mean over minima. Charged: Sella "
              "calls and time; soft arms the Hessian (once per minimum); model arms the rollout time per start.",
          f"    rollout per start: {np.median([o['rollout_seconds'] for o in over.values()]) if over else 0:.2f} s"
          f" = {np.median([o['rollout_seconds'] / o['force_seconds'] for o in over.values()]) if over else 0:.1f} force-call "
          f"equivalents (one force call {1000 * np.median([o['force_seconds'] for o in over.values()]) if over else 0:.0f} ms, "
          f"Hessian {np.median([o['hessian_calls'] for o in over.values()]) if over else 0:.0f} calls in "
          f"{np.median([o['hessian_seconds'] for o in over.values()]) if over else 0:.0f} s, medians over minima)",
          "arm        saddles within force-call budget: " + " ".join(f"{b // 1000:>4d}k" for b in budgets_calls)
          + "   | within worker time: " + " ".join(f"{b // 60:>4d}m" for b in budgets_s)
          + "   | calls to reach 1 / 5 / 10 (minima reaching)"]
    rng = np.random.default_rng(2)
    for a in ARMS:
        if a == "null" or not any(r["arm"] == a for r in runs.values()):
            continue
        by_calls, by_time, reach = [], [], {k: [] for k in firsts}
        for t in tids:
            rs = [r for r in runs.values() if r["tid"] == t and r["arm"] == a]
            if not rs or t not in over:
                continue
            rs = [rs[i] for i in rng.permutation(len(rs))]
            o = over[t]
            calls0 = o["hessian_calls"] if a in ("soft", "softpush") else 0
            secs0 = o["hessian_seconds"] if a in ("soft", "softpush") else 0.0
            per_start = o["rollout_seconds"] if a in MODEL_ARMS else 0.0
            c = calls0 + np.cumsum([r.get("sella_calls", 0) for r in rs])
            w = secs0 + np.cumsum([r.get("seconds", 0.0) + per_start for r in rs])
            seen, n_seen = set(), []
            for r in rs:
                if hit(r["id"]):
                    seen.add(owner[r["id"]])
                n_seen.append(len(seen))
            n_seen = np.array(n_seen)
            by_calls.append([n_seen[c <= b][-1] if np.any(c <= b) else 0 for b in budgets_calls])
            by_time.append([n_seen[w <= b][-1] if np.any(w <= b) else 0 for b in budgets_s])
            for k in firsts:
                if n_seen[-1] >= k:
                    reach[k].append(c[np.argmax(n_seen >= k)])
        if by_calls:
            reached = " / ".join(f"{np.median(reach[k]):.0f}" if reach[k] else "-" for k in firsts)
            L.append(f"{a:9s} {'':32s}" + " ".join(f"{v:5.1f}" for v in np.mean(by_calls, 0)) + f"   | {'':19s}"
                     + " ".join(f"{v:5.1f}" for v in np.mean(by_time, 0))
                     + f"   | {reached} ({'/'.join(str(len(reach[k])) for k in firsts)} of {len(by_calls)})")
    L.append("    push and softpush have half as many starts, so they stop growing once their 128 runs are spent; "
             "worker time is Sella time with 8 workers sharing a GPU, plus the rollouts and the Hessian")

    # 5. What the model adds: the same starts with and without the model.
    L += ["", "5. the same starts with the model (rollout, then Sella) and without it (Sella from the start)"]
    for control, a in WITHOUT_MODEL.items():
        pairs = [(runs.get(f"{a}:{r['tid']}:{r['idx']}"), r) for r in runs.values() if r["arm"] == control]
        pairs = [(x, y) for x, y in pairs if x is not None]
        if not pairs:
            continue
        L.append(f"  {a} starts ({len(pairs)}):")
        for name, k, arm in (("model + Sella", 0, a), ("Sella only", 1, control)):
            rs = [pr[k] for pr in pairs]
            ids = {r["id"] for r in rs}
            n = sum(len(found(t, arm, 1.0, ids)) for t in tids)
            calls = sum(r.get("sella_calls", 0) for r in rs)
            L.append(f"    {name:14s} converged {100 * np.mean([r.get('conv', False) for r in rs]):5.1f}%, reached a connected "
                     f"saddle {100 * np.mean([hit(r['id']) for r in rs]):5.1f}%, Sella calls per run {calls / len(rs):6.1f}, "
                     f"distinct connected saddles {n:4d}, calls per saddle {calls / max(n, 1):7.0f}")
        same_pt = [owner.get(x["id"]) == owner.get(y["id"]) for x, y in pairs if x["id"] in owner and y["id"] in owner]
        if same_pt:
            L.append(f"    both reached a verified point: the same one in {100 * np.mean(same_pt):.0f}% of {len(same_pt)} pairs")

    # 6. Null test: each landing point against a random soft-mode displacement of the same size.
    L += ["", "6. null test: model landing points against random soft-mode displacements of the same size (null)",
          "source   landings  size p50 (A)  energy above min p50: landing / null (eV)  largest force p50: landing / null (eV/A)"]
    for a in MODEL_ARMS:
        ls = [l for l in landings.values() if l["arm"] == a and l["tid"] in by_tid]
        if not ls:
            continue
        g = lambda k: np.median([l[k] for l in ls])
        L.append(f"{a:7s} {len(ls):9d}  {g('size'):11.2f}   {g('e_landing'):18.2f} / {g('e_null'):5.2f}"
                 f"{'':22s}{g('f_landing'):6.2f} / {g('f_null'):5.2f}")
    L.append("  Sella from the landing point against Sella from its null copy (the same landings):")
    for a in MODEL_ARMS:
        pairs = [(runs.get(f"{a}:{r['tid']}:{r['idx']}"), r) for r in runs.values() if r["arm"] == "null" and r.get("source") == a]
        pairs = [(x, y) for x, y in pairs if x is not None]
        if not pairs:
            continue
        for name, k in (("landing", 0), ("null", 1)):
            rs = [pr[k] for pr in pairs]
            conv = [r for r in rs if r.get("conv")]
            L.append(f"    {a:6s} {name:8s} n {len(rs):4d}: converged {100 * len(conv) / len(rs):5.1f}%, reached a connected "
                     f"saddle {100 * np.mean([hit(r['id']) for r in rs]):5.1f}%, Sella calls p50 "
                     f"{np.median([r.get('sella_calls', 0) for r in rs]):5.0f}, moved p50 "
                     f"{np.median([r['moved'] for r in conv]) if conv else float('nan'):.2f} A")
    for a in MODEL_ARMS:
        dl, dn = [], []
        for l in (l for l in landings.values() if l["arm"] == a and l["tid"] in by_tid):
            ks = [k for k in by_tid[l["tid"]] if points[k].get("nneg") == 1 and k != owner.get(l["id"])]
            if ks:
                P = np.array([points[k]["pos"] for k in ks]); m = minima[l["tid"]]
                dl.append(dist(m, P, np.array(l["landing"])).min()); dn.append(dist(m, P, np.array(l["null"])).min())
        if dl:
            dl, dn = np.array(dl), np.array(dn)
            L.append(f"  {a:6s} distance to the nearest index-1 point other than its own Sella result, p50: landing {np.median(dl):.2f} A, "
                     f"null {np.median(dn):.2f} A; within 0.3 A: {100 * np.mean(dl < 0.3):.0f}% vs {100 * np.mean(dn < 0.3):.0f}%")
    spacing = []
    for t in tids:
        ks = [k for k in by_tid[t] if points[k].get("nneg") == 1]
        P = np.array([points[k]["pos"] for k in ks])
        if len(ks) > 1:
            spacing += [np.sort(dist(minima[t], P, P[i]))[1] for i in range(len(ks))]
    if spacing:
        L.append(f"  spacing of index-1 points (distance to the nearest other one) p10 / p50 / p90: "
                 f"{np.percentile(spacing, 10):.2f} / {np.median(spacing):.2f} / {np.percentile(spacing, 90):.2f} A")

    # 7. Does the hint decide the saddle?
    L += ["", "7. does the hint decide the saddle? (hint runs that reached a connected saddle <= 1 eV)"]
    cos_h, cos_r, same_nb, same_rand = [], [], [], []
    rng = np.random.default_rng(1)
    for t in tids:
        m = minima[t]
        hs_runs = [r for r in runs.values() if r["tid"] == t and r["arm"] == "hint" and hit(r["id"])]
        for r in hs_runs:
            l = landings[r["id"]]
            d = mic(m, np.array(points[owner[r["id"]]]["pos"]) - m.positions); d -= d.mean(0)
            v, e = np.asarray(l["vector"]), l["epicenter"]
            if np.linalg.norm(d[e]) > 1e-3:
                cos_h.append(d[e] @ v / np.linalg.norm(d[e]))
                u = rng.normal(size=3); cos_r.append(d[e] @ u / np.linalg.norm(d[e]) / np.linalg.norm(u))
        by_e = {}
        for r in hs_runs:
            by_e.setdefault(landings[r["id"]]["epicenter"], []).append(r)
        for e, rs_e in by_e.items():
            if len(rs_e) < 3:
                continue
            V = np.array([landings[r["id"]]["vector"] for r in rs_e])
            for i, r in enumerate(rs_e):
                ang = V @ V[i]; ang[i] = -2
                j = int(np.argmax(ang)); k = int(rng.choice([x for x in range(len(rs_e)) if x != i]))
                same_nb.append(owner[r["id"]] == owner[rs_e[j]["id"]]); same_rand.append(owner[r["id"]] == owner[rs_e[k]["id"]])
    if cos_h:
        cos_h, cos_r = np.array(cos_h), np.array(cos_r)
        L.append(f"  epicenter's move at the saddle along the hint direction: mean cosine {cos_h.mean():+.2f} "
                 f"(random direction {cos_r.mean():+.2f}); cosine > 0.5 in {100 * np.mean(cos_h > 0.5):.0f}% "
                 f"(random {100 * np.mean(cos_r > 0.5):.0f}%); n = {len(cos_h)}")
    if same_nb:
        L.append(f"  the nearest direction on the same epicenter reaches the same saddle in {100 * np.mean(same_nb):.0f}%, "
                 f"a random direction on it in {100 * np.mean(same_rand):.0f}% (n = {len(same_nb)})")

    # 8. The two minima of the earlier hint search: the new runs against its full-Hessian labels.
    old = Path(OLD_DIR) / ms.VERIFY_DIR
    if old.exists():
        L += ["", "8. the two minima of the earlier hint search: connected saddles <= 1 eV"]
        old_rows = [c for c in read_rows(old / "verify_*.json") if c.get("nneg") == 1 and c.get("connected")
                    and c["barrier"] <= 1.0]
        for t in KEEP:
            if t not in by_tid:
                continue
            m = minima[t]
            O = []
            for c in old_rows:
                if c["tid"] == t and (not O or dist(m, np.array(O), np.array(c["pos"])).min() >= hs.MERGE):
                    O.append(c["pos"])
            new = [points[k]["pos"] for k in by_tid[t] if saddle(k, 1.0)]
            both = sum(dist(m, np.array(O), np.array(p)).min() < hs.MERGE for p in new) if new and O else 0
            L.append(f"  {t}: earlier union {len(O)}, this run {len(new)}, found by both {both}")

    text = "\n".join(L) + "\n"
    if args.stable_only:
        text = (f"Stable minima only: left out {unstable} (a Hessian eigenvalue below -0.05 eV/A^2 at the minimum)\n\n"
                + text)
    (out / ("report_stable.txt" if args.stable_only else "report.txt")).write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"select": select, "rollout": rollout, "sella": sella, "verify": verify, "analyze": analyze}[args.mode](args, out)


if __name__ == "__main__":
    main()
