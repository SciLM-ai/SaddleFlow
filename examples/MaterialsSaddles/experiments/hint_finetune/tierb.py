"""Tier B: hint labels on ~200 mp20bat training minima, built offline before the fine-tune.

Each stage is its own job (run_tierb.sh, chained by submit_tierb.sh); every stage skips work already written.

    select    (CPU) Read the mp20bat train split; candidates are the reactants and products of 8 to 40 atoms, one
              per element set; rank 400 of them by farthest-point sampling on composition, mobile ion, anion
              family and size. The train/test split is by element set, so none of these chemistries is in the
              test split.
    screen    (GPU; --shard/--nshards) Largest force and the full Hessian (forward differences, translations
              removed) at every ranked candidate. A candidate is a minimum under UMA-S when its lowest eigenvalue
              is >= -0.05 eV/A^2 and its largest force <= 0.05 eV/A. The first 215 such minima in rank order are
              accepted (about 20% of minima are not minima under UMA-S; the margin keeps ~200 after verification)
              and split into 4 groups balanced by atoms^1.5. The Hessian's modes are kept for the soft pushes.
    discover  (GPU; --group, --shard/--nshards) Sella straight from 128 starts per minimum, no model:
                atom   64 single-atom pushes drawn from the every-atom grid (16 Fibonacci directions per atom):
                       the atom moved 0.6 A along the direction, 0.05 A noise on every atom
                soft   64 soft-mode pushes: a random combination of the minimum's vibration modes, weighted by
                       1/sqrt(eigenvalue), scaled so the most-moving atom moves 0.6 A, plus 0.05 A noise
              Sella: UMA-S-1.2 omat, fmax 0.01, <= 300 steps, <= 600 calls, delta0 0.1; stopped beyond 2.5 A from
              the minimum or above both the minimum + 1.5 eV and the start's own energy.
    verify    (GPU; --group, --shard/--nshards) Converged points merged within 0.1 A per minimum; each checked
              once: energy above the minimum (above 1.5 eV out of scope), full central-difference Hessian (index =
              eigenvalues below -0.01), and for index 1 connectivity (push along the negative mode, LBFGS to
              0.005 eV/A, back within 0.2 A of the minimum). Same code as the 30-minimum experiment.
    labels    (CPU) Connected index-1 saddles <= 1 eV, merged into regions (hint_labels.merge: 0.15 A and 50 meV,
              most-reached first), 20% of regions per minimum held out (minima with at least 3 regions), and one
              hint row per region and atom moving >= 0.5x the largest move and >= 0.2 A. Writes labels.json and
              reports the merge-radius check (do points 0.1-0.25 A apart look like the same saddle?).
    checks    (CPU) The collision and coverage checks on the label table itself (hint_grid.py's measures).

    python examples/MaterialsSaddles/experiments/hint_finetune/tierb.py <mode> --out-dir <dir>
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hint_vs_noise"))
sys.path.insert(0, str(HERE.parent / "canonical_hints"))
import hint_labels as hl  # noqa: E402
import hint_vs_noise as hvn  # noqa: E402
from hint_vs_noise import hs, ms, read_rows, write_rows  # noqa: E402
from soft_modes import soft_displacement, soft_modes  # noqa: E402

TRAIN_SHARDS = "/scratch/gqs7zr/saddleflow/MaterialsSaddles/mp20bat/train/mp20bat_train_*.aselmdb"
N_RANKED, N_ACCEPT, N_GROUPS = 400, 215, 4
MIN_ATOMS, MAX_ATOMS = 8, 40
N_ATOM, N_SOFT, GRID_K, PUSH, NOISE = 64, 64, 16, 0.6, 0.05
CUTOFF, HOLD_OUT = 1.0, 0.2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["select", "screen", "discover", "verify", "labels", "checks"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--group", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="smoke tests: use only the first N minima (screen: candidates)")
    p.add_argument("--scale", type=float, default=1.0, help="smoke tests: scale the number of Sella starts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def candidates(out):
    return {int(a.info["tid"]): a for a in read(out / "candidates.traj", index=":")}


def accepted(out, limit=0):
    """{tid: group} for the first N_ACCEPT stable candidates in rank order (deterministic for every stage)."""
    rows = sorted(read_rows(out / "screen" / "screen_*.json"), key=lambda r: r["tid"])
    stable = [r for r in rows if r["stable"]][:N_ACCEPT]
    if limit:
        stable = stable[:limit]
    load, group = [0.0] * N_GROUPS, {}
    for r in sorted(stable, key=lambda r: -r["natoms"]):
        g = int(np.argmin(load)); group[r["tid"]] = g; load[g] += r["natoms"] ** 1.5
    return group


def group_minima(args, out):
    cands, group = candidates(out), accepted(out, args.limit)
    return {t: cands[t] for t, g in sorted(group.items()) if g == args.group}


# ---------------------------------------------------------------- select
def select(args, out):
    from ase.db import connect
    rng = np.random.default_rng(args.seed)
    # Rows are consecutive [R, S, P] triplets (MaterialsSaddlesDataset's convention); only R and P are kept.
    by_set, n_triplets = {}, 0
    for f in sorted(glob.glob(TRAIN_SHARDS)):
        with connect(f) as db:
            for i, row in enumerate(db.select()):
                if i % 3 == 1:
                    continue
                if i % 3 == 0:
                    n_triplets += 1
                if MIN_ATOMS <= row.natoms <= MAX_ATOMS:
                    m = row.toatoms()
                    by_set.setdefault(tuple(sorted(set(m.get_chemical_symbols()))), []).append(
                        (n_triplets - 1, "R" if i % 3 == 0 else "P", m))
    pool = [v[int(rng.integers(len(v)))] for v in by_set.values()]
    elements = sorted({e for _, _, m in pool for e in m.get_chemical_symbols()})
    mobiles, families = sorted(hs.MOBILE), [n for n, _ in hvn.ANIONS] + ["other"]

    def feature(m):
        sym = m.get_chemical_symbols()
        comp = np.array([sym.count(e) for e in elements], float) / len(sym)
        mob = np.array([e in sym for e in mobiles], float); mob /= max(mob.sum(), 1)
        fam = np.array([hvn.anion_family(sym) == f for f in families], float)
        return np.concatenate([comp, mob, fam, [len(sym) / MAX_ATOMS]])

    F = np.array([feature(m) for _, _, m in pool])
    first = int(rng.integers(len(pool)))
    picked, gap = [first], np.linalg.norm(F - F[first], axis=1)
    while len(picked) < min(N_RANKED, len(pool)):
        k = int(np.argmax(gap)); picked.append(k)
        gap = np.minimum(gap, np.linalg.norm(F - F[k], axis=1))
    out_atoms = []
    for rank, k in enumerate(picked):
        tid3, side, m = pool[k]
        a = m.copy(); a.info = dict(tid=rank, triplet=tid3, side=side, formula=m.get_chemical_formula())
        a.set_constraint()
        out_atoms.append(a)
    write(out / "candidates.traj", out_atoms)
    print(f"{n_triplets} triplets, {len(by_set)} element sets with {MIN_ATOMS}-{MAX_ATOMS}-atom minima; "
          f"ranked {len(out_atoms)}")


# ---------------------------------------------------------------- screen
def screen(args, out):
    from saddleflow.classical_saddle_search import PESBackend
    cands = candidates(out)
    tids = sorted(cands)[:args.limit or None]
    path = out / "screen" / f"screen_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["tid"] for r in read_rows(path.parent / "screen_*.json")}
    calc = hs.calculator(args.device)
    for tid in tids[args.shard::args.nshards]:
        if tid in done:
            continue
        m = cands[tid]
        pes = PESBackend(m.copy(), calc)
        x = pes.dofs.pack(m.positions)
        _, fmax = pes.dofs.force_metrics(pes.forces(x))
        values, vectors, calls = soft_modes(pes, x)
        stable = bool(values[0] >= -0.05 and fmax <= 0.05)
        if stable:
            np.savez(path.parent / f"modes_{tid}.npz", values=values, vectors=vectors)
        done_rows.append(dict(tid=tid, natoms=len(m), formula=m.info["formula"], fmax=fmax,
                              eigs=np.round(values[:4], 4).tolist(), hessian_calls=calls, stable=stable))
        write_rows(path, done_rows)
    print(f"shard {args.shard}: {len(done_rows)} screened, {sum(r['stable'] for r in done_rows)} stable")


# ---------------------------------------------------------------- discover
def starts(args, tid, m):
    """[(id, arm, positions, extra)] for one minimum: single-atom pushes and soft-mode pushes."""
    n_atom, n_soft = max(1, int(round(N_ATOM * args.scale))), max(1, int(round(N_SOFT * args.scale)))
    grid = [(a, k, d) for a in range(len(m)) for k, d in enumerate(hs.fibonacci(GRID_K, np.random.default_rng([tid, 10, a])))]
    pick = np.random.default_rng([tid, 11]).choice(len(grid), min(n_atom, len(grid)), replace=False)
    out = []
    for j, i in enumerate(sorted(pick)):
        a, k, d = grid[i]
        pos = m.positions + NOISE * np.random.default_rng([tid, 12, j]).normal(size=m.positions.shape)
        pos[a] += PUSH * d
        out.append((f"atom:{tid}:{a}:{k}", "atom", pos, dict(atom=int(a), direction=np.round(d, 6).tolist())))
    modes = np.load(out_dir_of(args) / "screen" / f"modes_{tid}.npz")
    for j in range(n_soft):
        g = np.random.default_rng([tid, 13, j])
        pos = m.positions + soft_displacement(modes["values"], modes["vectors"], g, PUSH) + NOISE * g.normal(size=m.positions.shape)
        out.append((f"soft:{tid}:{j}", "soft", pos, {}))
    return out


def out_dir_of(args):
    return Path(args.out_dir)


def discover(args, out):
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    minima = group_minima(args, out)
    jobs = [(t, *s) for t, m in minima.items() for s in starts(args, t, m)]
    path = out / "runs" / f"runs_{args.group}_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in read_rows(path.parent / f"runs_{args.group}_*.json")}
    calc = hs.calculator(args.device)
    e_min = {}
    for tid, rid, arm, start, extra in jobs[args.shard::args.nshards]:
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
            seconds = time.perf_counter() - t0
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            final = np.asarray(res["positions"])
            row.update(start_rise=e_start - e_min[tid], status=res["status"], sella_calls=pes.force_calls - before,
                       sella_steps=res["steps"], seconds=seconds, fmax=fmax,
                       conv=bool(fmax <= hs.FMAX and res["status"] == "converged"),
                       moved=float(hl.saddle_distance(m, final[None], np.array(start))[0]), pos=np.round(final, 5).tolist())
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"group {args.group} shard {args.shard}: {len(done_rows)} runs")


# ---------------------------------------------------------------- verify
def verify(args, out):
    from saddleflow.classical_saddle_search import PESBackend
    minima = group_minima(args, out)
    runs = [r for r in read_rows(out / "runs" / f"runs_{args.group}_*.json") if r["tid"] in minima]
    groups = hvn.distinct_points(runs, minima)
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
            row = dict(g, **hvn.check(pes, m, pes.dofs.pack(np.array(g["pos"])), e_min[g["tid"]], calc))
        except Exception as exc:
            row = dict(g, status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"group {args.group} shard {args.shard}: {len(done_rows)} distinct points checked")


# ---------------------------------------------------------------- labels
def verified_points(out, minima):
    """Distinct verified points per minimum (one row per point even if two shards checked it)."""
    points = {}
    for c in read_rows(out / "verify" / "verify_*.json"):
        if c["tid"] not in minima or c.get("status") == "error":
            continue
        P = points.setdefault(c["tid"], [])
        if not P or hl.saddle_distance(minima[c["tid"]], np.array([p["pos"] for p in P]), np.array(c["pos"])).min() >= hs.MERGE:
            P.append(c)
    return points


def labels(args, out):
    cands, group = candidates(out), accepted(out, args.limit)
    minima = {t: cands[t] for t in sorted(group)}
    points = verified_points(out, minima)
    if not points:
        raise SystemExit("no verified points: run discover and verify first (labels.json not written)")
    rng = np.random.default_rng(args.seed)
    L = ["Tier B labels", "=" * 100]

    # The merge-radius check: pairs of connected index-1 points closer than 0.25 A (they are > 0.1 A apart, the
    # verification merge). Same saddle loosely converged should differ little in energy and lowest eigenvalue.
    close = []
    for t, P in points.items():
        S = [p for p in P if p.get("nneg") == 1 and p.get("connected") and p["barrier"] <= CUTOFF]
        for i in range(len(S)):
            d = hl.saddle_distance(minima[t], np.array([s["pos"] for s in S]), np.array(S[i]["pos"]))
            for j in range(i + 1, len(S)):
                if d[j] < 0.25:
                    close.append((d[j], abs(S[i]["barrier"] - S[j]["barrier"]), abs(S[i]["eigs"][0] - S[j]["eigs"][0])))
    L += ["", "merge-radius check: pairs of connected saddles closer than 0.25 A (by distance band)",
          "   band (A)      pairs   |dE| p50 (meV)   |dE| <= 50 meV   |d lowest eigenvalue| p50 (eV/A^2)"]
    close = np.array(close) if close else np.zeros((0, 3))
    for lo, hi in ((0.1, 0.15), (0.15, 0.2), (0.2, 0.25)):
        k = (close[:, 0] >= lo) & (close[:, 0] < hi)
        if k.any():
            L.append(f"   {lo:.2f}-{hi:.2f}   {k.sum():8d}   {1000 * np.median(close[k, 1]):12.1f}   {100 * np.mean(close[k, 1] <= hl.MERGE_DE):12.0f}%"
                     f"   {np.median(close[k, 2]):12.3f}")

    entries, n_rows, n_held = [], 0, 0
    for t, m in minima.items():
        P = points.get(t, [])
        S = [p for p in P if p.get("nneg") == 1 and p.get("connected") and p["barrier"] <= CUTOFF]
        if not S:
            continue
        gid, reps = hl.merge(m, [s["pos"] for s in S], [s["barrier"] for s in S], [len(s["members"]) for s in S])
        n = len(reps)
        held = set(rng.choice(n, max(1, int(round(HOLD_OUT * n))), replace=False).tolist()) if n >= 3 else set()
        regions = []
        for k, r in enumerate(reps):
            s = S[r]
            hints = hl.make_hints(m, s["pos"])
            un = m.positions + hl.displacement(m, s["pos"])     # unwrapped to the minimum, overall shift removed
            regions.append(dict(region=k, held_out=k in held, barrier=s["barrier"], members=int(np.sum(gid == k)),
                                reached=int(sum(len(S[i]["members"]) for i in np.flatnonzero(gid == k))),
                                saddle=np.round(un, 5).tolist(),
                                hints=[dict(atom=a, direction=np.round(u, 6).tolist(), move=round(mv, 4)) for a, u, mv in hints]))
            n_rows += len(hints) * (k not in held); n_held += len(hints) * (k in held)
        entries.append(dict(tid=t, formula=m.get_chemical_formula(), natoms=len(m), numbers=m.numbers.tolist(),
                            cell=m.cell.array.round(6).tolist(), positions=np.round(m.positions, 6).tolist(),
                            triplet=int(m.info["triplet"]), side=m.info["side"], regions=regions))
    (out / "labels.json").write_text(json.dumps(dict(settings=dict(cutoff=CUTOFF, merge_radius=hl.MERGE_RADIUS,
                                                                   merge_de=hl.MERGE_DE, hint_thr=hl.HINT_THR,
                                                                   hint_min_move=hl.HINT_MIN_MOVE, hold_out=HOLD_OUT),
                                                     minima=entries)))
    nreg = [len(e["regions"]) for e in entries]
    L += ["", f"accepted minima {len(minima)}; with at least one connected saddle <= {CUTOFF} eV: {len(entries)}",
          f"connected saddles {sum(len([p for p in points.get(t, []) if p.get('nneg') == 1 and p.get('connected') and p['barrier'] <= CUTOFF]) for t in minima)}"
          f" -> regions {sum(nreg)} (per minimum p10 / p50 / p90 {np.percentile(nreg, 10):.0f} / {np.median(nreg):.0f} / "
          f"{np.percentile(nreg, 90):.0f}); held-out regions {sum(r['held_out'] for e in entries for r in e['regions'])}",
          f"hint rows: training {n_rows}, held out {n_held}"]
    text = "\n".join(L) + "\n"
    (out / "labels_report.txt").write_text(text)
    print(text)


# ---------------------------------------------------------------- checks
def checks(args, out):
    import hint_grid as hg
    from ase import Atoms
    data = json.loads((out / "labels.json").read_text())["minima"]
    rng = np.random.default_rng(1)
    L = ["Collision and coverage checks on the Tier B label table", "=" * 100, ""]
    for subset in ("all regions", "training regions only"):
        for theta, K in ((30, 16), (20, 32)):
            cov, crowd, need = [], [], []
            for e in data:
                regs = [r for r in e["regions"] if subset == "all regions" or not r["held_out"]]
                if not regs:
                    continue
                m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
                D = [hl.displacement(m, r["saddle"]) for r in regs]
                P = np.array([r["saddle"] for r in regs])
                apart = np.array([hl.saddle_distance(m, P, p) > 0.3 for p in P]); np.fill_diagonal(apart, False)
                lab = [(k, h["atom"], np.array(h["direction"])) for k, r in enumerate(regs) for h in r["hints"]]
                c, nd, cr, _ = hg.evaluate(m, D, apart, lab, list(range(len(m))), K, theta, rng)
                cov.append(c); crowd.append(cr); need.append(nd)
            crowd = np.concatenate(crowd) if crowd else np.zeros(0)
            need = np.array(need, float)
            L.append(f"{subset:22s} theta {theta:2d}, K {K:2d}: coverage median {100 * np.median(cov):5.1f}% (p10 {100 * np.percentile(cov, 10):5.1f}%), "
                     f"hints to 90% median {np.nanmedian(need):5.0f} ({np.sum(~np.isnan(need))} of {len(need)} minima reach it), "
                     f"least crowded cone <= 3: {100 * np.mean(crowd <= 3):5.1f}%, <= 10: {100 * np.mean(crowd <= 10):5.1f}%")
    # Held-out regions whose hint cones (30 degrees) also contain a training region more than 0.3 A away.
    shared, total = 0, 0
    for e in data:
        m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
        train = [r for r in e["regions"] if not r["held_out"]]
        for r in (r for r in e["regions"] if r["held_out"]):
            total += 1
            for h in r["hints"]:
                u = np.array(h["direction"])
                if any(t["hints"] and any(g["atom"] == h["atom"] and np.array(g["direction"]) @ u >= np.cos(np.radians(30))
                                           for g in t["hints"])
                       and hl.saddle_distance(m, np.array([t["saddle"]]), np.array(r["saddle"]))[0] > 0.3 for t in train):
                    shared += 1
                    break
    L += ["", f"held-out regions with a hint cone (30 deg) shared with a training region > 0.3 A away: {shared} of {total} "
              f"(report hint following separately for the {total - shared} that share none)"]
    text = "\n".join(L) + "\n"
    (out / "checks.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"select": select, "screen": screen, "discover": discover, "verify": verify, "labels": labels,
     "checks": checks}[args.mode](args, out)


if __name__ == "__main__":
    main()
