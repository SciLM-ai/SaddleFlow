"""The released model started from hints instead of from noise (the mp20bat counterpart of LiC's B0).

No training: the released SaddleFlow model is rolled out (5 passes x 4 steps) from the minimum with the
epicenter pushed along a hint, for the same epicenters and directions as the classical search on the plan's
minima, and every landing point is refined by Sella. Four arms:
    hs_spread_n30   the plan's 32 spread directions; the epicenter pushed 0.6 A; 0.3 A noise on every atom
    hs_spread_n05   the same, 0.05 A noise
    hs_mode_n30     the local-mode directions (both signs; mode_search.py's block and seeding), scaled so the
                    most-moving atom moves 0.6 A; 0.3 A noise
    hs_mode_n05     the same, 0.05 A noise
0.3 A is the training noise; 0.05 A keeps the hint clear (narrower than the training noise is safe). The
noise is seeded per start, so the two noise levels of a direction differ only in size.

    rollout  local blocks for the mode directions, then all rollouts (GPU)
    sella    Sella from every landing point, the mode search's caps and stop rules (GPU; --shard/--nshards)
Then mode_search.py verify and analyze add these arms to the comparison.

Soft-mode starts (a quick look before the 30-minimum hint_vs_noise experiment, same definitions):
    soft_rollout  the full Hessian at each minimum (soft_modes.py); 384 starts per minimum, a random soft-mode
                  displacement scaled so the most-moving atom moves 0.6 A plus 0.05 A noise, rolled out (hs_soft);
                  energy and force at every landing point of the noise (model, model_inv), hs_spread_n05 and
                  hs_soft arms and at its null copy, a random soft-mode displacement of the same size (GPU)
    soft_sella    Sella from every hs_soft landing point and from 32 null copies per source arm and minimum
                  (soft_null); the energy cap is raised to the start's energy for a start above it (GPU; sharded)
    soft_report   noise vs hint vs soft at equal numbers of starts, and the null test (CPU), after
                  mode_search.py verify has checked the new points

    python examples/MaterialsSaddles/hint_search/hint_start.py rollout --run-dir <failure-mode run> --out-dir <hint_search dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.geometry import find_mic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hint_search as hs  # noqa: E402
import mode_search as ms  # noqa: E402
from hint_search import read_rows, write_rows  # noqa: E402

PUSH = 0.6
NOISE = {"n30": 0.3, "n05": 0.05}
N_SOFT, N_NULL, SOFT_NOISE = 384, 32, 0.05
GROUPS = {"noise": ("model", "model_inv"), "hint": ("hs_spread_n05",), "soft": ("hs_soft",)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["rollout", "sella", "soft_rollout", "soft_sella", "soft_report"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-soft", type=int, default=N_SOFT, help="soft starts per minimum")
    p.add_argument("--n-null", type=int, default=N_NULL, help="null copies run through Sella per group and minimum")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def rollout(args, out):
    import torch
    from saddleflow import SaddleFlow
    from saddleflow.classical_saddle_search import PESBackend, local_hessian
    torch.backends.cuda.matmul.allow_tf32 = False
    plan, rows, minima = hs.plan_minima(args)
    plan_rows = {r["tid"]: r for r in rows}
    calc = hs.calculator(args.device)
    starts, meta = [], []
    for tid, e in ms.epicenters(rows):
        m = minima[tid]
        # Spread directions: the plan's, on the epicenter only.
        dirs = []
        for j, d in enumerate(plan_rows[tid]["directions"]):
            u = np.zeros_like(m.positions); u[e] = d
            dirs.append(("spread", j, u, 0))
        # Mode directions: mode_search.py's block and seeding, so they are the same directions.
        pes = PESBackend(m.copy(), calc)
        x_min = pes.dofs.pack(m.positions)
        cluster = np.argsort(find_mic(m.positions - m.positions[e], m.cell, m.pbc)[1])[:ms.CLUSTER]
        H, coords, block_calls = local_hessian(pes, x_min, cluster)
        for k, d in enumerate(ms.mode_directions(H, np.random.default_rng([tid, e]))):
            u = np.zeros(len(x_min)); u[coords] = d["u"]
            dirs.append(("mode", k, pes.dofs.expand(u) - pes.dofs.expand(np.zeros_like(u)), block_calls))
        n_mode = sum(d[0] == "mode" for d in dirs)
        for kind, k, u, calls in dirs:
            u = PUSH * u / np.linalg.norm(u, axis=1).max()           # the most-moving atom moves PUSH
            for tag, sigma in NOISE.items():
                rng = np.random.default_rng([tid, e, k, kind == "mode"])
                s = m.copy()
                s.set_positions(m.positions + u + sigma * rng.normal(size=m.positions.shape), apply_constraint=False)
                starts.append(s)
                meta.append(dict(id=f"hs_{kind}_{tag}:{tid}:{e}:{k}:0", arm=f"hs_{kind}_{tag}", tid=tid, epicenter=int(e),
                                 element=m.get_chemical_symbols()[e], direction=k, kind=kind, noise=sigma, repeat=0,
                                 block_calls=calls if kind == "mode" else 0, block_runs=2 * max(n_mode, 1)))
        print(f"  minimum {tid} epicenter {e}: {len(dirs)} directions", flush=True)
    sf = SaddleFlow.from_pretrained(plan["model"], device=args.device)
    props = sf.propose(starts, n=1, sigma=0.0, task_name="omat", batch_size=args.batch_size)
    rows_out = [dict(mt, landing=np.round(p[0].positions, 5).tolist()) for mt, p in zip(meta, props)]
    (out / "hint_start").mkdir(exist_ok=True)
    write_rows(out / "hint_start" / "landings.json", rows_out)
    print(f"{len(rows_out)} rollouts")


def sella(args, out):
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    plan, rows, minima = hs.plan_minima(args)
    landings = json.loads((out / "hint_start" / "landings.json").read_text())
    path = out / "hint_start" / f"runs_{args.shard}.json"
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in done_rows}
    calc = hs.calculator(args.device)
    e_min = {}
    for job in landings[args.shard::args.nshards]:
        if job["id"] in done:
            continue
        m = minima[job["tid"]]
        pes = PESBackend(m.copy(), calc)
        if job["tid"] not in e_min:
            e_min[job["tid"]] = pes.energy(pes.dofs.pack(m.positions))
        row = {k: v for k, v in job.items() if k != "landing"}
        try:
            x0 = pes.dofs.pack(np.array(job["landing"]))
            before = pes.force_calls
            res = run_sella(pes, x0, fmax_tol=hs.FMAX, max_steps=hs.SELLA_STEPS, max_force_calls=before + hs.SELLA_CALLS,
                            delta0=0.1, stop=ms.stop_rule(m, e_min[job["tid"]], pes.dofs), return_hessian=True)
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            row.update(status=res["status"], sella_calls=pes.force_calls - before, sella_steps=res["steps"], fmax=fmax,
                       conv=bool(fmax <= hs.FMAX and res["status"] == "converged"), pos=np.round(res["positions"], 5).tolist())
            if "hessian_evecs" in res:
                row["hessian_start"] = np.round(res["hessian_evecs"][:, :4], 5).tolist()
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"shard {args.shard}: {len(done_rows)} runs")


def energy_force(m, pos, calc):
    a = m.copy(); a.set_positions(pos, apply_constraint=False); a.calc = calc
    return float(a.get_potential_energy()), float(np.linalg.norm(a.get_forces(), axis=1).max())


def all_landings(out):
    """Every model landing point of the noise, hint and soft groups: id -> (arm, tid, positions)."""
    land = {f"{r['arm']}:{r['tid']}:{r['draw']}": (r["arm"], r["tid"], r["pos"])
            for r in json.loads((out / "model_landings.json").read_text())}
    land.update({r["id"]: (r["arm"], r["tid"], r["landing"]) for r in json.loads((out / "hint_start" / "landings.json").read_text())
                 if r["arm"] == "hs_spread_n05"})
    soft = out / "hint_start" / "soft_landings.json"
    if soft.exists():
        land.update({r["id"]: (r["arm"], r["tid"], r["landing"]) for r in json.loads(soft.read_text())})
    return land


def soft_rollout(args, out):
    import torch
    from saddleflow import SaddleFlow
    from saddleflow.classical_saddle_search import PESBackend
    from soft_modes import soft_displacement, soft_modes
    torch.backends.cuda.matmul.allow_tf32 = False
    plan, rows, minima = hs.plan_minima(args)
    calc = hs.calculator(args.device)
    path = out / "hint_start" / "soft_landings.json"
    modes = {}
    if not path.exists():
        starts, meta = [], []
        for tid, m in minima.items():
            pes = PESBackend(m.copy(), calc)
            t0 = time.perf_counter()
            values, vectors, calls = soft_modes(pes, pes.dofs.pack(m.positions))
            modes[tid] = (values, vectors, calls)
            hess_seconds = time.perf_counter() - t0
            for j in range(args.n_soft):
                g = np.random.default_rng([tid, 6, j])
                s = m.copy()
                s.set_positions(m.positions + soft_displacement(values, vectors, g, PUSH) + SOFT_NOISE * g.normal(size=m.positions.shape),
                                apply_constraint=False)
                starts.append(s)
                meta.append(dict(id=f"hs_soft:{tid}:{j}:0", arm="hs_soft", tid=tid, draw=j, repeat=0, hessian_calls=calls,
                                 hessian_seconds=hess_seconds,
                                 start=np.round(s.positions, 5).tolist()))
            print(f"  minimum {tid}: Hessian {calls} calls, softest modes {np.round(values[:4], 3).tolist()} eV/A^2", flush=True)
        sf = SaddleFlow.from_pretrained(plan["model"], device=args.device)
        t0 = time.perf_counter()
        props = sf.propose(starts, n=1, sigma=0.0, task_name="omat", batch_size=args.batch_size)
        per_start = (time.perf_counter() - t0) / len(starts)
        write_rows(path, [dict(mt, rollout_seconds=per_start, landing=np.round(p[0].positions, 5).tolist())
                          for mt, p in zip(meta, props)])
    # Energy and force at every landing point and at its null copy.
    rows_out = []
    for tid, m in minima.items():
        if tid not in modes:
            pes = PESBackend(m.copy(), calc)
            modes[tid] = soft_modes(pes, pes.dofs.pack(m.positions))
        values, vectors, _ = modes[tid]
        e_min, _ = energy_force(m, m.positions, calc)
        for k, (lid, (arm, t, pos)) in enumerate(sorted(all_landings(out).items())):
            if t != tid:
                continue
            pos = np.array(pos)
            size = float(hs.max_mic_untranslated(m, hs.at(m, pos)))
            null = m.positions + soft_displacement(values, vectors, np.random.default_rng([tid, 4, k]), size)
            e_l, f_l = energy_force(m, pos, calc); e_n, f_n = energy_force(m, null, calc)
            rows_out.append(dict(id=lid, arm=arm, tid=tid, size=size, e_landing=e_l - e_min, f_landing=f_l,
                                 e_null=e_n - e_min, f_null=f_n, null=np.round(null, 5).tolist()))
        print(f"  minimum {tid}: null copies done", flush=True)
    write_rows(out / "hint_start" / "null_points.json", rows_out)
    print(f"{len(rows_out)} landing points with null copies")


def soft_sella(args, out):
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    plan, rows, minima = hs.plan_minima(args)
    jobs = [(r["id"], "hs_soft", r["tid"], r["landing"], dict(draw=r["draw"], repeat=0))
            for r in json.loads((out / "hint_start" / "soft_landings.json").read_text())]
    nulls = json.loads((out / "hint_start" / "null_points.json").read_text())
    for tid in minima:
        for group, arms in GROUPS.items():
            src = sorted((r for r in nulls if r["tid"] == tid and r["arm"] in arms), key=lambda r: r["id"])
            g = np.random.default_rng([tid, 7, list(GROUPS).index(group)])
            for i in np.sort(g.choice(len(src), min(args.n_null, len(src)), replace=False)):
                r = src[i]
                jobs.append((f"soft_null:{tid}:{r['id']}", "soft_null", tid, r["null"], dict(landing_id=r["id"], group=group, repeat=0)))
    path = out / "hint_start" / f"runs_soft_{args.shard}.json"
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in read_rows(out / "hint_start" / "runs_soft_*.json")}
    calc = hs.calculator(args.device)
    e_min = {}
    for rid, arm, tid, start, extra in jobs[args.shard::args.nshards]:
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
            row.update(start_rise=e_start - e_min[tid], status=res["status"], sella_calls=pes.force_calls - before,
                       sella_steps=res["steps"], fmax=fmax, conv=bool(fmax <= hs.FMAX and res["status"] == "converged"),
                       pos=np.round(res["positions"], 5).tolist())
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"shard {args.shard}: {len(done_rows)} runs")


def soft_report(args, out):
    plan, rows, minima = hs.plan_minima(args)
    runs = ms.all_runs(out)
    land = all_landings(out)
    nulls = {r["id"]: r for r in json.loads((out / "hint_start" / "null_points.json").read_text())}

    def dist(m, P, q):
        d = np.asarray(P) - np.asarray(q)[None]
        A = m.cell.array; f = d @ np.linalg.inv(A); d = (f - np.round(f)) @ A
        d -= d.mean(-2, keepdims=True)
        return np.linalg.norm(d, axis=-1).max(-1)

    points = []
    for c in read_rows(out / ms.VERIFY_DIR / "verify_*.json"):
        if c["tid"] not in minima:
            continue
        P = [p["pos"] for p in points if p["tid"] == c["tid"]]
        if not P or dist(minima[c["tid"]], P, c["pos"]).min() >= hs.MERGE:
            points.append(c)
    by_tid = {t: [k for k, p in enumerate(points) if p["tid"] == t] for t in minima}
    owner = {}
    for rid, r in runs.items():
        if r.get("conv") and r["tid"] in minima and by_tid[r["tid"]]:
            ks = by_tid[r["tid"]]
            d = dist(minima[r["tid"]], [points[k]["pos"] for k in ks], r["pos"])
            if d.min() < hs.MERGE:
                owner[rid] = ks[int(d.argmin())]
    saddle = lambda k, cut: points[k].get("nneg") == 1 and points[k].get("connected") and points[k]["barrier"] <= cut
    hit = lambda rid, cut=1.0: rid in owner and saddle(owner[rid], cut)
    first = lambda r: r.get("repeat", 0) == 0
    group_runs = {g: [rid for rid, r in runs.items() if ms.arm_of(r) in arms and first(r) and r["tid"] in minima]
                  for g, arms in GROUPS.items()}
    L = ["Soft-mode starts on the two hint-search minima: noise vs hint (spread, 0.05 A) vs soft, with the model",
         "=" * 104, "Coverage is counted against the union of every arm ever run on these minima (classical included).", ""]
    L += ["connected saddles found (share of the union):",
          "group   starts  converged%   <= 0.3 eV       <= 1.0 eV      Sella calls/run  calls per saddle (<= 1 eV)"]
    union = {cut: [k for k in range(len(points)) if saddle(k, cut)] for cut in (0.3, 1.0)}
    for g, ids in group_runs.items():
        found = {cut: {owner[i] for i in ids if hit(i, cut)} for cut in (0.3, 1.0)}
        calls = sum(runs[i].get("sella_calls", 0) for i in ids)
        L.append(f"{g:6s} {len(ids):7d}  {100 * np.mean([runs[i].get('conv', False) for i in ids]):9.1f}   "
                 + "   ".join(f"{len(found[c]):3d} ({100 * len(found[c]) / max(len(union[c]), 1):4.1f}%)" for c in (0.3, 1.0))
                 + f"   {calls / max(len(ids), 1):12.0f}  {calls / max(len(found[1.0]), 1):10.0f}")
    L.append(f"union  {len(union[0.3]):26d}          {len(union[1.0]):3d}")
    soft_only = {owner[i] for i in group_runs["soft"] if hit(i)} - {owner[i] for g in ("noise", "hint") for i in group_runs[g] if hit(i)}
    L.append(f"  connected saddles (<= 1 eV) found by soft starts and by neither noise nor hint starts: {len(soft_only)}")

    L += ["", "distinct connected saddles (<= 1 eV) against starts, per minimum summed (starts in random order):"]
    rng = np.random.default_rng(0)
    marks = [64, 128, 256, 384, 512, 768]
    for g, ids in group_runs.items():
        total = np.zeros(len(marks))
        for t in minima:
            mine = [i for i in ids if runs[i]["tid"] == t]
            mine = [mine[i] for i in rng.permutation(len(mine))]
            total += [len({owner[i] for i in mine[:n] if hit(i)}) if n <= len(mine) else np.nan for n in marks]
        L.append(f"  {g:6s} " + "  ".join(f"{n}: {v:4.0f}" for n, v in zip(marks, total) if not np.isnan(v)))

    # Cost to find distinct saddles in force calls (all groups recorded their Sella calls; the soft group is also
    # charged its Hessian). Wall time exists only for the soft runs, so it is reported for them alone.
    soft_rows = json.loads((out / "hint_start" / "soft_landings.json").read_text())
    hess = {r["tid"]: r for r in soft_rows}
    budgets = [5000, 10000, 20000, 40000, 80000]
    L += ["", "distinct connected saddles (<= 1 eV) within force-call budgets, summed over the two minima "
              "(runs in random order; soft charged its Hessian):",
          "group   " + " ".join(f"{b // 1000:>5d}k" for b in budgets) + "   calls to reach 5 / 10 / 20 (summed budget)"]
    rng = np.random.default_rng(3)
    for g, ids in group_runs.items():
        tot, reach = np.zeros(len(budgets)), []
        curves = []
        for t in minima:
            mine = [ids[i] for i in rng.permutation(len(ids)) if runs[ids[i]]["tid"] == t]
            c0 = hess[t]["hessian_calls"] if g == "soft" and t in hess else 0
            c = c0 + np.cumsum([runs[i].get("sella_calls", 0) for i in mine])
            seen, n = set(), []
            for i in mine:
                if hit(i):
                    seen.add(owner[i])
                n.append(len(seen))
            n = np.array(n)
            tot += [n[c <= b][-1] if np.any(c <= b) else 0 for b in budgets]
            curves.append((c, n))
        # Interleave the minima's budgets equally: total saddles at a total budget split evenly.
        for k in (5, 10, 20):
            best = next((B for B in np.arange(1000, 400001, 1000)
                         if sum(n[c <= B / len(curves)][-1] if np.any(c <= B / len(curves)) else 0 for c, n in curves) >= k), None)
            reach.append(f"{best / 1000:.0f}k" if best else "-")
        L.append(f"{g:6s} " + " ".join(f"{v:6.0f}" for v in tot) + "   " + " / ".join(reach))
    secs = [r["seconds"] for r in runs.values() if ms.arm_of(r) == "hs_soft" and "seconds" in r]
    if secs:
        L.append(f"  soft wall time: Sella {np.median(secs):.1f} s per run (median, 8 workers sharing a GPU), rollout "
                 f"{np.median([r['rollout_seconds'] for r in soft_rows]):.2f} s per start, Hessian "
                 + ", ".join(f"{t}: {r['hessian_calls']} calls / {r['hessian_seconds']:.0f} s" for t, r in hess.items()))

    L += ["", "landing accuracy (converged runs): how far Sella moved the landing point",
          "group   runs  moved p50 / p90 (A)  within 0.1 A  Sella calls p50"]
    for g, ids in group_runs.items():
        mv, c = [], []
        for i in ids:
            r = runs[i]
            if r.get("conv") and i in land:
                mv.append(dist(minima[r["tid"]], [r["pos"]], land[i][2])[0]); c.append(r["sella_calls"])
        if mv:
            L.append(f"{g:6s} {len(mv):6d}  {np.median(mv):6.2f} / {np.percentile(mv, 90):5.2f}       "
                     f"{100 * np.mean(np.array(mv) < 0.1):9.0f}%  {np.median(c):14.0f}")

    L += ["", "null test: each landing point against a random soft-mode displacement of the same size",
          "group   landings  size p50 (A)  energy above min p50: landing / null (eV)  largest force p50: landing / null (eV/A)"]
    for g, arms in GROUPS.items():
        ls = [r for r in nulls.values() if r["arm"] in arms and r["tid"] in minima]
        if ls:
            q = lambda k: np.median([r[k] for r in ls])
            L.append(f"{g:6s} {len(ls):9d}  {q('size'):11.2f}   {q('e_landing'):18.2f} / {q('e_null'):5.2f}"
                     f"{'':22s}{q('f_landing'):6.2f} / {q('f_null'):5.2f}")
    L.append("  Sella from the landing point against Sella from its null copy (the same landings):")
    nr = [r for r in runs.values() if ms.arm_of(r) == "soft_null" and r["tid"] in minima]
    for g in GROUPS:
        pairs = [(runs.get(r["landing_id"]), r) for r in nr if r["group"] == g]
        pairs = [(a, b) for a, b in pairs if a is not None]
        for name, k in (("landing", 0), ("null", 1)):
            rs = [p[k] for p in pairs]
            if rs:
                L.append(f"    {g:6s} {name:8s} n {len(rs):3d}: converged {100 * np.mean([r.get('conv', False) for r in rs]):5.1f}%, "
                         f"reached a connected saddle (<= 1 eV) {100 * np.mean([hit(r['id']) for r in rs]):5.1f}%, "
                         f"distinct {len({owner[r['id']] for r in rs if hit(r['id'])}):3d}, Sella calls p50 "
                         f"{np.median([r.get('sella_calls', 0) for r in rs]):4.0f}")
    text = "\n".join(L) + "\n"
    (out / "hint_start" / "soft_report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    {"rollout": rollout, "sella": sella, "soft_rollout": soft_rollout, "soft_sella": soft_sella,
     "soft_report": soft_report}[args.mode](args, out)


if __name__ == "__main__":
    main()
