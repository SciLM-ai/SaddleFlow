"""Mode-based search directions and cheaper verification, compared with the first hint-search run.

Reads a hint_search.py output directory (plan.json, search/, model_landings.json) and adds:

    search   Stage 2. For every epicenter of the plan's minima: the dense Hessian block of its 15 nearest
             atoms with the rest fixed (one forward difference per coordinate), its lowest 6 modes, both signs,
             near-degenerate modes (within 10%) replaced by 3 directions spread across their span. Two arms:
               modeA  start = minimum + the mode push that costs 0.1 eV in the block, then Sella with v0 = mode
               modeB  climb_along the mode, then Sella with v0 = mode
             Every start also gets a 0.01 eV random jitter on the cluster (seeded per direction, so repeats
             are identical). Sella: fmax 0.01, <= 300 steps, <= 600 calls, stops when the structure drifts
             more than 2.5 A from the minimum or rises more than 1.5 eV above it. GPU; --shard/--nshards.
    verify   Stage 1, on every converged point of every arm (old and new), merged per minimum within 0.1 A:
             energy above the minimum (points above 1.5 eV are out of scope); the lowest 2 modes by a block
             eigensolver (central differences, translations removed, started from Sella's approximate Hessian
             when the run kept it), <= 300 calls; then, for index 1 only, connectivity: push along the negative
             mode (0.1-0.3 A, sized for a starting force >= 0.1 eV/A) and relax (LBFGS to 0.005 eV/A), stopping as
             soon as it comes within 0.2 A of the minimum; the side pointing toward the minimum first, the other
             only if it did not return; a relaxation that ends within 0.5 A and 10 meV of the minimum also
             counts. A 5% sample of eigenvalues is re-checked by central differences along the found modes. GPU.
    analyze  per arm: outcomes, force calls per run and per connected saddle, connected saddles found at
             barrier cutoffs 0.3 / 0.5 / 1.0 eV against the union of all arms, repeatability; verification cost
             and agreement with the first verification (CPU).

    python examples/MaterialsSaddles/hint_search/mode_search.py <mode> --run-dir <failure-mode run> --out-dir <hint_search dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hint_search as hs  # noqa: E402
from hint_search import at, max_mic_untranslated, read_rows, same, write_rows  # noqa: E402

CLUSTER, N_MODES, DEGENERATE = 15, 6, 0.10
PUSH_EV, JITTER_EV = 0.1, 0.01
DRIFT, MAX_RISE, CUTOFF = 2.5, 1.5, 1.0
MODE_ARMS = ("modeA", "modeB")
# Index labels come from the full central-difference Hessian (6N force calls), written to verify3/. The
# block eigensolver (verify2/) missed a negative mode on 26% of the points the first verification's full
# Hessian had checked, so its labels are kept only for comparison.
VERIFY_DIR = "verify3"
HINT_ARMS = ("hs_spread_n30", "hs_spread_n05", "hs_mode_n30", "hs_mode_n05")    # hint_start.py
SOFT_ARMS = ("hs_soft", "soft_null")                                                # hint_start.py soft_*
OLD_RANDOM = ("nudge0.3", "nudge0.6", "climb")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["search", "verify", "analyze"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=2, help="first N minima of the plan (the two cost-check minima)")
    p.add_argument("--max-directions", type=int, default=0, help="cap directions per epicenter (smoke tests only)")
    p.add_argument("--max-epicenters", type=int, default=0, help="cap epicenters (smoke tests only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="", help="verify: prefix for this job's result files, so jobs never share one")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def epicenters(rows):
    return [(r["tid"], e) for r in rows for e in r["mobile"] + r["others"]]


def mode_directions(H, rng):
    """Lowest modes of a local block, both signs; a near-degenerate set gives 3 directions across its span."""
    values, vectors = np.linalg.eigh(H)
    dirs, i = [], 0
    while i < N_MODES:
        j = i + 1
        while j < len(values) and values[j] - values[j - 1] < DEGENERATE * abs(values[j]):
            j += 1
        span = vectors[:, i:j]
        picks = [span[:, 0]] if j - i == 1 else [span @ (c / np.linalg.norm(c)) for c in rng.normal(size=(3, j - i))]
        for k, u in enumerate(picks):
            for sign in (1, -1):
                dirs.append(dict(mode=i, size=j - i, pick=k, sign=sign, u=sign * u / np.linalg.norm(u)))
        i = j
    return dirs


def energy_scaled(H, u, energy):
    """Amplitude a with 1/2 a^2 u^T H u = energy (u unit), capped at 0.5 A."""
    return float(min(np.sqrt(2 * energy / max(u @ H @ u, 1e-6)), 0.5))


def stop_rule(m, e_min, dofs, e_start=None):
    """Stop above e_min + MAX_RISE (or above e_start, for a start that already lies higher) or beyond DRIFT."""
    cap = max(e_min + MAX_RISE, e_start if e_start is not None else -np.inf)

    def rule(x, energy):
        if energy > cap:
            return "energy"
        if max_mic_untranslated(m, at(m, dofs.expand(x))) > DRIFT:
            return "drift"
    return rule


def search(args, out):
    from saddleflow.classical_saddle_search import PESBackend, climb_along, local_hessian, run_sella
    plan, rows, minima = hs.plan_minima(args)
    path = out / "mode_search" / f"runs_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in read_rows(path.parent / "runs_*.json")}
    calc = hs.calculator(args.device)
    centers = epicenters(rows)[:args.max_epicenters or None]
    for tid, e in centers[args.shard::args.nshards]:
        m = minima[tid]
        pes = PESBackend(m.copy(), calc)
        x_min = pes.dofs.pack(m.positions)
        e_min = pes.energy(x_min)
        d = find_mic(m.positions - m.positions[e], m.cell, m.pbc)[1]
        cluster = np.argsort(d)[:CLUSTER]
        H, coords, block_calls = local_hessian(pes, x_min, cluster)
        rng = np.random.default_rng([tid, e])
        dirs = mode_directions(H, rng)[:args.max_directions or None]
        specs = [(arm, k, 0) for k in range(len(dirs)) for arm in MODE_ARMS]
        specs += [(arm, int(rng.integers(len(dirs))), 1) for arm in MODE_ARMS]      # identical-start repeats
        for arm, k, rep in specs:
            rid = f"{arm}:{tid}:{e}:{k}:{rep}"
            if rid in done:
                continue
            u_block = dirs[k]["u"]
            u = np.zeros_like(x_min); u[coords] = u_block
            jr = np.random.default_rng([tid, e, k])                      # same jitter for both arms and repeats
            jitter = np.zeros_like(x_min); g = jr.normal(size=len(coords))
            jitter[coords] = g * energy_scaled(H, g / np.linalg.norm(g), JITTER_EV) / np.linalg.norm(g)
            row = dict(id=rid, arm=arm, tid=tid, epicenter=int(e), element=m.get_chemical_symbols()[e], direction=k,
                       repeat=rep, mode=dirs[k]["mode"], degenerate=dirs[k]["size"], sign=dirs[k]["sign"],
                       block_calls=block_calls, block_runs=len(specs), eig_block=float(u_block @ H @ u_block))
            try:
                x0 = x_min + jitter
                if arm == "modeA":
                    x0 = x0 + energy_scaled(H, u_block, PUSH_EV) * u
                else:
                    # Push so that the most-moving atom moves 0.05 A per step (u is unit over the cluster).
                    big = float(np.linalg.norm(u.reshape(-1, 3), axis=1).max())
                    c = climb_along(pes, x0, u, push=0.05 / big, max_distance=2.0 / big, max_rise=MAX_RISE)
                    row.update(climb_status=c["status"], climb_calls=c["force_calls"], climb_steps=c["steps"],
                               perp_force_rms=c["perp_force_rms"])
                    if c["status"] != "left_bowl":
                        row.update(status="climb_" + c["status"]); done_rows.append(row)
                        write_rows(path, done_rows); continue
                    x0 = c["x"]
                before = pes.force_calls
                res = run_sella(pes, x0, fmax_tol=hs.FMAX, max_steps=hs.SELLA_STEPS, max_force_calls=before + hs.SELLA_CALLS,
                                delta0=0.1, v0=u, stop=stop_rule(m, e_min, pes.dofs), return_hessian=True)
                pes.max_force_calls = None
                _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
                row.update(status=res["status"], sella_calls=pes.force_calls - before, sella_steps=res["steps"],
                           fmax=fmax, conv=bool(fmax <= hs.FMAX and res["status"] == "converged"),
                           pos=np.round(res["positions"], 5).tolist())
                if "hessian_evecs" in res:
                    row["hessian_start"] = np.round(res["hessian_evecs"][:, :4], 5).tolist()
            except Exception as exc:
                row.update(status="error", err=str(exc)[:200])
            done_rows.append(row)
            write_rows(path, done_rows)
        print(f"  minimum {tid} epicenter {e}: {len(dirs)} directions, block {block_calls} calls", flush=True)
    print(f"shard {args.shard}: {len(done_rows)} runs")


def all_runs(out):
    runs = {r["id"]: dict(r, source="first") for r in read_rows(out / "search" / "runs_*.json")}
    runs.update({r["id"]: dict(r, source="mode") for r in read_rows(out / "mode_search" / "runs_*.json")})
    runs.update({r["id"]: dict(r, source="hintstart") for r in read_rows(out / "hint_start" / "runs_*.json")})
    return runs


def arm_of(run):
    return "all_atoms" if run.get("all_atoms") else run["id"].split(":")[0]


def groups_v2(out, minima):
    """Distinct converged points per minimum over every arm, merged within 0.1 A, in a fixed order."""
    runs = all_runs(out)
    groups = []
    for tid, m in minima.items():
        reps = []
        for rid in sorted(runs):
            r = runs[rid]
            if r["tid"] != tid or not r.get("conv"):
                continue
            for c in reps:
                if same(m, c["pos"], r["pos"]):
                    c["members"].append(rid); break
            else:
                reps.append(dict(tid=tid, pos=r["pos"], members=[rid]))
        groups += reps
    return groups, runs


def returns_to(m, start, calc, e_min, max_steps=500):
    """Relax `start` (LBFGS to 0.005 eV/A); stop once within 0.2 A of the minimum. (returned, steps, distance).

    The threshold must be tight: at 0.05 eV/A relaxations stop on flat ground far from any minimum."""
    from ase.optimize import LBFGS
    a = start.copy(); a.calc = calc
    opt, steps, dist = LBFGS(a, logfile=None, maxstep=0.2), 0, np.inf
    for _ in opt.irun(fmax=0.005, steps=max_steps):
        steps += 1
        dist = max_mic_untranslated(m, a)
        if dist < hs.RETURN:
            return True, steps, dist
    near = dist < 0.5 and abs(a.get_potential_energy() - e_min) < 0.01
    return near, steps, dist


def verify(args, out):
    from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend
    plan, rows, minima = hs.plan_minima(args)
    groups, runs = groups_v2(out, minima)
    path = out / VERIFY_DIR / f"verify_{args.tag}{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    # Runs can be added after an earlier verification (another arm), which regroups the points; a point
    # that matches an already verified one (within 0.1 A) is not checked again.
    verified = {}
    for r in read_rows(path.parent / "verify_*.json"):
        verified.setdefault(r["tid"], []).append(r["pos"])
    calc = hs.calculator(args.device)
    e_min = {}
    for g in groups[args.shard::args.nshards]:
        m = minima[g["tid"]]
        if any(same(m, p, g["pos"]) for p in verified.get(g["tid"], [])):
            continue
        verified.setdefault(g["tid"], []).append(g["pos"])
        pes = PESBackend(m.copy(), calc)
        if g["tid"] not in e_min:
            e_min[g["tid"]] = pes.energy(pes.dofs.pack(m.positions))
        x = pes.dofs.pack(np.array(g["pos"]))
        row = dict(g, barrier=pes.energy(x) - e_min[g["tid"]])
        if row["barrier"] > MAX_RISE:
            row.update(status="above_cutoff"); done_rows.append(row); write_rows(path, done_rows); continue
        # Full central-difference Hessian over the free coordinates (6N calls). The three translations
        # appear as ~0 eigenvalues and never count toward the index (threshold -0.01).
        before = pes.force_calls
        values, vectors, _ = MinimumModeEstimator(backend="dense", eps=0.002).lowest_modes(pes, x, n_modes=4)
        row.update(eigs=values.tolist(), nneg=int(np.sum(values < -0.01)), index_calls=pes.force_calls - before)
        if row["nneg"] == 1:
            # Push far enough that the force is >= 0.1 eV/A (0.1-0.3 A on the most-moving atom), and relax
            # first the side that points toward the minimum: it usually returns, so the other side is skipped.
            size = float(np.clip(0.1 / abs(values[0]), 0.1, 0.3))
            v = vectors[:, 0].reshape(-1, 3); v = size * v / np.linalg.norm(v, axis=1).max()
            toward = np.dot(v.ravel(), pes.dofs.pack(m.positions) - x)
            sides = []
            for sign in ((1, -1) if toward >= 0 else (-1, 1)):
                ok, steps, dist = returns_to(m, at(m, pes.dofs.expand(x + sign * v.ravel())), calc, e_min[g["tid"]])
                sides.append(dict(returned=ok, steps=steps, distance=dist))
                if ok:
                    break
            row.update(connected=any(s["returned"] for s in sides), sides=sides,
                       relax_calls=sum(s["steps"] for s in sides))
        row["status"] = "checked"
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"shard {args.shard}: {len(done_rows)} distinct points checked")


def analyze(args, out):
    plan, rows, minima = hs.plan_minima(args)
    runs = all_runs(out)
    # One row per distinct point (two jobs may check the same point), and every converged run assigned to
    # its point by position, so arms added after a verification are counted correctly.
    checked = []
    for c in read_rows(out / VERIFY_DIR / "verify_*.json"):
        if not any(d["tid"] == c["tid"] and same(minima[c["tid"]], d["pos"], c["pos"]) for d in checked):
            checked.append(dict(c, key=(c["tid"], c["members"][0]), members=[]))
    for rid in sorted(runs):
        r = runs[rid]
        if r.get("conv") and r["tid"] in minima:
            hit = next((c for c in checked if c["tid"] == r["tid"] and same(minima[r["tid"]], c["pos"], r["pos"])), None)
            if hit is not None:
                hit["members"].append(rid)
    first = lambda r: r.get("repeat", 0) == 0
    arms = list(OLD_RANDOM) + ["all_atoms"] + list(hs.MODEL_ARMS) + list(MODE_ARMS) + list(HINT_ARMS) + list(SOFT_ARMS)
    saddles = [c for c in checked if c.get("nneg") == 1 and c.get("connected")]
    lines = ["Mode-based directions and cheaper verification vs the first hint-search run", "=" * 104,
             f"{len(rows)} minima; {len(runs)} runs ({sum(r['source'] == 'mode' for r in runs.values())} new); "
             f"{len(checked)} distinct converged points; {len(saddles)} connected index-1 saddles in scope "
             f"(barrier <= {MAX_RISE} eV). Coverage counts saddles with barrier <= the cutoff, against the union.", ""]

    lines += ["run outcomes and search cost per arm (first runs):",
              "arm         runs  converged%  budget%  drift%  energy%  climb_fail%   calls/run"]
    cost = {}
    for arm in arms:
        rs = [r for r in runs.values() if arm_of(r) == arm and first(r)]
        if not rs:
            continue
        st = [r.get("status", "") for r in rs]
        # A mode run also pays its share of the epicenter's local block.
        calls = [r.get("climb_calls", 0) + r.get("sella_calls", 0) + r.get("block_calls", 0) / max(r.get("block_runs", 1), 1)
                 for r in rs]
        cost[arm] = sum(calls)
        lines.append(f"{arm:10s} {len(rs):5d}  {100 * np.mean([r.get('conv', False) for r in rs]):9.1f}  "
                     f"{100 * np.mean([s == 'query_budget' for s in st]):7.1f}  {100 * np.mean([s == 'drift' for s in st]):6.1f}  "
                     f"{100 * np.mean([s == 'energy' for s in st]):7.1f}  {100 * np.mean([s.startswith('climb_') for s in st]):11.1f}   "
                     f"{np.mean(calls):9.1f}")

    def found(arm, cut):
        return [c for c in saddles if c["barrier"] <= cut and any(arm_of(runs[i]) == arm and first(runs[i]) for i in c["members"])]
    union = {cut: [c for c in saddles if c["barrier"] <= cut] for cut in (0.3, 0.5, CUTOFF)}
    lines += ["", "connected saddles found (share of the union) and force calls per connected saddle (barrier <= 1 eV):",
              "arm           <= 0.3 eV       <= 0.5 eV       <= 1.0 eV     calls/saddle   only this arm"]
    for arm in arms:
        if arm not in cost:
            continue
        cells = [f"{len(found(arm, c)):3d} ({100 * len(found(arm, c)) / max(len(union[c]), 1):5.1f}%)" for c in (0.3, 0.5, CUTOFF)]
        only = sum(1 for c in union[CUTOFF] if {arm_of(runs[i]) for i in c["members"] if first(runs[i])} == {arm})
        lines.append(f"{arm:10s}  {cells[0]}  {cells[1]}  {cells[2]}   {cost[arm] / max(len(found(arm, CUTOFF)), 1):10.0f}   {only:5d}")
    lines.append(f"union        {len(union[0.3]):3d}             {len(union[0.5]):3d}             {len(union[CUTOFF]):3d}")
    old_union = {i for c in union[CUTOFF] for i in c["members"] if runs[i]["source"] == "first"}
    new_only = sum(1 for c in union[CUTOFF] if all(runs[i]["source"] == "mode" for i in c["members"]))
    lines.append(f"  saddles found only by the new mode arms: {new_only}; only by the first run's arms: "
                 f"{sum(1 for c in union[CUTOFF] if all(runs[i]['source'] == 'first' for i in c['members']))}")

    lines += ["", "repeatability (identical starts; share whose repeat lands on a different distinct point):"]
    owner = {i: n for n, c in enumerate(checked) for i in c["members"]}
    for arm in list(OLD_RANDOM) + list(MODE_ARMS):
        reps = [r for r in runs.values() if arm_of(r) == arm and r.get("repeat") == 1]
        flips = [owner.get(r["id"]) != owner.get(r["id"][:-1] + "0") for r in reps]
        if flips:
            lines.append(f"  {arm:10s} {len(flips):3d} repeats, {100 * np.mean(flips):5.1f}% differ")

    lines += ["", "per minimum (connected saddles <= 1 eV):", "  tid  formula          union  first-run arms  mode arms  model"]
    for r in rows:
        u = [c for c in union[CUTOFF] if c["tid"] == r["tid"]]
        src = lambda c, s: any(runs[i]["source"] == s and arm_of(runs[i]) not in hs.MODEL_ARMS for i in c["members"])
        mod = lambda c: any(arm_of(runs[i]) in hs.MODEL_ARMS for i in c["members"])
        lines.append(f"  {r['tid']:4d} {r['formula']:16s} {len(u):5d}  {sum(src(c, 'first') for c in u):14d}  "
                     f"{sum(src(c, 'mode') for c in u):9d}  {sum(map(mod, u)):5d}")

    lines += ["", f"verification (full central-difference Hessian, then connectivity): "
                  f"{np.mean([c.get('index_calls', 0) + c.get('relax_calls', 0) for c in checked]):.0f} force calls per distinct "
                  f"point (index {np.mean([c.get('index_calls', 0) for c in checked]):.0f}, connectivity "
                  f"{np.mean([c.get('relax_calls', 0) for c in checked]):.0f}); above the 1.5 eV scope "
                  f"{sum(c.get('status') == 'above_cutoff' for c in checked)} of {len(checked)}"]
    # How the two earlier checks compare with the full-Hessian labels, point by point (matched by position).
    def compare(name, rows_other):
        pairs = []
        for o in rows_other:
            if o.get("nneg") is None:
                continue
            c = next((c for c in checked if c["tid"] == o["tid"] and c.get("nneg") is not None
                      and same(minima[o["tid"]], c["pos"], o["pos"])), None)
            if c is not None:
                pairs.append((c, o))
        if not pairs:
            return
        table = {}
        for c, o in pairs:
            table[(c["nneg"], o["nneg"])] = table.get((c["nneg"], o["nneg"]), 0) + 1
        con = [bool(c.get("connected")) == bool(o.get("connected")) for c, o in pairs if c["nneg"] == 1 and o["nneg"] == 1]
        lines.append(f"  {name}: index agrees on {100 * np.mean([c['nneg'] == o['nneg'] for c, o in pairs]):.1f}% of {len(pairs)} "
                     f"points; (full-Hessian index, its index): count {dict(sorted(table.items()))}"
                     + (f"; connectivity agrees on {100 * np.mean(con):.1f}% of {len(con)}" if con else ""))
    compare("first verification (full Hessian, older connectivity test)", read_rows(out / "verify" / "verify_*.json"))
    compare("block eigensolver (verify2, not used)", read_rows(out / "verify2" / "verify_*.json"))
    text = "\n".join(lines) + "\n"
    (out / "mode_report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    {"search": search, "verify": verify, "analyze": analyze}[args.mode](args, out)


if __name__ == "__main__":
    main()
