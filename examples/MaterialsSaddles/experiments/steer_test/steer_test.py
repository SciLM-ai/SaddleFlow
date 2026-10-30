"""Steering test on the dataset's own saddles: given a hint pointing at a stored mp20bat test saddle (usually the
working ion hopping into a vacancy, often a long hop), does each model reach that saddle?

The hint evaluation (eval_hints.py) scores against Sella-found reference regions, and every Sella run there stops
once an atom drifts 2.5 A from the minimum: no Tier B label and no reference region lies beyond 2.5 A, while 10% of
the stored dataset saddles do. This test uses the stored saddles themselves as targets and raises the drift stop.

Cases: the local mp20bat test split (element sets disjoint from every training set: the release train file, the
local train copy, Tier B). Per element set, up to PER_SET (minimum, stored saddle) pairs with a stored barrier of
0.05-1 eV, seeded, distinct minima (and triplets) first, then distinct saddles on a picked minimum.

Stages (each skips work already written):
    refine  (GPU; --shard/--nshards) the stored saddle (CI-NEB climbing image, DFT) on UMA-S: Sella from it
            (UMA-S-1.2 omat, fmax 0.01, <= 300 steps, <= 600 calls, delta0 0.1, no stop rule: the start is already
            the saddle); when it converges within MAX_MOVE of the stored point, hint_vs_noise.check (full-Hessian
            index, connectivity to the minimum, barrier). Valid: index 1, connected, <= 1 eV. The refined point
            is the target.
    land    (GPU) one hint start per valid case: make_hints on (minimum, target), its largest-move atom, the
            direction snapped to the nearest of that atom's 16 grid directions (as inference would offer it);
            the minimum + 0.15 A noise on every atom, the atom pushed 0.6 A along the direction (eval_hints'
            hint start). Landing points (5 passes x 4 Euler steps) for the released model (no hint input) and
            each fine-tune in MODELS (hint given at every step); a fine-tune whose checkpoint is missing is
            skipped (land again once it is there).
    sella   (GPU; --shard/--nshards) Sella from every landing, and from the hint start itself (push, no model):
            the settings above with the energy stop of the evaluation (above both the minimum + 1.5 eV and the
            start) and the drift stop at DRIFT instead of 2.5 A; every run records the largest drift it reached.
    report  (CPU) report.txt. Hit: a converged Sella run ending within END of the target. Landing: within LAND of
            the target before Sella. Per arm, overall and by the target's distance from the minimum; paired
            (McNemar, exact) comparisons of the hit rates.

    python examples/MaterialsSaddles/experiments/steer_test/steer_test.py <stage> --out-dir <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hint_finetune"))
import eval_hints as ev  # noqa: E402
from eval_hints import hl, hs, hvn, ms, read_rows, write_rows  # noqa: E402
from saddle_reach_check import LOCAL, size, triplets  # noqa: E402

PER_SET, LOW, CUTOFF, SAME, MAX_MOVE = 2, 0.05, 1.0, 0.1, 0.3
DRIFT, OLD_DRIFT, LAND, END = 6.0, ms.DRIFT, 0.2, 0.15
PASSES, STEPS = 5, 4
MODELS = {"pilot": f"{ev.DATA}/hint_finetune/runs/pilot_21872158/checkpoint_final",
          "armA": f"{ev.DATA}/hint_finetune/runs/armA_6747347/checkpoint_final",
          "inversion": f"{ev.DATA}/hint_finetune/runs/inversion_6751338/checkpoint_final"}
ARMS = ("released", *MODELS, "push")
BINS = ((0, 1.5), (1.5, 2.5), (2.5, np.inf))


def atoms(c, key="minimum"):
    return Atoms(numbers=c["numbers"], positions=c[key], cell=c["cell"], pbc=True)


def cases():
    """[case dict] in a fixed order: up to PER_SET (minimum, stored saddle) pairs per test element set."""
    by = {}
    for i, (Z, cell, R, S, P, bR, bP) in enumerate(triplets(LOCAL.format(split="test"))):
        for side, E, b in (("R", R, bR), ("P", P, bP)):
            if LOW <= b <= CUTOFF:
                key = "-".join(sorted(set(Atoms(numbers=Z).get_chemical_symbols())))
                by.setdefault(key, []).append(dict(id=f"{i}:{side}", triplet=i, elements=key, numbers=Z.tolist(),
                                                   cell=cell.tolist(), minimum=E.tolist(), stored=S.tolist(),
                                                   stored_barrier=float(b)))
    rng = np.random.default_rng(0)
    out = []
    for key, cands in sorted(by.items()):
        cands = [cands[j] for j in rng.permutation(len(cands))]
        pick = []

        def same_minimum(a, b):
            return a["numbers"] == b["numbers"] and size(atoms(a), np.array(b["minimum"])) < SAME

        for c in cands:         # distinct minima and triplets first
            if len(pick) < PER_SET and all(c["triplet"] != p["triplet"] and not same_minimum(c, p) for p in pick):
                pick.append(c)
        for c in cands:         # then distinct saddles on a picked minimum
            if len(pick) < PER_SET and c not in pick and all(
                    not same_minimum(c, p) or hl.saddle_distance(atoms(p), np.array([p["stored"]]), np.array(c["stored"]))[0] >= SAME
                    for p in pick):
                pick.append(c)
        out += pick
    return out


def valid(r):
    return (r.get("conv") and r["moved"] <= MAX_MOVE and r.get("nneg") == 1 and r.get("connected")
            and r.get("barrier", np.inf) <= CUTOFF)


def targets(out):
    """[(case index, case, refined row)] for the valid cases, in case order."""
    R = {r["id"]: r for r in read_rows(out / "refined" / "refined_*.json")}
    return [(i, c, R[c["id"]]) for i, c in enumerate(cases()) if c["id"] in R and valid(R[c["id"]])]


# ---------------------------------------------------------------- refine
def refine(args, out):
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    C = cases()
    path = out / "refined" / f"refined_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in rows}
    calc = hs.calculator(args.device)
    for c in C[args.shard::args.nshards]:
        if c["id"] in done:
            continue
        m, S = atoms(c), np.array(c["stored"])
        pes = PESBackend(m.copy(), calc)
        e_min = pes.energy(pes.dofs.pack(m.positions))
        row = dict(id=c["id"], stored_size=size(m, S))
        try:
            before, t0 = pes.force_calls, time.perf_counter()
            res = run_sella(pes, pes.dofs.pack(S), fmax_tol=hs.FMAX, max_steps=hs.SELLA_STEPS,
                            max_force_calls=before + hs.SELLA_CALLS, delta0=0.1)
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            final = np.asarray(res["positions"])
            row.update(status=res["status"], sella_calls=pes.force_calls - before, seconds=time.perf_counter() - t0,
                       fmax=fmax, conv=bool(fmax <= hs.FMAX and res["status"] == "converged"),
                       moved=float(hl.saddle_distance(m, final[None], S)[0]), pos=np.round(final, 5).tolist())
            if row["conv"] and row["moved"] <= MAX_MOVE:
                before = pes.force_calls
                chk = hvn.check(pes, m, res["final"], e_min, calc)
                row.update(chk, check_status=chk.pop("status"), check_calls=pes.force_calls - before)
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        rows.append(row)
        write_rows(path, rows)
    print(f"shard {args.shard}: {len(rows)} refined ({len(C)} cases in all)")


# ---------------------------------------------------------------- land
def starts(out):
    """[(id, key, minimum, start, hint (N, 3), extra)] per valid case, in eval_hints' job layout."""
    jobs = []
    for i, c, r in targets(out):
        m = atoms(c)
        a, u, move = hl.make_hints(m, np.array(r["pos"]))[0]
        dirs = hs.fibonacci(ev.GRID_K, np.random.default_rng([i, 20, a]))
        d = dirs[int(np.argmax(dirs @ u))]
        pos = m.positions + ev.HINT_NOISE * np.random.default_rng([i, 21]).normal(size=m.positions.shape)
        pos[a] += ev.PUSH * d
        h = np.zeros((len(m), 3)); h[a] = d
        jobs.append((c["id"], c["id"], m, pos, h, dict(atom=int(a), move=move, snap_deg=float(np.degrees(np.arccos(np.clip(d @ u, -1, 1)))),
                                                       direction=np.round(d, 6).tolist())))
    return jobs


def land(args, out):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (out / "landings").mkdir(exist_ok=True)
    jobs = starts(out)
    target = {c["id"]: np.array(r["pos"]) for _, c, r in targets(out)}
    print(f"{len(jobs)} valid cases")
    for arm in ("released", *MODELS):
        path = out / "landings" / f"{arm}.json"
        if path.exists():
            continue
        if arm == "released":
            from saddleflow import SaddleFlow
            model = SaddleFlow.from_pretrained("SciLM/SaddleFlow", device=args.device)
        elif not (Path(MODELS[arm]) / "ema.pt").exists():
            print(f"{arm}: no checkpoint at {MODELS[arm]}, skipped", flush=True)
            continue
        else:
            model = ev.load_finetuned(MODELS[arm], args.device)
        t0 = time.perf_counter()
        if arm == "released":
            lands = ev.roll_released(model, jobs, PASSES, STEPS, args.batch_size)
        else:
            lands = ev.roll_finetuned(model, jobs, True, PASSES, STEPS, args.batch_size)
        seconds = (time.perf_counter() - t0) / max(1, len(jobs))
        rows = [dict(id=jid, arm=arm, **extra, rollout_seconds=seconds, start=np.round(pos, 5).tolist(),
                     landing=np.round(x, 5).tolist(),
                     start_distance=float(hl.saddle_distance(m, pos[None], target[jid])[0]),
                     target_distance=float(hl.saddle_distance(m, x[None], target[jid])[0]))
                for (jid, _, m, pos, _, extra), x in zip(jobs, lands)]
        write_rows(path, rows)
        print(f"{arm}: {len(rows)} landings, {seconds:.3f} s per start", flush=True)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- sella
def stop_rule(m, e_min, dofs, e_start, reached):
    """The evaluation's energy stop, the drift stop at DRIFT; reached[0] tracks the largest drift."""
    cap = max(e_min + ms.MAX_RISE, e_start)

    def rule(x, energy):
        d = ms.max_mic_untranslated(m, ms.at(m, dofs.expand(x)))
        reached[0] = max(reached[0], d)
        if energy > cap:
            return "energy"
        if d > DRIFT:
            return "drift"
    return rule


def sella(args, out):
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    T = {c["id"]: (c, r) for _, c, r in targets(out)}
    jobs = []
    for arm in ARMS:
        path = out / "landings" / f"{'released' if arm == 'push' else arm}.json"
        if path.exists():
            jobs += [(arm, l["id"], l["start"] if arm == "push" else l["landing"])
                     for l in sorted(json.loads(path.read_text()), key=lambda l: l["id"])]
    path = out / "runs" / f"runs_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["arm"], r["id"]) for r in read_rows(path.parent / "runs_*.json")}
    calc = hs.calculator(args.device)
    e_ref = {}
    for arm, cid, start in jobs[args.shard::args.nshards]:
        if (arm, cid) in done:
            continue
        c, r = T[cid]
        m = atoms(c)
        pes = PESBackend(m.copy(), calc)
        if cid not in e_ref:
            e_ref[cid] = pes.energy(pes.dofs.pack(m.positions))
        row, reached = dict(id=cid, arm=arm), [0.0]
        try:
            x0 = pes.dofs.pack(np.array(start))
            e_start = pes.energy(x0)
            before, t0 = pes.force_calls, time.perf_counter()
            res = run_sella(pes, x0, fmax_tol=hs.FMAX, max_steps=hs.SELLA_STEPS, max_force_calls=before + hs.SELLA_CALLS,
                            delta0=0.1, stop=stop_rule(m, e_ref[cid], pes.dofs, e_start, reached))
            seconds = time.perf_counter() - t0
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            final = np.asarray(res["positions"])
            row.update(status=res["status"], sella_calls=pes.force_calls - before, seconds=seconds, fmax=fmax,
                       conv=bool(fmax <= hs.FMAX and res["status"] == "converged"), max_drift=reached[0],
                       target_distance=float(hl.saddle_distance(m, final[None], np.array(r["pos"]))[0]),
                       pos=np.round(final, 5).tolist())
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        rows.append(row)
        write_rows(path, rows)
    print(f"shard {args.shard}: {len(rows)} runs ({len(jobs)} in all)")


# ---------------------------------------------------------------- report
def mcnemar(x, y):
    """Exact two-sided McNemar p on paired booleans."""
    from scipy.stats import binomtest
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    return b, c, (binomtest(b, b + c).pvalue if b + c else 1.0)


def report(args, out):
    C = cases()
    R = {r["id"]: r for r in read_rows(out / "refined" / "refined_*.json")}
    T = targets(out)
    dist = {c["id"]: size(atoms(c), np.array(r["pos"])) for _, c, r in T}
    L = ["Steering test on stored mp20bat test saddles", "=" * 110, "",
         f"cases: {len(C)} (minimum, stored saddle) pairs, barrier {LOW}-{CUTOFF} eV, <= {PER_SET} per element set "
         f"({len({c['elements'] for c in C})} test element sets); refined {len(R)}", "",
         "REFINEMENT: Sella from the stored saddle on UMA-S, then index, connectivity, barrier",
         "  stored distance  cases | converged  moved <= 0.3 | index 1  connected  <= 1 eV | valid  share | moved p50"]
    for lo, hi in BINS:
        S = [R[c["id"]] for c in C if c["id"] in R and lo <= R[c["id"]]["stored_size"] < hi]
        if not S:
            continue
        conv = [r for r in S if r.get("conv")]
        near = [r for r in conv if r["moved"] <= MAX_MOVE]
        L.append(f"  {lo:4.1f}-{hi:<4.1f} A    {len(S):6d} | {len(conv):9d} {len(near):12d} | "
                 f"{sum(r.get('nneg') == 1 for r in near):7d} {sum(bool(r.get('connected')) for r in near):10d} "
                 f"{sum(r.get('barrier', np.inf) <= CUTOFF for r in near):8d} | {sum(bool(valid(r)) for r in S):5d} "
                 f"{100 * np.mean([bool(valid(r)) for r in S]):5.0f}% | {np.median([r['moved'] for r in conv]) if conv else np.nan:8.3f}")
    L.append(f"  valid targets: {len(T)}; distance from the minimum p50 {np.median(list(dist.values())) if dist else np.nan:.2f} A")

    lands = {a: {l["id"]: l for l in json.loads((out / "landings" / f"{a}.json").read_text())}
             for a in ("released", *MODELS) if (out / "landings" / f"{a}.json").exists()}
    runs = {}
    for r in read_rows(out / "runs" / "runs_*.json"):
        runs.setdefault(r["arm"], {})[r["id"]] = r
    if "released" in lands:
        snap = [l["snap_deg"] for l in lands["released"].values()]
        L += ["", f"HINT: the largest-move atom, snapped to the nearest of its {ev.GRID_K} grid directions "
              f"(angle to the true direction p50 {np.median(snap):.0f}, p90 {np.percentile(snap, 90):.0f} deg)"]
    if not runs:
        print("\n".join(L)); (out / "report.txt").write_text("\n".join(L) + "\n"); return

    ids = [c["id"] for _, c, _ in T]

    def hit(arm, i):
        r = runs.get(arm, {}).get(i)
        return bool(r and r.get("conv") and r["target_distance"] <= END)

    L += ["", f"RESULTS: hit = converged Sella run ending within {END} A of the target; landing = within {LAND} A before "
          f"Sella; elsewhere = converged, not the target (not verified); drift stop {DRIFT} A ('cut': hits whose run "
          f"drifted beyond the evaluation's {OLD_DRIFT} A)",
          "  arm         target dist   n | landing  land dist p50 | hit    elsewhere  not conv  drift stop | calls p50 (hits)  cut"]
    for arm in ARMS:
        if arm not in runs:
            L.append(f"  {arm:11s} not run"); continue
        for lo, hi in ((0, np.inf), *BINS):
            I = [i for i in ids if lo <= dist[i] < hi and i in runs[arm]]
            if not I:
                continue
            rr = [runs[arm][i] for i in I]
            H = [r for i, r in zip(I, rr) if hit(arm, i)]
            la = lands.get("released" if arm == "push" else arm, {})
            ld = [la[i]["start_distance" if arm == "push" else "target_distance"] for i in I if i in la]
            label = "all" if hi == np.inf and lo == 0 else f"{lo:.1f}-{hi:.1f} A"
            L.append(f"  {arm if lo == 0 and hi == np.inf else '':11s} {label:11s} {len(I):4d} | "
                     f"{100 * np.mean(np.array(ld) <= LAND):6.1f}% {np.median(ld):11.3f}   | "
                     f"{100 * len(H) / len(I):5.1f}% {100 * np.mean([bool(r.get('conv')) for r in rr]) - 100 * len(H) / len(I):8.1f}% "
                     f"{100 * np.mean([not r.get('conv') for r in rr]):8.1f}% {100 * np.mean([r.get('status') == 'drift' for r in rr]):9.1f}% | "
                     f"{np.median([r['sella_calls'] for r in H]) if H else np.nan:15.0f}  "
                     f"{sum(r['max_drift'] > OLD_DRIFT for r in H):4d}")
    L += ["", "PAIRED hit rates (same cases; McNemar exact): hits only the first arm / only the second / p"]
    for a, b in (("armA", "pilot"), ("armA", "released"), ("pilot", "released"), ("inversion", "pilot"),
                 ("armA", "push"), ("pilot", "push"), ("released", "push")):
        if a in runs and b in runs:
            I = [i for i in ids if i in runs[a] and i in runs[b]]
            x, y = np.array([hit(a, i) for i in I]), np.array([hit(b, i) for i in I])
            n1, n2, p = mcnemar(x, y)
            L.append(f"  {a:10s} vs {b:10s} {len(I):4d} cases: {100 * x.mean():5.1f}% vs {100 * y.mean():5.1f}%   {n1:3d} / {n2:3d}   p {p:.3g}")
    text = "\n".join(L)
    print(text)
    (out / "report.txt").write_text(text + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["refine", "land", "sella", "report"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"refine": refine, "land": land, "sella": sella, "report": report}[a.stage](a, out)


if __name__ == "__main__":
    main()
