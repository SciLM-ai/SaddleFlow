"""Put the stored mp20bat dataset saddles (CI-NEB climbing images, converged to 0.05 eV/A, never checked our way)
through the standard Tier B used: Sella from the stored saddle, then hint_vs_noise.check (full-Hessian index,
connectivity to that minimum, barrier). Then redo the comparisons with the refined saddles.

Sets (all from the local mp20bat copy the pilot and Tier B used):
    tierb   every stored saddle <= 1 eV of a train triplet whose reactant or product is a Tier B minimum
    test24  every stored saddle <= 1 eV of a test or train triplet whose reactant or product is one of the 24 test
            minima of the evaluation
    steer   one (minimum, saddle) pair per element set of the test split, barrier 0.05-1 eV, seeded, at most 300:
            the reference for a later steering test on dataset-type saddles

Modes:
    refine  (GPU; --shard/--nshards) Sella from the stored saddle (UMA-S-1.2 omat, fmax 0.01, <= 300 steps,
            <= 600 calls, delta0 0.1, no drift stop: the start is already the saddle), then the check when it
            converges
    report  (CPU) per set: converged, moved from the stored point, index 1, connected, barrier <= 1 eV; tierb:
            how many valid refined saddles match a Tier B region (0.15 A); test24: whether they are in the
            reference and which methods reached them (a landing within 0.2 A, a Sella end within 0.15 A);
            writes steer_reference.json

    python examples/MaterialsSaddles/experiments/hint_finetune/refine_stored.py <mode> --out-dir <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_hints as ev  # noqa: E402
from eval_hints import hl, hs, hvn, read_rows, write_rows  # noqa: E402
from saddle_reach_check import LOCAL, size, triplets  # noqa: E402

SAME, REGION, LAND, LOW, CUTOFF, N_STEER = 0.1, 0.15, 0.2, 0.05, 1.0, 300


def jobs():
    """[(set, id, minimum, stored saddle, stored barrier)] in a fixed order."""
    out = []
    train, test = triplets(LOCAL.format(split="train")), triplets(LOCAL.format(split="test"))

    def touching(m, T):
        found = []
        for Z, cell, R, S, P, bR, bP in T:
            if len(Z) != len(m) or np.any(Z != m.numbers):
                continue
            for E, b in ((R, bR), (P, bP)):
                if b <= CUTOFF and size(m, E) < SAME and (not found or hl.saddle_distance(m, np.array([f[0] for f in found]), S).min() >= SAME):
                    found.append((S, b))
        return found

    for e in json.loads(Path(ev.LABELS).read_text())["minima"]:
        m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
        out += [("tierb", f"{e['tid']}:{i}", m, S, b) for i, (S, b) in enumerate(touching(m, train))]
    args = SimpleNamespace(old_dir=ev.OLD_DIR, limit=0)
    for t, (m, _) in ev.test_minima(args).items():
        out += [("test24", f"{t}:{i}", m, S, b) for i, (S, b) in enumerate(touching(m, test + train))]
    by_set = {}
    for i, (Z, cell, R, S, P, bR, bP) in enumerate(test):
        for side, E, b in (("R", R, bR), ("P", P, bP)):
            if LOW <= b <= CUTOFF:
                m = Atoms(numbers=Z, positions=E, cell=cell, pbc=True)
                by_set.setdefault(tuple(sorted(set(m.get_chemical_symbols()))), []).append((f"{i}:{side}", m, S, b))
    rng = np.random.default_rng(0)
    pick = [v[int(rng.integers(len(v)))] for _, v in sorted(by_set.items())]
    pick = [pick[i] for i in np.sort(rng.permutation(len(pick))[:N_STEER])]
    out += [("steer", k, m, S, b) for k, m, S, b in pick]
    return out


def refine(args, out):
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    J = jobs()
    path = out / f"refined_{args.shard}.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["set"], r["id"]) for r in rows}
    calc = hs.calculator(args.device)
    for kind, jid, m, S, b in J[args.shard::args.nshards]:
        if (kind, jid) in done:
            continue
        pes = PESBackend(m.copy(), calc)
        e_min = pes.energy(pes.dofs.pack(m.positions))
        row = dict(set=kind, id=jid, numbers=m.numbers.tolist(), cell=m.cell.array.tolist(), minimum=m.positions.tolist(),
                   stored=np.round(S, 5).tolist(), stored_barrier=b, stored_size=size(m, S))
        try:
            x0 = pes.dofs.pack(np.array(S))
            before, t0 = pes.force_calls, time.perf_counter()
            res = run_sella(pes, x0, fmax_tol=hs.FMAX, max_steps=hs.SELLA_STEPS, max_force_calls=before + hs.SELLA_CALLS,
                            delta0=0.1)
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            final = np.asarray(res["positions"])
            row.update(status=res["status"], sella_calls=pes.force_calls - before, seconds=time.perf_counter() - t0,
                       fmax=fmax, conv=bool(fmax <= hs.FMAX and res["status"] == "converged"),
                       moved=float(hl.saddle_distance(m, final[None], np.array(S))[0]), pos=np.round(final, 5).tolist())
            if row["conv"]:
                row.update(hvn.check(pes, m, res["final"], e_min, calc))
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        rows.append(row)
        write_rows(path, rows)
    print(f"shard {args.shard}: {len(rows)} refined ({len(J)} jobs in all)")


def valid(r):
    return r.get("conv") and r.get("nneg") == 1 and r.get("connected") and r.get("barrier", np.inf) <= CUTOFF


def report(args, out):
    R = read_rows(out / "refined_*.json")
    L = ["Stored mp20bat saddles refined with Sella and checked as Tier B's were", "=" * 100, "",
         "  set       n | converged  moved p50 | index 1  connected  <= 1 eV | valid | distance from minimum p50: stored  valid"]
    for kind in ("tierb", "test24", "steer"):
        S = [r for r in R if r["set"] == kind]
        if not S:
            continue
        conv = [r for r in S if r.get("conv")]
        moved = [r["moved"] for r in conv]
        V = [r for r in S if valid(r)]
        L.append(f"  {kind:7s} {len(S):4d} | {100 * len(conv) / len(S):8.0f}% {np.median(moved) if moved else np.nan:9.3f} | "
                 f"{100 * np.mean([r.get('nneg') == 1 for r in conv]) if conv else 0:6.0f}% "
                 f"{100 * np.mean([bool(r.get('connected')) for r in conv]) if conv else 0:9.0f}% "
                 f"{100 * np.mean([r.get('barrier', np.inf) <= CUTOFF for r in conv]) if conv else 0:7.0f}% | {len(V):5d} | "
                 f"{np.median([r['stored_size'] for r in S]):30.2f} "
                 + (f"{np.median([size(Atoms(numbers=r['numbers'], positions=r['minimum'], cell=r['cell'], pbc=True), r['pos']) for r in V]):6.2f}" if V else "     -"))

    # Tier B: do the valid refined saddles match a Tier B region?
    tb = {e["tid"]: e for e in json.loads(Path(ev.LABELS).read_text())["minima"]}
    V = [r for r in R if r["set"] == "tierb" and valid(r)]
    hit = 0
    for r in V:
        e = tb[int(r["id"].split(":")[0])]
        m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
        hit += hl.saddle_distance(m, np.array([g["saddle"] for g in e["regions"]]), np.array(r["pos"])).min() < REGION
    L += ["", f"TIER B: of {len(V)} valid refined stored saddles on the Tier B minima, {hit} match a Tier B region "
          f"(within {REGION} A), so Tier B's 128 Sella runs found {100 * hit / max(len(V), 1):.0f}% of them"]

    # The 24 test minima: in the reference, and reached by which method.
    args_ = SimpleNamespace(old_dir=ev.OLD_DIR, limit=0)
    minima = {t: m for t, (m, _) in ev.test_minima(args_).items()}
    ref = ev.reference(args_, Path(args.eval_dir), minima)
    V = [r for r in R if r["set"] == "test24" and valid(r)]
    stored = {t: [np.array(r["pos"]) for r in V if int(r["id"].split(":")[0]) == t] for t in minima}
    n = len(V)
    in_ref = sum(bool(ref[t]["regions"]) and hl.saddle_distance(m, np.array([g["pos"] for g in ref[t]["regions"]]), s).min() < REGION
                 for t, m in minima.items() for s in stored[t])
    lands, runs = {}, {}
    ed = Path(args.eval_dir)
    for l in read_rows(ed / "landings" / "*.json"):
        if l["key"] in minima:
            lands.setdefault(l["arm"], {}).setdefault(l["key"], []).append(l["landing"])
    for l in read_rows(Path(ev.OLD_DIR) / "landings" / "landings_*.json"):
        if l["tid"] in minima:
            lands.setdefault(ev.OLD_ARMS[l["arm"]], {}).setdefault(l["tid"], []).append(l["landing"])
    for r in read_rows(ed / "runs" / "runs_*.json"):
        if r["key"] in minima and r.get("conv"):
            runs.setdefault(r["arm"], {}).setdefault(r["key"], []).append(r["pos"])
    for r in read_rows(Path(ev.OLD_DIR) / "runs" / "runs_*.json"):
        if r["tid"] in minima and r.get("conv") and r["arm"] in ev.OLD_ARMS:
            runs.setdefault(ev.OLD_ARMS[r["arm"]], {}).setdefault(r["tid"], []).append(r["pos"])

    def hits(points, radius):
        return sum(bool(points.get(t)) and hl.saddle_distance(m, np.array(points[t]), s).min() <= radius
                   for t, m in minima.items() for s in stored[t])

    L += ["", f"TEST MINIMA: {n} valid refined stored saddles on {sum(bool(v) for v in stored.values())} of the 24 test "
          f"minima; {in_ref} are in the 24-minimum reference (within {REGION} A)",
          f"  stage 1: a landing within {LAND} A; stage 2: a converged Sella run ending within {REGION} A",
          "  method              stage 1   stage 2"]
    for arm in ("ft_hint_1x8", "ft_hint", "ft_noise", "rel_hint", "rel_noise", "rel_mobile_hint", "rel_soft", "push", "softpush"):
        s1 = f"{hits(lands[arm], LAND)}/{n}" if arm in lands else "-"
        s2 = f"{hits(runs[arm], REGION)}/{n}" if arm in runs else "-"
        L.append(f"  {arm:18s} {s1:>8s}  {s2:>8s}")

    steer = [dict(id=r["id"], numbers=r["numbers"], cell=r["cell"], minimum=r["minimum"], saddle=r["pos"],
                  barrier=r["barrier"], stored=r["stored"]) for r in R if r["set"] == "steer" and valid(r)]
    (out / "steer_reference.json").write_text(json.dumps(steer))
    L += ["", f"STEER: {len(steer)} valid (minimum, saddle) pairs written to steer_reference.json"]
    text = "\n".join(L)
    print(text)
    (out / "report.txt").write_text(text + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["refine", "report"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-dir", default="/scratch/gqs7zr/saddleflow/hint_finetune/eval/pilot_21872158_checkpoint_final")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"refine": refine, "report": report}[a.mode](a, out)


if __name__ == "__main__":
    main()
