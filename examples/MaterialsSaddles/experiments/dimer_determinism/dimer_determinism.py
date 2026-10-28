"""Is the ASE dimer on UMA reproducible, and how sensitive is it to its start? The Sella determinism experiment,
repeated for the dimer (Henkelman and Jonsson, JCP 111, 7010 (1999)) on the same 38 starts.

The Sella experiment found that UMA's forces change by ~1e-6 eV/A on every GPU call whatever the settings, that a
single-threaded, re-seeded CPU run is bitwise reproducible, and that Sella from a start moved 1e-6 A ends at the same
point only ~2/3 of the time. The dimer has one more input than Sella: its initial mode. ASE draws it at random
(seeded here, never from the clock), so "the same start" means the same positions AND the same initial-mode seed
(a hash of the start id). The D5 and D9 conditions vary that seed per repeat instead.

    dimer     one --condition per process; every start run --repeats times with fmax 0.01, <= 300 steps,
              <= 600 force calls and the drift/energy stops of the Sella runs; dimer settings are run_dimer's
              (ASE's defaults with a 0.01 A image separation) plus --dimer-params, recorded with every run:
                D0  UMA on the GPU as in production, initial-mode seed fixed per start
                D1  D0 + PyTorch re-seeded before every force call (fixes UMA's random edge rolls)
                D2  D1 + PyTorch deterministic mode, TF32 off
                D3  D2 with the start moved 1e-6 A (a different random move per repeat)
                D4  D2 with the start moved 1e-4 A
                D5  D2 with a different initial-mode seed per repeat
                D6  CPU, one thread, re-seeded per force call: the deterministic reference (6 starts, 2 repeats)
                D7  D6 with the start moved 1e-6 A: chaos on a fully deterministic setup
                D8  D6 with the start moved 1e-4 A
                D9  D6 with a different initial-mode seed per repeat
    analyze   per condition: convergence and force calls, bitwise-identical repeats, same final point (0.1 A)
              and status, the step at which repeats first differ, final distances; the perturbed conditions
              against their unperturbed reference; dimer endpoints against Sella's from the same start (CPU)

Tier 1 (UMA's force noise) is not repeated: it is the same potential, measured in the Sella experiment.

    python examples/MaterialsSaddles/experiments/dimer_determinism/dimer_determinism.py <mode> --out-dir <dir> [...]

Run on one GPU model only (the batch script pins A100 80GB, as for Sella); different GPU models need not agree.
"""
import os
import sys

CONDITIONS = {
    "D0": dict(seed_uma=False, deterministic=False, device="cuda", perturb=0.0, vary_mode=False),
    "D1": dict(seed_uma=True, deterministic=False, device="cuda", perturb=0.0, vary_mode=False),
    "D2": dict(seed_uma=True, deterministic=True, device="cuda", perturb=0.0, vary_mode=False),
    "D3": dict(seed_uma=True, deterministic=True, device="cuda", perturb=1e-6, vary_mode=False),
    "D4": dict(seed_uma=True, deterministic=True, device="cuda", perturb=1e-4, vary_mode=False),
    "D5": dict(seed_uma=True, deterministic=True, device="cuda", perturb=0.0, vary_mode=True),
    "D6": dict(seed_uma=True, deterministic=False, device="cpu", perturb=0.0, vary_mode=False, threads=1),
    "D7": dict(seed_uma=True, deterministic=False, device="cpu", perturb=1e-6, vary_mode=False, threads=1),
    "D8": dict(seed_uma=True, deterministic=False, device="cpu", perturb=1e-4, vary_mode=False, threads=1),
    "D9": dict(seed_uma=True, deterministic=False, device="cpu", perturb=0.0, vary_mode=True, threads=1),
}
# Against which unperturbed run each condition is compared, and the matching Sella condition.
REFERENCE = {"D3": "D2", "D4": "D2", "D5": "D2", "D7": "D6", "D8": "D6", "D9": "D6"}
SELLA_MATCH = {"D2": "C3", "D6": "C6"}


def _argv_value(flag):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


# This must be in place before torch is imported.
if CONDITIONS.get(_argv_value("--condition"), {}).get("deterministic"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sella_determinism"))
import sella_determinism as sd  # noqa: E402  (also puts hint_search on the path)
import hint_search as hs  # noqa: E402
import mode_search as ms  # noqa: E402
from hint_search import max_mic_untranslated, read_rows, write_rows  # noqa: E402

SELLA_DIR = "/scratch/gqs7zr/saddleflow/sella_determinism"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["dimer", "analyze"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sella-dir", default=SELLA_DIR, help="the Sella experiment: its starts.json and runs")
    p.add_argument("--condition", choices=sorted(CONDITIONS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-starts", type=int, default=1, help="starts of each kind per minimum on the CPU (Sella: 1)")
    p.add_argument("--max-starts", type=int, default=0, help="first N starts only (tests)")
    p.add_argument("--max-steps", type=int, default=hs.SELLA_STEPS, help="dimer step cap (lower only for tests)")
    p.add_argument("--dimer-params", type=json.loads, default={},
                   help='DimerControl settings on top of run_dimer\'s, as JSON, e.g. \'{"max_num_rot": 10}\'')
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def mode_seed(start_id, rep, vary):
    """The initial-mode seed: fixed per start, or (vary) different for every repeat and from the fixed one, so
    no D5/D9 run shares its mode with the D2/D6 reference."""
    return (zlib.crc32(start_id.encode()) + (rep + 1 if vary else 0)) % 2 ** 32


def load_starts(args, out):
    path = out / "starts.json"
    if not path.exists():
        shutil.copy(Path(args.sella_dir) / "starts.json", path)      # the Sella experiment's starts, unchanged
    return json.loads(path.read_text())


# ---------------------------------------------------------------- repeated dimer runs
def dimer_runs(args, out):
    import torch
    cond = CONDITIONS[args.condition]
    if cond.get("threads"):
        torch.set_num_threads(cond["threads"])     # multithreaded CPU reductions are not reproducible
    if cond["deterministic"]:
        sd.make_torch_deterministic()
    from saddleflow.classical_saddle_search import PESBackend, run_dimer
    plan, rows, minima = sd.minima_and_plan()
    starts = load_starts(args, out)
    if cond["device"] == "cpu":
        # The first --cpu-starts starts of each kind per minimum (with 1, the same 6 starts as Sella's C6..C8).
        starts = [x for t in minima for k in ("nudge", "model", "hint")
                  for x in [x for x in starts if x["tid"] == t and x["kind"] == k][:args.cpu_starts]]
        args.repeats = min(args.repeats, 2)
    if args.max_starts:
        starts = starts[:args.max_starts]
    calc = sd.uma_calculator(cond["device"], cond["seed_uma"])
    if cond["deterministic"]:
        sd.make_torch_deterministic()
    path = out / f"runs_{args.condition}_{args.shard}.json"
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["start"], r["repeat"]) for r in done_rows}
    e_min = {}
    for s in starts[args.shard::args.nshards]:
        m = minima[s["tid"]]
        for rep in range(args.repeats):
            if (s["id"], rep) in done:
                continue
            pes = PESBackend(m.copy(), calc)
            if s["tid"] not in e_min:
                e_min[s["tid"]] = pes.energy(pes.dofs.pack(m.positions))
            x0 = pes.dofs.pack(np.array(s["pos"]))
            if cond["perturb"]:
                u = np.random.default_rng([rep, 7]).normal(size=x0.shape).reshape(-1, 3)
                x0 = x0 + (cond["perturb"] * u / np.linalg.norm(u, axis=1).max()).ravel()
            seed = mode_seed(s["id"], rep, cond["vary_mode"])
            pes.force_calls = 0
            res = run_dimer(pes, x0, seed=seed, fmax_tol=hs.FMAX, max_steps=args.max_steps,
                            max_force_calls=hs.SELLA_CALLS, stop=ms.stop_rule(m, e_min[s["tid"]], pes.dofs),
                            **args.dimer_params)
            centres = res["trace"][1:]
            steps = [hashlib.sha1(np.ascontiguousarray(r["x"]).tobytes()).hexdigest()[:16] for r in res["trace"]]
            done_rows.append(dict(start=s["id"], tid=s["tid"], kind=s["kind"], condition=args.condition, repeat=rep,
                                  mode_seed=seed, dimer_params=args.dimer_params, status=res["status"], error=res["error"], steps=res["steps"],
                                  force_calls=res["force_calls"], curvature=res["curvature"],
                                  final=np.asarray(res["positions"]).tolist(), step_hashes=steps,
                                  step_fmax=[float(r["force_max"]) for r in res["trace"]],
                                  step_curvature=[float(r["eigs"][0]) for r in centres],
                                  step_rotations=[int(r["rotations"]) for r in centres],
                                  step_energy=[float(r["energy"]) for r in centres]))
            write_rows(path, done_rows)
            print(f"  {s['id']} repeat {rep}: {res['status']}, {res['steps']} steps, {res['force_calls']} calls, "
                  f"curvature {res['curvature']:.3f}", flush=True)
    print(f"{args.condition} shard {args.shard}: {len(done_rows)} runs")


# ---------------------------------------------------------------- analyze
def dist(minima, a, b):
    m = minima[a["tid"]]
    return max_mic_untranslated(hs.at(m, a["final"]), hs.at(m, b["final"]))


def repeat_pairs(rs):
    rs = sorted(rs, key=lambda r: r["repeat"])
    return [(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs))]


def against(minima, runs, ref):
    """Every run against the repeat-0 run of the same start in ref: final distances."""
    first = {s: min(rs, key=lambda r: r["repeat"]) for s, rs in ref.items()}
    return [dist(minima, r, first[s]) for s, rs in runs.items() if s in first for r in rs]


def analyze(args, out):
    plan, rows, minima = sd.minima_and_plan()
    by = {}
    for r in read_rows(out / "runs_*.json"):
        by.setdefault(r["condition"], {}).setdefault(r["start"], []).append(r)
    sella = {}
    for r in read_rows(Path(args.sella_dir) / "runs_*.json"):
        sella.setdefault(r["condition"], {}).setdefault(r["start"], []).append(r)
    lines = ["ASE dimer on UMA: determinism and sensitivity", "=" * 100, "",
             "cond  runs  converged  force calls p50/p90  pairs  bitwise-identical  same point+status  "
             "median first-differing step  final distance p50/p90 (A)"]
    for cond in sorted(by):
        rs_all = [r for rs in by[cond].values() for r in rs]
        calls = [r["force_calls"] for r in rs_all]
        pairs = [p for rs in by[cond].values() for p in repeat_pairs(rs)]
        head = (f"{cond:4s} {len(rs_all):5d}   {100 * np.mean([r['status'] == 'converged' for r in rs_all]):5.1f}%"
                f"       {np.median(calls):5.0f} / {np.quantile(calls, 0.9):5.0f}    {len(pairs):5d}")
        if not pairs:
            lines.append(head)
            continue
        d = [dist(minima, a, b) for a, b in pairs]
        same_bits = [np.array_equal(np.array(a["final"]), np.array(b["final"])) for a, b in pairs]
        same_pt = [x < hs.MERGE and a["status"] == b["status"] for (a, b), x in zip(pairs, d)]
        diverged = [f for f in (sd.first_difference(a["step_hashes"], b["step_hashes"]) for a, b in pairs) if f is not None]
        lines.append(f"{head}   {100 * np.mean(same_bits):6.1f}%           {100 * np.mean(same_pt):6.1f}%            "
                     f"{np.median(diverged) if diverged else float('nan'):6.1f} ({len(diverged)}/{len(pairs)} differ)"
                     f"       {np.median(d):6.3f} / {np.quantile(d, 0.9):6.3f}")
    lines.append("")
    for cond, base in REFERENCE.items():
        if cond in by and base in by:
            d = against(minima, by[cond], by[base])
            if d:
                lines.append(f"  {cond} against the unperturbed {base} run: {100 * np.mean([x < hs.MERGE for x in d]):.1f}% "
                             f"end at the same point (final distance p50 {np.median(d):.3f}, p90 {np.quantile(d, 0.9):.3f} A)")
    for cond, sc in SELLA_MATCH.items():
        if cond in by and sc in sella:
            d = against(minima, by[cond], sella[sc])
            if d:
                lines.append(f"  {cond} against Sella {sc} from the same start: {100 * np.mean([x < hs.MERGE for x in d]):.1f}% "
                             f"end at the same point (final distance p50 {np.median(d):.3f}, p90 {np.quantile(d, 0.9):.3f} A)")
    lines += ["", "by kind of start: converged; same point+status between repeats:"]
    for cond in sorted(by):
        cells = []
        for kind in ("nudge", "model", "hint"):
            rs = [r for rr in by[cond].values() for r in rr if r["kind"] == kind]
            pairs = [p for rr in by[cond].values() for p in repeat_pairs(rr) if p[0]["kind"] == kind]
            if rs:
                cell = f"{kind} {100 * np.mean([r['status'] == 'converged' for r in rs]):5.1f}% of {len(rs)}"
                if pairs:
                    ok = [dist(minima, a, b) < hs.MERGE and a["status"] == b["status"] for a, b in pairs]
                    cell += f"; {100 * np.mean(ok):5.1f}% of {len(pairs)}"
                cells.append(cell)
        lines.append(f"  {cond}: " + " | ".join(cells))
    lines += ["", "statuses:"]
    for cond in sorted(by):
        st = [r["status"] for rs in by[cond].values() for r in rs]
        lines.append(f"  {cond}: " + ", ".join(f"{s} {st.count(s)}" for s in sorted(set(st))))
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "dimer" and not args.condition:
        raise SystemExit("dimer needs --condition")
    {"dimer": dimer_runs, "analyze": analyze}[args.mode](args, out)


if __name__ == "__main__":
    main()
