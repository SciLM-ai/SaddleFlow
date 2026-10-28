"""Is Sella on UMA reproducible, and if not, why? Identical starts gave different results 59-72% of the time.

Four possible sources are separated:
  - UMA's random edge rotation: eSCN rotates each edge onto a fixed axis, and the roll about the edge is drawn at
    random on every call (fairchem/core/models/uma/common/rotation.py, gamma = rand * 2 pi). The result is the same
    in exact arithmetic but not in float32, so forces change at ~1e-6 eV/A on every call, CPU or GPU. Re-seeding
    PyTorch's generator before every force call fixes the rolls.
  - UMA on the GPU: atomic additions in message passing (PyTorch's deterministic mode).
  - Sella's own GPU linear algebra, used for matrices of dimension >= 200 (the 78-atom cell, not the 35-atom one).
  - Sella's unseeded random fallback in its eigensolver.
A last question is chaos: even a fully deterministic run may change its result when the start moves by 1e-6 A.

    select    38 starts on the two cost-check minima: the repeated nudge starts of the hint search
              (whose repeats mostly disagreed), model landing points from noise and from hint starts (CPU)
    forces    tier 1, one --config per process (gpu_default | gpu_det | cpu | cpu_1thread): UMA forces repeated on the same
              structures (largest change, bitwise-identical count, PyTorch's warnings about nondeterministic
              operations), and Sella's GPU eigensolver repeated on one Hessian (GPU configs)
    sella     tier 2, one --condition per process; every start run --repeats times with the production settings
              (fmax 0.01, <= 300 steps, <= 600 calls, delta0 0.1, drift/energy stops):
                C0  production: UMA on the GPU, Sella's GPU linear algebra on, numpy unseeded
                C1  C0 + PyTorch re-seeded before every force call (fixes UMA's random edge rolls)
                C2  C1 + Sella's GPU linear algebra off (SELLA_DISABLE_GPU=1) + numpy seeded per run
                C3  C2 + PyTorch deterministic mode, TF32 off
                C4  C3 with the start moved 1e-6 A (a different random move per repeat)
                C5  C3 with the start moved 1e-4 A
                C6  CPU, one thread, re-seeded per force call, Sella CPU, numpy seeded: the deterministic
                    reference (one nudge, one model and one hint start per minimum, 2 repeats; slow)
                C7  C6 with the start moved 1e-6 A: chaos on a fully deterministic setup
                C8  C6 with the start moved 1e-4 A
              On the GPU no setting makes UMA's forces bitwise reproducible (index_reduce has no deterministic
              CUDA implementation), so the clean chaos test is C7/C8 against C6, on the CPU.
              Every run counts how often Sella's random fallback and its GPU paths were used.
    analyze   per condition: bitwise-identical repeats, same final point (0.1 A) and status, the step at which
              repeats first differ, final distances; tier-1 numbers; one report (CPU)

    python examples/MaterialsSaddles/experiments/sella_determinism/sella_determinism.py <mode> --out-dir <dir> [...]

Run on one GPU model only (the batch script pins A100 80GB); different GPU models need not agree.
"""
import os
import sys

CONDITIONS = {
    "C0": dict(seed_uma=False, sella_gpu=True, seed=False, deterministic=False, device="cuda", perturb=0.0),
    "C1": dict(seed_uma=True, sella_gpu=True, seed=False, deterministic=False, device="cuda", perturb=0.0),
    "C2": dict(seed_uma=True, sella_gpu=False, seed=True, deterministic=False, device="cuda", perturb=0.0),
    "C3": dict(seed_uma=True, sella_gpu=False, seed=True, deterministic=True, device="cuda", perturb=0.0),
    "C4": dict(seed_uma=True, sella_gpu=False, seed=True, deterministic=True, device="cuda", perturb=1e-6),
    "C5": dict(seed_uma=True, sella_gpu=False, seed=True, deterministic=True, device="cuda", perturb=1e-4),
    "C6": dict(seed_uma=True, sella_gpu=False, seed=True, deterministic=False, device="cpu", perturb=0.0, threads=1),
    "C7": dict(seed_uma=True, sella_gpu=False, seed=True, deterministic=False, device="cpu", perturb=1e-6, threads=1),
    "C8": dict(seed_uma=True, sella_gpu=False, seed=True, deterministic=False, device="cpu", perturb=1e-4, threads=1),
}
FORCE_CONFIGS = {"gpu_default": dict(device="cuda", deterministic=False, seed_uma=False),
                 "gpu_seeded": dict(device="cuda", deterministic=False, seed_uma=True),
                 "gpu_det_seeded": dict(device="cuda", deterministic=True, seed_uma=True),
                 "cpu": dict(device="cpu", deterministic=False, seed_uma=False),
                 "cpu_seeded": dict(device="cpu", deterministic=False, seed_uma=True),
                 "cpu_1thread_seeded": dict(device="cpu", deterministic=False, seed_uma=True, threads=1)}


def _argv_value(flag):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


# These settings must be in place before torch, fairchem or sella are imported.
_SETTING = CONDITIONS.get(_argv_value("--condition")) or FORCE_CONFIGS.get(_argv_value("--config")) or {}
if _SETTING.get("sella_gpu") is False:
    os.environ["SELLA_DISABLE_GPU"] = "1"
if _SETTING.get("deterministic"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "hint_search"))
import hint_search as hs  # noqa: E402
import mode_search as ms  # noqa: E402
from hint_search import max_mic_untranslated, read_rows, write_rows  # noqa: E402

RUN_DIR = "/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_21421000"
HINT_DIR = "/scratch/gqs7zr/saddleflow/mp20bat_hint_search"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["select", "forces", "sella", "analyze"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--condition", choices=sorted(CONDITIONS))
    p.add_argument("--config", choices=sorted(FORCE_CONFIGS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--force-repeats", type=int, default=50)
    p.add_argument("--max-starts", type=int, default=0, help="first N starts only (tests; the CPU conditions use 6)")
    p.add_argument("--max-steps", type=int, default=hs.SELLA_STEPS, help="Sella step cap (lower only for tests)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def make_torch_deterministic():
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def uma_calculator(device, seed_uma):
    """The UMA calculator; with seed_uma, PyTorch is re-seeded before every evaluation, which fixes the random
    edge rolls (and any other draw) so identical structures give identical forces."""
    import torch
    calc = hs.calculator(device)
    if seed_uma:
        original = calc.calculate

        def calculate(*a, **k):
            torch.manual_seed(0)          # also seeds every CUDA device
            return original(*a, **k)
        calc.calculate = calculate
    return calc


def minima_and_plan():
    plan = json.loads((Path(HINT_DIR) / "plan.json").read_text())
    from ase.io import read
    by_tid = {int(m.info["tid"]): m for m in read(Path(RUN_DIR) / "minima.traj", index=":")}
    rows = plan["minima"][:2]
    return plan, rows, {r["tid"]: by_tid[r["tid"]] for r in rows}


# ---------------------------------------------------------------- select
def select(args, out):
    plan, rows, minima = minima_and_plan()
    rng = np.random.default_rng(0)
    plan_rows = {r["tid"]: r for r in rows}
    model = json.loads((Path(HINT_DIR) / "model_landings.json").read_text())
    hints = json.loads((Path(HINT_DIR) / "hint_start" / "landings.json").read_text())
    starts = []
    for tid, m in minima.items():
        # The nudge starts that were repeated in the hint search (their repeats mostly disagreed).
        reps = [s for s in plan["specs"] if s["tid"] == tid and s["repeat"] == 1 and s["arm"].startswith("nudge")]
        for s in [reps[i] for i in rng.permutation(len(reps))[:6]]:
            pos, d = hs.search_start(m, s, plan_rows[tid])
            pos = pos + float(s["arm"][len("nudge"):]) * d
            starts.append(dict(id=f"{s['arm']}:{tid}:{s['epicenter']}:{s['direction']}", tid=tid, kind="nudge",
                               pos=pos.tolist()))
        for kind, pool, key in (("model", [r for r in model if r["tid"] == tid and r["arm"] == "model"], "pos"),
                                ("hint", [r for r in hints if r["tid"] == tid], "landing")):
            for r in [pool[i] for i in rng.permutation(len(pool))[:7]]:
                rid = f"{kind}:{tid}:{r.get('draw', r.get('id'))}"
                starts.append(dict(id=rid, tid=tid, kind=kind, pos=r[key]))
    (out / "starts.json").write_text(json.dumps(starts))
    print(f"{len(starts)} starts: " + ", ".join(f"{k} {sum(s['kind'] == k for s in starts)}" for k in ("nudge", "model", "hint")))


# ---------------------------------------------------------------- tier 1: forces and Sella's GPU eigensolver
def forces(args, out):
    import torch
    cfg = FORCE_CONFIGS[args.config]
    if cfg.get("threads"):
        torch.set_num_threads(cfg["threads"])
    if cfg["deterministic"]:
        make_torch_deterministic()
    plan, rows, minima = minima_and_plan()
    hints = json.loads((Path(HINT_DIR) / "hint_start" / "landings.json").read_text())
    structures = [(f"minimum {tid} ({len(m)} atoms)", m) for tid, m in minima.items()]
    landing = next(r for r in hints if r["tid"] == rows[1]["tid"])
    a = minima[rows[1]["tid"]].copy(); a.positions = np.array(landing["landing"])
    structures.append((f"a landing point of minimum {rows[1]['tid']}", a))
    calc = uma_calculator(cfg["device"], cfg["seed_uma"])
    if cfg["deterministic"]:
        make_torch_deterministic()          # again, in case loading the model changed a setting
    n = args.force_repeats if cfg["device"] == "cuda" else min(args.force_repeats, 6)
    result = dict(config=args.config, torch=torch.__version__, tf32=torch.backends.cuda.matmul.allow_tf32,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, structures=[])
    seen_warnings = set()
    for name, atoms in structures:
        F, E = [], []
        for _ in range(n):
            b = atoms.copy(); calc.reset(); b.calc = calc
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                F.append(b.get_forces().copy()); E.append(float(b.get_potential_energy()))
            seen_warnings |= {str(w.message)[:200] for w in caught}
        F = np.array(F)
        # The first call can differ only because of warm-up, so evaluations 2..n are also compared with call 2.
        result["structures"].append(dict(name=name, repeats=n, max_force_change=float(np.abs(F - F[0]).max()),
                                         max_energy_change=float(np.ptp(E)),
                                         identical=int(sum(np.array_equal(f, F[0]) for f in F)),
                                         identical_after_first=int(sum(np.array_equal(f, F[1]) for f in F[1:])),
                                         max_change_after_first=float(np.abs(F[1:] - F[1]).max())))
        print(f"  {name}: largest force change {result['structures'][-1]['max_force_change']:.3e} eV/A, "
              f"{result['structures'][-1]['identical']}/{n} bitwise identical", flush=True)
    result["nondeterminism_warnings"] = sorted(seen_warnings)
    if cfg["device"] == "cuda":
        # Sella's eigensolver (GPU for dimension >= 200) on one Hessian with three zero translation modes.
        from saddleflow.classical_saddle_search import PESBackend
        import sella._gpu as sg
        m = minima[rows[1]["tid"]]
        pes = PESBackend(m.copy(), calc)
        H, _ = pes.free_hessian(pes.dofs.pack(m.positions), eps=0.002)
        outs = [sg.gpu_eigh(H.copy()) for _ in range(20)]
        result["eigh"] = dict(dimension=int(H.shape[0]), uses_gpu=bool(sg._gpu_ok(H.shape[0])),
                              identical_values=int(sum(np.array_equal(o[0], outs[0][0]) for o in outs)),
                              identical_vectors=int(sum(np.array_equal(o[1], outs[0][1]) for o in outs)),
                              max_vector_change=float(max(np.abs(o[1] - outs[0][1]).max() for o in outs)),
                              lowest_values=np.round(outs[0][0][:5], 6).tolist())
        print(f"  Sella eigh ({H.shape[0]} dims, GPU {result['eigh']['uses_gpu']}): {result['eigh']['identical_vectors']}/20 "
              f"identical eigenvectors", flush=True)
    write_rows(out / f"forces_{args.config}.json", [result])


# ---------------------------------------------------------------- tier 2: repeated Sella runs
class Counters:
    """Counts uses of Sella's random fallback and of its GPU paths during one run."""

    def __init__(self):
        import sella._gpu as sg
        self.random_normal = 0
        self.gpu = 0
        original_normal, original_ok, original_mm = np.random.normal, sg._gpu_ok, sg._gpu_matmul_ok

        def normal(*a, **k):
            self.random_normal += 1
            return original_normal(*a, **k)

        def ok(n):
            r = original_ok(n); self.gpu += bool(r); return r

        def mm(n):
            r = original_mm(n); self.gpu += bool(r); return r
        np.random.normal, sg._gpu_ok, sg._gpu_matmul_ok = normal, ok, mm

    def reset(self):
        self.random_normal = 0
        self.gpu = 0


def sella_runs(args, out):
    import torch
    cond = CONDITIONS[args.condition]
    if cond.get("threads"):
        torch.set_num_threads(cond["threads"])     # multithreaded CPU reductions are not reproducible
    if cond["deterministic"]:
        make_torch_deterministic()
    from saddleflow.classical_saddle_search import PESBackend, run_sella
    plan, rows, minima = minima_and_plan()
    starts = json.loads((out / "starts.json").read_text())
    if cond["device"] == "cpu":
        # One start of each kind per minimum: 6 starts, 2 repeats.
        starts = [next(x for x in starts if x["tid"] == t and x["kind"] == k) for t in minima for k in ("nudge", "model", "hint")]
        args.repeats = min(args.repeats, 2)
    if args.max_starts:
        starts = starts[:args.max_starts]
    calc = uma_calculator(cond["device"], cond["seed_uma"])
    if cond["deterministic"]:
        make_torch_deterministic()
    counters = Counters()
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
            if cond["seed"]:
                np.random.seed(12345)
            counters.reset()
            pes.force_calls = 0
            res = run_sella(pes, x0, fmax_tol=hs.FMAX, max_steps=args.max_steps, max_force_calls=hs.SELLA_CALLS,
                            delta0=0.1, stop=ms.stop_rule(m, e_min[s["tid"]], pes.dofs))
            steps = [hashlib.sha1(np.ascontiguousarray(r["x"]).tobytes()).hexdigest()[:16] for r in res["trace"]]
            done_rows.append(dict(start=s["id"], tid=s["tid"], kind=s["kind"], condition=args.condition, repeat=rep,
                                  status=res["status"], steps=res["steps"], force_calls=res["force_calls"],
                                  final=np.asarray(res["positions"]).tolist(), step_hashes=steps,
                                  step_fmax=[float(r["force_max"]) for r in res["trace"]],
                                  random_normal_calls=counters.random_normal, gpu_calls=counters.gpu))
            write_rows(path, done_rows)
            print(f"  {s['id']} repeat {rep}: {res['status']}, {res['steps']} steps, random {counters.random_normal}, "
                  f"GPU {counters.gpu}", flush=True)
    print(f"{args.condition} shard {args.shard}: {len(done_rows)} runs")


# ---------------------------------------------------------------- analyze
def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def analyze(args, out):
    plan, rows, minima = minima_and_plan()
    runs = read_rows(out / "runs_*.json")
    lines = ["Sella-on-UMA determinism", "=" * 100, ""]
    for path in sorted(out.glob("forces_*.json")):
        r = json.loads(path.read_text())[0]
        lines.append(f"tier 1, {r['config']} (torch {r['torch']}, GPU {r['gpu']}, TF32 {r['tf32']}):")
        for s in r["structures"]:
            lines.append(f"  {s['name']:34s} {s['identical']:3d}/{s['repeats']} bitwise-identical to the first call "
                         f"({s.get('identical_after_first', 0)}/{s['repeats'] - 1} identical among calls 2..n); largest force change "
                         f"{s['max_force_change']:.2e} eV/A ({s.get('max_change_after_first', float('nan')):.2e} after the first call), "
                         f"energy {s['max_energy_change']:.2e} eV")
        if "eigh" in r:
            e = r["eigh"]
            lines.append(f"  Sella eigensolver, {e['dimension']} dims (GPU path {e['uses_gpu']}): {e['identical_values']}/20 identical "
                         f"eigenvalues, {e['identical_vectors']}/20 identical eigenvectors (largest change {e['max_vector_change']:.2e}); "
                         f"lowest eigenvalues {e['lowest_values']}")
        for w in r["nondeterminism_warnings"][:5]:
            lines.append(f"  warning: {w}")
        lines.append("")
    lines += ["tier 2: repeated Sella runs from identical starts",
              "cond  runs  pairs  bitwise-identical  same point+status  median first-differing step  final distance p50/p90 (A)  "
              "random fallback runs  GPU-path runs"]
    by = {}
    for r in runs:
        by.setdefault(r["condition"], {}).setdefault(r["start"], []).append(r)
    for cond in sorted(by):
        pairs = []
        for start, rs in by[cond].items():
            rs = sorted(rs, key=lambda r: r["repeat"])
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    pairs.append((rs[i], rs[j]))
        if not pairs:
            continue
        m_of = lambda r: minima[r["tid"]]
        same_bits = [np.array_equal(np.array(a["final"]), np.array(b["final"])) for a, b in pairs]
        dist = [max_mic_untranslated(hs.at(m_of(a), a["final"]), hs.at(m_of(a), b["final"])) for a, b in pairs]
        same_pt = [d < hs.MERGE and (a["status"] == b["status"]) for (a, b), d in zip(pairs, dist)]
        firsts = [first_difference(a["step_hashes"], b["step_hashes"]) for a, b in pairs]
        diverged = [f for f in firsts if f is not None]
        rs_all = [r for rs in by[cond].values() for r in rs]
        lines.append(f"{cond:4s} {len(rs_all):5d} {len(pairs):6d}   {100 * np.mean(same_bits):6.1f}%           "
                     f"{100 * np.mean(same_pt):6.1f}%            "
                     f"{np.median(diverged) if diverged else float('nan'):6.1f} ({len(diverged)}/{len(pairs)} differ)       "
                     f"{np.median(dist):6.3f} / {np.quantile(dist, 0.9):6.3f}           "
                     f"{sum(r['random_normal_calls'] > 0 for r in rs_all):4d}             {sum(r['gpu_calls'] > 0 for r in rs_all):4d}")
    # Sensitivity: C4/C5 against the unperturbed C3 run (GPU), C7/C8 against C6 (deterministic CPU).
    for base, perturbed in (("C3", ("C4", "C5")), ("C6", ("C7", "C8"))):
        if base not in by:
            continue
        ref = {s: sorted(rs, key=lambda r: r["repeat"])[0] for s, rs in by[base].items()}
        for cond in perturbed:
            if cond not in by:
                continue
            d = [max_mic_untranslated(hs.at(minima[r["tid"]], r["final"]), hs.at(minima[r["tid"]], ref[s]["final"]))
                 for s, rs in by[cond].items() if s in ref for r in rs]
            if not d:
                continue
            same = [x < hs.MERGE for x in d]
            lines.append(f"  {cond} against the unperturbed {base} run: {100 * np.mean(same):.1f}% end at the same point "
                         f"(final distance p50 {np.median(d):.3f}, p90 {np.quantile(d, 0.9):.3f} A)")
    lines += ["", "by kind of start, same point+status between repeats:"]
    for cond in sorted(by):
        cells = []
        for kind in ("nudge", "model", "hint"):
            pairs = [(a, b) for rs in by[cond].values() for i, a in enumerate(sorted(rs, key=lambda r: r["repeat"]))
                     for b in sorted(rs, key=lambda r: r["repeat"])[i + 1:] if a["kind"] == kind]
            if pairs:
                ok = [max_mic_untranslated(hs.at(minima[a["tid"]], a["final"]), hs.at(minima[a["tid"]], b["final"])) < hs.MERGE
                      and a["status"] == b["status"] for a, b in pairs]
                cells.append(f"{kind} {100 * np.mean(ok):5.1f}% of {len(pairs)}")
        lines.append(f"  {cond}: " + "; ".join(cells))
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "sella" and not args.condition:
        raise SystemExit("sella needs --condition")
    if args.mode == "forces" and not args.config:
        raise SystemExit("forces needs --config")
    {"select": select, "forces": forces, "sella": sella_runs, "analyze": analyze}[args.mode](args, out)


if __name__ == "__main__":
    main()
