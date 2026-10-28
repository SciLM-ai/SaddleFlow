"""Evaluate the hint fine-tune: the model's landing points alone (stage 1), then Sella and verification (stage 2).

Test minima: the 24 stable minima of the 30-minimum experiment (hint_vs_noise.py; minima with a Hessian eigenvalue
below -0.05 eV/A^2 left out). Reference saddles: every connected index-1 saddle <= 1 eV verified there, merged into
regions as the training labels are (hint_labels.merge: 0.15 A and 50 meV, most-reached first). Saddles verified in
stage 2 are added to the reference, and stage 1 is scored again against the larger set.

Arms on the test minima (starts seeded per minimum and identical across models):
    ft_hint       fine-tuned model on the hint grid: every atom x 16 Fibonacci directions (randomly rotated per
                  atom). Start: the minimum + 0.15 A noise on every atom, the hinted atom pushed 0.6 A along the
                  direction (the training hint start); hint field given at every step; 5 passes x 4 Euler steps
    ft_hint_1x8   the same starts and hints, 1 pass x 8 steps (do the restarts pull different hints together?)
    inference ablations of ft_hint_1x8 (same checkpoint and hints): ft_hint_1x8_s2 a second start-noise draw,
                  ft_hint_1x16 16 Euler steps, ft_hint_heun 8 Heun (second-order) steps, ft_hint_1x4 4 Euler steps
    ft_nohint     the same starts, hint field zero (does the hint input matter, or only the push?)
    rel_hint      the released model from the same starts (it has no hint input)
    ft_noise      fine-tuned model, hint field zero, the 256 noise starts of the 30-minimum experiment (minimum +
                  0.3 A noise, same seeds), paired with the released model's landings from those starts
    rel_noise, rel_mobile_hint, rel_soft, push, softpush
                  the 30-minimum experiment's arms, read from its files (no new runs)

Hint following on the held-out Tier B regions (barrier >= --min-barrier): one start per region from its
largest-move label, the direction tilted within 30 degrees (as the grid would offer it); arms follow_ft (hint
given), follow_nohint (hint zero) and follow_rel (released model).

Forgetting: --n-forget mp20bat test records (seeded), the stored endpoint + 0.3 A noise (one draw, shared by both
models); arms forget_ft (fine-tuned, hint zero) and forget_rel (released).

Stages (each its own job: run_eval.sh, chained by submit_eval.sh; every stage skips work already written):
    land     (GPU; --group) every landing point of every arm, and one UMA-S energy and force evaluation at each
    sella    (GPU; --group, --shard/--nshards) Sella (UMA-S-1.2 omat, fmax 0.01, <= 300 steps, <= 600 calls,
             delta0 0.1, the 30-minimum experiment's stop rule) from the landing points. On the test minima each
             arm's landings are first merged within 0.15 A (distance only: landings are not stationary points, so
             their energies say little), one run per distinct landing, the members sharing its result. Hint
             following and forgetting: every landing; forgetting also gets the full-Hessian index check.
    verify   (GPU; --group, --shard/--nshards) the test minima's converged Sella points, merged within 0.1 A across
             arms; a point within 0.1 A of one the 30-minimum experiment verified keeps that result, the rest get
             hint_vs_noise.check (energy, full Hessian, connectivity)
    report   (CPU) report.txt: stage 1 always, stage 2 where its files exist

    python examples/MaterialsSaddles/experiments/hint_finetune/eval_hints.py <stage> --out-dir <dir> [--ckpt <dir>]
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hint_vs_noise"))
import hint_labels as hl  # noqa: E402
import hint_vs_noise as hvn  # noqa: E402
from hint_vs_noise import hs, ms, read_rows, write_rows  # noqa: E402

OLD_DIR = "/scratch/gqs7zr/saddleflow/hint_vs_noise"
LABELS = "/scratch/gqs7zr/saddleflow/hint_finetune/tierb/labels.json"
TEST_SHARDS = "/scratch/gqs7zr/saddleflow/MaterialsSaddles/mp20bat/test/mp20bat_test_*.aselmdb"
N_GROUPS, GRID_K, PUSH, HINT_NOISE, NOISE_SIGMA, N_NOISE, TILT = 4, 16, 0.6, 0.15, 0.3, 256, 30.0
LAND_MERGE, CUTOFF, RADII, BUDGETS = 0.15, 1.0, (0.05, 0.1, 0.2, 0.3), (5000, 10000, 20000, 40000, 80000)
# Stage-2 cost of a model rollout, in UMA-S force-call equivalents per network evaluation: 0.85 s per start of 20
# evaluations (UMA-M, strict fp32, batch 32) against 42 force calls per GPU-second (8 Sella workers), both measured
# on an RTX Pro 6000. Network evaluations per start for each arm (the Sella-only arms have none).
ROLLOUT_CALLS_PER_EVAL = 0.85 / 20 * 42
EVALS = {"ft_hint": 20, "ft_hint_1x8": 8, "ft_hint_1x8_s2": 8, "ft_hint_1x16": 16, "ft_hint_heun": 16, "ft_hint_1x4": 4,
         "ft_noise": 20, "rel_hint": 20, "rel_noise": 20, "rel_mobile_hint": 20, "rel_soft": 20}

# arm -> (model, starts, hint given, passes, steps)
TEST_ARMS = {"ft_hint": ("ft", "grid", True, 5, 4), "ft_hint_1x8": ("ft", "grid", True, 1, 8),
             "ft_nohint": ("ft", "grid", False, 5, 4), "ft_noise": ("ft", "noise", False, 5, 4),
             "rel_hint": ("rel", "grid", False, 5, 4),
             "ft_hint_1x8_s2": ("ft", "grid", True, 1, 8), "ft_hint_1x16": ("ft", "grid", True, 1, 16),
             "ft_hint_heun": ("ft", "grid", True, 1, 8), "ft_hint_1x4": ("ft", "grid", True, 1, 4)}
ABLATIONS = ("ft_hint_1x8_s2", "ft_hint_1x16", "ft_hint_heun", "ft_hint_1x4")
HEUN = {"ft_hint_heun"}                                 # arms integrated with Heun's method instead of Euler's
START_SEED = {"ft_hint_1x8_s2": 2}                      # arms with a different start-noise draw (default 0)
FOLLOW_ARMS = {"follow_ft": ("ft", "follow", True, 5, 4), "follow_nohint": ("ft", "follow", False, 5, 4),
               "follow_rel": ("rel", "follow", False, 5, 4)}
FORGET_ARMS = {"forget_ft": ("ft", "forget", False, 5, 4), "forget_rel": ("rel", "forget", False, 5, 4)}
ARMS = {**TEST_ARMS, **FOLLOW_ARMS, **FORGET_ARMS}
SELLA_ARMS = ("ft_hint", "ft_hint_1x8", "ft_noise", "rel_hint", *ABLATIONS, "follow_ft", "follow_rel", "forget_ft", "forget_rel")
OLD_ARMS = {"noise": "rel_noise", "hint": "rel_mobile_hint", "soft": "rel_soft", "push": "push", "softpush": "softpush"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["land", "sella", "verify", "report"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", help="land: fine-tune checkpoint directory (model.safetensors, ema.pt; config.json one level up)")
    p.add_argument("--live", action="store_true", help="land: the live weights instead of the EMA")
    p.add_argument("--old-dir", default=OLD_DIR, help="the 30-minimum experiment (hint_vs_noise.py) output directory")
    p.add_argument("--labels", default=LABELS, help="Tier B labels.json (held-out regions for hint following)")
    p.add_argument("--min-barrier", type=float, default=0.05, help="eV; held-out regions below it are not followed")
    p.add_argument("--n-forget", type=int, default=400)
    p.add_argument("--arms", default="", help="land/sella: comma-separated subset of the arms (default all)")
    p.add_argument("--group", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="smoke tests: first N test minima, regions and records per group")
    p.add_argument("--max-hints", type=int, default=0, help="smoke tests: at most N grid hints per minimum")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def chosen(args, pool):
    return [a for a in pool if not args.arms or a in args.arms.split(",")]


# ---------------------------------------------------------------- inputs
def test_minima(args):
    """{tid: (minimum, group)} for the stable minima of the 30-minimum experiment."""
    old = Path(args.old_dir)
    plan = json.loads((old / "plan.json").read_text())
    by_tid = {int(m.info["tid"]): m for m in read(Path(plan["run_dir"]) / "minima.traj", index=":")}
    eig = {}
    for l in read_rows(old / "landings" / "landings_*.json"):
        eig.setdefault(l["tid"], l["soft_eigs"][0])
    rows = [r for r in plan["minima"] if eig.get(r["tid"], -1) >= -0.05]
    out = {}
    for g in range(N_GROUPS):
        mine = [r for r in rows if r["group"] == g]
        for r in mine[:args.limit or None]:
            out[r["tid"]] = (by_tid[r["tid"]], g)
    return out


def follow_regions(args):
    """[(key, minimum, region, label, group)] for the held-out Tier B regions at or above --min-barrier."""
    data = json.loads(Path(args.labels).read_text())["minima"]
    rows = []
    for e in data:
        m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
        for r in e["regions"]:
            if r["held_out"] and r["barrier"] >= args.min_barrier:
                rows.append((f"{e['tid']}:{r['region']}", m, r, r["hints"][0], e))
    out = []
    for g in range(N_GROUPS):
        mine = [(k, m, r, h, e) for i, (k, m, r, h, e) in enumerate(rows) if i % N_GROUPS == g]
        out += [(k, m, r, h, g, e) for k, m, r, h, e in mine[:args.limit or None]]
    return out


def forget_records(args):
    """[(record index, endpoint as Atoms, group)]: --n-forget seeded mp20bat test records."""
    from saddleflow.data import MaterialsSaddlesDataset
    ds = MaterialsSaddlesDataset(sorted(glob.glob(TEST_SHARDS)))
    pick = np.sort(np.random.default_rng(0).choice(len(ds), min(args.n_forget, len(ds)), replace=False))
    out = []
    for i, k in enumerate(pick):
        if args.limit and i // N_GROUPS >= args.limit:
            break
        r = ds[int(k)]
        m = Atoms(numbers=r["Z"].numpy(), positions=r["start_pos"].double().numpy(), cell=r["cell"].double().numpy(), pbc=True)
        if r["fixed"].any():
            from ase.constraints import FixAtoms
            m.set_constraint(FixAtoms(indices=np.flatnonzero(r["fixed"].numpy())))
        out.append((int(k), m, i % N_GROUPS))
    return out


def grid(tid, m, max_hints=0):
    """[(atom, k, unit direction)]: every atom x GRID_K Fibonacci directions, rotated per atom."""
    g = [(a, k, d) for a in range(len(m)) for k, d in enumerate(hs.fibonacci(GRID_K, np.random.default_rng([tid, 20, a])))]
    if max_hints and len(g) > max_hints:
        g = [g[i] for i in np.sort(np.random.default_rng([tid, 22]).choice(len(g), max_hints, replace=False))]
    return g


def starts(args, group, kind, seed=0):
    """[(id, key, minimum, start positions, hint (N, 3) or None, extra)] for one kind of start in one group.
    seed (grid only): 0 the standard start noise, otherwise another draw for the same hints."""
    out = []
    if kind == "grid":
        for tid, (m, g) in test_minima(args).items():
            if g != group:
                continue
            for a, k, d in grid(tid, m, args.max_hints):
                rng = np.random.default_rng([tid, 21, a, k] + ([seed] if seed else []))
                pos = m.positions + HINT_NOISE * rng.normal(size=m.positions.shape)
                pos[a] += PUSH * d
                h = np.zeros((len(m), 3)); h[a] = d
                out.append((f"{tid}:{a}:{k}", tid, m, pos, h, dict(atom=int(a), direction=np.round(d, 6).tolist())))
    elif kind == "noise":
        for tid, (m, g) in test_minima(args).items():
            if g != group:
                continue
            for j in range(N_NOISE):            # the seeds of hint_vs_noise.starts_for (released arm "noise")
                pos = m.positions + NOISE_SIGMA * np.random.default_rng([tid, 1, j]).normal(size=m.positions.shape)
                out.append((f"{tid}:{j}", tid, m, pos, None, dict(idx=j)))
    elif kind == "follow":
        for key, m, r, lab, g, e in follow_regions(args):
            if g != group:
                continue
            rng = np.random.default_rng([e["tid"], r["region"], 30])
            d = hl.tilt(np.array(lab["direction"]), TILT, rng)
            pos = m.positions + HINT_NOISE * rng.normal(size=m.positions.shape)
            pos[lab["atom"]] += PUSH * d
            h = np.zeros((len(m), 3)); h[lab["atom"]] = d
            out.append((key, key, m, pos, h, dict(region=r["region"], atom=lab["atom"], direction=np.round(d, 6).tolist())))
    elif kind == "forget":
        for k, m, g in forget_records(args):
            if g != group:
                continue
            mobile = np.ones((len(m), 1))
            for c in m.constraints:
                mobile[c.get_indices()] = 0
            pos = m.positions + NOISE_SIGMA * mobile * np.random.default_rng([k, 40]).normal(size=m.positions.shape)
            out.append((str(k), k, m, pos, None, {}))
    return out


# ---------------------------------------------------------------- models
def load_finetuned(ckpt, device, live=False):
    """The fine-tuned FlowMatchingLoss module (backbone with HintFiLM, head) with the checkpoint's EMA weights."""
    import torch
    from safetensors.torch import load_file
    import finetune as ft
    ckpt = Path(ckpt)
    cfg = json.loads((ckpt.parent / "config.json").read_text())["extras"]["finetune"]
    loss, _, _ = ft.build(SimpleNamespace(**cfg), device)
    missing, unexpected = loss.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=False)
    missing = [k for k in missing if not k.startswith("backbone.film.")]
    if missing or unexpected:
        raise RuntimeError(f"{ckpt}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    if not live:
        shadow = torch.load(ckpt / "ema.pt", map_location="cpu")["shadow"]
        params = [p for p in loss.parameters() if p.requires_grad]
        if len(shadow) != len(params):
            raise RuntimeError(f"ema.pt has {len(shadow)} tensors, the model {len(params)} trainable ones")
        with torch.no_grad():
            for p, s in zip(params, shadow):
                p.copy_(s.to(p.device))
    return loss.eval()


def roll_finetuned(model, jobs, hinted, passes, steps, batch_size, heun=False):
    """Landing positions for the fine-tuned model: `passes` x `steps` Euler (or Heun) steps, the hint at every step."""
    import torch
    from fairchem.core.datasets.collaters.simple_collater import data_list_collater
    from saddleflow.data.transforms import wrap_positions
    from saddleflow.flow.matching import apply_output_projections, build_atomic_data
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s0 in range(0, len(jobs), batch_size):
            chunk = jobs[s0:s0 + batch_size]
            recs = []
            for _, _, m, _, _, _ in chunk:
                fixed = np.zeros(len(m), bool)
                for c in m.constraints:
                    fixed[c.get_indices()] = True
                recs.append(dict(Z=torch.as_tensor(m.numbers, dtype=torch.long), fixed=torch.as_tensor(fixed),
                                 cell=torch.as_tensor(m.cell.array, dtype=torch.float32)))
            xs = [wrap_positions(torch.as_tensor(j[3], dtype=torch.float32), r["cell"]) for j, r in zip(chunk, recs)]
            h = torch.cat([torch.as_tensor(j[4] if hinted and j[4] is not None else np.zeros((len(j[2]), 3)),
                                           dtype=torch.float32) for j in chunk]).to(dev)
            sizes, B = [len(j[2]) for j in chunk], len(chunk)
            fixed = torch.cat([r["fixed"] for r in recs]).to(dev)

            def velocity(xs, time):
                t = torch.full((B,), time, dtype=torch.float32, device=dev)
                data = [build_atomic_data(x, r["Z"], r["cell"], "omat", 0, 0, r["fixed"]) for x, r in zip(xs, recs)]
                batch = data_list_collater(data, otf_graph=True).to(dev)
                feat = model.backbone(batch, t, batch.batch, hint=h)["node_embedding"]
                feat = model.global_attn(feat, batch.batch)
                v = model.velocity_head(feat, t, batch.batch)
                return torch.split(apply_output_projections(v, fixed, batch.batch, B).cpu().float(), sizes)

            for _ in range(passes):
                for k in range(steps):
                    v = velocity(xs, k / steps)
                    if heun:                    # average the slope here and at the Euler prediction
                        pred = [wrap_positions(x + dv / steps, r["cell"]) for x, dv, r in zip(xs, v, recs)]
                        v = [(a + b) / 2 for a, b in zip(v, velocity(pred, (k + 1) / steps))]
                    xs = [wrap_positions(x + dv / steps, r["cell"]) for x, dv, r in zip(xs, v, recs)]
            out += [x.double().numpy() for x in xs]
    return out


def roll_released(sf, jobs, passes, steps, batch_size):
    structures = []
    for _, _, m, pos, _, _ in jobs:
        s = m.copy(); s.set_positions(pos, apply_constraint=False); structures.append(s)
    props = sf.propose(structures, n=1, sigma=0.0, passes=passes, steps=steps, task_name="omat", batch_size=batch_size)
    return [p[0].positions for p in props]


# ---------------------------------------------------------------- land
def land(args, out):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (out / "landings").mkdir(exist_ok=True)
    todo = [a for a in chosen(args, ARMS) if not (out / "landings" / f"{a}_{args.group}.json").exists()]
    if not todo:
        print(f"group {args.group}: every arm landed"); return
    calc = hs.calculator(args.device)
    cache, e_ref = {}, {}
    for model_kind in ("ft", "rel"):
        arms = [a for a in todo if ARMS[a][0] == model_kind]
        if not arms:
            continue
        if model_kind == "ft":
            if not args.ckpt:
                raise SystemExit("--ckpt is required for the fine-tuned arms")
            model = load_finetuned(args.ckpt, args.device, args.live)
        else:
            from saddleflow import SaddleFlow
            model = SaddleFlow.from_pretrained("SciLM/SaddleFlow", device=args.device)
        for arm in arms:
            _, kind, hinted, passes, steps = ARMS[arm]
            seed = START_SEED.get(arm, 0)
            if (kind, seed) not in cache:
                cache[(kind, seed)] = starts(args, args.group, kind, seed)
            jobs = cache[(kind, seed)]
            t0 = time.perf_counter()
            if model_kind == "ft":
                lands = roll_finetuned(model, jobs, hinted, passes, steps, args.batch_size, heun=arm in HEUN)
            else:
                lands = roll_released(model, jobs, passes, steps, args.batch_size)
            seconds = (time.perf_counter() - t0) / max(1, len(jobs))
            rows = []
            for (jid, key, m, pos, _, extra), x in zip(jobs, lands):
                if key not in e_ref:
                    e_ref[key] = hvn.energy_force(m, m.positions, calc)[0]
                e, f = hvn.energy_force(m, x, calc)
                rows.append(dict(id=f"{arm}:{jid}", arm=arm, key=key, **extra, e_landing=e - e_ref[key], f_landing=f,
                                 size=float(hl.saddle_distance(m, x[None], m.positions)[0]), rollout_seconds=seconds,
                                 start=np.round(pos, 5).tolist(), landing=np.round(x, 5).tolist()))
            write_rows(out / "landings" / f"{arm}_{args.group}.json", rows)
            print(f"group {args.group} {arm}: {len(rows)} landings, {seconds:.3f} s per start", flush=True)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- sella
def atoms_of(args, kind):
    """{key: minimum or record endpoint} for one kind of start (every group)."""
    if kind in ("grid", "noise"):
        return {t: m for t, (m, _) in test_minima(args).items()}
    if kind == "follow":
        return {k: m for k, m, *_ in follow_regions(args)}
    return {k: m for k, m, _ in forget_records(args)}


def sella_jobs(args, out):
    """Every Sella run of the group, in a fixed order: (id, arm, key, start)."""
    jobs = []
    for arm in chosen(args, SELLA_ARMS):
        path = out / "landings" / f"{arm}_{args.group}.json"
        if not path.exists():
            continue
        L = sorted(json.loads(path.read_text()), key=lambda l: l["id"])
        if ARMS[arm][1] in ("grid", "noise"):
            minima = atoms_of(args, "grid")
            for key in sorted({l["key"] for l in L}):
                mine = [l for l in L if l["key"] == key]
                gid, reps = hl.merge(minima[key], [l["landing"] for l in mine], [0.0] * len(mine), [0] * len(mine),
                                     radius=LAND_MERGE, de=np.inf)
                for k, r in enumerate(reps):
                    jobs.append((mine[r]["id"], arm, key, mine[r]["landing"],
                                 [mine[i]["id"] for i in np.flatnonzero(gid == k)]))
        else:
            jobs += [(l["id"], arm, l["key"], l["landing"], [l["id"]]) for l in L]
    return jobs


def sella(args, out):
    from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, run_sella
    jobs = sella_jobs(args, out)
    atoms = {kind: atoms_of(args, kind) for kind in {"grid" if ARMS[a][1] == "noise" else ARMS[a][1] for _, a, _, _, _ in jobs}}
    path = out / "runs" / f"runs_{args.group}_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    done_rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in read_rows(path.parent / f"runs_{args.group}_*.json")}
    calc = hs.calculator(args.device)
    e_ref = {}
    for rid, arm, key, start, members in jobs[args.shard::args.nshards]:
        if rid in done:
            continue
        kind = ARMS[arm][1]
        m = atoms["grid" if kind == "noise" else kind][key]
        pes = PESBackend(m.copy(), calc)
        if (kind, key) not in e_ref:
            e_ref[(kind, key)] = pes.energy(pes.dofs.pack(m.positions))
        row = dict(id=rid, arm=arm, key=key, members=members)
        try:
            x0 = pes.dofs.pack(np.array(start))
            e_start = pes.energy(x0)
            before, t0 = pes.force_calls, time.perf_counter()
            res = run_sella(pes, x0, fmax_tol=hs.FMAX, max_steps=hs.SELLA_STEPS, max_force_calls=before + hs.SELLA_CALLS,
                            delta0=0.1, stop=ms.stop_rule(m, e_ref[(kind, key)], pes.dofs, e_start))
            seconds = time.perf_counter() - t0
            pes.max_force_calls = None
            _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
            final = np.asarray(res["positions"])
            row.update(start_rise=e_start - e_ref[(kind, key)], status=res["status"], sella_calls=pes.force_calls - before,
                       sella_steps=res["steps"], seconds=seconds, fmax=fmax,
                       conv=bool(fmax <= hs.FMAX and res["status"] == "converged"),
                       moved=float(hl.saddle_distance(m, final[None], np.array(start))[0]), pos=np.round(final, 5).tolist())
            if kind == "forget" and row["conv"]:
                before = pes.force_calls
                values, _, _ = MinimumModeEstimator(backend="dense", eps=0.002).lowest_modes(pes, pes.dofs.pack(final), n_modes=2)
                row.update(eigs=values.tolist(), nneg=int(np.sum(values < -0.01)), index_calls=pes.force_calls - before,
                           barrier=pes.energy(pes.dofs.pack(final)) - e_ref[(kind, key)])
        except Exception as exc:
            row.update(status="error", err=str(exc)[:200])
        done_rows.append(row)
        write_rows(path, done_rows)
    print(f"group {args.group} shard {args.shard}: {len(done_rows)} runs")


# ---------------------------------------------------------------- verify
def old_points(args, minima):
    """{tid: [verified point]} of the 30-minimum experiment, one row per distinct point."""
    points = {}
    for c in read_rows(Path(args.old_dir) / "verify" / "verify_*.json"):
        if c["tid"] not in minima or c.get("status") == "error":
            continue
        P = points.setdefault(c["tid"], [])
        if not P or hl.saddle_distance(minima[c["tid"]], np.array([p["pos"] for p in P]), np.array(c["pos"])).min() >= hs.MERGE:
            P.append(dict(c, source="old"))
    return points


def verify(args, out):
    from saddleflow.classical_saddle_search import PESBackend
    minima = {t: m for t, (m, g) in test_minima(args).items() if g == args.group}
    runs = [dict(r, tid=r["key"]) for r in read_rows(out / "runs" / f"runs_{args.group}_*.json")
            if ARMS[r["arm"]][1] in ("grid", "noise")]
    old = old_points(args, minima)
    path = out / "verify" / f"verify_{args.group}_{args.shard}.json"; path.parent.mkdir(exist_ok=True)
    for c in read_rows(path.parent / f"verify_{args.group}_*.json"):    # verified here before (later arms added)
        if c["tid"] in minima:
            old.setdefault(c["tid"], []).append(c)
    groups = [g for g in hvn.distinct_points(runs, minima)
              if not old.get(g["tid"]) or hl.saddle_distance(minima[g["tid"]], np.array([p["pos"] for p in old[g["tid"]]]),
                                                             np.array(g["pos"])).min() >= hs.MERGE]
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
    print(f"group {args.group} shard {args.shard}: {len(done_rows)} new distinct points checked "
          f"({len(groups)} not verified before)")


# ---------------------------------------------------------------- report
def reference(args, out, minima):
    """Per test minimum: the verified points (old and new), the connected index-1 points <= 1 eV with their region,
    and the regions (representative position, barrier, labels, whether only stage 2 found them)."""
    points = old_points(args, minima)
    for c in read_rows(out / "verify" / "verify_*.json"):
        if c.get("status") != "error":
            points.setdefault(c["tid"], []).append(dict(c, source="new"))
    ref = {}
    for t, m in minima.items():
        S = [p for p in points.get(t, []) if p.get("nneg") == 1 and p.get("connected") and p["barrier"] <= CUTOFF]
        if not S:
            ref[t] = dict(S=[], P=np.zeros((0, len(m), 3)), region=np.zeros(0, int), regions=[]); continue
        gid, reps = hl.merge(m, [s["pos"] for s in S], [s["barrier"] for s in S], [len(s.get("members", [])) for s in S])
        regions = [dict(pos=np.array(S[r]["pos"]), barrier=S[r]["barrier"], labels=hl.make_hints(m, S[r]["pos"]),
                        new=all(S[i]["source"] == "new" for i in np.flatnonzero(gid == k))) for k, r in enumerate(reps)]
        ref[t] = dict(S=S, P=np.array([s["pos"] for s in S]), region=gid, regions=regions)
    return ref


def nearest_region(m, ref, pos, radius):
    """(region index or -1, distance to the nearest region representative) for one structure."""
    if not ref["regions"]:
        return -1, np.inf
    d = hl.saddle_distance(m, np.array([r["pos"] for r in ref["regions"]]), np.array(pos))
    k = int(d.argmin())
    return (k if d[k] <= radius else -1), float(d[k])


def endpoint_region(m, ref, pos):
    """The region of a Sella endpoint: that of the nearest connected point within 0.1 A, else -1."""
    if not len(ref["S"]):
        return -1
    d = hl.saddle_distance(m, ref["P"], np.array(pos))
    k = int(d.argmin())
    return int(ref["region"][k]) if d[k] < hs.MERGE else -1


def followed(region, atom, direction):
    """Does the hint (atom, direction) point to the region: a label on that atom within TILT degrees?"""
    return any(a == atom and np.dot(u, direction) >= np.cos(np.radians(TILT)) for a, u, _ in region["labels"])


def paired(x, y):
    from scipy.stats import wilcoxon
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = ~(np.isnan(x) | np.isnan(y))
    if k.sum() < 3 or np.allclose(x[k], y[k]):
        return "n/a"
    return f"{wilcoxon(x[k], y[k]).pvalue:.3g}"


def report(args, out):
    minima = {t: m for t, (m, _) in test_minima(args).items()}
    ref = reference(args, out, minima)
    n_new = sum(r["new"] for t in ref for r in ref[t]["regions"])
    n_reg = sum(len(ref[t]["regions"]) for t in ref)
    L = [f"Hint fine-tune evaluation: {out}", "=" * 118, "",
         f"test minima {len(minima)}; reference: {n_reg} regions (connected index-1 saddles <= {CUTOFF} eV, merged at "
         f"{hl.MERGE_RADIUS} A / {1000 * hl.MERGE_DE:.0f} meV), {n_new} of them found only in stage 2", ""]

    # Landings of every arm on the test minima.
    lands = {}
    for l in read_rows(out / "landings" / "*.json"):
        if l["arm"] in TEST_ARMS and l["key"] in minima:
            lands.setdefault(l["arm"], []).append(l)
    for l in read_rows(Path(args.old_dir) / "landings" / "landings_*.json"):
        if l["tid"] in minima:
            lands.setdefault(OLD_ARMS[l["arm"]], []).append(dict(l, key=l["tid"], atom=l.get("epicenter"),
                                                                direction=l.get("vector")))
    order = [a for a in ("ft_hint", "ft_hint_1x8", *ABLATIONS, "ft_nohint", "rel_hint", "ft_noise", "rel_noise", "rel_mobile_hint", "rel_soft")
             if a in lands]

    # Stage 1.
    per_min = {}
    L += ["STAGE 1: the landing points alone, against the reference (no Sella)",
          "  precision: distance from each landing to the nearest reference saddle (largest per-atom move, minimum image)",
          "  coverage: per minimum, the share of its reference saddles with a landing within the radius (mean over minima)",
          "  distinct: landings merged within 0.15 A, per minimum (mean); unknown: distinct landings > 0.3 A from every "
          "reference saddle",
          "  hint followed: of the landings within 0.2 A of a reference saddle, the share whose saddle has a label on the "
          "hinted atom within 30 deg of the hint",
          "",
          "  arm               landings | dist p50 | <=0.05  <=0.1  <=0.2  <=0.3 | coverage 0.1  0.2  0.3  (0.2, >=0.05 eV) "
          "(0.2, first 256) | distinct unknown | fmax p50  E p50 | followed"]
    for arm in order:
        A = lands[arm]
        dists, cov, cov_floor, cov256, distinct, unknown, follow = [], {r: [] for r in RADII}, [], [], [], [], []
        per_min[arm] = {}
        for t, m in minima.items():
            mine = sorted((l for l in A if l["key"] == t), key=lambda l: l["id"])
            regs = ref[t]["regions"]
            if not mine:
                continue
            X = np.array([l["landing"] for l in mine])
            D = (np.stack([hl.saddle_distance(m, X, r["pos"]) for r in regs], 1) if regs else np.full((len(mine), 1), np.inf))
            near = D.min(1); dists += near.tolist()
            for r in RADII:
                cov[r].append(np.mean((D <= r).any(0)) if regs else np.nan)
            floor = [k for k, r in enumerate(regs) if r["barrier"] >= args.min_barrier]
            cov_floor.append(np.mean((D[:, floor] <= 0.2).any(0)) if floor else np.nan)
            pick = np.random.default_rng([t, 50]).permutation(len(mine))[:256]
            cov256.append(np.mean((D[pick] <= 0.2).any(0)) if regs else np.nan)
            gid, reps = hl.merge(m, X, [0.0] * len(X), [0] * len(X), radius=LAND_MERGE, de=np.inf)
            distinct.append(len(reps)); unknown.append(int(np.sum(near[reps] > 0.3)))
            if mine[0].get("atom") is not None and regs:
                for i, l in enumerate(mine):
                    if near[i] <= 0.2:
                        follow.append(followed(regs[int(D[i].argmin())], l["atom"], np.array(l["direction"])))
            per_min[arm][t] = dict(cov=cov[0.2][-1], cov256=cov256[-1], distinct=len(reps))
        f = np.array([l["f_landing"] for l in A if l["key"] in minima]); e = np.array([l["e_landing"] for l in A if l["key"] in minima])
        dists = np.array(dists)
        L.append(f"  {arm:16s} {len(dists):8d} | {np.median(dists):7.3f}  | " + "  ".join(f"{100 * np.mean(dists <= r):5.1f}" for r in RADII)
                 + " | " + "  ".join(f"{100 * np.nanmean(cov[r]):5.1f}" for r in RADII[1:])
                 + f"   {100 * np.nanmean(cov_floor):5.1f}             {100 * np.nanmean(cov256):5.1f}         "
                 f"| {np.mean(distinct):7.1f} {np.mean(unknown):6.1f}  | {np.median(f):7.2f} {np.median(e):6.2f} | "
                 + (f"{100 * np.mean(follow):5.1f}%" if follow else "  -"))
    L += ["", "  paired across minima (Wilcoxon p): coverage within 0.2 A, and within 0.2 A after the first 256 landings"]
    for a, b in (("ft_hint", "rel_hint"), ("ft_hint", "ft_nohint"), ("ft_hint", "ft_hint_1x8"), ("ft_hint", "rel_noise"),
                 ("ft_noise", "rel_noise"), ("ft_hint", "rel_mobile_hint"), *(("ft_hint_1x8", a) for a in ABLATIONS)):
        if a in per_min and b in per_min:
            ts = [t for t in minima if t in per_min[a] and t in per_min[b]]
            x = [per_min[a][t]["cov"] for t in ts]; y = [per_min[b][t]["cov"] for t in ts]
            x2 = [per_min[a][t]["cov256"] for t in ts]; y2 = [per_min[b][t]["cov256"] for t in ts]
            L.append(f"    {a:12s} vs {b:16s} {100 * np.nanmean(x):5.1f}% vs {100 * np.nanmean(y):5.1f}%  p {paired(x, y):>7s}   |  "
                     f"first 256: {100 * np.nanmean(x2):5.1f}% vs {100 * np.nanmean(y2):5.1f}%  p {paired(x2, y2):>7s}")

    # Stage 2 on the test minima.
    runs = {}
    for r in read_rows(out / "runs" / "runs_*.json"):
        runs.setdefault(r["arm"], []).append(r)
    old_runs = {}
    for r in read_rows(Path(args.old_dir) / "runs" / "runs_*.json"):
        if r["tid"] in minima and r["arm"] in OLD_ARMS:
            old_runs.setdefault(OLD_ARMS[r["arm"]], []).append(dict(r, key=r["tid"], members=[r["id"]]))
    # The released model's arms as if merged like ours: only the representative of each merged group runs.
    for arm in ("rel_noise", "rel_mobile_hint", "rel_soft"):
        if arm in old_runs and arm in lands:
            by_id = {r["id"]: r for r in old_runs[arm]}
            merged = []
            for t, m in minima.items():
                mine = sorted((l for l in lands[arm] if l["key"] == t), key=lambda l: l["id"])
                if not mine:
                    continue
                gid, reps = hl.merge(m, [l["landing"] for l in mine], [0.0] * len(mine), [0] * len(mine), radius=LAND_MERGE, de=np.inf)
                for k, rp in enumerate(reps):
                    if mine[rp]["id"] in by_id:
                        merged.append(dict(by_id[mine[rp]["id"]], members=[mine[i]["id"] for i in np.flatnonzero(gid == k)]))
            old_runs[arm + "_merged"] = merged
    stage2 = [a for a in ("ft_hint", "ft_hint_1x8", *ABLATIONS, "ft_noise", "rel_hint") if a in runs] + \
             [a for a in ("rel_noise_merged", "rel_mobile_hint_merged", "rel_soft_merged", "rel_noise", "rel_mobile_hint",
                          "rel_soft", "push", "softpush") if a in old_runs]
    if any(a in runs for a in TEST_ARMS):
        allruns = {**runs, **old_runs}
        hessian = {}                                    # the soft-mode Hessian each soft arm needs, once per minimum
        for l in lands.get("rel_soft", []):
            hessian.setdefault(l["key"], l["hessian_calls"])
        # Regions the Sella-only arms (push, softpush) reach on each minimum.
        sella_only = {t: {endpoint_region(m, ref[t], r["pos"]) for a in ("push", "softpush") for r in old_runs.get(a, [])
                          if r["key"] == t and r.get("conv")} - {-1} for t, m in minima.items()}
        L += ["", "", "STAGE 2: Sella from the landing points (test minima)",
              "  runs: Sella runs (merged arms: one per distinct landing); conv: converged; calls: Sella force calls per run;",
              "  moved: distance from the landing to the Sella saddle, for runs ending at a connected index-1 saddle <= 1 eV",
              "  (the paper metric); saddles: distinct reference regions reached per minimum (mean), its share of the minimum's",
              "  regions, and new: regions only stage 2 found; vs Sella only: regions per minimum (mean) this arm reaches",
              "  that push and softpush together do not / that they reach and this arm does not.",
              f"  cost: every start is charged its model rollout, {ROLLOUT_CALLS_PER_EVAL:.2f} force-call equivalents per "
              "network evaluation (0.85 s per 20-evaluation start against 42",
              "  UMA-S force calls per GPU-second, both measured on an RTX Pro 6000), whether or not its landing got its own",
              "  Sella run; the first start of each merged group also pays that group's Sella calls; soft arms pay the",
              "  minimum's soft-mode Hessian up front. total: mean compute per minimum. budget: regions reached within the",
              "  first N force-call equivalents (starts in random order), mean over minima.",
              "",
              "  arm                       runs  conv%  calls p50 mean | moved p50 mean | saddles share   new | vs Sella only "
              "|  total | " + "  ".join(f"{b // 1000:>3d}k" for b in BUDGETS)]
        keep = {}                                       # arm -> minimum -> (regions reached, (cost, region) in order, moved per region)
        for arm in stage2:
            R = [r for r in allruns[arm] if r["key"] in minima]
            calls = np.array([r.get("sella_calls", 0) for r in R])
            per_start = ROLLOUT_CALLS_PER_EVAL * EVALS.get(arm.removesuffix("_merged"), 0)
            moved, found, share, new, budget = [], [], [], [], {b: [] for b in BUDGETS}
            gain, miss, total = [], [], []
            for t, m in minima.items():
                mine = sorted((r for r in R if r["key"] == t), key=lambda r: r["id"])
                if not mine:
                    continue
                regs = ref[t]["regions"]
                reg = [endpoint_region(m, ref[t], r["pos"]) if r.get("conv") else -1 for r in mine]
                moved += [r["moved"] for r, k in zip(mine, reg) if k >= 0]
                hit = {k for k in reg if k >= 0}
                found.append(len(hit)); share.append(len(hit) / len(regs) if regs else np.nan)
                new.append(sum(regs[k]["new"] for k in hit))
                gain.append(len(hit - sella_only[t])); miss.append(len(sella_only[t] - hit))
                # Every start of the arm (a run's members), in random order: the rollout always, the run's Sella calls
                # when the first of its members comes up.
                starts = [(i, mid) for i, r in enumerate(mine) for mid in r.get("members", [r["id"]])]
                cost = hessian.get(t, 0) if "soft" in arm else 0
                seen, reached = set(), []
                for j in np.random.default_rng([t, 51]).permutation(len(starts)):
                    i = starts[j][0]
                    cost += per_start
                    if i not in seen:
                        seen.add(i); cost += mine[i].get("sella_calls", 0)
                        if reg[i] >= 0:
                            reached.append((cost, reg[i]))
                total.append(cost)
                for b in BUDGETS:
                    budget[b].append(len({k for c, k in reached if c <= b}))
                by_region = {}
                for r, k in zip(mine, reg):
                    if k >= 0:
                        by_region.setdefault(k, []).append(r["moved"])
                keep.setdefault(arm, {})[t] = (hit, reached, by_region)
            moved = np.array(moved) if moved else np.array([np.nan])
            L.append(f"  {arm:24s} {len(R):6d} {100 * np.mean([r.get('conv', False) for r in R]):5.1f}  {np.median(calls):6.0f} {np.mean(calls):5.0f}"
                     f" | {np.nanmedian(moved):5.3f} {np.nanmean(moved):5.3f} | {np.mean(found):7.1f} {100 * np.nanmean(share):5.1f}% {np.mean(new):5.1f} | "
                     f"{np.mean(gain):5.1f} / {np.mean(miss):5.1f} | {np.mean(total) / 1000:5.0f}k | "
                     + "  ".join(f"{np.mean(budget[b]):4.1f}" for b in BUDGETS))

        # Model and Sella together: half of each budget on the 1x8 model starts, half on Sella from soft-mode pushes,
        # each in its own random order (as above); the saddles are the union of what the two reach.
        a, b = "ft_hint_1x8", "softpush"
        if a in keep and b in keep:
            ts = [t for t in minima if t in keep[a] and t in keep[b]]
            row = [np.mean([len({k for c, k in keep[a][t][1] if c <= B / 2} | {k for c, k in keep[b][t][1] if c <= B / 2})
                            for t in ts]) for B in BUDGETS]
            both = np.mean([len(keep[a][t][0] | keep[b][t][0]) for t in ts])
            L += ["", f"  combined: half of each budget on {a}, half on {b} (union of the regions reached); all starts of both: "
                  f"{both:.1f} saddles per minimum",
                  f"  {a + ' + ' + b:24s} {'':74s} | " + "  ".join(f"{x:4.1f}" for x in row)]

        # Precision on the regions both models reach from the same hint starts: the paper metric (landing to Sella
        # saddle) per region, the median over the runs ending there, paired over regions.
        L += ["", "  precision on shared regions: regions both the fine-tuned arm and rel_hint (same hint starts) reach;",
              "  per region the median distance moved (landing to the Sella saddle) over the runs ending there",
              "  arm                   regions | moved p50: arm  rel_hint | arm closer in | Wilcoxon p"]
        for arm in [x for x in ("ft_hint", "ft_hint_1x8", *ABLATIONS) if x in keep and "rel_hint" in keep]:
            x, y = [], []
            for t in minima:
                if t in keep[arm] and t in keep["rel_hint"]:
                    for k in keep[arm][t][0] & keep["rel_hint"][t][0]:
                        x.append(np.median(keep[arm][t][2][k])); y.append(np.median(keep["rel_hint"][t][2][k]))
            x, y = np.array(x), np.array(y)
            L.append(f"  {arm:20s} {len(x):8d} | {np.median(x):14.3f} {np.median(y):9.3f} | {100 * np.mean(x < y):11.0f}% | {paired(x, y):>10s}")

        # The regions the Sella-only arms reach and ft_hint_1x8 does not, by the first reason that applies:
        # no grid hint points at it (a label on that atom within TILT degrees); a 1x8 landing came within 0.2 A but
        # its Sella run ended elsewhere or failed; every hint pointing at it also points at another region (crowded);
        # an uncrowded hint points at it but no landing came within 0.2 A.
        a = "ft_hint_1x8"
        if a in keep and a in lands:
            why = {"no hint points at it": [], "landed within 0.2 A, Sella went elsewhere": [],
                   "every pointing hint is shared": [], "unshared hint, no landing within 0.2 A": []}
            barriers = {k: [] for k in why}
            for t, m in minima.items():
                if t not in keep[a]:
                    continue
                regs = ref[t]["regions"]
                missed = sella_only[t] - keep[a][t][0]
                X = np.array([l["landing"] for l in lands[a] if l["key"] == t])
                hints = grid(t, m)
                counts = {k: 0 for k in why}
                for k in missed:
                    points = [i for i, (atom, _, d) in enumerate(hints) if followed(regs[k], atom, d)]
                    if not points:
                        c = "no hint points at it"
                    elif len(X) and hl.saddle_distance(m, X, regs[k]["pos"]).min() <= 0.2:
                        c = "landed within 0.2 A, Sella went elsewhere"
                    elif all(any(followed(regs[j], hints[i][0], hints[i][2]) for j in range(len(regs)) if j != k) for i in points):
                        c = "every pointing hint is shared"
                    else:
                        c = "unshared hint, no landing within 0.2 A"
                    counts[c] += 1; barriers[c].append(regs[k]["barrier"])
                for c in why:
                    why[c].append(counts[c])
            n = sum(sum(v) for v in why.values())
            L += ["", f"  regions push and softpush reach that {a} does not, by reason ({n} in all)",
                  "  reason                                       per minimum  share  barrier p50"]
            for c in why:
                L.append(f"  {c:44s} {np.mean(why[c]):11.1f} {100 * sum(why[c]) / max(n, 1):5.0f}%  "
                         + (f"{np.median(barriers[c]):10.3f}" if barriers[c] else "         -"))

    # Hint following on the held-out Tier B regions.
    fol = {}
    for l in read_rows(out / "landings" / "*.json"):
        if l["arm"] in FOLLOW_ARMS:
            fol.setdefault(l["arm"], []).append(l)
    if fol:
        regions = {k: (m, r, e) for k, m, r, _, _, e in follow_regions(args)}
        clean = {}
        for k, (m, r, e) in regions.items():
            train = [t for t in e["regions"] if not t["held_out"]]
            clean[k] = not any(any(g["atom"] == h["atom"] and np.dot(g["direction"], h["direction"]) >= np.cos(np.radians(TILT))
                                   for g in t["hints"] for h in r["hints"])
                               and hl.saddle_distance(m, np.array([t["saddle"]]), np.array(r["saddle"]))[0] > 0.3 for t in train)
        L += ["", "", f"HINT FOLLOWING: held-out Tier B regions (barrier >= {args.min_barrier} eV), one start per region "
              f"({sum(clean.values())} clean: no training region > 0.3 A away shares a hint cone)",
              "  landing: within 0.2 A of the target saddle / of another known region of that minimum; distance p50 to the target",
              "  after Sella: the endpoint is the target (within 0.15 A) / another known region / not converged",
              "",
              "  arm             subset   n | landing: target  other  dist p50 | Sella: target  other  not conv"]
        sruns = {r["id"]: r for a in FOLLOW_ARMS for r in runs.get(a, [])}
        for arm in (a for a in FOLLOW_ARMS if a in fol):
            for subset in ("all", "clean"):
                rows = [l for l in fol[arm] if subset == "all" or clean.get(l["key"])]
                tgt, oth, dist, s_t, s_o, s_n, n_s = 0, 0, [], 0, 0, 0, 0
                for l in rows:
                    m, r, e = regions[l["key"]]
                    others = [o["saddle"] for o in e["regions"] if o["region"] != r["region"]]
                    d = float(hl.saddle_distance(m, np.array([l["landing"]]), np.array(r["saddle"]))[0]); dist.append(d)
                    do = hl.saddle_distance(m, np.array(others), np.array(l["landing"])).min() if others else np.inf
                    tgt += d <= 0.2; oth += d > 0.2 and do <= 0.2
                    s = sruns.get(l["id"])
                    if s is not None:
                        n_s += 1
                        if not s.get("conv"):
                            s_n += 1
                        elif hl.saddle_distance(m, np.array([s["pos"]]), np.array(r["saddle"]))[0] <= LAND_MERGE:
                            s_t += 1
                        elif others and hl.saddle_distance(m, np.array(others), np.array(s["pos"])).min() <= LAND_MERGE:
                            s_o += 1
                n = max(1, len(rows)); ns = max(1, n_s)
                L.append(f"  {arm:14s} {subset:6s} {len(rows):4d} |        {100 * tgt / n:5.1f}% {100 * oth / n:5.1f}%  {np.median(dist) if dist else np.nan:6.3f}  | "
                         + (f"      {100 * s_t / ns:5.1f}% {100 * s_o / ns:5.1f}%  {100 * s_n / ns:5.1f}%" if n_s else "      -"))

    # Forgetting.
    fr = {a: {r["key"]: r for r in runs.get(a, [])} for a in FORGET_ARMS}
    if any(fr.values()):
        L += ["", "", "FORGETTING: mp20bat test records, hint zero, one shared noise draw per record",
              "  ok: Sella converged to an index-1 saddle; moved: landing to that saddle (paper metric)",
              "", "  arm          runs   ok%  moved p50 mean  calls p50"]
        ok = {}
        for arm, R in fr.items():
            if not R:
                continue
            ok[arm] = {k: (r.get("conv") and r.get("nneg") == 1) for k, r in R.items()}
            mv = np.array([r["moved"] for k, r in R.items() if ok[arm][k]])
            L.append(f"  {arm:12s} {len(R):5d} {100 * np.mean(list(ok[arm].values())):5.1f}  {np.median(mv):6.3f} {np.mean(mv):5.3f}"
                     f"  {np.median([r.get('sella_calls', 0) for r in R.values()]):6.0f}")
        if len(ok) == 2:
            both = [k for k in ok["forget_ft"] if k in ok["forget_rel"] and ok["forget_ft"][k] and ok["forget_rel"][k]]
            x = [fr["forget_ft"][k]["moved"] for k in both]; y = [fr["forget_rel"][k]["moved"] for k in both]
            only_ft = sum(ok["forget_ft"][k] and not ok["forget_rel"].get(k, False) for k in ok["forget_ft"])
            only_rel = sum(ok["forget_rel"][k] and not ok["forget_ft"].get(k, False) for k in ok["forget_rel"])
            L.append(f"  paired on {len(both)} records both reach: moved median {np.median(x):.3f} vs {np.median(y):.3f} "
                     f"(Wilcoxon p {paired(x, y)}); reached by only the fine-tuned model {only_ft}, only the released {only_rel}")
    text = "\n".join(L) + "\n"
    (out / "report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"land": land, "sella": sella, "verify": verify, "report": report}[args.mode](args, out)


if __name__ == "__main__":
    main()
