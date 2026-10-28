"""LiC hint pilot: the A0-after-Sella baseline, the mirror test, and the classical hint search A3.

All three read the n12_uncond coverage archive (source_study.npz, sigma = 0.30 A): its reference structure,
the 63 minima, the 179 catalog saddles and their stored connectivity, and the 8,064 rollout endpoints.

    export-a0    Write every rollout endpoint as a full structure (a0/endpoints.traj, info['tid'] = draw).
                 Sella then runs on it with ../../../MaterialsSaddles/sella_eval.py (run_hint_pilot.sh).
    analyze-a0   Coverage from where Sella converged, with caps on how far Sella moved the endpoint,
                 next to the original 0.10 A criterion, on the same draws. CPU.
    mirror       Rerun the model from the first --mirror-draws starts of every minimum on the mirrored
                 (x -> -x) and the 180-degree-rotated system, map the endpoints back and compare them with
                 the original endpoints. The rotation is the control: the model is rotation-equivariant
                 by construction. GPU, minutes. With --vsym, also the symmetrized-velocity test (vsym_test):
                 the same starts rolled out three ways (the model as is with its K steps; v_sym, the velocity
                 averaged with its mirror image, K steps = 2K evaluations; 2K plain steps, the compute control)
                 and scored per saddle group (training, rotation copy, mirror copy, other).
    a3           Classical hint search from every minimum: 12 in-plane directions plus one toward each
                 nearby C-C bond midpoint, in three arms (Sella from a 0.4 A nudge, from a 0.8 A nudge, and
                 climb_along the direction then Sella), plus --repeats repeated runs per arm. GPU;
                 --shard/--nshards split the runs; finished runs are skipped on resubmission.
    analyze-a3   Outcomes, coverage of connected pairs (the hint ceiling), hint fidelity, neighbouring-
                 direction consistency, Sella repeatability and force calls per arm. CPU.
    report       Concatenate the three reports into hint_pilot_report.txt.

    python examples/LiC/experiments/hint_pilot/hint_pilot.py <mode> --out-dir <dir>

The search, verification and transform code is general (PESBackend, run_sella, climb_along,
MinimumModeEstimator, transform_structure); only the hint directions and catalog matching are LiC-specific.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic

from saddleflow.data.transforms import transform_structure

LIC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(LIC / "rollout_eval"), str(LIC / "experiments" / "source_study")]
ARCHIVE = "/scratch/gqs7zr/saddleflow/lic_source_study_seed42/coverage_curve/sigma0.30/source_study.npz"
CAPS = (0.1, 0.2, 0.3, np.inf)            # A: largest Sella move that still credits the model
MATCH = 0.05                              # A: a converged point within this of a catalog saddle is that saddle
ARMS = ("nudge0.4", "nudge0.8", "climb")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["export-a0", "analyze-a0", "mirror", "a3", "analyze-a3", "report"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--archive", default=ARCHIVE)
    p.add_argument("--method", default="n12_uncond")
    p.add_argument("--mirror-draws", type=int, default=16, help="Starts per minimum in the mirror test")
    p.add_argument("--vsym", action="store_true", help="mirror: also run the symmetrized-velocity test")
    p.add_argument("--vsym-draws", type=int, default=128, help="Starts per minimum in the v_sym test")
    p.add_argument("--repeats", type=int, default=100, help="Repeated runs per A3 arm (Sella repeatability)")
    p.add_argument("--fmax", type=float, default=0.005, help="eV/A; the LiC catalog's threshold")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------- shared LiC context and general helpers
def context(args):
    from coverage_curve import load
    data = np.load(args.archive, allow_pickle=True)
    run = load(args.archive, args.method)          # minimum and draw rank of every endpoint
    use = (data["method"] == args.method) & (data["source_population"][data["source_index"]] == "connected")
    reference = Atoms(numbers=data["numbers"], positions=data["positions"], cell=data["cell"], pbc=data["pbc"])
    reference.set_constraint(FixAtoms(mask=~data["free_mask"].astype(bool)))
    return dict(data=data, run=run, reference=reference, li=int(data["li_index"]), cell=data["cell"],
                pbc=data["pbc"], minima=data["minima"], edges=data["catalog_edges"],
                saddles=np.vstack([data["train_saddles"], data["test_saddles"]]),
                starts=data["starts"][data["source_index"][use]], finals=data["final"][use],
                outcome=data["outcome"][use], found=data["nearest_true_index"][use])


def mic_norm(vectors, cell, pbc):
    return find_mic(np.asarray(vectors).reshape(-1, 3), cell, pbc)[1]


def match(points, ctx):
    """Nearest catalog saddle of each Li position and its distance."""
    d = np.stack([mic_norm(ctx["saddles"] - p, ctx["cell"], ctx["pbc"]) for p in np.atleast_2d(points)])
    return d.argmin(1), d.min(1)


def table_line(cells, widths):
    return "  ".join(f"{c:>{w}}" if not isinstance(c, str) else f"{c:<{w}}" for c, w in zip(cells, widths))


# ---------------------------------------------------------------- A0 after Sella
def export_a0(args, out):
    from ase.io import Trajectory
    ctx = context(args)
    path = out / "a0" / "endpoints.traj"
    path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(path, "w") as traj:
        for tid, (final, m) in enumerate(zip(ctx["finals"], ctx["run"]["minimum"])):
            a = ctx["reference"].copy()
            a.positions[ctx["li"]] = final
            a.info.update(tid=tid, src=f"min{int(m)}")
            traj.write(a)
    print(f"wrote {path} ({len(ctx['finals'])} endpoints)")


def coverage(found, minimum, rank, edges, k):
    """Mean over minima of the share of its connected saddles found by its first k draws, and the pairs."""
    fractions, reached, missed = [], 0, []
    for m in np.unique(minimum):
        expected = np.flatnonzero(np.any(edges == m, axis=1))
        hits = set(found[(minimum == m) & (rank < k)]) & set(expected)
        fractions.append(len(hits) / len(expected)); reached += len(hits)
        missed += [(int(m), int(s)) for s in expected if s not in hits]
    return 100 * np.mean(fractions), reached, missed


def analyze_a0(args, out):
    ctx = context(args)
    records = {r["tid"]: r for path in sorted(glob.glob(str(out / "a0" / "sella*.json")))
               for r in json.loads(Path(path).read_text())}
    n = len(ctx["finals"])
    minimum, rank, edges = ctx["run"]["minimum"], ctx["run"]["rank"], ctx["edges"]
    raw = np.where(ctx["outcome"] == 0, ctx["found"], -1)
    sella_saddle, moved = np.full(n, -1), np.full(n, np.inf)
    status = np.array(["missing"] * n, dtype="U16")
    for tid, r in records.items():
        if "err" in r:
            status[tid] = "error"; continue
        if not r["conv"]:
            status[tid] = "not_converged"; continue
        if r.get("nneg") != 1:
            status[tid] = "not_index1"; continue
        s, dist = match(np.array(r["pos"])[ctx["li"]], ctx)
        moved[tid] = r["maxd"]
        if dist[0] < MATCH:
            sella_saddle[tid], status[tid] = s[0], "catalog"
        else:
            status[tid] = "new_index1"
    connected = lambda f: np.array([f[i] >= 0 and minimum[i] in edges[f[i]] for i in range(n)])
    lines = ["A0 after Sella: n12_uncond endpoints (sigma 0.30 A) scored by where Sella converges", "=" * 100,
             f"{len(records)}/{n} endpoints have a Sella result. Sella: UMA-S-1.2 omat, fmax {args.fmax}, <= 300 steps, "
             f"delta0 0.1, index 1. A converged point within {MATCH} A of a catalog saddle is that saddle.",
             "Sella outcomes: " + ", ".join(f"{s} {np.sum(status == s)}" for s in np.unique(status)),
             f"Sella force calls per endpoint: median {np.median([r['nfc'] for r in records.values() if r.get('nfc', -1) >= 0]):.0f}",
             "", "criterion                         k  coverage%  pairs reached  draws on a connected saddle%"]
    criteria = [("raw endpoint within 0.10 A", raw)]
    for cap in CAPS:
        credited = np.where(moved <= cap, sella_saddle, -1)
        criteria.append((f"Sella, moved <= {cap:g} A" if np.isfinite(cap) else "Sella, any move", credited))
    missed_at = {}
    for name, f in criteria:
        hit = connected(f)
        for k in (8, 32, 128):
            cov, reached, missed = coverage(f, minimum, rank, edges, k)
            lines.append(f"{name:30s} {k:4d}  {cov:9.2f}  {reached:6d}/{len(edges) * 2:<6d}  "
                         f"{100 * hit[rank < k].mean():10.2f}")
            missed_at[name] = missed
    lines += ["", "never-reached pairs at k = 128 (minimum, saddle):"]
    lines += [f"  {name:30s} {len(m):3d}  {m}" for name, m in missed_at.items()]
    new = status == "new_index1"
    if new.any():
        pos = np.array([np.array(records[t]["pos"])[ctx["li"]] for t in np.flatnonzero(new)])
        distinct = []
        for p in pos:
            if not distinct or mic_norm(np.array(distinct) - p, ctx["cell"], ctx["pbc"]).min() > MATCH:
                distinct.append(p)
        lines.append(f"\nindex-1 points not in the catalog: {new.sum()} endpoints, {len(distinct)} distinct "
                     f"(not checked for connectivity)")
    write_report(out / "a0" / "report.txt", lines)


# ---------------------------------------------------------------- mirror test
def symmetry_groups(ctx):
    """Each test saddle as an exact rotation copy (<0.1 A) or approximate mirror copy (<0.35 A) of a
    training saddle, by the in-plane rotation or mirror of the sheet that maps one onto the other."""
    return {s: kind for s, (kind, _) in symmetry_sources(ctx).items()}


def symmetry_sources(ctx):
    """symmetry_groups with the training saddle each copy comes from: {saddle: (group, training saddle or -1)}."""
    from scipy.spatial import cKDTree
    C = ctx["reference"].positions[ctx["data"]["carbon_indices"], :2]
    A = ctx["cell"][:2, :2]
    tiles = np.vstack([C + i * A[0] + j * A[1] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    tree, S = cKDTree(tiles), ctx["saddles"][:, :2]
    ops = []
    for k in range(12):
        c, s = np.cos(np.pi * k / 6), np.sin(np.pi * k / 6)
        R = np.array([[c, -s], [s, c]])
        ops += [("rotation", R), ("mirror", R @ np.diag([1, -1]))]
    n_train = len(ctx["data"]["train_saddles"])
    group = {s: ("training", s) for s in range(n_train)}
    for dst in range(n_train, len(S)):
        best = min((tree.query(C @ R.T + S[dst] - R @ S[src])[0].max(), kind, src)
                   for src in range(n_train) for kind, R in ops)
        group[dst] = (best[1], best[2]) if best[0] < (0.1 if best[1] == "rotation" else 0.35) else ("other", -1)
    return group


def mirror(args, out):
    import torch
    from force_guided_sampling import load_flow_model
    from robustness_shells import rollout_from_step
    from saddleflow.data import atoms_to_sample_dict
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ctx = context(args)
    config = json.loads(str(ctx["data"]["config_json"]))
    _, _, backbone, attention, head = load_flow_model(Path(config["checkpoints"][args.method]), args.device)
    pick = np.flatnonzero(ctx["run"]["rank"] < args.mirror_draws)
    li, ref = ctx["li"], ctx["reference"]
    starts = []
    for i in pick:
        p = ref.positions.copy(); p[li] = ctx["starts"][i]; starts.append(p)
    starts = np.array(starts)
    maps = {"identity": np.eye(3), "rotation 180 about z": np.diag([-1.0, -1.0, 1.0]),
            "mirror x -> -x": np.diag([-1.0, 1.0, 1.0])}
    results = {}
    for name, R in maps.items():
        moved = transform_structure(ref, R)
        sample = atoms_to_sample_dict(moved)
        ends = []
        for b in range(0, len(starts), 32):
            x = torch.as_tensor(starts[b:b + 32] @ R.T, dtype=torch.float32)
            final, _ = rollout_from_step(x, 0, sample=sample, backbone=backbone, attention=attention,
                                         velocity_head=head, device=args.device, K=int(config["K"]))
            ends.append(final.numpy()[:, li] @ R)          # back to the original frame
        results[name] = np.concatenate(ends)
        print(f"  {name} done", flush=True)
    stored = ctx["finals"][pick]
    group = symmetry_groups(ctx)
    s_near, d_near = match(stored, ctx)
    near_group = np.array([group[s] if d < 0.3 else "away" for s, d in zip(s_near, d_near)])
    lines = ["Mirror test: is n12_uncond mirror-equivariant?", "=" * 100,
             f"{len(pick)} starts ({args.mirror_draws} per minimum). Each map is applied to the whole system; the model "
             "is run from the mapped start and its endpoint mapped back. Distance (A) to the stored endpoint of the "
             "same start. The identity checks reproducibility; the rotation is the control (exact by construction).",
             "", table_line(["map", "p50", "p90", "p99", "max"], [24, 8, 8, 8, 8])]
    for name, back in list(results.items()):
        d = mic_norm(back - stored, ctx["cell"], ctx["pbc"])
        results[name + " distance"] = d
        lines.append(table_line([name] + [f"{v:.4f}" for v in np.quantile(d, [.5, .9, .99, 1])], [24, 8, 8, 8, 8]))
    d = results["mirror x -> -x distance"]
    lines += ["", "mirror discrepancy by the saddle the stored endpoint is nearest (within 0.3 A):",
              table_line(["nearest saddle is", "n", "p50", "p90"], [24, 6, 8, 8])]
    for g in ("training", "rotation", "mirror", "other", "away"):
        m = near_group == g
        if m.any():
            lines.append(table_line([g, int(m.sum())] + [f"{v:.4f}" for v in np.quantile(d[m], [.5, .9])], [24, 6, 8, 8]))
    (out / "mirror").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mirror" / "mirror.npz", pick=pick, stored=stored, near_group=near_group,
                        **{k.replace(" ", "_").replace(">", ""): v for k, v in results.items()})
    if args.vsym:
        lines += [""] + vsym_test(args, ctx, backbone, attention, head, int(config["K"]), out)
    write_report(out / "mirror" / "report.txt", lines)


def rollout_vsym(starts, sample, mirrored, M, backbone, attention, head, device, K, symmetrize):
    """Euler from t = 0 in K steps. symmetrize: the velocity is averaged with its mirror image,
    v_sym(x) = (v(x) + M v(M x)) / 2 (M improper and orthogonal); with exact rotation-equivariance this makes the
    field equivariant under every mirror. Two network evaluations per step."""
    import torch
    from force_guided_sampling import model_velocity
    from saddleflow.data.transforms import wrap_positions
    Mt = torch.as_tensor(M, dtype=starts.dtype)
    x = wrap_positions(starts.clone(), sample["cell"])
    for step in range(K):
        v = model_velocity(x, step / K, sample, backbone, attention, head, device).cpu()
        if symmetrize:
            xm = wrap_positions(x @ Mt.T, mirrored["cell"])
            vm = model_velocity(xm, step / K, mirrored, backbone, attention, head, device).cpu()
            v = 0.5 * (v + vm @ Mt)
        x = wrap_positions(x + v / K, sample["cell"])
    return x


def vsym_test(args, ctx, backbone, attention, head, K, out):
    """Baseline (K steps), v_sym (K steps, 2K evaluations) and the compute control (2K steps) from the same
    starts; hits (an endpoint within 0.10 A of a catalog saddle connected to its minimum) scored per group."""
    import torch
    from saddleflow.data import atoms_to_sample_dict
    M = np.diag([-1.0, 1.0, 1.0])                  # the mirror of the mirror test
    li, ref, edges = ctx["li"], ctx["reference"], ctx["edges"]
    pick = np.flatnonzero(ctx["run"]["rank"] < args.vsym_draws)
    minimum = ctx["run"]["minimum"][pick]
    full = []
    for i in pick:
        p = ref.positions.copy(); p[li] = ctx["starts"][i]; full.append(p)
    full = torch.as_tensor(np.array(full), dtype=torch.float32)
    sample, mirrored = atoms_to_sample_dict(ref), atoms_to_sample_dict(transform_structure(ref, M))
    arms = {f"baseline, {K} steps": (K, False), f"v_sym, {K} steps ({2 * K} evaluations)": (K, True),
            f"control, {2 * K} steps": (2 * K, False)}
    ends = {}
    for name, (steps, sym) in arms.items():
        ends[name] = np.concatenate([rollout_vsym(full[b:b + 32], sample, mirrored, M, backbone, attention, head,
                                                  args.device, steps, sym).numpy()[:, li]
                                     for b in range(0, len(full), 32)])
        print(f"  v_sym test: {name} done", flush=True)
    base = list(arms)[0]
    repro = mic_norm(ends[base] - ctx["finals"][pick], ctx["cell"], ctx["pbc"])
    groups = symmetry_sources(ctx)
    pairs = [(int(m), s) for s in range(len(edges)) for m in edges[s]]
    kinds = ("training", "rotation", "mirror", "other")
    lines = ["v_sym test: the velocity averaged with its mirror image (x -> -x), against the plain model at the same "
             "and at twice the steps", "=" * 100,
             f"{len(pick)} starts ({args.vsym_draws} per minimum). A hit: an endpoint within 0.10 A of a catalog saddle "
             "connected to its minimum. Groups as in symmetry_groups: the 12 training saddles, exact rotation copies "
             "(< 0.1 A) and approximate mirror copies (< 0.35 A) of them, and the other test saddles.",
             f"baseline rerun against the stored endpoints: max {repro.max():.4f} A", "",
             table_line(["arm", "group", "pairs", "reach rate", "hit dist p50", "never reached"], [36, 9, 6, 10, 12, 13])]
    per_saddle = {}
    for name in arms:
        s_near, d_near = match(ends[name], ctx)
        hit = np.array([d < 0.10 and m in edges[s] for m, s, d in zip(minimum, s_near, d_near)])
        p50 = {}
        for kind in kinds:
            mine = [(m, s) for m, s in pairs if groups[s][0] == kind]
            if not mine:
                continue
            rate = [np.mean(hit[minimum == m] & (s_near[minimum == m] == s)) for m, s in mine]
            dist = d_near[hit & np.isin(s_near, [s for _, s in mine])]
            never = sum(r == 0 for r in rate)
            lines.append(table_line([name, kind, len(mine), f"{np.mean(rate):.3f}",
                                     f"{np.median(dist):.4f}" if len(dist) else "-", f"{never}/{len(mine)}"],
                                    [36, 9, 6, 10, 12, 13]))
        per_saddle[name] = {s: np.median(d_near[hit & (s_near == s)]) for s in set(s_near[hit])}
        reached = {(m, s) for m, s, h in zip(minimum, s_near, hit) if h}
        lines.append(table_line([name, "all", len(pairs), f"draws on a connected saddle {100 * hit.mean():.1f}%",
                                 "", f"{len(pairs) - len(reached)}/{len(pairs)}"], [36, 9, 6, 10, 12, 13]))
    # Within a pair: each copy's hit distance against that of the training saddle it copies.
    lines += ["", "within pairs: hit distance p50 of a copy minus that of its training saddle (median over copies)",
              table_line(["arm", "rotation copies", "mirror copies", "training saddles p50"], [36, 16, 14, 20])]
    for name in arms:
        d = per_saddle[name]
        delta = {kind: [d[s] - d[src] for s, (k, src) in groups.items() if k == kind and s in d and src in d]
                 for kind in ("rotation", "mirror")}
        train = [d[s] for s, (k, _) in groups.items() if k == "training" and s in d]
        lines.append(table_line([name] + [f"{np.median(delta[k]):+.4f} (n {len(delta[k])})" if delta[k] else "-"
                                          for k in ("rotation", "mirror")]
                                + [f"{np.median(train):.4f}" if train else "-"], [36, 16, 14, 20]))
    np.savez_compressed(out / "mirror" / "vsym.npz", pick=pick, **{n.split(",")[0].replace(" ", "_"): e for n, e in ends.items()})
    return lines


# ---------------------------------------------------------------- A3 classical hint search
def directions(ctx, m):
    """12 evenly spaced in-plane directions plus one toward each C-C bond midpoint within 1.8 A in-plane."""
    dirs = [("even", np.array([np.cos(a), np.sin(a), 0.0])) for a in np.arange(12) * np.pi / 6]
    C = ctx["reference"].positions[ctx["data"]["carbon_indices"]]
    x = ctx["minima"][m]
    flat = find_mic(C - x, ctx["cell"], ctx["pbc"])[0]
    near = C[np.linalg.norm(flat[:, :2], axis=1) < 2.2]          # in-plane: the Li sits above the sheet
    seen = []
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            bond = find_mic((near[j] - near[i])[None], ctx["cell"], ctx["pbc"])[0][0]
            if np.linalg.norm(bond) < 1.8:
                v = find_mic((near[i] + bond / 2 - x)[None], ctx["cell"], ctx["pbc"])[0][0]
                v[2] = 0.0
                if 0.3 < np.linalg.norm(v) < 1.8 and all(np.dot(v, s) / np.linalg.norm(v) / np.linalg.norm(s) < 0.999 for s in seen):
                    seen.append(v); dirs.append(("bond", v / np.linalg.norm(v)))
    return dirs


def a3_specs(ctx, repeats):
    specs = [(arm, m, j, 0) for arm in ARMS for m in range(len(ctx["minima"])) for j in range(len(directions(ctx, m)))]
    rng = np.random.default_rng(0)
    for arm in ARMS:
        first = [s for s in specs if s[0] == arm and s[3] == 0]
        specs += [(arm, m, j, 1) for arm, m, j, _ in (first[i] for i in rng.choice(len(first), repeats, replace=False))]
    return specs


def a3(args, out):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, climb_along, run_sella

    ctx = context(args)
    folder = out / "a3"; folder.mkdir(parents=True, exist_ok=True)
    mine = a3_specs(ctx, args.repeats)[args.shard::args.nshards]
    path = folder / f"runs_{args.shard}.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["id"] for r in rows}            # shards are fixed, so this shard's file holds its finished runs
    calc = FAIRChemCalculator(pretrained_mlip.get_predict_unit("uma-s-1p2", device=args.device), task_name="omat")
    dir_cache = {}
    for arm, m, j, rep in mine:
        rid = f"{arm}:{m}:{j}:{rep}"
        if rid in done:
            continue
        kind, v = dir_cache.setdefault(m, directions(ctx, m))[j]
        atoms = ctx["reference"].copy(); atoms.positions[ctx["li"]] = ctx["minima"][m]
        pes = PESBackend(atoms, calc)
        full = np.zeros((len(atoms), 3)); full[ctx["li"]] = v
        x_min, d = pes.dofs.pack(atoms.positions), pes.dofs.pack(full)
        row = dict(id=rid, arm=arm, minimum=m, direction=j, kind=kind, vector=v.tolist(), repeat=rep)
        try:
            if arm == "climb":
                climb = climb_along(pes, x_min, d)
                row.update(climb_status=climb["status"], climb_steps=climb["steps"], climb_calls=climb["force_calls"],
                           climb_rise=climb["energy_rise"], perp_force_rms=climb["perp_force_rms"])
                if climb["status"] != "left_bowl":
                    row.update(outcome="climb_" + climb["status"]); rows.append(row)
                    path.write_text(json.dumps(rows)); continue
                x0 = climb["x"]
            else:
                x0 = x_min + float(arm[len("nudge"):]) * d
            before = pes.force_calls
            result = run_sella(pes, x0, fmax_tol=args.fmax, max_steps=300, max_force_calls=10 ** 6, delta0=0.1)
            pes.max_force_calls = None
            row.update(sella_calls=pes.force_calls - before, sella_status=result["status"], sella_steps=result["steps"])
            before = pes.force_calls
            _, fmax = pes.dofs.force_metrics(pes.forces(result["final"]))
            values, _, _ = MinimumModeEstimator().lowest_modes(pes, result["final"], n_modes=2)
            row.update(index_calls=pes.force_calls - before, fmax=fmax, conv=bool(fmax <= args.fmax),
                       eigs=values.tolist(), nneg=int(np.sum(values < -0.01)),
                       li=np.asarray(result["positions"])[ctx["li"]].tolist())
        except Exception as exc:          # one failing run must not lose the shard
            row.update(outcome="error", err=str(exc)[:200])
        rows.append(row)
        path.write_text(json.dumps(rows))
    print(f"shard {args.shard}: {len(rows)} runs in {path}")


def a3_outcome(row, ctx):
    if "outcome" in row:
        return row["outcome"], -1
    if not row["conv"]:
        return "not_converged", -1
    if row["nneg"] != 1:
        return "not_index1", -1
    s, dist = match(np.array(row["li"]), ctx)
    if dist[0] >= MATCH:
        return "new_index1", -1
    return ("connected" if row["minimum"] in ctx["edges"][s[0]] else "wandered"), int(s[0])


def analyze_a3(args, out):
    ctx = context(args)
    rows = [r for path in sorted(glob.glob(str(out / "a3" / "runs_*.json"))) for r in json.loads(Path(path).read_text())]
    for r in rows:
        r["result"], r["saddle"] = a3_outcome(r, ctx)
    edges, minima = ctx["edges"], ctx["minima"]
    n_train = len(ctx["data"]["train_saddles"])
    train_minima = sorted({int(m) for m in edges[:n_train].ravel()})
    (out / "a3" / "training_minima.json").write_text(json.dumps(train_minima))
    held = [m for m in range(len(minima)) if m not in train_minima]
    total = {"all": 2 * len(edges), "training": sum(int(np.isin(edges[s], train_minima).sum()) for s in range(len(edges))),
             "held-out": sum(int(np.isin(edges[s], held).sum()) for s in range(len(edges)))}
    outcomes = ["connected", "wandered", "new_index1", "not_index1", "not_converged", "climb_wall", "climb_too_far",
                "climb_max_steps", "error"]
    lines = ["A3: classical hint search from every minimum", "=" * 100,
             f"{len(rows)} runs. Directions per minimum: 12 evenly spaced in-plane + one toward each C-C bond midpoint "
             f"within 1.8 A. Sella fmax {args.fmax}; index 1 = one eigenvalue < -0.01; catalog match {MATCH} A.",
             f"Training minima (touched by the {n_train} training saddles): {len(train_minima)}; held-out: {len(held)}.",
             "", "outcome shares, first runs only (%):",
             table_line(["arm", "runs"] + outcomes, [10, 6] + [9] * len(outcomes))]
    pairs_by_arm = {}
    for arm in ARMS:
        first = [r for r in rows if r["arm"] == arm and r["repeat"] == 0]
        if not first:
            continue
        lines.append(table_line([arm, len(first)] + [f"{100 * np.mean([r['result'] == o for r in first]):.1f}"
                                                      for o in outcomes], [10, 6] + [9] * len(outcomes)))
        pairs_by_arm[arm] = {(r["minimum"], r["saddle"]) for r in first if r["result"] == "connected"}
    lines += ["", "connected pairs reached (the hint ceiling):",
              table_line(["arm", "all", "training", "held-out", "distinct per minimum"], [14, 10, 10, 10, 20])]
    union = set().union(*pairs_by_arm.values()) if pairs_by_arm else set()
    for arm, pairs in list(pairs_by_arm.items()) + [("union", union)]:
        count = lambda ms: sum(m in ms for m, _ in pairs)
        per_min = np.mean([sum(m == mm for mm, _ in pairs) for m in range(len(minima))])
        lines.append(table_line([arm, f"{len(pairs)}/{total['all']}", f"{count(train_minima)}/{total['training']}",
                                 f"{count(held)}/{total['held-out']}", f"{per_min:.2f}"], [14, 10, 10, 10, 20]))
    all_pairs = {(int(m), int(s)) for s in range(len(edges)) for m in edges[s]}
    unreached = sorted(all_pairs - union)
    lines.append(f"  pairs no arm reaches: {len(unreached)}" + (f"  {unreached}" if len(unreached) <= 40 else ""))

    lines += ["", "hint fidelity and consistency, first runs:",
              table_line(["arm", "fidelity%", "neighbours agree%", "distinct of 12 even", "repeat differs%"], [10, 10, 18, 20, 16])]
    for arm in pairs_by_arm:
        first = {(r["minimum"], r["direction"]): r for r in rows if r["arm"] == arm and r["repeat"] == 0}
        faithful = []
        for r in first.values():
            if r["result"] != "connected":
                continue
            cands = np.flatnonzero(np.any(edges == r["minimum"], axis=1))
            vecs = find_mic(ctx["saddles"][cands] - minima[r["minimum"]], ctx["cell"], ctx["pbc"])[0]
            vecs[:, 2] = 0
            cos = vecs @ np.array(r["vector"]) / np.linalg.norm(vecs, axis=1)
            faithful.append(cands[cos.argmax()] == r["saddle"])
        agree, distinct = [], []
        for m in range(len(minima)):
            even = [first.get((m, j)) for j in range(12)]
            labels = [e["saddle"] if e and e["result"] == "connected" else None for e in even]
            agree += [labels[j] is not None and labels[j] == labels[(j + 1) % 12] for j in range(12)]
            distinct.append(len({x for x in labels if x is not None}))
        reps = [r for r in rows if r["arm"] == arm and r["repeat"] == 1]
        differs = [(r["result"], r["saddle"]) != (first[(r["minimum"], r["direction"])]["result"],
                                                  first[(r["minimum"], r["direction"])]["saddle"])
                   for r in reps if (r["minimum"], r["direction"]) in first]
        lines.append(table_line([arm, f"{100 * np.mean(faithful):.1f}", f"{100 * np.mean(agree):.1f}",
                                 f"{np.mean(distinct):.2f}", f"{100 * np.mean(differs):.1f}" if differs else "n/a"],
                                [10, 10, 18, 20, 16]))
    lines += ["  fidelity: a connected result is the connected saddle closest in angle to the hint direction.",
              "  neighbours agree: adjacent even directions (30 degrees apart) reach the same connected saddle.",
              "", "force calls per run (mean) and per connected pair found:",
              table_line(["arm", "climb", "sella", "index", "total", "per pair"], [10, 8, 8, 8, 8, 10])]
    for arm in pairs_by_arm:
        first = [r for r in rows if r["arm"] == arm and r["repeat"] == 0]
        c = {k: np.mean([r.get(k, 0) for r in first]) for k in ("climb_calls", "sella_calls", "index_calls")}
        spent = sum(r.get("climb_calls", 0) + r.get("sella_calls", 0) for r in first)
        lines.append(table_line([arm, f"{c['climb_calls']:.1f}", f"{c['sella_calls']:.1f}", f"{c['index_calls']:.1f}",
                                 f"{sum(c.values()):.1f}", f"{spent / max(len(pairs_by_arm[arm]), 1):.1f}"],
                                [10, 8, 8, 8, 8, 10]))
    climbs = [r for r in rows if r["arm"] == "climb" and r["repeat"] == 0 and "climb_steps" in r]
    if climbs:
        lines += ["", f"climb arm: median {np.median([r['climb_steps'] for r in climbs]):.0f} climb steps, median "
                      f"perpendicular force at handover {np.median([r['perp_force_rms'] for r in climbs]):.3f} eV/A"]
    write_report(out / "a3" / "report.txt", lines)


def write_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)


def report(args, out):
    parts = [out / name / "report.txt" for name in ("a0", "mirror", "a3")]
    text = "\n\n".join(p.read_text() if p.exists() else f"[{p} missing]" for p in parts)
    (out / "hint_pilot_report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"export-a0": export_a0, "analyze-a0": analyze_a0, "mirror": mirror, "a3": a3,
     "analyze-a3": analyze_a3, "report": report}[args.mode](args, out)


if __name__ == "__main__":
    main()
