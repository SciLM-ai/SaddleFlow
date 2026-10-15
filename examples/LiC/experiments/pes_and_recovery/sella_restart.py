"""Michael: one Sella kick at a failed FM endpoint, then a fresh FM clock.

Inference only. The full-Sella comparator starts at EVERY original source,
not just failed endpoints. NPZ/report snapshots precede CPU-only plotting.
"""

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from ase.geometry import wrap_positions

from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, make_uma_calculator, run_sella

try:
    from compare_saddle_search import diagnose_states, save_archive
    from saddle_search_data import classify, generate_starts, load_geometry, read_npz
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.compare_saddle_search import diagnose_states, save_archive
    from examples.LiC.rollout_eval.saddle_search_data import classify, generate_starts, load_geometry, read_npz


DEFAULT_CKPT = "/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final"
COST_KEYS = ["proposal_seconds", "gate_seconds", "kick_seconds", "restart_seconds", "full_sella_seconds"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[2]  # examples/LiC
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Original unconditioned EMA checkpoint; no fine-tuning")
    p.add_argument("--out-dir")
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))
    p.add_argument("--starts", default="mixed", help="mixed or comma-separated false,true,shells,grid")
    p.add_argument("--starts-npz", help="Reuse ORIGINAL t=0 sources, then run the baseline afresh for honest timing")
    p.add_argument("--limit-starts", type=int)
    p.add_argument("--samples-per-site", type=int, default=1)
    p.add_argument("--false-noise-sigma", type=float, default=.1)
    p.add_argument("--true-noise-sigma", type=float, default=.25)
    p.add_argument("--radii", default="0.6,0.8,1.0")
    p.add_argument("--directions-per-radius", type=int, default=1)
    p.add_argument("--shell-dimensions", choices=["xy", "xyz"], default="xy")
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--task-name", default="omat")
    p.add_argument("--alphas", default="1", help="Predeclared kick scales; optional sweep .25,.5,1,2. No best-alpha selection")
    p.add_argument("--max-kick", type=float, help="Optional Li-displacement cap, A; controls use the same capped norm")
    p.add_argument("--gate", choices=["force", "force-index"], default="force")
    p.add_argument("--fmax", type=float, default=.010, help="Free-force gate, final certification, and full-Sella stopping, eV/A")
    p.add_argument("--eigenvalue-tol", type=float, default=.01)
    p.add_argument("--fd-eps", type=float, default=.002)
    p.add_argument("--sella-delta0", type=float, default=.1)
    p.add_argument("--max-steps", type=int, default=100, help="Full-Sella maximum iterations ONLY; the hybrid always takes ONE")
    p.add_argument("--max-force-calls", type=int, default=800, help="Per Sella search, including curvature probes")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-only", metavar="NPZ")
    return p.parse_args()


def matched_kicks(delta, force, alpha, seed, max_kick=None):
    """Same displacement norm for Sella/random/downhill-force, including caps."""
    length = alpha * np.linalg.norm(delta)
    if max_kick is not None:
        length = min(length, max_kick)
    random = np.random.default_rng(seed).normal(size=delta.shape)
    if length > 0 and np.linalg.norm(force) < 1e-30:
        raise ValueError("Michael: a zero-force state has no matched force-kick direction")
    unit = lambda v: v / max(np.linalg.norm(v), 1e-30)
    return dict(sella=length * unit(delta), random=length * unit(random), force=length * unit(force))


def flow_result(method, i, path, seconds, calls, model_calls, **costs):
    return dict(method=method, source_index=i, restart=0, final=path[-1].copy(),
                trace=[dict(x=x.copy(), q=None, eigs=None) for x in path],
                status="fixed_K", error="", steps=model_calls, force_calls=int(calls),
                model_forward_calls=model_calls, wall_seconds=seconds, **costs)


def groups(data):
    index = data["source_index"]
    result = {"all": np.ones(len(index), dtype=bool), "flagged": data["flagged"][index],
              "passed_gate": ~data["flagged"][index]}
    result.update({str(p): data["source_population"][index] == p for p in np.unique(data["source_population"])})
    result.update({f"endpoint_basin_{b}": data["baseline_basin"][index] == b
                   for b in np.unique(data["baseline_basin"]) if b >= 0})
    return {name: mask for name, mask in result.items() if mask.any()}


def make_report(data):
    cfg = json.loads(str(data["config_json"]))
    methods = list(dict.fromkeys(data["method"]))
    lines = ["Michael: Sella-preconditioned endpoint restart", "=" * 145,
        f"Original EMA checkpoint: {cfg['ckpt']}",
        f"Original starts={len(data['starts'])}; flagged={data['flagged'].sum()}; gate={cfg['gate']}; fmax={cfg['fmax']} eV/A.",
        f"Autonomous Euler K={cfg['K']} before repair AND K={cfg['K']} on a NEW clock after the kick. No splicing or training.",
        "Only UMA force/index decides repair eligibility. True/false catalog labels and parent IDs NEVER guide the kick or flow.",
        "full_sella starts at EVERY ORIGINAL source. It does NOT inherit flow endpoints, curvature history, or the hybrid kick.",
        "baseline is the original flow alone; restart_none controls for an extra rollout at the failed endpoint with no kick.",
        "Random/force controls match each Sella kick's norm. Their reported cost INCLUDES Sella norm calibration, not a standalone cheap policy.",
        "Scaled kicks are extrapolations, not accepted Sella steps (only alpha=1 without clipping is the actual update).",
        "work_s/PES include proposal + all-source failure detection + correction + repair, but EXCLUDE post-hoc final certification.",
        "total_s includes final force + index-one certification for EVERY output; model loading/warmup and report/PDF I/O are excluded.",
        "Flow timings are GPU-synchronized batch throughput allocated per source; Sella searches are serial. Same GPU/job, no historical timings.",
        "Control branches share measured proposal/gate/kick work; each table row reports its hypothetical standalone pipeline cost, not job runtime.",
        "Any catalog true <.10 A; false region <.30 A with true precedence. Physical success requires force <= fmax and strict index-one.",
        "One random direction per endpoint; noisy draws/endpoint episodes can be correlated within basins. No best-alpha selection.",
        f"Progress: {str(data['stage'])}; accepted nonzero kicks={np.sum(data['kick_norm'] > 1e-10)}.", "",
        "method          group                   N   true%  false%   phys%  med fmax  mean PES  mean work_s  mean total_s", "-" * 145]
    for name, group in groups(data).items():
        for method in methods:
            use = group & (data["method"] == method)
            if not use.any(): continue
            y = data["outcome"][use]
            lines.append(f"{method:15s} {name:22s} {use.sum():5d} {100*np.mean(y==0):7.2f} {100*np.mean(y==1):7.2f} "
                f"{100*np.mean(data['physical_success'][use]):7.2f} {np.median(data['force_max'][use]):9.5f} "
                f"{np.mean(data['force_calls'][use]):9.2f} {np.mean(data['wall_seconds'][use]):12.4f} "
                f"{np.mean(data['wall_seconds'][use]+data['diagnostic_seconds'][use]):13.4f}")
    lines += ["", "ALL-source timing breakdown (sums; compare only rows with the same N)",
              "method            N   proposal_s     gate_s     kick_s   repair_s  full_Sel_s     work_s     cert_s    total_s   PES+cert", "-"*145]
    for method in methods:
        use = data["method"] == method
        times = [data[key][use].sum() for key in COST_KEYS]
        work, cert = data["wall_seconds"][use].sum(), data["diagnostic_seconds"][use].sum()
        lines.append(f"{method:15s} {use.sum():4d}" + "".join(f" {t:10.3f}" for t in times) +
                     f" {work:10.3f} {cert:10.3f} {work+cert:10.3f} "
                     f"{data['force_calls'][use].sum()+data['diagnostic_force_calls'][use].sum():10.0f}")
    kicked = data["flagged"] & np.isfinite(data["kick_norm"])
    if kicked.any():
        y = classify(data["proposal_path"][kicked, -1]+data["sella_delta"][kicked], data)[0]
        lines.append(f"One-step Sella output alone (before repair flow): N={kicked.sum()}, "
                     f"true={100*np.mean(y==0):.2f}%, false={100*np.mean(y==1):.2f}%, "
                     f"median displacement={np.median(data['kick_norm'][kicked]):.6f} A, "
                     f"median fmax={np.median(data['kick_force_max'][kicked]):.6f} eV/A; no kick index certification.")
    for method in methods:
        if not method.startswith("sella_a"): continue
        for control in ["restart_none", method.replace("sella_a", "random_a"), method.replace("sella_a", "force_a")]:
            left = {int(data["source_index"][j]): data["outcome"][j] == 0 for j in np.flatnonzero(data["method"] == control)}
            right = {int(data["source_index"][j]): data["outcome"][j] == 0 for j in np.flatnonzero(data["method"] == method)}
            both = sorted(set(left) & set(right))
            if both:
                gains = sum(right[i] and not left[i] for i in both)
                losses = sum(left[i] and not right[i] for i in both)
                lines.append(f"Paired {method} vs {control}: new true hits={gains}, lost true hits={losses}, N={len(both)}.")
    return "\n".join(lines) + "\n"


def save_progress(out, args, geometry, sources, results, extra, stage):
    # Reuse the established pickle-free solver schema; atomic NPZ precedes report/PDF.
    scratch = out / "sella_restart.tmp.npz"
    data = save_archive(scratch, args, geometry, sources, results, [])
    data.update(extra, stage=np.asarray(stage))
    for key in COST_KEYS:
        data[key] = np.asarray([r.get(key, 0.) for r in results])
    np.savez_compressed(scratch, **data)
    scratch.replace(out / "sella_restart.npz")
    report = make_report(data)
    (out / "report.txt").write_text(report)
    print(f"[save] Michael: stage={stage}; completed outputs={len(results)}; {out/'sella_restart.npz'}", flush=True)
    return report


def plot_results(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try: from plot_saddle_search import draw_paths, draw_sheet, path_segments
    except ModuleNotFoundError: from examples.LiC.rollout_eval.plot_saddle_search import draw_paths, draw_sheet, path_segments
    path = Path(path); data = read_npz(path); methods = list(dict.fromkeys(data["method"]))
    with PdfPages(path.with_suffix(".pdf")) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for ax, field, title in zip(axes, ["outcome", "physical_success", "wall_seconds"],
                                   ["Any-true hit (%)", "Force + index-one (%)", "Mean work / output (seconds)"]):
            values = [np.mean(data[field][data["method"] == m] == 0) * 100 if field == "outcome" else
                      np.mean(data[field][data["method"] == m]) * (100 if field == "physical_success" else 1) for m in methods]
            ax.bar(methods, values); ax.set_title(title); ax.tick_params(axis="x", rotation=70)
        fig.suptitle(f"Michael: endpoint restart, {str(data['stage'])}; work includes detection and repair")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # All outputs and flagged-only panels; trajectories include the original proposal.
        for group in ["all", "flagged"]:
            selection = groups(data).get(group)
            if selection is None: continue
            for first in range(0, len(methods), 4):
                page = methods[first:first+4]
                fig, axes = plt.subplots(2, 2, figsize=(14, 14))
                for ax, method in zip(axes.flat, page):
                    use = np.flatnonzero(selection & (data["method"] == method))
                    draw_sheet(ax, data); draw_paths(ax, data, use, show_starts=False, path_limit=100)
                    if "_a" in method:
                        K = json.loads(str(data["config_json"]))["K"]
                        for j in use:
                            if not data["flagged"][data["source_index"][j]]: continue
                            for segment in path_segments(data["path"][j, K:K+2], data["cell"], data["pbc"]):
                                ax.plot(segment[:, 0], segment[:, 1], c="darkorange", lw=1.5, zorder=3)
                    ax.set_title(f"{method}: N={len(use)}; true={100*np.mean(data['outcome'][use]==0):.1f}%; "
                                 f"physical={100*np.mean(data['physical_success'][use]):.1f}%")
                for ax in list(axes.flat)[len(page):]: ax.set_visible(False)
                fig.suptitle(f"Michael: {group}; blue trajectories, green/red/purple finals = true/false/other; blue rings = certified\n"
                             "Orange: endpoint kick. Hybrid paths include proposal + repair; full Sella starts at the original source.")
                fig.tight_layout(rect=(0, 0, 1, .95)); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] Michael: {path.with_suffix('.pdf')}", flush=True)


def main():
    args = parse_args()
    if args.plot_only: plot_results(args.plot_only); return
    alphas = list(dict.fromkeys(float(a) for a in args.alphas.split(",")))
    if not alphas or not all(np.isfinite(a) and a > 0 for a in alphas): raise ValueError("Michael: alphas must be positive")
    if min(args.K, args.batch_size, args.max_steps, args.max_force_calls) <= 0 or args.fmax <= 0:
        raise ValueError("Michael: step/query counts and force tolerance must be positive")
    if args.max_kick is not None and args.max_kick <= 0: raise ValueError("Michael: max-kick must be positive")
    out = Path(args.out_dir or Path(args.ckpt).parent / "sella_restart"); out.mkdir(parents=True, exist_ok=True)
    if (out / "sella_restart.npz").exists(): raise ValueError("Michael: use a fresh output directory to preserve previous results")
    reference, geometry = load_geometry(args.train_traj, args.test_traj)
    sources = generate_starts(args, geometry); n = len(sources["starts"])
    if not n: raise ValueError("Michael: no starting points selected")
    import torch
    from saddleflow.data import atoms_to_sample_dict
    try:
        from force_guided_sampling import load_flow_model
        from robustness_shells import sample_points
    except ModuleNotFoundError:
        from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
        from examples.LiC.rollout_eval.robustness_shells import sample_points
    sample = atoms_to_sample_dict(reference)
    print(f"[setup] Michael: {n} original starts; unchanged EMA checkpoint; endpoint-only repair; alpha={alphas}", flush=True)
    _, _, backbone, attention, head = load_flow_model(Path(args.ckpt).resolve(), args.device)
    if getattr(head, "_physics_conditioner", None) is not None:
        raise ValueError("Michael: this one-kick ablation assumes the original unconditioned model; use robustness_shells for conditioned checkpoints")
    calculator, batch = make_uma_calculator(args.model, args.device, args.task_name)
    estimator = MinimumModeEstimator("dense", args.fd_eps)

    def sync():
        if args.device.startswith("cuda"): torch.cuda.synchronize()

    def flow(points, label, K=None):
        sync(); before = perf_counter()
        with torch.no_grad():
            path = sample_points(points, reference.positions, sample=sample, backbone=backbone,
                attention=attention, velocity_head=head, device=args.device, K=K or args.K,
                batch_size=args.batch_size, li_index=geometry["li_index"], label=label)
        sync(); return path, perf_counter()-before

    def sella(point, steps, label):
        pes = PESBackend(reference, calculator, batch)
        sync(); before = perf_counter()
        result = run_sella(pes, point, max_steps=steps, max_force_calls=args.max_force_calls,
                           fmax_tol=0. if steps == 1 else args.fmax, delta0=args.sella_delta0)
        sync(); result["wall_seconds"] = perf_counter()-before
        print(f"[Sella] Michael: {label}; iterations={result['steps']}, PES={result['force_calls']}, "
              f"time={result['wall_seconds']:.3f}s, status={result['status']}", flush=True)
        return result, pes

    # Exclude first-use initialization, but include fresh curvature initialization in EVERY timed Sella trial.
    flow(sources["starts"][:1], "warmup (excluded)", K=1)
    sella(sources["starts"][0], 1, "warmup (excluded)")
    proposal, proposal_seconds = flow(sources["starts"], "original proposal")
    if not np.isfinite(proposal).all(): raise ValueError("Michael: baseline produced nonfinite positions; inspect the model before repair")
    endpoints = proposal[:, -1]
    flagged = np.zeros(n, dtype=bool); gate_force = np.zeros((n, 3)); gate_eigs = np.full((n, 2), np.nan)
    gate_calls = np.zeros(n, dtype=int); gate_seconds = np.zeros(n)
    for i, z in enumerate(endpoints):
        pes = PESBackend(reference, calculator, batch); sync(); before = perf_counter()
        gate_force[i] = pes.forces(z)
        force_bad = np.linalg.norm(gate_force[i]) > args.fmax
        index_bad = False
        if args.gate == "force-index":
            eigs, _, _ = estimator.lowest_modes(pes, z)
            gate_eigs[i] = eigs[:2]
            index_bad = not (eigs[0] < -args.eigenvalue_tol and eigs[1] > args.eigenvalue_tol)
        flagged[i] = force_bad or index_bad
        gate_calls[i] = pes.force_calls; sync(); gate_seconds[i] = perf_counter()-before
        if (i+1) % 10 == 0 or i+1 == n: print(f"[gate] Michael: {i+1}/{n}; flagged so far={flagged[:i+1].sum()}", flush=True)
    original_outcome = classify(endpoints, geometry)[0]
    false_delta = endpoints[:, None]-geometry["false_bridges"][None]
    from ase.geometry import find_mic
    mic, _ = find_mic(false_delta.reshape(-1, 3), geometry["cell"], geometry["pbc"])
    basin = np.linalg.norm(mic.reshape(n, -1, 3), axis=2).argmin(axis=1)
    basin[original_outcome != 1] = -1
    extra = dict(flagged=flagged, proposal_path=proposal, gate_force=gate_force, gate_eigenvalues=gate_eigs,
                 gate_force_calls=gate_calls, gate_wall_seconds=gate_seconds, baseline_basin=basin,
                 sella_delta=np.full((n, 3), np.nan), kick_norm=np.full(n, np.nan),
                 kick_force_calls=np.zeros(n, dtype=int), kick_wall_seconds=np.zeros(n),
                 kick_force_max=np.full(n, np.nan),
                 kick_status=np.full(n, "not_needed", dtype="U32"))
    results = []

    def score(result, pes=None):
        pes = pes or PESBackend(reference, calculator, batch)
        diagnose_states(result, pes, estimator, [], args)
        results.append(result)

    def save(stage): return save_progress(out, args, geometry, sources, results, extra, stage)

    for i in range(n):
        score(flow_result("baseline", i, proposal[i], proposal_seconds/n, 0, args.K, proposal_seconds=proposal_seconds/n))
        if (i+1) % 10 == 0: save("baseline certification")
    save("baseline complete")
    failed = np.flatnonzero(flagged)
    for count, i in enumerate(failed):
        result, pes = sella(endpoints[i], 1, f"one-step endpoint {count+1}/{len(failed)}")
        # The adapter returns the last accepted state, never a temporary Hessian probe.
        delta = pes.dofs.displacement(result["final"], endpoints[i])
        extra["sella_delta"][i] = delta; extra["kick_norm"][i] = np.linalg.norm(delta)
        extra["kick_force_calls"][i] = result["force_calls"]; extra["kick_wall_seconds"][i] = result["wall_seconds"]
        extra["kick_status"][i] = result["status"]
        extra["kick_force_max"][i] = result["trace"][-1]["force_max"]
        if (count+1) % 10 == 0: save("one-step kicks")
    save("kicks complete")

    variants = [("restart_none", None, 0.)] + [(f"{kind}_a{alpha:g}", kind, alpha)
                                               for alpha in alphas for kind in ["random", "force", "sella"]]
    for method, kind, alpha in variants:
        repaired, restart_seconds = {}, 0.
        if len(failed):
            points = []
            for i in failed:
                dx = np.zeros(3) if kind is None else matched_kicks(extra["sella_delta"][i], gate_force[i], alpha,
                                                    args.seed+100003*int(i), args.max_kick)[kind]
                points.append(endpoints[i]+dx)
            points = wrap_positions(np.asarray(points), geometry["cell"], geometry["pbc"])
            path, restart_seconds = flow(points, method)
            repaired = dict(zip(failed, path))
        for i in range(n):
            path = np.vstack([proposal[i], repaired[i]]) if flagged[i] else proposal[i]
            kick_s = extra["kick_wall_seconds"][i] if flagged[i] and kind is not None else 0.
            kick_n = extra["kick_force_calls"][i] if flagged[i] and kind is not None else 0
            repair_s = restart_seconds/len(failed) if flagged[i] else 0.
            work = proposal_seconds/n + gate_seconds[i] + kick_s + repair_s
            score(flow_result(method, i, path, work, gate_calls[i]+kick_n, args.K*(1+int(flagged[i])),
                proposal_seconds=proposal_seconds/n, gate_seconds=gate_seconds[i], kick_seconds=kick_s, restart_seconds=repair_s))
            if (i+1) % 10 == 0: save(f"{method} certification")
        save(f"{method} complete")
    # The fair all-Sella alternative starts BEFORE the original model proposal, on every source.
    for i, point in enumerate(sources["starts"]):
        result, pes = sella(point, args.max_steps, f"full original start {i+1}/{n}")
        result.update(method="full_sella", source_index=i, restart=0, model_forward_calls=0,
                      full_sella_seconds=result["wall_seconds"])
        score(result, pes); save(f"full Sella {i+1}/{n}")
    report = save("complete"); print(report, flush=True)
    print(f"[done] Michael: CPU replot: python examples/LiC/experiments/pes_and_recovery/sella_restart.py --plot-only {out/'sella_restart.npz'}", flush=True)
    if not args.no_plot: plot_results(out / "sella_restart.npz")


if __name__ == "__main__":
    main()
