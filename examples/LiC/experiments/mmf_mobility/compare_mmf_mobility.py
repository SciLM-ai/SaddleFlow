"""Compare raw/global/state MMF at a matched Kmax (default training horizon: 100).

Michael: use --baseline-archive to reuse the EXACT 68 starts and completed
Sella/SaddleFlow comparator rows from the previous benchmark. A/B/C are rerun
with shared physics settings; the old oracle row is never substituted for A.
NPZ and report are saved before the separate plotting function is called.
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
import torch

from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, evaluate_endpoint, make_uma_calculator
from learned_mobility import MobilityPolicy, run_mobility_flow

try:
    from compare_saddle_search import save_archive
    from saddle_search_data import classify, generate_starts, load_geometry
    from train_mmf_mobility import load_policy, rollout_options
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.compare_saddle_search import save_archive
    from examples.LiC.rollout_eval.saddle_search_data import classify, generate_starts, load_geometry
    from examples.LiC.experiments.mmf_mobility.train_mmf_mobility import load_policy, rollout_options


def parse_args():
    here = Path(__file__).resolve().parents[2]  # examples/LiC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Directory produced by train_mmf_mobility.py.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline-archive", help="Previous saddle_search.npz: shared starts + Sella/SaddleFlow finals.")
    p.add_argument("--train-traj", default=None)
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))
    p.add_argument("--step-budgets", default="20,40,60,80,100")
    p.add_argument("--starts", default="mixed")
    p.add_argument("--starts-npz", help="Alternatively reuse starts from a robustness-shell NPZ.")
    p.add_argument("--limit-starts", type=int)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--samples-per-site", type=int, default=1)
    p.add_argument("--false-noise-sigma", type=float, default=0.1)
    p.add_argument("--true-noise-sigma", type=float, default=0.25)
    p.add_argument("--radii", default="0.6,0.8,1.0")
    p.add_argument("--directions-per-radius", type=int, default=1)
    p.add_argument("--shell-dimensions", choices=("xy", "xyz"), default="xy")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load_baseline(path, args, geometry, config):
    """Reuse evidence only when source geometries, potential, and certificate match."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    old = json.loads(str(data["config_json"]))
    for key in ("model", "task_name", "fd_eps", "fmax", "eigenvalue_tol"):
        if old[key] != config[key]:
            raise ValueError(f"Baseline uses different {key}; do not mix certificates/potentials.")
    for key in ("cell", "pbc", "numbers", "free_mask", "train_saddles", "test_saddles", "false_bridges"):
        if not np.allclose(data[key], geometry[key]):
            raise ValueError(f"Baseline {key} differs from current LiC data.")
    free = geometry["free_mask"]
    if not np.allclose(data["positions"][~free], geometry["positions"][~free]):
        raise ValueError("Baseline fixed scaffold differs.")
    count = len(data["starts"])
    for method in ("sella", "saddleflow"):
        use = data["method"] == method
        if use.any() and (use.sum() != count or not np.array_equal(np.sort(data["source_index"][use]), np.arange(count))):
            raise ValueError(f"Baseline {method} does not have one completed trial for every source.")
    sources = dict(starts=data["starts"], population=data["source_population"],
                   radius=data["source_radius"], parent=data["source_parent"])
    args.ckpt, args.K = old.get("ckpt"), old.get("K", 20)
    return data, sources


def metrics_from_row(row, config):
    eigs = row["eigs"]
    index_one = bool(eigs[0] < -config["eigenvalue_tol"] and eigs[1] > config["eigenvalue_tol"])
    stationary = bool(row["force_max"] <= config["fmax"])
    return dict(force_rms=row["force_rms"], force_max=row["force_max"], eigs=eigs[:2],
                stationary=stationary, index_one=index_one,
                physical_success=stationary and index_one,
                hessian_asymmetry=row["hessian_asymmetry"], diagnostic_force_calls=0)


def diagnose_steps(result, pes, config, budgets):
    """Anytime snapshots use already evaluated states; early stops carry forward.

    The policy always sees k/Kmax, even at the 20-step snapshot. We NEVER change
    its clock to k/20. Each final endpoint gets an additional independent check.
    That verification cost is recorded separately from search.
    """
    observed = [row for row in result["trace"] if row["q"] is not None]
    result["budget_results"] = []
    for budget in budgets:
        row = [row for row in observed if row["step"] <= budget][-1]
        result["budget_results"].append(dict(x=row["x"].copy(), **metrics_from_row(row, config)))
    began = perf_counter()
    estimator = MinimumModeEstimator("dense", config["fd_eps"])
    result["evaluation"] = evaluate_endpoint(pes, result["final"], estimator,
                                               config["fmax"], config["eigenvalue_tol"])
    result["diagnostic_force_calls"] = result["evaluation"]["diagnostic_force_calls"]
    result["diagnostic_seconds"] = perf_counter() - began
    result["budget_available"] = [True] * len(budgets)


def imported_comparators(data, budgets, selected_sources):
    """Copy completed Sella/FM results, never select a restart or a best endpoint."""
    remap = {old: new for new, old in enumerate(selected_sources)}
    results = []
    for i in range(len(data["method"])):
        method, source = str(data["method"][i]), int(data["source_index"][i])
        if method not in ("sella", "saddleflow") or source not in remap:
            continue
        trace = []
        for j in range(int(data["trace_length"][i])):
            q, eigs = data["q"][i, j], data["trace_eigenvalues"][i, j]
            row = dict(x=data["path"][i, j].copy(), q=q.copy() if np.isfinite(q).all() else None,
                       eigs=eigs.copy() if np.isfinite(eigs).all() else None)
            for key in ("force_rms", "force_max", "query_count", "force_parallel", "force_perpendicular",
                        "shift_rms", "hessian_asymmetry"):
                row[key] = float(data["trace_" + key][i, j])
            trace.append(row)
        evaluation = {key: data[key][i].item() for key in
                      ("force_rms", "force_max", "index_one", "stationary", "physical_success")}
        evaluation.update(eigs=data["final_eigenvalues"][i].copy(),
                          hessian_asymmetry=float(data["final_hessian_asymmetry"][i]),
                          diagnostic_force_calls=int(data["diagnostic_force_calls"][i]))
        result = dict(method=method, source_index=remap[source], restart=0, final=data["final"][i].copy(),
                      trace=trace, status=str(data["status"][i]), error=str(data["error"][i]),
                      evaluation=evaluation, budget_available=[False]*len(budgets),
                      budget_results=[dict(x=data["final"][i].copy(), **evaluation) for _ in budgets])
        for key in ("steps", "force_calls", "wall_seconds", "model_forward_calls",
                    "diagnostic_force_calls", "diagnostic_seconds"):
            result[key] = data[key][i].item()
        results.append(result)
    return results


def make_report(data):
    cfg = json.loads(str(data["config_json"]))
    methods = list(dict.fromkeys(data["method"].tolist()))
    lines = ["Michael: LiC learned oracle-MMF translation benchmark", "="*110,
             f"Kmax={cfg['max_steps']}; h={cfg['step_size']} A^2/eV; trust radius={cfg['max_step']} A",
             f"UMA={cfg['model']}; FD eps={cfg['fd_eps']} A; identical starts={len(data['starts'])}",
             f"Physical stop/certificate: fmax <= {cfg['fmax']}, lambda1 < -{cfg['eigenvalue_tol']}, lambda2 > {cfg['eigenvalue_tol']}",
             "A raw_mmf=(1,1); B global_mmf=learned constants; C learned_mmf=state-dependent MLP.",
             "B/C trained only on noisy training saddles; endpoint loss only. A has no trained parameters.",
             "Anytime rows use the SAME Kmax clock; physically early-stopped states carry forward.",
             "Each evaluated Li state costs ~7 PES queries; terminal verification is extra.",
             f"Imported Sella/SaddleFlow provenance: {cfg.get('baseline_archive') or 'not included'}", "",
             "Final outcomes", "method            N   true%  false%   phys%  med fmax  med lambda1  med lambda2  med steps  med PES  med sec", "-"*116]
    for method in methods:
        use = data["method"] == method
        eigs = data["final_eigenvalues"][use]
        lines.append(f"{method:16s} {use.sum():3d} {100*np.mean(data['outcome'][use]==0):7.1f}"
                     f" {100*np.mean(data['outcome'][use]==1):7.1f} {100*np.mean(data['physical_success'][use]):7.1f}"
                     f" {np.median(data['force_max'][use]):9.5f} {np.median(eigs[:,0]):12.5f} {np.median(eigs[:,1]):12.5f}"
                     f" {np.median(data['steps'][use]):10.1f} {np.median(data['force_calls'][use]):8.1f}"
                     f" {np.median(data['wall_seconds'][use]):8.2f}")
    lines += ["", "Anytime performance (translation-step budget, not FM time or PES-query count)",
              "method           budget    N   true%  false%   phys%  median fmax", "-"*76]
    for j, budget in enumerate(data["step_budgets"]):
        for method in methods:
            use = (data["method"] == method) & data["budget_available"][:, j]
            if use.any():
                lines.append(f"{method:16s} {budget:6d} {use.sum():4d}"
                             f" {100*np.mean(data['budget_outcome'][use,j]==0):7.1f}"
                             f" {100*np.mean(data['budget_outcome'][use,j]==1):7.1f}"
                             f" {100*np.mean(data['budget_physical_success'][use,j]):7.1f}"
                             f" {np.median(data['budget_force_max'][use,j]):12.6f}")
    lines += ["", "Final outcomes by source population", "method           population       N   true%  false%   phys%", "-"*76]
    population = data["source_population"][data["source_index"]]
    for group in np.unique(population):
        for method in methods:
            use = (data["method"] == method) & (population == group)
            if use.any():
                lines.append(f"{method:16s} {group:14s} {use.sum():4d} {100*np.mean(data['outcome'][use]==0):7.1f}"
                             f" {100*np.mean(data['outcome'][use]==1):7.1f} {100*np.mean(data['physical_success'][use]):7.1f}")
    lines += ["", "Status counts and total search cost"]
    for method in methods:
        use = data["method"] == method
        names, counts = np.unique(data["status"][use], return_counts=True)
        lines.append(f"{method}: " + ", ".join(f"{name}={count}" for name, count in zip(names, counts))
                     + f"; search PES={data['force_calls'][use].sum():.0f}, search seconds={data['wall_seconds'][use].sum():.1f}"
                     + f", verification PES={data['diagnostic_force_calls'][use].sum():.0f}")
    lines += ["", "Catalog success means ANY stored saddle, not necessarily the training parent.",
              "A false-region endpoint at Kmax is not evidence of a fixed point of MMF.",
              "Training uses an approximate local derivative: dF/dx=-H, dq/dx=0, dlambda/dx=0.",
              "Equal outer steps are not equal cost to SaddleFlow. Imported comparator timings are historical."]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    torch.set_num_threads(1)
    run = Path(args.run_dir)
    config = json.loads((run / "config.json").read_text())
    args.training_config = config  # Keep training provenance inside the self-contained NPZ metadata.
    args.policy_updates = {}
    policies = [("raw_mmf", MobilityPolicy("raw", config["mobility_min"], config["mobility_max"]))]
    for kind, name in (("global", "global_mmf"), ("state", "learned_mmf")):
        policy, state = load_policy(run / kind / "checkpoint_final.pt")
        for key in ("model", "task_name", "max_steps", "step_size", "max_step", "fd_eps", "fmax", "eigenvalue_tol"):
            if state["config"][key] != config[key]:
                raise ValueError(f"Policy {kind} has mismatched {key}.")
        policy.eval()
        policies.append((name, policy))
        args.policy_updates[kind] = state["update"]
    budgets = sorted(set(int(value) for value in args.step_budgets.split(",")))
    if not budgets or min(budgets) < 1 or max(budgets) > config["max_steps"] or args.log_every_steps < 1:
        raise ValueError("Step budgets must lie in [1,Kmax]; logging interval must be positive.")
    for key in ("model", "task_name", "max_steps", "step_size", "max_step", "fd_eps", "fmax", "eigenvalue_tol"):
        setattr(args, key, config[key])
    args.train_traj = args.train_traj or config["train_traj"]
    args.ckpt, args.K, args.restarts = None, 20, 1  # Metadata for the shared archive/plot helpers.
    reference, geometry = load_geometry(args.train_traj, args.test_traj)
    baseline = None
    if args.baseline_archive:
        if args.starts_npz:
            raise ValueError("Choose baseline-archive OR starts-npz, not both.")
        baseline, sources = load_baseline(args.baseline_archive, args, geometry, config)
        chosen = np.arange(len(sources["starts"]))
        if args.limit_starts is not None:
            chosen = np.sort(np.random.default_rng(args.seed).choice(chosen, min(args.limit_starts, len(chosen)), replace=False))
        sources = {key: values[chosen] for key, values in sources.items()}
    else:
        sources = generate_starts(args, geometry)
    if not len(sources["starts"]):
        raise ValueError("No starts selected.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "saddle_search.npz"
    if archive_path.exists():
        raise FileExistsError(f"Use a new output directory: {archive_path}")
    print(f"[setup] Michael: {len(sources['starts'])} shared starts; Kmax={args.max_steps}; anytime={budgets}", flush=True)
    print("[UMA] loading frozen oracle and warming up center + FD probes", flush=True)
    calculator, batch_evaluate = make_uma_calculator(args.model, args.device, args.task_name)
    warm = PESBackend(reference, calculator, batch_evaluate)
    warm.forces(sources["starts"][0])
    warm.free_hessian(sources["starts"][0], args.fd_eps)
    results = []

    def save_progress():
        data = save_archive(archive_path, args, geometry, sources, results, budgets)
        (out / "report.txt").write_text(make_report(data))

    for method, policy in policies:
        for index, source in enumerate(sources["starts"]):
            print(f"[search] {method} start {index+1}/{len(sources['starts'])} begin", flush=True)
            pes = PESBackend(reference, calculator, batch_evaluate)

            def log_step(step, row):
                if step % args.log_every_steps == 0:
                    print(f"  step {step:3d}/{args.max_steps} PES={row['query_count']:4d}"
                          f" fmax={row['force_max']:.6f} eigs={row['eigs'][:2]}"
                          f" a={row['mobility_a']:.3f} b={row['mobility_b']:.3f}", flush=True)

            with torch.no_grad():
                result = run_mobility_flow(pes, source, policy, callback=log_step, **rollout_options(config))
            result.pop("terminal_tensor")
            result.update(method=method, source_index=index, restart=0)
            diagnose_steps(result, pes, config, budgets)
            results.append(result)
            save_progress()
            outcome, distance, _ = classify(np.asarray([result["final"]]), geometry)
            print(f"[done trial] {len(results)}/{3*len(sources['starts'])} {method}; {result['status']}"
                  f" steps={result['steps']} d_true={distance[0]:.4f} A phys={result['evaluation']['physical_success']}", flush=True)
    if baseline is not None:
        results.extend(imported_comparators(baseline, budgets, chosen))
    save_progress()
    with np.load(archive_path, allow_pickle=False) as data:
        print("\n" + make_report(data), flush=True)
    print(f"[save] {archive_path}; {out / 'report.txt'}", flush=True)
    if not args.no_plot:
        try:
            try:
                from plot_saddle_search import render_pdf
            except ModuleNotFoundError:
                from examples.LiC.rollout_eval.plot_saddle_search import render_pdf
            render_pdf(archive_path)
        except Exception as exc:
            print(f"[plot] failed: {exc}; NPZ/report already saved.", flush=True)
    print(f"[done] replot: python examples/LiC/rollout_eval/plot_saddle_search.py {archive_path}", flush=True)


if __name__ == "__main__":
    main()
