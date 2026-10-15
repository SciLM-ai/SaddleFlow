"""Michael: autonomous 20-step E0/E1/E2 sampling, then separate UMA verification.

Reuses the previous benchmark's source coordinates, never its model endpoints.
All requested training checkpoints are reported: no test-based checkpoint
selection. Numerical results are saved before the existing graphene PDF renderer.
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

from saddleflow.data import atoms_to_sample_dict
from saddleflow.classical_saddle_search import FreeDOFs, MinimumModeEstimator, PESBackend, evaluate_endpoint, make_uma_calculator

try:
    from compare_saddle_search import save_archive
    from force_guided_sampling import load_flow_model
    from robustness_shells import rollout_from_step
    from saddle_search_data import load_geometry
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.compare_saddle_search import save_archive
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
    from examples.LiC.rollout_eval.robustness_shells import rollout_from_step
    from examples.LiC.rollout_eval.saddle_search_data import load_geometry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Parent of E0/, E1/, E2/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--updates", default="0,250,500,1000,2000")
    p.add_argument("--starts-archive", default="/scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_21107961/saddle_search.npz")
    p.add_argument("--test-traj", default=str((Path(__file__).resolve().parents[2] / "test_set.traj")))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fmax", type=float, default=0.005)
    p.add_argument("--eigenvalue-tol", type=float, default=0.01)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def checkpoint_at(run, update, total):
    return run / ("checkpoint_final" if update == total else f"checkpoint_step_{update:08d}")


def shared_sources(path, geometry):
    with np.load(path, allow_pickle=False) as data:
        for key in ["numbers", "cell", "pbc", "free_mask", "train_saddles", "test_saddles", "false_bridges"]:
            if not np.allclose(data[key], geometry[key]):
                raise ValueError(f"Source archive differs in {key}")
        if not np.allclose(data["positions"][~geometry["free_mask"]], geometry["positions"][~geometry["free_mask"]]):
            raise ValueError("Source archive scaffold differs")
        return dict(starts=data["starts"].copy(), population=data["source_population"].copy(),
                    radius=data["source_radius"].copy(), parent=data["source_parent"].copy())


def check_matching_logs(root):
    """Verify pairs by full bridge hashes and selected-example indices."""
    signatures = []
    for experiment in ["E0", "E1", "E2"]:
        rows = [json.loads(line) for line in (root / experiment / "training_steps.jsonl").read_text().splitlines()]
        signatures.append([(r["update"], r["batch_sha256"], r["selected_example"]) for r in rows])
    if not signatures[0] or signatures[0] != signatures[1] or signatures[0] != signatures[2]:
        raise ValueError("Training streams are not matched; inspect training_steps.jsonl")
    return len(signatures[0])


def make_report(data):
    config = json.loads(str(data["config_json"]))
    rows = ["Michael: matched PES-direction FM continuation", "="*100,
            "All sampling is 20-step autonomous Euler; no force, Hessian, or minimum-mode oracle calls.",
            "Student UMA backbone forwards ARE executed; forces/curvature below are post-hoc measurements.",
            "True means any stored saddle (<0.10 A); false region <0.30 A, with true precedence.",
            f"Physical = free fmax <={config['fmax']} AND lambda1<-{config['eigenvalue_tol']} AND lambda2>{config['eigenvalue_tol']}.",
            "Checkpoints are predeclared, not selected using evaluation results.", "",
            "method        N    true%   false%   physical%   index1%   med fmax   med lambda1   med lambda2", "-"*100]
    for method in dict.fromkeys(data["method"]):
        use = data["method"] == method
        rows.append(f"{method:12s} {use.sum():4d} {100*np.mean(data['outcome'][use]==0):8.2f}"
                    f" {100*np.mean(data['outcome'][use]==1):8.2f} {100*data['physical_success'][use].mean():11.2f}"
                    f" {100*data['index_one'][use].mean():9.2f} {np.median(data['force_max'][use]):10.6f}"
                    f" {np.median(data['final_eigenvalues'][use,0]):13.5f} {np.median(data['final_eigenvalues'][use,1]):13.5f}")
    population = data["source_population"][data["source_index"]]
    rows += ["", "method       population       N    true%   false%   physical%", "-"*70]
    for group in np.unique(population):
        for method in dict.fromkeys(data["method"]):
            use = (data["method"] == method) & (population == group)
            rows.append(f"{method:12s} {group:14s} {use.sum():4d} {100*np.mean(data['outcome'][use]==0):8.2f}"
                        f" {100*np.mean(data['outcome'][use]==1):8.2f} {100*data['physical_success'][use].mean():11.2f}")
    return "\n".join(rows) + "\n"


def main():
    args = parse_args()
    root, out = Path(args.run_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = [json.loads((root / e / "config.json").read_text()) for e in ["E0", "E1", "E2"]]
    settings = ["init_weights", "updates", "learning_rate", "uma_lr", "batch_size", "noise_sigma", "seed", "ema_decay"]
    for config in configs[1:]:
        if any(config[k] != configs[0][k] for k in settings):
            raise ValueError("Continuation settings differ across E0/E1/E2")
        for key in ["lambda_pes", "teacher_deadband"]:
            if config["extras"][key] != configs[0]["extras"][key]:
                raise ValueError(f"Shared calibration differs: {key}")
    matched = check_matching_logs(root)
    if matched != configs[0]["updates"]:
        raise ValueError("Training did not complete the requested number of updates")
    reference, geometry = load_geometry(configs[0]["train_traj"], args.test_traj)
    sources = shared_sources(args.starts_archive, geometry)
    dofs, sample = FreeDOFs(reference), atoms_to_sample_dict(reference)
    # Metadata expected by the existing archive and PDF renderer.
    args.K, args.max_steps, args.step_budgets, args.restarts = 20, 20, "20", 1
    args.train_traj = configs[0]["train_traj"]
    args.model = configs[0]["extras"]["backbone"]
    args.fd_eps = configs[0]["fd_eps"]
    args.checkpoints = {}
    calculator, batch_evaluate = make_uma_calculator(args.model, args.device, sample["task_name"])
    results = []
    initial_paths = None
    archive = out / "saddle_search.npz"
    updates = [int(value) for value in args.updates.split(",")]
    for experiment, config in zip(["E0", "E1", "E2"], configs):
        for update in updates:
            checkpoint = checkpoint_at(root / experiment, update, config["updates"])
            method = f"{experiment}_s{update:04d}"
            args.checkpoints[method] = str(checkpoint)
            _, _, backbone, attention, head = load_flow_model(checkpoint, args.device)
            paths, elapsed = [], []
            print(f"[sample] Michael: {method}; {len(sources['starts'])} identical starts; no PES queries", flush=True)
            # Sampling phase: calculator is NOT referenced anywhere in this loop.
            for first in range(0, len(sources["starts"]), args.batch_size):
                points = sources["starts"][first:first+args.batch_size]
                positions = torch.as_tensor(np.asarray([dofs.expand(x) for x in points]), dtype=torch.float32)
                began = perf_counter()
                _, path = rollout_from_step(positions, 0, sample=sample, backbone=backbone,
                        attention=attention, velocity_head=head, device=args.device, K=20)
                paths.extend(path.numpy())
                elapsed.extend([(perf_counter()-began)/len(points)]*len(points))
                print(f"[sample] {method} {min(first+args.batch_size,len(sources['starts']))}/{len(sources['starts'])}", flush=True)
            if update == 0:
                if initial_paths is None:
                    initial_paths = np.asarray(paths)
                elif not np.allclose(initial_paths, paths, atol=1e-5, rtol=0):
                    raise ValueError("Update-zero samplers differ; check shared EMA initialization")
            del backbone, attention, head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Verification phase only: seven force geometries per Li endpoint.
            for i, path in enumerate(paths):
                pes = PESBackend(reference, calculator, batch_evaluate)
                began = perf_counter()
                evaluation = evaluate_endpoint(pes, path[-1], MinimumModeEstimator(eps=args.fd_eps), args.fmax, args.eigenvalue_tol)
                trace = [dict(x=x.astype(float), q=None, eigs=None, force_rms=np.nan,
                              force_max=np.nan, query_count=0, force_parallel=np.nan,
                              force_perpendicular=np.nan, shift_rms=0, hessian_asymmetry=np.nan) for x in path]
                results.append(dict(method=method, source_index=i, restart=0, final=path[-1], trace=trace,
                    status="fixed_K", error="", steps=20, force_calls=0, wall_seconds=elapsed[i],
                    model_forward_calls=20, diagnostic_seconds=perf_counter()-began,
                    diagnostic_force_calls=evaluation["diagnostic_force_calls"], evaluation=evaluation,
                    budget_results=[dict(x=path[-1], **evaluation)]))
                if (i+1) % 16 == 0 or i+1 == len(paths):
                    print(f"[verify] {method} {i+1}/{len(paths)} endpoints (post-hoc UMA)", flush=True)
            data = save_archive(archive, args, geometry, sources, results, [20])
            (out / "report.txt").write_text(make_report(data))
            print(f"[save] Michael: {archive}", flush=True)
    print(make_report(data), flush=True)
    if not args.no_plot:
        try:
            from plot_pes_directions import render_pdf
        except ModuleNotFoundError:
            from examples.LiC.experiments.pes_and_recovery.plot_pes_directions import render_pdf
        render_pdf(archive)
    print(f"[done] Michael: replot with python examples/LiC/experiments/pes_and_recovery/plot_pes_directions.py {archive}", flush=True)


if __name__ == "__main__":
    main()
