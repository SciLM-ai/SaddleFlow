"""Physics ceiling for LiC: Sella, oracle MMF, two-flow, optional SaddleFlow.

Run from the repository root. Numerical results are saved before plotting.
The same starts are used for every method; the two-flow method additionally
repeats each start with independent, target-free initial orientations.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from saddleflow.classical_saddle_search import (
    MinimumModeEstimator, PESBackend, evaluate_endpoint, make_uma_calculator,
    run_oracle_mmf, run_sella, run_two_flow,
)

try:
    from saddle_search_data import classify, generate_starts, load_geometry
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.saddle_search_data import classify, generate_starts, load_geometry


def parse_args():
    here = Path(__file__).resolve().parents[1]  # examples/LiC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", help="Optional existing unconditioned LiC checkpoint; adds SaddleFlow.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))
    p.add_argument("--methods", default="sella,oracle,two_flow")
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--task-name", default="omat")
    p.add_argument("--device", default="cuda")
    p.add_argument("--starts", default="mixed", help="mixed or comma-separated false,true,shells,grid")
    p.add_argument("--starts-npz", help="Reuse t=0 Li starts from an existing shell/throw NPZ.")
    p.add_argument("--limit-starts", type=int)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--samples-per-site", type=int, default=1, help="Noisy draws per true/false site.")
    p.add_argument("--false-noise-sigma", type=float, default=0.1)
    p.add_argument("--true-noise-sigma", type=float, default=0.25)
    p.add_argument("--radii", default="0.6,0.8,1.0")
    p.add_argument("--directions-per-radius", type=int, default=1)
    p.add_argument("--shell-dimensions", choices=("xy", "xyz"), default="xy")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--restarts", type=int, default=2, help="Independent two-flow trials per source.")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-force-calls", type=int, default=800, help="Per independent trial; includes rotation.")
    p.add_argument("--force-budgets", default="10,20,40,80")
    p.add_argument("--fmax", type=float, default=0.005, help="Free-atom fmax stopping tolerance, eV/A.")
    p.add_argument("--eigenvalue-tol", type=float, default=0.01, help="Index-one certification tolerance, eV/A^2.")
    p.add_argument("--fd-eps", type=float, default=0.002, help="Finite-difference displacement, A.")
    p.add_argument("--mode-backend", choices=("auto", "dense", "hvp"), default="auto")
    p.add_argument("--orientation-backend", choices=("dense", "hvp"), default="dense")
    p.add_argument("--step-size", type=float, default=0.2, help="Translation mobility, A^2/eV.")
    p.add_argument("--max-step", type=float, default=0.1, help="Maximum translation RMS per free atom, A.")
    p.add_argument("--orientation-step-size", type=float, default=0.1, help="Rotation step, A^2/eV.")
    p.add_argument("--rotations-per-step", type=int, default=3)
    p.add_argument("--sella-delta0", type=float, default=0.1)
    p.add_argument("--K", type=int, default=20, help="SaddleFlow Euler steps, NOT physics iterations.")
    p.add_argument("--batch-size", type=int, default=64, help="SaddleFlow batch size.")
    p.add_argument("--log-every-steps", type=int, default=10)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def diagnose_states(result, pes, estimator, budgets, args):
    """Evaluate final and last available state at each PES-query budget.

    Verification calls are separate from search calls. No extra searches and no
    label-based restart selection. Repeated identical checkpoints are evaluated
    once. A physical iteration interrupted during rotation keeps its last x.
    """
    cache = {}

    def score(x):
        key = np.asarray(x).tobytes()
        if key not in cache:
            before = perf_counter()
            cache[key] = evaluate_endpoint(pes, x, estimator, args.fmax, args.eigenvalue_tol)
            cache[key]["diagnostic_seconds"] = perf_counter() - before
        return cache[key]

    result["evaluation"] = score(result["final"])
    budget_results = []
    for budget in budgets:
        available = [row for row in result["trace"] if row["query_count"] <= budget]
        x = available[-1]["x"] if available else result["trace"][0]["x"]
        budget_results.append(dict(x=x.copy(), **score(x)))
    result["budget_results"] = budget_results
    result["diagnostic_force_calls"] = sum(row["diagnostic_force_calls"] for row in cache.values())
    result["diagnostic_seconds"] = sum(row["diagnostic_seconds"] for row in cache.values())


def run_physics_trials(args, reference, geometry, sources, calculator, batch_evaluate, methods, budgets,
                       results, save_progress):
    runners = {"sella": run_sella, "oracle": run_oracle_mmf, "two_flow": run_two_flow}
    total = sum(len(sources["starts"]) * (args.restarts if m == "two_flow" else 1) for m in methods)
    completed = 0
    for method in methods:
        restarts = args.restarts if method == "two_flow" else 1
        for source_index, point in enumerate(sources["starts"]):
            for restart in range(restarts):
                seed = args.seed + 100003 * source_index + restart
                pes = PESBackend(reference, calculator, batch_evaluate)
                estimator = MinimumModeEstimator(args.mode_backend, args.fd_eps, seed=seed)
                label = f"{method} start {source_index + 1}/{len(sources['starts'])} restart {restart + 1}/{restarts}"
                print(f"[search] {label} begin", flush=True)

                def log_step(step, row):
                    if step % args.log_every_steps == 0:
                        eig = row["eigs"]
                        text = "not computed" if eig is None else ",".join(f"{v:.4f}" for v in eig[:2])
                        print(f"  step {step:4d}  PES {row['query_count']:4d}/{args.max_force_calls}"
                              f"  fmax {row['force_max']:.6f}  eigs [{text}]", flush=True)

                options = dict(max_steps=args.max_steps, max_force_calls=args.max_force_calls,
                               fmax_tol=args.fmax, callback=log_step)
                if method == "sella":
                    options["delta0"] = args.sella_delta0
                else:
                    options.update(estimator=estimator, step_size=args.step_size, max_step=args.max_step,
                                   eigenvalue_tol=args.eigenvalue_tol)
                    if method == "two_flow":
                        options.update(orientation_backend=args.orientation_backend,
                                       orientation_step_size=args.orientation_step_size,
                                       rotations_per_step=args.rotations_per_step, seed=seed)
                result = runners[method](pes, point, **options)
                result.update(method=method, source_index=source_index, restart=restart,
                              model_forward_calls=0)
                print(f"[verify] {label}; {result['status']}, search PES={result['force_calls']},"
                      f" time={result['wall_seconds']:.1f}s", flush=True)
                diagnose_states(result, pes, estimator, budgets, args)
                results.append(result)
                completed += 1
                save_progress()
                _, distance, _ = classify(np.asarray([result["final"]]), geometry)
                print(f"[done trial] {completed}/{total}  {label};"
                      f" d_true={distance[0]:.4f} A,"
                      f" fmax={result['evaluation']['force_max']:.6f},"
                      f" index1={result['evaluation']['index_one']}", flush=True)


def run_saddleflow_trials(args, reference, sources, calculator, batch_evaluate, budgets, results, save_progress):
    """Reuse the ordinary checkpoint loader/sampler; no physics guidance added."""
    import torch
    from saddleflow.data import atoms_to_sample_dict
    try:
        from force_guided_sampling import load_flow_model
        from robustness_shells import rollout_from_step
    except ModuleNotFoundError:
        from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
        from examples.LiC.rollout_eval.robustness_shells import rollout_from_step

    print("[saddleflow] loading checkpoint and warming up the ordinary sampler", flush=True)
    _, _, backbone, attention, head = load_flow_model(Path(args.ckpt).resolve(), args.device)
    sample = atoms_to_sample_dict(reference)
    try: from conditioning import conditioning_queries
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import conditioning_queries
    dofs = PESBackend(reference, calculator, batch_evaluate).dofs
    # Exclude first-use graph/model preparation from the learned row's timing.
    warm_positions = torch.as_tensor(dofs.expand(sources["starts"][0]), dtype=torch.float32)[None]
    rollout_from_step(warm_positions, args.K - 1, sample=sample, backbone=backbone,
                      attention=attention, velocity_head=head, device=args.device, K=args.K,
                      source_positions=warm_positions)
    for first in range(0, len(sources["starts"]), args.batch_size):
        before_queries = conditioning_queries(head)
        points = sources["starts"][first:first + args.batch_size]
        full_positions = np.asarray([dofs.expand(point) for point in points])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = perf_counter()
        _, paths = rollout_from_step(
            torch.as_tensor(full_positions, dtype=torch.float32), 0,
            sample=sample, backbone=backbone, attention=attention,
            velocity_head=head, device=args.device, K=args.K,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds_per_source = (perf_counter() - start) / len(points)
        queries_per_source = (conditioning_queries(head)-before_queries)//len(points)
        for i, path in enumerate(paths.numpy()):
            # Geometry/source-memory students spend zero PES queries. Force/
            # curvature inputs pay for current-state oracle queries, including FD.
            trace = [dict(x=x.astype(float), query_count=j*queries_per_source//args.K, force_rms=np.nan,
                          force_max=np.nan, eigs=None, q=None, force_parallel=np.nan,
                          force_perpendicular=np.nan, shift_rms=0.0,
                          hessian_asymmetry=np.nan) for j, x in enumerate(path)]
            result = dict(method="saddleflow", source_index=first + i, restart=0,
                          final=path[-1].astype(float), trace=trace, status="fixed_K",
                          steps=args.K, force_calls=queries_per_source, wall_seconds=seconds_per_source,
                          model_forward_calls=args.K, error="")
            if getattr(args, "geometry_only", False):
                evaluation = dict(force_rms=np.nan, force_max=np.nan, index_one=np.nan,
                    stationary=np.nan, physical_success=np.nan, eigs=np.full(2, np.nan), hessian_asymmetry=np.nan)
                result.update(evaluation=evaluation, diagnostic_force_calls=0, diagnostic_seconds=0.,
                              budget_results=[dict(x=path[-1].copy(), **evaluation) for _ in budgets])
            else:
                pes = PESBackend(reference, calculator, batch_evaluate)
                estimator = MinimumModeEstimator(args.mode_backend, args.fd_eps, seed=args.seed)
                diagnose_states(result, pes, estimator, budgets, args)
            results.append(result)
        save_progress()
        print(f"[saddleflow] {min(first + args.batch_size, len(sources['starts']))}/"
              f"{len(sources['starts'])} identical starts", flush=True)
    return backbone, attention, head


def save_archive(path, args, geometry, sources, results, budgets):
    """A self-contained, pickle-free archive; plotting needs no model or GPU."""
    n = len(results)
    length = max((len(r["trace"]) for r in results), default=1)
    paths = np.full((n, length, 3), np.nan)
    orientations = np.full_like(paths, np.nan)
    scalar_keys = ["force_rms", "force_max", "query_count", "force_parallel",
                   "force_perpendicular", "shift_rms", "hessian_asymmetry"]
    # Optional learned-mobility diagnostics. Existing solver archives retain
    # their original schema when these fields are absent.
    for key in ("step", "mobility_a", "mobility_b", "clipped", "eigenvalue3"):
        if any(key in row for result in results for row in result["trace"]):
            scalar_keys.append(key)
    history = {"trace_" + key: np.full((n, length), np.nan) for key in scalar_keys}
    eigenvalues = np.full((n, length, 2), np.nan)
    for i, result in enumerate(results):
        for j, row in enumerate(result["trace"]):
            paths[i, j] = row["x"]
            if row["q"] is not None:
                orientations[i, j] = row["q"]
            if row["eigs"] is not None:
                eigenvalues[i, j, :len(row["eigs"][:2])] = row["eigs"][:2]
            for key in scalar_keys:
                history["trace_" + key][i, j] = row.get(key, np.nan)
    final = np.asarray([r["final"] for r in results]).reshape(n, 3)
    outcome, d_true, d_false = classify(final, geometry) if n else (np.array([]),) * 3
    array = dict(config_json=json.dumps(vars(args)), **geometry,
                 starts=sources["starts"], source_population=sources["population"],
                 source_radius=sources["radius"], source_parent=sources["parent"],
                 method=np.asarray([r["method"] for r in results], dtype="U16"),
                 source_index=np.asarray([r["source_index"] for r in results], dtype=int),
                 restart=np.asarray([r["restart"] for r in results], dtype=int),
                 status=np.asarray([r["status"] for r in results], dtype="U32"),
                 error=np.asarray([r.get("error", "") for r in results], dtype="U512"),
                 path=paths, q=orientations, trace_eigenvalues=eigenvalues, **history,
                 final=final, outcome=outcome, distance_to_true=d_true, distance_to_false=d_false,
                 trace_length=np.asarray([len(r["trace"]) for r in results]))
    budget_key = "step_budgets" if getattr(args, "step_budgets", None) else "force_budgets"
    array[budget_key] = np.asarray(budgets)
    array["budget_available"] = np.asarray([r.get("budget_available", [True] * len(budgets))
                                             for r in results], dtype=bool).reshape(n, len(budgets))
    for key in ("steps", "force_calls", "wall_seconds", "model_forward_calls",
                "diagnostic_force_calls", "diagnostic_seconds"):
        array[key] = np.asarray([r[key] for r in results])
    for key in ("force_rms", "force_max", "index_one", "stationary", "physical_success"):
        array[key] = np.asarray([r["evaluation"][key] for r in results])
        array["budget_" + key] = np.asarray([[b[key] for b in r["budget_results"]] for r in results])
    array["final_eigenvalues"] = np.asarray([r["evaluation"]["eigs"][:2] for r in results]).reshape(n, 2)
    array["final_hessian_asymmetry"] = np.asarray([r["evaluation"]["hessian_asymmetry"] for r in results])
    array["budget_eigenvalues"] = np.asarray([[b["eigs"][:2] for b in r["budget_results"]]
                                              for r in results]).reshape(n, len(budgets), 2)
    array["budget_positions"] = np.asarray([[b["x"] for b in r["budget_results"]] for r in results]).reshape(n, len(budgets), 3)
    array["budget_outcome"] = (classify(array["budget_positions"].reshape(-1, 3), geometry)[0]
                               .reshape(n, len(budgets)) if n else np.empty((0, len(budgets)), dtype=int))
    np.savez_compressed(path, **array)
    return array


def make_report(data):
    config = json.loads(str(data["config_json"]))
    lines = ["LiC physics saddle-search benchmark", "=" * 110,
             f"Pretrained UMA: {config['model']}; task={config['task_name']}; FD displacement={config['fd_eps']} A",
             f"Starts: {len(data['starts'])}; checkpoint: {config['ckpt'] or 'not used'}",
             "Catalog true hit: distance <0.10 A; false region: not a true hit and distance <0.30 A.",
             f"Physical success: free fmax <= {config['fmax']} eV/A AND lambda1 < -{config['eigenvalue_tol']}"
             f" AND lambda2 > {config['eigenvalue_tol']} eV/A^2.",
             "Rows count independent trials; two-flow repeats every source with seeded orientations.",
             "No ground-truth selection among restarts. Index1 alone does not imply force convergence.",
             "Search PES calls count energy+force geometries, including FD probes; cached reads cost zero.",
             "Verification of finals/budget checkpoints is EXTRA, outside search budgets and wall times.",
             "SaddleFlow uses K learned velocity evaluations; architecture-conditioning PES queries count when enabled.",
             "SaddleFlow wall time is batched amortized time per source; physics searches are serial.", "",
             "Final outcomes (all populations)",
             "method           N  true%  false%  other%  stat%  index1%  phys%  med fmax  med lambda1  med lambda2  med PES  med sec",
             "-" * 124]
    methods = list(dict.fromkeys(data["method"].tolist()))
    for method in methods:
        use = data["method"] == method
        outcome = data["outcome"][use]
        eigs = data["final_eigenvalues"][use]
        lines.append(f"{method:14s} {use.sum():4d} {100*np.mean(outcome==0):6.1f}"
                     f" {100*np.mean(outcome==1):7.1f} {100*np.mean(outcome==2):7.1f}"
                     f" {100*np.mean(data['stationary'][use]):6.1f} {100*np.mean(data['index_one'][use]):8.1f}"
                     f" {100*np.mean(data['physical_success'][use]):6.1f} {np.median(data['force_max'][use]):9.5f}"
                     f" {np.median(eigs[:,0]):12.5f} {np.median(eigs[:,1]):12.5f}"
                     f" {np.median(data['force_calls'][use]):8.1f} {np.median(data['wall_seconds'][use]):8.2f}")
    lines += ["", "Final outcomes by start population", "method         population       N   true%  false%  phys%  median fmax", "-" * 78]
    for group in np.unique(data["source_population"]):
        for method in methods:
            use = (data["method"] == method) & (data["source_population"][data["source_index"]] == group)
            if use.any():
                lines.append(f"{method:14s} {group:14s} {use.sum():4d}"
                             f" {100*np.mean(data['outcome'][use]==0):7.1f}"
                             f" {100*np.mean(data['outcome'][use]==1):7.1f}"
                             f" {100*np.mean(data['physical_success'][use]):6.1f}"
                             f" {np.median(data['force_max'][use]):12.6f}")
    lines += ["", "Anytime checkpoints: last usable translation/accepted Sella state at or below the query budget",
              "method         budget    N   true%  false%  phys%  median fmax", "-" * 76]
    for j, budget in enumerate(data["force_budgets"]):
        for method in methods:
            use = data["method"] == method
            lines.append(f"{method:14s} {budget:6d} {use.sum():4d}"
                         f" {100*np.mean(data['budget_outcome'][use,j]==0):7.1f}"
                         f" {100*np.mean(data['budget_outcome'][use,j]==1):7.1f}"
                         f" {100*np.mean(data['budget_physical_success'][use,j]):6.1f}"
                         f" {np.median(data['budget_force_max'][use,j]):12.6f}")
    lines += ["", "Status counts and total cost (including all restarts)"]
    for method in methods:
        use = data["method"] == method
        status, counts = np.unique(data["status"][use], return_counts=True)
        lines.append(f"{method}: " + ", ".join(f"{s}={c}" for s, c in zip(status, counts))
                     + f"; search PES={data['force_calls'][use].sum():.0f},"
                     f" verification PES={data['diagnostic_force_calls'][use].sum():.0f},"
                     f" search seconds={data['wall_seconds'][use].sum():.1f},"
                     f" verification seconds={data['diagnostic_seconds'][use].sum():.1f}")
    lines += ["", "Interpretation: catalog hit, force convergence and index-one certification are distinct.",
              "A false-region endpoint at the iteration/query limit is NOT a fixed point of MMF.",
              "Low-force minima, soft/degenerate modes and numerical failures are not certified saddles.",
              "Check FD displacement sensitivity before drawing curvature conclusions."]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    methods = args.methods.split(",")
    if not methods or not set(methods) <= {"sella", "oracle", "two_flow"} or len(set(methods)) != len(methods):
        raise ValueError("Methods must be sella,oracle,two_flow (SaddleFlow is added with --ckpt).")
    if args.restarts < 1 or args.log_every_steps < 1 or args.K < 1 or args.batch_size < 1:
        raise ValueError("Restart, logging, K and batch sizes must be positive.")
    if args.ckpt:
        checkpoint = Path(args.ckpt).resolve()
        config = json.loads((checkpoint.parent / "config.json").read_text())
        extras = config.get("extras", {})
        if int(extras.get("delta_endpoint_channels", 0)) != 0 or float(extras.get("ts_denoise_sigma", 0)) <= 0:
            raise ValueError("The learned comparator requires an unconditioned LiC TS-denoising checkpoint.")
        if not (checkpoint / "ema.pt").is_file():
            raise FileNotFoundError(f"Expected EMA weights at {checkpoint / 'ema.pt'}.")
    if "sella" in methods:
        import importlib.util
        if importlib.util.find_spec("sella") is None:
            raise ImportError("Install Sella first: CC=gcc CXX=g++ pip install 'saddleflow[eval]'")
    budgets = sorted(set(int(b) for b in args.force_budgets.split(",")))
    if not budgets or min(budgets) < 1 or max(budgets) > args.max_force_calls:
        raise ValueError("Force budgets must be positive and <= max-force-calls.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "saddle_search.npz"
    if archive_path.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {archive_path}.")
    reference, geometry = load_geometry(args.train_traj, args.test_traj)
    sources = generate_starts(args, geometry)
    if not len(sources["starts"]):
        raise ValueError("No starting geometries were selected.")
    print(f"[setup] {len(sources['starts'])} identical starts; {len(geometry['train_saddles'])} train"
          f" + {len(geometry['test_saddles'])} test saddles; {len(geometry['false_bridges'])} false bridges", flush=True)
    print(f"[setup] methods={methods}; two-flow restarts={args.restarts}; budgets={budgets}", flush=True)
    print("[UMA] loading pretrained potential (model setup time is not search time)", flush=True)
    calculator, batch_evaluate = make_uma_calculator(args.model, args.device, args.task_name)
    print("[UMA] warming up scalar and batched force evaluations before timed searches", flush=True)
    warm_pes = PESBackend(reference, calculator, batch_evaluate)
    warm_x = warm_pes.dofs.pack(reference.positions)
    warm_pes.forces(warm_x)
    warm_pes.forces_many([warm_x, warm_x])
    print(f"[UMA] ready; {warm_pes.force_calls} setup-only PES evaluations (excluded from search cost)", flush=True)
    results = []

    def save_progress():
        data = save_archive(archive_path, args, geometry, sources, results, budgets)
        (out / "report.txt").write_text(make_report(data))

    run_physics_trials(args, reference, geometry, sources, calculator, batch_evaluate, methods, budgets, results, save_progress)
    if args.ckpt:
        run_saddleflow_trials(args, reference, sources, calculator, batch_evaluate, budgets, results, save_progress)
    save_progress()
    with np.load(archive_path, allow_pickle=False) as archive:
        print("\n" + make_report(archive), flush=True)
    print(f"[save] numerical results: {archive_path}; report: {out / 'report.txt'}", flush=True)
    if not args.no_plot:
        try:
            try:
                from plot_saddle_search import render_pdf
            except ModuleNotFoundError:
                from examples.LiC.rollout_eval.plot_saddle_search import render_pdf
            render_pdf(archive_path, out / "saddle_search.pdf")
        except Exception as exc:
            print(f"[plot] failed: {exc}. Numerical results and report are already saved.", flush=True)
    print(f"[done] replot without sampling: python examples/LiC/rollout_eval/plot_saddle_search.py {archive_path}", flush=True)


if __name__ == "__main__":
    main()
