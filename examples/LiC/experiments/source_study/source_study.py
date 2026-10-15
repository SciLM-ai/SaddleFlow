"""Michael: nested 12/48-saddle source-memory study; no PES-guided sampling.

prepare: freeze the split once. compare: evaluate matched EMA checkpoints.
All teacher/parent information is evaluation metadata, never a student input.
"""

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write

try:
    from saddle_search_data import file_hash, read_npz
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.saddle_search_data import file_hash, read_npz


METHODS = ("n12_uncond", "n12_source", "n48_uncond", "n48_source")
HERE = Path(__file__).resolve().parents[2]  # examples/LiC
DEFAULT_ROOT = "/scratch/gqs7zr/saddleflow/lic_source_study_seed42"


def load_split(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"split.json").read_text())
    for name, checksum in manifest["files"].items():
        if file_hash(directory/name) != checksum:
            raise ValueError(f"Michael: source-study split file changed: {name}")
    return manifest, read_npz(directory/"split.npz")


def prepare(args):
    directory = Path(args.study_dir).resolve()/"split"
    inputs = {str(Path(p).resolve()): file_hash(p) for p in [args.train_traj, args.test_traj]}
    if (directory/"split.json").exists():
        manifest, _ = load_split(directory)
        if manifest["seed"] != args.split_seed or manifest["inputs"] != inputs:
            raise ValueError("Michael: split already exists with different seed/data; use a new study directory")
        print(f"[split] Michael: reusing immutable split {directory}")
        return
    train, test = read(args.train_traj, index=":"), read(args.test_traj, index=":")
    if len(train) != 36 or len(test) != 501:
        raise ValueError("Michael: this study expects the original 12 train + 167 test triplets")
    frames = train+test
    extra = np.sort(np.random.default_rng(args.split_seed).choice(np.arange(12, 179), 36, replace=False))
    small = np.arange(12); large = np.r_[small, extra]
    heldout = np.setdiff1d(np.arange(179), large)
    directory.mkdir(parents=True, exist_ok=True)
    groups = {"train12": small, "train48": large, "heldout131": heldout,
              "catalog_rest": np.arange(12, 179), "catalog": np.arange(179)}
    for name, indices in groups.items():
        write(directory/f"{name}.traj", [frames[3*int(i)+j] for i in indices for j in range(3)])
    np.savez_compressed(directory/"split.npz", train12=small, train48=large,
                        extra36=extra, heldout131=heldout, seed=np.asarray(args.split_seed))
    names = [f"{name}.traj" for name in groups]+["split.npz"]
    manifest = dict(seed=args.split_seed, inputs=inputs, train12=small.tolist(), train48=large.tolist(),
                    extra36=extra.tolist(), heldout131=heldout.tolist(),
                    files={name: file_hash(directory/name) for name in names})
    (directory/"split.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"[split] Michael: {directory}: original 12, nested 48, fixed held-out 131; seed={args.split_seed}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", nargs="?", choices=["prepare", "compare"], default="compare")
    p.add_argument("--study-dir", default=DEFAULT_ROOT)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-traj", default=str(HERE/"train_set.traj"))
    p.add_argument("--test-traj", default=str(HERE/"test_set.traj"))
    p.add_argument("--models", default=",".join(METHODS))
    p.add_argument("--checkpoint", default="checkpoint_final", help="Same named checkpoint for every arm; never auto-select unequal steps")
    p.add_argument("--out-dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1, help="Evaluation-only RNG; independent of split/training seeds")
    p.add_argument("--samples-per-site", type=int, default=1)
    p.add_argument("--false-noise-sigma", type=float, default=.05)
    p.add_argument("--true-noise-sigma", type=float, default=.25)
    p.add_argument("--radii", default="0.25,0.43,0.6,0.8,1.0")
    p.add_argument("--directions-per-radius", type=int, default=4)
    p.add_argument("--coverage-draws", type=int, default=8)
    p.add_argument("--coverage-noise-sigma", type=float, default=.15)
    p.add_argument("--minimum-tol", type=float, default=.05)
    p.add_argument("--populations", default="all", help="Comma-separated groups or all; false_exact gives the old ten-bridge test")
    p.add_argument("--limit-starts", type=int)
    p.add_argument("--fmax", type=float, default=.005)
    p.add_argument("--eigenvalue-tol", type=float, default=.01)
    p.add_argument("--fd-eps", type=float, default=.002)
    p.add_argument("--geometry-only", "--no-uma", action="store_true", help="Skip post-hoc certificates, not reported as physical failures")
    p.add_argument("--probe-times", default="0,0.25,0.5,0.75")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-only", help="CPU replot from source_study.npz; loads no model or oracle")
    return p.parse_args()


def evaluation_sources(args, geometry, split):
    """Generate ALL starts before population/subset filtering, independent of arm."""
    all_saddles = np.vstack([geometry["train_saddles"], geometry["test_saddles"]])
    rng = np.random.default_rng(args.seed)
    points, groups, parents, radii, minimum_ids = [], [], [], [], []
    def add(x, group, parent=-1, radius=np.nan, minimum=-1):
        points.append(x); groups.append(group); parents.append(parent); radii.append(radius); minimum_ids.append(minimum)
    for point in geometry["false_bridges"]:
        add(point.copy(), "false_exact")
        for _ in range(args.samples_per_site):
            add(point+rng.normal(0, args.false_noise_sigma, 3), "false_noise")
    for name in ["train12", "extra36", "heldout131"]:
        for i in split[name]:
            for _ in range(args.samples_per_site):
                add(all_saddles[i]+rng.normal(0, args.true_noise_sigma, 3), f"true_noise_{name}", int(i))
    for i in split["train12"]:
        for radius in map(float, args.radii.split(",")):
            for _ in range(args.directions_per_radius):
                direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
                add(all_saddles[i]+radius*direction, "shells", int(i), radius)
    # Catalog connectivity from existing R/S/P labels, NOT inferred from solver outcomes.
    frames = read(Path(args.study_dir)/"split/catalog.traj", index=":")
    minima = []; edges = []
    for saddle_id in range(179):
        endpoints = []
        for j in [0, 2]:
            point = frames[3*saddle_id+j].positions[geometry["li_index"]]
            if minima:
                _, distances = find_mic(np.asarray(minima)-point, geometry["cell"], geometry["pbc"])
                nearest = int(np.argmin(distances))
            if not minima or distances[nearest] >= args.minimum_tol:
                nearest = len(minima); minima.append(point.copy())
            endpoints.append(nearest)
        edges.append(endpoints)
    for i, point in enumerate(minima):
        for _ in range(args.coverage_draws):
            add(point+rng.normal(0, args.coverage_noise_sigma, 3), "connected", minimum=i)
    sources = dict(starts=np.asarray(points), population=np.asarray(groups), parent=np.asarray(parents),
                   radius=np.asarray(radii), minimum_index=np.asarray(minimum_ids))
    use = np.arange(len(points))
    if args.populations != "all":
        wanted = set(args.populations.split(","))
        if not wanted <= set(groups): raise ValueError(f"Michael: unknown/empty populations {wanted-set(groups)}")
        use = use[np.isin(sources["population"], list(wanted))]
    if args.limit_starts:
        use = np.sort(rng.choice(use, min(args.limit_starts, len(use)), replace=False))
    return {key: value[use] for key, value in sources.items()}, np.asarray(minima), np.asarray(edges)


def report(data):
    config = json.loads(str(data["config_json"]))
    lines = ["Michael: 12/48-saddle x source-memory study (EMA)", "="*125,
        f"Checkpoint={config['checkpoint']}; equal update={config['checkpoint_update']}; K={config['K']}; split seed={config['split_seed']}",
        "Original 12 nested in 48; fixed holdout=131. Extra36 are NOT common held-out saddles.",
        "Identical evaluation starts; source-conditioned students retain actual x0, never the target/noise vector.",
        "No online PES queries during sampling. UMA-derived student network executes normally.",
        "Catalog hit <.10 A; false region <.30 A with true precedence. Parent is diagnostic, not a unique nearest-saddle claim.",
        f"Post-hoc physical certificate: fmax<={config['fmax']}, lambda1<-{config['eigenvalue_tol']}, lambda2>{config['eigenvalue_tol']}; NaN means skipped.",
        "Connectivity means stored R/S/P catalog adjacency, NOT independent downhill verification.",
        "Noise draws share centers; coverage is conditional on the sampled starts. Partial models have smaller N.", "",
        "method        population                  N   true%  false%   phys%  held131% parent% unique_true  med fmax", "-"*125]
    methods = list(dict.fromkeys(data["method"].tolist()))
    population = data["source_population"][data["source_index"]]
    for method in methods:
        for group in ["all"]+list(dict.fromkeys(population.tolist())):
            use = (data["method"] == method) & ((population == group) if group != "all" else True)
            if not use.any(): continue
            rows = data["source_index"][use]; hit = data["outcome"][use] == 0; ids = data["nearest_true_index"][use]
            owned = data["source_parent"][rows] >= 0
            parent_hit = np.mean(hit[owned] & (ids[owned] == data["source_parent"][rows][owned])) if owned.any() else np.nan
            lines.append(f"{method:13s} {group:25s} {use.sum():5d} {100*hit.mean():7.2f}"
                f" {100*np.mean(data['outcome'][use]==1):7.2f} {100*np.mean(data['physical_success'][use]):7.2f}"
                f" {100*np.mean(hit & np.isin(ids,data['heldout131'])):8.2f} {100*parent_hit:7.2f}"
                f" {len(np.unique(ids[hit])):11d} {np.median(data['force_max'][use]):9.5f}")
    lines += ["", "Robustness shells: identical sources around the common 12 training saddles",
              "method        radius(A)     N   true%  false%   phys%"]
    radius = data["source_radius"][data["source_index"]]
    for method in methods:
        for r in np.unique(radius[population == "shells"]):
            use = (data["method"] == method) & (population == "shells") & (radius == r)
            if use.any():
                lines.append(f"{method:13s} {r:9.3f} {use.sum():5d}"
                    f" {100*np.mean(data['outcome'][use]==0):7.2f}"
                    f" {100*np.mean(data['outcome'][use]==1):7.2f}"
                    f" {100*np.mean(data['physical_success'][use]):7.2f}")
    lines += ["", "Physical diagnostics and costs (post-hoc certification excluded from sampling time/PES)",
              "method             N   index1%  med lambda1  med lambda2  sampling_s/source  cert_PES/source"]
    for method in methods:
        use = data["method"] == method
        eig = np.median(data["final_eigenvalues"][use], axis=0)
        lines.append(f"{method:13s} {use.sum():6d} {100*np.mean(data['index_one'][use]):9.2f}"
            f" {eig[0]:12.5f} {eig[1]:12.5f} {np.mean(data['wall_seconds'][use]):18.5f}"
            f" {np.mean(data['diagnostic_force_calls'][use]):16.2f}")
    lines += ["", "Catalog-connected saddle coverage (macro over starting minima represented in this evaluation)",
              "method         minima  mean_coverage%  connected_hit%  reached_edges/available_edges"]
    edges = data["catalog_edges"]
    for method in methods:
        use = (data["method"] == method) & (population == "connected")
        rows = data["source_index"][use]; initial = data["source_minimum_index"][rows]
        found = data["nearest_true_index"][use]; hit = data["outcome"][use] == 0
        fractions = []; reached = total = valid = 0
        for minimum in np.unique(initial):
            expected = np.flatnonzero(np.any(edges == minimum, axis=1))
            observed = np.unique(found[(initial == minimum) & hit])
            matched = np.intersect1d(expected, observed)
            reached += len(matched); total += len(expected); fractions.append(len(matched)/len(expected))
            valid += np.sum(hit & (initial == minimum) & np.isin(found, expected))
        if fractions:
            lines.append(f"{method:13s} {len(fractions):7d} {100*np.mean(fractions):15.2f}"
                         f" {100*valid/use.sum():15.2f} {reached}/{total}")
    lines += ["", "Fixed teacher-forced FM probes: free-Li VECTOR RMSE (A/FM-time). Interior loss can exploit (x_t-x0)/t.",
              "method        population                  t      N   vector_RMSE"]
    for model, values in zip(data.get("probe_method", []), data.get("probe_mse", [])):
        valid = data["source_parent"] >= 0
        for group in np.unique(data["source_population"][valid]):
            mask = data["source_population"][valid] == group
            for t, error in zip(data["probe_times"], values):
                lines.append(f"{model:13s} {group:25s} {t:5.2f} {mask.sum():6d} {np.sqrt(error[mask].mean()):13.6f}")
    return "\n".join(lines)+"\n"


def compare(args):
    import torch
    from saddleflow.data import atoms_to_sample_dict
    from saddleflow.data.transforms import wrap_positions
    from saddleflow.classical_saddle_search import make_uma_calculator
    try:
        from saddle_search_data import load_geometry
        from compare_saddle_search import run_saddleflow_trials, save_archive
        from robustness_shells import nearest_labels, make_full_structures
        from force_guided_sampling import model_velocity
    except ModuleNotFoundError:
        from examples.LiC.rollout_eval.saddle_search_data import load_geometry
        from examples.LiC.rollout_eval.compare_saddle_search import run_saddleflow_trials, save_archive
        from examples.LiC.rollout_eval.robustness_shells import nearest_labels, make_full_structures
        from examples.LiC.rollout_eval.force_guided_sampling import model_velocity
    root = Path(args.study_dir).resolve(); manifest, split = load_split(root/"split")
    methods = args.models.split(",")
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("Michael: select n12_uncond,n12_source,n48_uncond,n48_source")
    if args.K < 1 or args.batch_size < 1 or args.samples_per_site < 1 or args.coverage_draws < 0:
        raise ValueError("Michael: invalid sampling counts")
    configs = [json.loads((root/m/"config.json").read_text()) for m in methods]
    shared = ["batch_size", "learning_rate", "weight_decay", "warmup_steps", "min_lr_ratio", "grad_clip_norm", "ema_decay", "mixed_precision", "seed", "max_steps", "init_weights", "num_workers"]
    architecture = ["backbone", "head_depth", "attn_layers", "attn_heads", "unfreeze_uma_all", "early_time_film", "early_time_film_blocks", "uma_lr", "ts_denoise_sigma", "source_channels"]
    updates = []
    args.checkpoints = {}
    for method, config in zip(methods, configs):
        ex = config["extras"]
        if ex.get("study_split_sha256") != file_hash(root/"split/split.json"):
            raise ValueError("Michael: checkpoint uses a different split")
        if ex["source_conditioning"] != method.endswith("source") or ex["train_saddle_count"] != int(method[1:3]):
            raise ValueError("Michael: model directory does not match its study arm")
        if any(config[k] != configs[0][k] for k in shared) or any(ex[k] != configs[0]["extras"][k] for k in architecture):
            raise ValueError("Michael: model settings/budgets are not matched")
        if ex.get("inject_force") or ex.get("curvature_conditioning") or ex.get("force_constraint_enabled"):
            raise ValueError("Michael: this study requires plain FM and no oracle inputs")
        checkpoint = root/method/args.checkpoint
        updates.append(json.loads((checkpoint/"meta.json").read_text())["global_step"])
        args.checkpoints[method] = str(checkpoint)
    if len(set(updates)) != 1: raise ValueError("Michael: compare checkpoints at identical optimizer updates")
    args.checkpoint_update = updates[0]; args.split_seed = manifest["seed"]
    args.train_traj = str(root/"split/train12.traj"); args.test_traj = str(root/"split/catalog_rest.traj")
    reference, geometry = load_geometry(args.train_traj, args.test_traj)
    sources, minima, edges = evaluation_sources(args, geometry, split)
    if not len(sources["starts"]): raise ValueError("Michael: no evaluation starts selected")
    args.model = configs[0]["extras"]["backbone"]; args.mode_backend = "dense"
    args.out_dir = args.out_dir or str(root/f"comparison_{args.checkpoint}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    archive = out/"source_study.npz"
    args.step_budgets = None
    calculator, batch = (None, None) if args.geometry_only else make_uma_calculator(args.model, args.device, "omat")
    results = []; probes = []; probe_methods = []
    probe_times = np.asarray(list(map(float, args.probe_times.split(","))))
    if np.any((probe_times < 0) | (probe_times >= 1)): raise ValueError("Michael: probe times must be in [0,1)")
    all_saddles = np.vstack([geometry["train_saddles"], geometry["test_saddles"]])
    def save():
        temporary = out/"source_study.partial.npz"
        data = save_archive(temporary, args, geometry, sources, results, [0])
        _, ids = nearest_labels(data["final"], all_saddles, geometry["cell"])
        data.update(nearest_true_index=ids, source_minimum_index=sources["minimum_index"], minima=minima,
                    catalog_edges=edges, probe_method=np.asarray(probe_methods, dtype="U16"),
                    probe_times=probe_times, probe_mse=np.asarray(probes), **split)
        np.savez_compressed(temporary, **data); temporary.replace(archive)
        text = report(data); (out/"report.txt").write_text(text)
        return data, text
    sample = atoms_to_sample_dict(reference)
    known = sources["parent"] >= 0
    initial = sources["starts"][known]
    delta = np.empty_like(initial)
    if known.any():
        delta, _ = find_mic(all_saddles[sources["parent"][known]]-initial, geometry["cell"], geometry["pbc"])
    for method in methods:
        args.ckpt = args.checkpoints[method]; first_result = len(results)
        def progress():
            for row in results[first_result:]: row["method"] = method
            save()
        backbone, attention, head = run_saddleflow_trials(args, reference, sources, calculator, batch, [0], results, progress)
        error = []
        for t in probe_times:
            squared = []
            for first in range(0, len(initial), args.batch_size):
                stop = first+args.batch_size
                x0 = make_full_structures(torch.as_tensor(geometry["positions"], dtype=torch.float32), initial[first:stop])
                xt = make_full_structures(torch.as_tensor(geometry["positions"], dtype=torch.float32), initial[first:stop]+t*delta[first:stop])
                kwargs = dict(source_positions=x0) if head.source_conditioning else {}
                v = model_velocity(wrap_positions(xt, sample["cell"]), float(t), sample, backbone, attention, head, args.device, **kwargs)
                squared.extend(np.sum((v[:, geometry["li_index"]].numpy()-delta[first:stop])**2, axis=-1))
            error.append(squared)
        probes.append(error); probe_methods.append(method)
        del backbone, attention, head
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _, text = save()
    print(text, flush=True)
    if not args.no_plot: plot(archive)
    print(f"[done] Michael: CPU replot: python examples/LiC/experiments/source_study/source_study.py --plot-only {archive}", flush=True)


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    data = read_npz(path); methods = list(dict.fromkeys(data["method"].tolist()))
    with PdfPages(Path(path).with_suffix(".pdf")) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, name in zip(axes, ["all", "false_exact"]):
            population = data["source_population"][data["source_index"]]
            masks = [(data["method"] == m) & ((population == name) if name != "all" else True) for m in methods]
            for j, (label, code) in enumerate([("True", 0), ("False region", 1)]):
                rates = [100*np.mean(data["outcome"][mask] == code) if mask.any() else np.nan for mask in masks]
                ax.bar(np.arange(len(methods))+.3*j, rates, .3, label=label)
            ax.set(xticks=np.arange(len(methods))+.15, xticklabels=methods, ylim=(0, 105), title=name, ylabel="Percent")
            ax.tick_params(axis="x", rotation=30); ax.legend()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        saddles = np.vstack([data["train_saddles"], data["test_saddles"]])
        for ax, method in zip(axes.flat, methods):
            carbon = data["positions"][data["carbon_indices"]]
            ax.scatter(carbon[:, 0], carbon[:, 1], s=3, color=".7")
            for indices, color, label in [(data["train12"], "green", "Common 12"), (data["extra36"], "orange", "Extra 36"), (data["heldout131"], "black", "Held-out 131")]:
                ax.scatter(saddles[indices, 0], saddles[indices, 1], s=12, facecolors="none", edgecolors=color, label=label)
            use = np.flatnonzero(data["method"] == method)
            for row in use[::max(1, len(use)//120)]:
                points = data["path"][row, :data["trace_length"][row], :2]
                if len(points)>1 and np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 2:
                    ax.plot(points[:, 0], points[:, 1], color="skyblue", alpha=.35, lw=.6)
                endpoint = data["final"][row]
                ax.scatter(endpoint[0], endpoint[1], s=7, color=["green", "red", "gray"][int(data["outcome"][row])])
            ax.set_title(method); ax.set_aspect("equal"); ax.legend(fontsize=7)
        for ax in axes.flat[len(methods):]: ax.set_visible(False)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] Michael: {Path(path).with_suffix('.pdf')}")


def main():
    args = parse_args()
    if args.plot_only:
        plot(args.plot_only)
    elif args.stage == "prepare":
        prepare(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
