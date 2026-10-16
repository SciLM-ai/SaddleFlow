"""Map how robust a trained LiC TS-denoiser is around its training saddles.

This is a *robustness/support diagnostic*, not a new training procedure and
not test-time adaptation.  It deliberately creates every perturbed state from
one of the 12 training saddle labels.  The 167 held-out saddle labels and the
ten known saddle-less C--C bridge sites are never used to construct a start;
they only classify where a rollout landed.

For a parent training saddle S_i, a fixed-radius shell point is

    x_t + delta,      ||delta_Li|| = r,

where x_t is either a point on LiC's prescribed TS-denoising bridge or a
state actually visited by the baseline sampler.  The ordinary learned flow is
then resumed from its original Euler step t=k/K using the original dt=1/K.
The resulting endpoint is classified as:

* parent saddle: within 0.10 A of S_i;
* other true saddle: within 0.10 A of another known saddle;
* known false bridge: within 0.30 A of one of the ten saddle-less bridges;
* other failure: none of the above.

The result estimates a recovery tube radius rho_i(t): how far the state can
move away from a known training transition before the present vector field no
longer returns to that transition's saddle.  A later recovery-training
experiment may use failures found here, whose target ownership is justified by
their training parent.  This script itself never uses those failures to train.

Examples, from the repository root:

    # Practical first pass: a controlled fixed-radius source-noise sweep
    # around the 12 training targets.
    CUDA_VISIBLE_DEVICES=0 python examples/LiC/rollout_eval/robustness_shells.py \
        --ckpt /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final

    # Probe actual learned states at four ordinary sampler times.  The source
    # noise now matches the original TS-denoising training sigma.
    CUDA_VISIBLE_DEVICES=0 python examples/LiC/rollout_eval/robustness_shells.py \
        --ckpt /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final \
        --state-source rollout --bridge-noise-sigma 0.25 \
        --start-times 0,0.25,0.5,0.75 --directions-per-radius 24

The numerical archive is written by this script only.  Render it later with
``plot_robustness_shells.py``; plotting is intentionally separate so changing
a figure never requires repeating the GPU sampling experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from saddleflow.constr_gen import force_rms_and_max
from saddleflow.data import atoms_to_sample_dict
from saddleflow.data.transforms import wrap_positions

# Reuse the exact checkpoint reconstruction, frozen-UMA interface, LiC label
# discovery, and standard velocity forward used by the existing LiC runners.
# This file adds only the controlled shell construction and its diagnostics.
try:  # Works for `python examples/LiC/rollout_eval/robustness_shells.py ...`.
    from force_guided_sampling import (  # noqa: E402
        FALSE_BASIN_RADIUS,
        LI,
        TRUE_HIT_RADIUS,
        find_false_midpoints,
        load_flow_model,
        make_force_function,
        model_velocity,
    )
    # The standalone plotting script imports plot_throws itself.  Keeping this
    # sampler free of Matplotlib makes archive generation independent of PDF
    # rendering failures and avoids importing plotting dependencies on GPU jobs.
except ModuleNotFoundError:  # Also permit `python -m examples.LiC.rollout_eval.robustness_shells` / imports.
    from examples.LiC.rollout_eval.force_guided_sampling import (  # noqa: E402
        FALSE_BASIN_RADIUS,
        LI,
        TRUE_HIT_RADIUS,
        find_false_midpoints,
        load_flow_model,
        make_force_function,
        model_velocity,
    )


OUTCOME_PARENT = 0
OUTCOME_OTHER_TRUE = 1
OUTCOME_FALSE_BRIDGE = 2
OUTCOME_OTHER_FAILURE = 3
OUTCOME_NAMES = np.asarray(
    ["parent_saddle", "other_true_saddle", "known_false_bridge", "other_failure"]
)


def parse_args() -> argparse.Namespace:
    """Parse options while keeping scientific choices visible at the CLI."""
    here = Path(__file__).resolve().parents[1]  # examples/LiC
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="LiC checkpoint_final directory")
    parser.add_argument("--train-traj", default=None)
    parser.add_argument("--test-traj", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--K", type=int, default=20, help="ordinary Euler step count")
    parser.add_argument("--batch-size", type=int, default=64)
    # The existing LiC force-guidance smoke test safely evaluated UMA in
    # batches of 64 on its GPU.  Keep that efficient default, while exposing
    # a lower value for smaller-memory devices.
    parser.add_argument("--force-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--state-source", choices=("bridge", "rollout"), default="bridge",
        help=(
            "Where the unperturbed x_t comes from: 'bridge' uses the prescribed "
            "TS-denoising line S+(1-t)*noise; 'rollout' uses a state actually "
            "visited by the ordinary learned sampler."
        ),
    )
    parser.add_argument(
        "--bridge-noise-sigma", type=float, default=0.0,
        help=(
            "A; Gaussian Li noise at t=0 before constructing bridge/rollout "
            "sources.  Keep the default 0 for the primary fixed-radius shell sweep "
            "x0=S+r*u; use 0.25 only when probing a random training-style bridge "
            "or an actual learned trajectory."
        ),
    )
    parser.add_argument(
        "--base-noise-samples", type=int, default=1,
        help="Independent base-noise draws per training saddle before shell directions.",
    )
    parser.add_argument(
        "--start-times", default="0",
        help=(
            "Comma-separated flow times on the Euler grid, e.g. 0,0.25,0.5,0.75. "
            "The sampler resumes from this exact original step."
        ),
    )
    parser.add_argument(
        "--radii", default="0,0.05,0.10,0.20,0.30,0.50,0.75,1.00",
        help="Comma-separated Li shell radii in A.",
    )
    parser.add_argument(
        "--directions-per-radius", type=int, default=48,
        help="Random shell directions for every positive radius.  Radius zero uses one point.",
    )
    parser.add_argument(
        "--parents", default=None,
        help="Optional zero-based comma-separated subset of the 12 training saddle indices.",
    )
    parser.add_argument(
        "--no-uma", action="store_true",
        help="Skip frozen-UMA endpoint force metrics to make a faster geometry-only pass.",
    )
    return parser.parse_args()


def parse_radii(text: str) -> np.ndarray:
    """Parse nonnegative, unique shell radii in a stable order."""
    try:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError("--radii must be comma-separated numbers") from error
    if not values or any(value < 0 for value in values):
        raise ValueError("--radii must contain at least one nonnegative radius")
    return np.asarray(sorted(set(values)), dtype=np.float32)


def parse_start_steps(text: str, K: int) -> list[int]:
    """Convert requested times to Euler-grid step indices without rounding silently."""
    try:
        times = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError("--start-times must be comma-separated numbers") from error
    if not times:
        raise ValueError("--start-times must contain at least one time")
    steps: list[int] = []
    for time in times:
        step = round(time * K)
        if not 0.0 <= time < 1.0 or abs(time - step / K) > 1e-8:
            raise ValueError(
                f"start time {time:g} is not a valid Euler-grid time for K={K}. "
                f"Use one of 0, 1/{K}, ..., {(K - 1)}/{K}."
            )
        steps.append(step)
    return sorted(set(steps))


def parse_parents(text: str | None, n_train: int) -> np.ndarray:
    """Return the requested zero-based training-parent indices."""
    if text is None:
        return np.arange(n_train, dtype=np.int64)
    try:
        parents = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    except ValueError as error:
        raise ValueError("--parents must be zero-based comma-separated integers") from error
    if not parents or any(parent < 0 or parent >= n_train for parent in parents):
        raise ValueError(f"--parents must lie between 0 and {n_train - 1}")
    return np.asarray(parents, dtype=np.int64)


def make_base_noises(
    n_parents: int,
    n_samples: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one Li-only source perturbation for each parent/noise replicate."""
    if sigma < 0 or n_samples <= 0:
        raise ValueError("bridge-noise-sigma must be nonnegative and base-noise-samples positive")
    return rng.normal(0.0, sigma, size=(n_parents, n_samples, 3)).astype(np.float32)


def unit_shell_directions(count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw isotropic 3D unit directions for the single mobile Li atom."""
    values = rng.normal(size=(count, 3)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)


def make_full_structures(reference_positions: torch.Tensor, li_positions: np.ndarray, li_index: int = LI) -> torch.Tensor:
    """Insert a Li-coordinate population into the fixed carbon-sheet template."""
    positions = reference_positions.unsqueeze(0).repeat(len(li_positions), 1, 1)
    positions[:, li_index] = torch.as_tensor(li_positions, dtype=positions.dtype)
    return positions


def wrap_li_coordinates(li_positions: np.ndarray, cell: torch.Tensor) -> np.ndarray:
    """Put Li-only coordinates in the same canonical cell image as sampler paths."""
    coordinate_tensor = torch.as_tensor(li_positions, dtype=torch.float32).unsqueeze(1)
    return wrap_positions(coordinate_tensor, cell)[:, 0].numpy()


def rollout_from_step(
    starts: torch.Tensor,
    start_step: int,
    *,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    device: str,
    K: int,
    source_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resume the ordinary Euler sampler from `start_step` using the original dt.

    This intentionally does *not* rescale the remaining interval or use a
    force-guidance term.  The returned Li paths are NaN before `start_step`,
    making the stored archive explicit about where each branch began.
    """
    x = wrap_positions(starts.clone(), sample["cell"])
    source_kwargs = {}
    if getattr(velocity_head, "source_conditioning", False):
        if source_positions is None:
            if start_step:
                raise ValueError("a source-conditioned restart at t>0 needs the ORIGINAL x0, not x_k")
            source_positions = starts.detach().clone()
        source_kwargs["source_positions"] = source_positions
    path = torch.full((len(x), K + 1, 3), torch.nan, dtype=x.dtype)
    path[:, start_step] = x[:, LI]
    dt = 1.0 / K
    for step in range(start_step, K):
        velocity = model_velocity(
            x, step / K, sample, backbone, attention, velocity_head, device, **source_kwargs
        )
        x = wrap_positions(x + dt * velocity, sample["cell"])
        path[:, step + 1] = x[:, LI]
    return x, path


def sample_points(points, reference_positions, *, sample, backbone, attention,
                  velocity_head, device, K, batch_size, start_step=0, li_index=LI, label="rollout"):
    """Shared LiC batching; keep the original clock and fixed scaffold."""
    reference=torch.as_tensor(reference_positions,dtype=torch.float32)
    try: from conditioning import conditioning_queries
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import conditioning_queries
    paths=[]
    for first in range(0,len(points),batch_size):
        before = conditioning_queries(velocity_head)
        full=make_full_structures(reference,points[first:first+batch_size],li_index)
        _,path=rollout_from_step(full,start_step,sample=sample,backbone=backbone,
            attention=attention,velocity_head=velocity_head,device=device,K=K)
        paths.append(path.numpy())
        print(f"[sample] {label} {min(first+batch_size,len(points))}/{len(points)}; "
              f"t={start_step/K:.2f}, remaining={K-start_step}; no PES guidance; "
              f"conditioning PES={conditioning_queries(velocity_head)-before}",flush=True)
    return np.concatenate(paths)


def baseline_source_paths(
    initial_positions: torch.Tensor,
    *,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    device: str,
    K: int,
    batch_size: int,
) -> np.ndarray:
    """Run ordinary paths once to provide actual learned-trajectory x_t sources."""
    paths: list[np.ndarray] = []
    for start in range(0, len(initial_positions), batch_size):
        batch = initial_positions[start:start + batch_size]
        _, li_path = rollout_from_step(
            batch, 0, sample=sample, backbone=backbone, attention=attention,
            velocity_head=velocity_head, device=device, K=K,
        )
        paths.append(li_path.numpy())
        print(
            f"[source rollout] {min(start + len(batch), len(initial_positions))}/"
            f"{len(initial_positions)} baseline paths",
            flush=True,
        )
    return np.concatenate(paths, axis=0)


def build_shell_population(
    parent_saddles: np.ndarray,
    parent_indices: np.ndarray,
    base_noises: np.ndarray,
    source_paths: np.ndarray | None,
    *,
    source_kind: str,
    start_steps: list[int],
    radii: np.ndarray,
    directions_per_radius: int,
    K: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Construct all training-owned shell starts and their provenance columns.

    `source_paths` is used only for state-source=rollout.  Evaluation labels
    are intentionally absent here: this function knows only the selected
    training parents and the random perturbations drawn around them.
    """
    reference_li: list[np.ndarray] = []
    start_li: list[np.ndarray] = []
    parent_column: list[int] = []
    noise_column: list[int] = []
    radius_column: list[float] = []
    direction_column: list[int] = []
    step_column: list[int] = []
    base_noise_column: list[np.ndarray] = []

    n_noise = base_noises.shape[1]
    for parent in parent_indices:
        for noise_index in range(n_noise):
            noise = base_noises[parent, noise_index]
            for step in start_steps:
                if source_kind == "bridge":
                    # x_t = S + (1-t) epsilon for the current TS-denoising line.
                    source = parent_saddles[parent] + (1.0 - step / K) * noise
                else:
                    assert source_paths is not None
                    source_row = parent * n_noise + noise_index
                    source = source_paths[source_row, step]

                for radius in radii:
                    n_directions = 1 if radius == 0 else directions_per_radius
                    directions = (
                        np.zeros((1, 3), dtype=np.float32)
                        if radius == 0
                        else unit_shell_directions(n_directions, rng)
                    )
                    for direction_index, direction in enumerate(directions):
                        reference_li.append(source)
                        start_li.append(source + radius * direction)
                        parent_column.append(int(parent))
                        noise_column.append(noise_index)
                        radius_column.append(float(radius))
                        direction_column.append(-1 if radius == 0 else direction_index)
                        step_column.append(step)
                        base_noise_column.append(noise)

    return {
        "reference_li": np.asarray(reference_li, dtype=np.float32),
        "starts_li": np.asarray(start_li, dtype=np.float32),
        "parent_index": np.asarray(parent_column, dtype=np.int64),
        "base_noise_index": np.asarray(noise_column, dtype=np.int64),
        "radius": np.asarray(radius_column, dtype=np.float32),
        "direction_index": np.asarray(direction_column, dtype=np.int64),
        "start_step": np.asarray(step_column, dtype=np.int64),
        "start_time": np.asarray(step_column, dtype=np.float32) / K,
        "base_noise": np.asarray(base_noise_column, dtype=np.float32),
    }


def sample_shell_population(
    population: dict[str, np.ndarray],
    reference_positions: torch.Tensor,
    *,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    device: str,
    K: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every shell state from its declared original Euler step to t=1."""
    n = len(population["starts_li"])
    finals = np.empty((n, 3), dtype=np.float32)
    paths = np.full((n, K + 1, 3), np.nan, dtype=np.float32)
    completed = 0

    for step in sorted(np.unique(population["start_step"])):
        rows_at_step = np.flatnonzero(population["start_step"] == step)
        for offset in range(0, len(rows_at_step), batch_size):
            rows = rows_at_step[offset:offset + batch_size]
            full_starts = make_full_structures(reference_positions, population["starts_li"][rows])
            endpoint, li_path = rollout_from_step(
                full_starts, int(step), sample=sample, backbone=backbone,
                attention=attention, velocity_head=velocity_head, device=device, K=K,
            )
            finals[rows] = endpoint[:, LI].numpy()
            paths[rows] = li_path.numpy()
            completed += len(rows)
            print(f"[sample] {completed}/{n} shell rollouts", flush=True)
    return finals, paths


def nearest_labels(
    points: np.ndarray,
    references: np.ndarray,
    cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest 3D LiC labels using the same in-plane PBC convention.

    The carbon sheet is periodic in x/y.  Its Li adsorption height remains an
    ordinary Cartesian coordinate, so only the 3x3 neighbouring in-plane
    images are considered, exactly as in the existing LiC metrics.
    """
    best_distance = np.full(len(points), np.inf)
    best_index = np.full(len(points), -1, dtype=np.int64)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            offset = i * cell[0] + j * cell[1]
            distance = np.linalg.norm(
                points[:, None, :] - (references[None, :, :] + offset), axis=2
            )
            nearest = distance.argmin(axis=1)
            values = distance[np.arange(len(points)), nearest]
            use = values < best_distance
            best_distance[use] = values[use]
            best_index[use] = nearest[use]
    return best_distance, best_index


def classify_endpoints(
    finals: np.ndarray,
    parent_index: np.ndarray,
    train_saddles: np.ndarray,
    test_saddles: np.ndarray,
    false_bridges: np.ndarray,
    cell: np.ndarray,
) -> dict[str, np.ndarray]:
    """Classify endpoints with fixed, predeclared LiC hit/basin radii."""
    true_saddles = np.vstack([train_saddles, test_saddles])
    d_parent = np.empty(len(finals), dtype=np.float32)
    for parent in np.unique(parent_index):
        rows = parent_index == parent
        distance, _ = nearest_labels(finals[rows], train_saddles[parent:parent + 1], cell)
        d_parent[rows] = distance
    d_true, nearest_true = nearest_labels(finals, true_saddles, cell)
    d_false, nearest_false = nearest_labels(finals, false_bridges, cell)

    outcome = np.full(len(finals), OUTCOME_OTHER_FAILURE, dtype=np.int8)
    is_parent = d_parent < TRUE_HIT_RADIUS
    is_any_true = d_true < TRUE_HIT_RADIUS
    is_false = d_false < FALSE_BASIN_RADIUS
    outcome[is_false] = OUTCOME_FALSE_BRIDGE
    outcome[is_any_true] = OUTCOME_OTHER_TRUE
    outcome[is_parent] = OUTCOME_PARENT
    return {
        "distance_to_parent": d_parent,
        "distance_to_any_true": d_true.astype(np.float32),
        "distance_to_false_bridge": d_false.astype(np.float32),
        "nearest_true_index": nearest_true,
        "nearest_false_index": nearest_false,
        "outcome_code": outcome,
    }


def evaluate_uma_forces(
    finals: np.ndarray,
    reference_positions: torch.Tensor,
    *,
    sample: dict,
    backbone_name: str,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate frozen UMA at endpoints, logging only the mobile Li force."""
    force_function = make_force_function(sample, backbone_name, device)
    mobile_mask = ~sample["fixed"]
    force_rms: list[np.ndarray] = []
    force_max: list[np.ndarray] = []
    for start in range(0, len(finals), batch_size):
        full_positions = make_full_structures(reference_positions, finals[start:start + batch_size])
        forces = force_function(full_positions)
        rms, maximum = force_rms_and_max(forces, mobile_mask)
        force_rms.append(rms.cpu().numpy())
        force_max.append(maximum.cpu().numpy())
        print(
            f"[UMA] {min(start + len(full_positions), len(finals))}/{len(finals)} endpoints",
            flush=True,
        )
    return np.concatenate(force_rms), np.concatenate(force_max)


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    """Compute a percentile while preserving the explicit no-UMA NaN case."""
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if len(finite) else float("nan")


def make_summary_rows(
    population: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    force_rms: np.ndarray,
) -> list[dict]:
    """Aggregate transparent probability and endpoint statistics by time/radius."""
    rows: list[dict] = []
    for step in sorted(np.unique(population["start_step"])):
        for radius in sorted(np.unique(population["radius"])):
            use = (population["start_step"] == step) & np.isclose(population["radius"], radius)
            outcomes = metrics["outcome_code"][use]
            rows.append({
                "step": int(step),
                "radius": float(radius),
                "n": int(use.sum()),
                "parent_pct": float(100 * np.mean(outcomes == OUTCOME_PARENT)),
                "other_true_pct": float(100 * np.mean(outcomes == OUTCOME_OTHER_TRUE)),
                "false_pct": float(100 * np.mean(outcomes == OUTCOME_FALSE_BRIDGE)),
                "other_pct": float(100 * np.mean(outcomes == OUTCOME_OTHER_FAILURE)),
                "median_d_parent": float(np.median(metrics["distance_to_parent"][use])),
                "median_d_any_true": float(np.median(metrics["distance_to_any_true"][use])),
                "median_force_rms": percentile_or_nan(force_rms[use], 50),
            })
    return rows


def write_report(
    output: Path,
    *,
    args: argparse.Namespace,
    n_train: int,
    n_selected_parents: int,
    n_test: int,
    n_false: int,
    n_shells: int,
    summary_rows: list[dict],
) -> str:
    """Create the stdout/report.txt table requested for the shell experiment."""
    lines = [
        "LiC training-saddle robustness shells",
        "=" * 66,
        f"checkpoint: {Path(args.ckpt).resolve()}",
        f"state source: {args.state_source}; K={args.K}; bridge noise sigma={args.bridge_noise_sigma:.3f} A",
        f"parents: {n_selected_parents} selected from {n_train} stored training saddles only; shell rollouts: {n_shells}",
        f"evaluation labels (never used to construct starts): {n_train} training + {n_test} held-out true saddles; {n_false} known false bridges",
        "",
        "Outcome definitions: parent / other true = endpoint within 0.10 A; false = endpoint within 0.30 A of a known saddle-less bridge.",
        "The shell radius is the Li displacement from the selected bridge or baseline-rollout source state.",
        "",
        "step    t   radius(A)    N   parent(%)  other_true(%)  false(%)  other(%)  med d_parent(A)  med d_any(A)  med F_Li(eV/A)",
        "-" * 126,
    ]
    for row in summary_rows:
        time = row["step"] / args.K
        force_text = f"{row['median_force_rms']:14.6f}" if np.isfinite(row["median_force_rms"]) else "           n/a"
        lines.append(
            f"{row['step']:4d}  {time:4.2f}  {row['radius']:9.3f}  {row['n']:3d}  "
            f"{row['parent_pct']:9.2f}  {row['other_true_pct']:13.2f}  "
            f"{row['false_pct']:8.2f}  {row['other_pct']:8.2f}  "
            f"{row['median_d_parent']:15.6f}  {row['median_d_any_true']:12.6f}  {force_text}"
        )
    lines += [
        "",
        "Interpretation: a fall in parent(%) as radius increases marks the empirical boundary of this checkpoint's recovery tube.",
        "A nonzero false(%) is particularly useful for later recovery-data mining because every such state retains the shown training-parent ownership.",
    ]
    text = "\n".join(lines) + "\n"
    output.write_text(text)
    return text


def main() -> None:
    args = parse_args()
    if args.K < 2:
        raise SystemExit("--K must be at least 2")
    if args.batch_size <= 0 or args.force_batch_size <= 0 or args.directions_per_radius <= 0:
        raise SystemExit("batch sizes and directions-per-radius must be positive")
    checkpoint = Path(args.ckpt)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    try: from conditioning import catalog_paths
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import catalog_paths
    extras = json.loads((checkpoint.parent/"config.json").read_text()).get("extras", {})
    args.train_traj, args.test_traj = catalog_paths(extras, args.train_traj, args.test_traj)
    output_dir = Path(args.out_dir) if args.out_dir else checkpoint.parent / "robustness_shells"
    output_dir.mkdir(parents=True, exist_ok=True)

    radii = parse_radii(args.radii)
    start_steps = parse_start_steps(args.start_times, args.K)
    rng = np.random.default_rng(args.seed)

    # This is the only start-data source.  The test file is read by
    # find_false_midpoints too, but only to assemble post-rollout evaluation
    # labels and to reconstruct the pre-existing false-bridge catalogue.
    reference, train_saddles, test_saddles, false_bridges, _ = find_false_midpoints(
        args.train_traj, args.test_traj
    )
    parents = parse_parents(args.parents, len(train_saddles))
    false_bridges = false_bridges.copy()
    false_bridges[:, 2] = float(np.mean(np.vstack([train_saddles, test_saddles])[:, 2]))
    sample = atoms_to_sample_dict(reference)
    reference_positions = torch.tensor(reference.get_positions(), dtype=torch.float32)
    base_noises = make_base_noises(
        len(train_saddles), args.base_noise_samples, args.bridge_noise_sigma, rng
    )

    config, backbone_name, backbone, attention, velocity_head = load_flow_model(checkpoint, args.device)
    try: from conditioning import conditioning_queries
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import conditioning_queries

    source_paths = None
    if args.state_source == "rollout":
        source_li = []
        for parent in range(len(train_saddles)):
            for noise_index in range(args.base_noise_samples):
                source_li.append(train_saddles[parent] + base_noises[parent, noise_index])
        source_starts = make_full_structures(reference_positions, np.asarray(source_li, dtype=np.float32))
        print(
            f"[setup] building {len(source_starts)} ordinary baseline paths for rollout-state sources",
            flush=True,
        )
        source_paths = baseline_source_paths(
            source_starts, sample=sample, backbone=backbone, attention=attention,
            velocity_head=velocity_head, device=args.device, K=args.K, batch_size=args.batch_size,
        )

    population = build_shell_population(
        train_saddles, parents, base_noises, source_paths,
        source_kind=args.state_source, start_steps=start_steps, radii=radii,
        directions_per_radius=args.directions_per_radius, K=args.K, rng=rng,
    )
    # Shells may cross a periodic cell edge.  Store the same canonical image
    # that the Euler sampler receives, so NPZ starts, plotted starts, and the
    # first recorded path point agree exactly.
    population["reference_li"] = wrap_li_coordinates(population["reference_li"], sample["cell"])
    population["starts_li"] = wrap_li_coordinates(population["starts_li"], sample["cell"])
    print(
        f"[setup] {len(population['starts_li'])} training-owned shell starts: "
        f"{len(parents)} parents x {args.base_noise_samples} base noise sample(s) x "
        f"{len(start_steps)} time(s); source={args.state_source}; radii={','.join(f'{r:g}' for r in radii)}",
        flush=True,
    )
    print(
        f"[setup] evaluation only: {len(train_saddles)} train + {len(test_saddles)} held-out true saddles; "
        f"{len(false_bridges)} known false bridges",
        flush=True,
    )

    finals, paths = sample_shell_population(
        population, reference_positions, sample=sample, backbone=backbone,
        attention=attention, velocity_head=velocity_head, device=args.device,
        K=args.K, batch_size=args.batch_size,
    )
    metrics = classify_endpoints(
        finals, population["parent_index"], train_saddles, test_saddles, false_bridges,
        np.asarray(reference.get_cell()),
    )

    if args.no_uma:
        force_rms = np.full(len(finals), np.nan, dtype=np.float32)
        force_max = np.full(len(finals), np.nan, dtype=np.float32)
        print("[UMA] skipped (--no-uma)", flush=True)
    else:
        force_rms, force_max = evaluate_uma_forces(
            finals, reference_positions, sample=sample, backbone_name=backbone_name,
            device=args.device, batch_size=args.force_batch_size,
        )
    summary_rows = make_summary_rows(population, metrics, force_rms)

    # The numerical archive contains all paths and is the hand-off point to
    # the separate CPU-only plotting script.
    npz_path = output_dir / "robustness_shells.npz"
    print(f"[save] writing numerical archive: {npz_path}", flush=True)
    np.savez_compressed(
        npz_path,
        starts=population["starts_li"],
        reference_states=population["reference_li"],
        finals=finals,
        paths=paths,
        parent_index=population["parent_index"],
        base_noise_index=population["base_noise_index"],
        base_noise=population["base_noise"],
        radius=population["radius"],
        direction_index=population["direction_index"],
        start_step=population["start_step"],
        start_time=population["start_time"],
        distance_to_parent=metrics["distance_to_parent"],
        distance_to_any_true=metrics["distance_to_any_true"],
        distance_to_false_bridge=metrics["distance_to_false_bridge"],
        nearest_true_index=metrics["nearest_true_index"],
        nearest_false_index=metrics["nearest_false_index"],
        outcome_code=metrics["outcome_code"],
        outcome_names=OUTCOME_NAMES,
        force_rms=force_rms,
        force_max=force_max,
        conditioning_pes_calls=np.asarray(conditioning_queries(velocity_head)),
        student_conditioning_json=np.asarray(json.dumps({key:config.get("extras",{}).get(key) for key in
            ["inject_force","force_field_channels","curvature_conditioning","curvature_fd_eps","physics_batch_size"]})),
        train_saddles=train_saddles,
        test_saddles=test_saddles,
        false_bridges=false_bridges,
        cell=np.asarray(reference.get_cell()),
        state_source=np.asarray(args.state_source),
        configuration_json=np.asarray(json.dumps(vars(args), sort_keys=True)),
    )

    report_path = output_dir / "report.txt"
    report = write_report(
        report_path, args=args, n_train=len(train_saddles), n_selected_parents=len(parents), n_test=len(test_saddles),
        n_false=len(false_bridges), n_shells=len(finals), summary_rows=summary_rows,
    )
    print("\n" + report, flush=True)

    print(
        f"[done] wrote {npz_path} and {report_path}\n"
        "[done] render without rerunning sampling:\n"
        f"  python examples/LiC/experiments/diagnostics/plot_robustness_shells.py {npz_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
