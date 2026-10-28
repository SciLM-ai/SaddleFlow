"""Benchmark generic UMA endpoint guidance on the LiC TS-denoiser.

The reusable, system-size-independent guidance term lives in
``saddleflow.constr_gen`` and is enabled on the shared
``saddleflow.flow.sample_saddles`` interface with ``force_guidance={...}``.
This file is intentionally LiC-specific only because it creates the 64x64
sheet starts and scores against LiC's 179 true and 10 false bridge labels.

This is the inference experiment for the LiC false-saddle problem.  It runs
the same starts twice:

By default this script runs the trained SaddleFlow velocity field alone.  Pass
``--force_guided_sampling`` to run the paired baseline-versus-guided ablation.
Only that explicit flag enables endpoint force guidance. A checkpoint trained
with force/curvature INPUTS still queries frozen UMA for those inputs at every
step, even when endpoint guidance is off.

At a guidance step, the current velocity gives an algebraic terminal estimate

    S_hat = x_t + (1 - t) * v_theta(x_t, t).

Frozen pretrained UMA evaluates that estimate.  If the mobile-atom force RMS
exceeds ``--force-tol``, a finite-difference H F direction is tried.  A small
endpoint displacement is accepted only if UMA confirms that it lowers force
RMS.  The accepted endpoint shift is converted back into a velocity correction
for the remaining Euler steps.

The current LiC checkpoint is unconditioned: there is no reactant/product pair
in this script.  The algebraic endpoint formula remains valid for its straight
TS-denoising flow.

Example from the repository root:

    CUDA_VISIBLE_DEVICES=0 python examples/LiC/rollout_eval/force_guided_sampling.py \
        --ckpt /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final \
        --grid 64 --K 20 --device cuda

The baseline run writes ``results.npz`` and ``report.txt`` under
``<checkpoint-parent>/force_guided_sampling``.  With
``--force_guided_sampling``, it additionally writes ``guidance_trace.json``
and stores the paired guided trajectories and UMA force metrics in
``results.npz``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.constr_gen import (
    estimate_hf_direction,
    force_rms_and_max,
    guide_terminal_estimates,
    uma_forces,
)
from saddleflow.data import atoms_to_sample_dict
from saddleflow.data.transforms import mic_displacement, wrap_positions
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import load_ema_weights, load_uma_backbone
from saddleflow.utils.forces import load_uma_force_head


N_CARBON = 126
LI = 126
TRUE_HIT_RADIUS = 0.10
FALSE_BASIN_RADIUS = 0.30
FALSE_MIDPOINT_CUTOFF = 0.50


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]  # examples/LiC
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="checkpoint_final directory")
    parser.add_argument("--train-traj", default=None, help="Saved study training catalog, otherwise original train_set.traj")
    parser.add_argument("--test-traj", default=None, help="Saved study catalog complement, otherwise original test_set.traj")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")

    # Starts mirror random_throw.py.  The modest random default is appropriate
    # for an initial ablation; --grid 64 reproduces the full sheet experiment.
    parser.add_argument("--n", type=int, default=256, help="random starts when --grid is zero")
    parser.add_argument("--grid", type=int, default=0, help="regular GRID x GRID starts; 0 = random")
    parser.add_argument("--z-jitter", type=float, default=0.30, help="A; random-start Li height noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--false_saddles_only", "--false-saddles-only",
        dest="false_saddles_only", action="store_true",
        help=(
            "Use exactly the ten known saddle-less C-C bridge midpoints as starts, "
            "at the common reference true-saddle height. Combine with "
            "--false-saddle-noise-sigma to make a local cloud around each bridge."
        ),
    )
    parser.add_argument(
        "--false-saddle-noise-sigma", type=float, default=0.0,
        help=(
            "A; isotropic Gaussian Li-coordinate noise around every false-bridge "
            "midpoint. Default 0 retains one exact start per bridge."
        ),
    )
    parser.add_argument(
        "--false-saddle-samples-per-bridge", type=int, default=1,
        help=(
            "Independent starts per false bridge. With sigma=0, values above one "
            "would duplicate exact starts, so use this with nonzero noise."
        ),
    )
    parser.add_argument("--K", type=int, default=20, help="Euler integration steps")
    parser.add_argument("--batch-size", type=int, default=64)

    # These are deliberately exposed rather than hidden scientific constants.
    # The defaults correspond to t = 0.50, 0.60, 0.75, 0.85 for K = 20.
    parser.add_argument(
        "--force_guided_sampling", "--force-guided-sampling",
        dest="force_guided_sampling", action="store_true",
        help="Opt in to UMA force guidance and paired baseline-versus-guided evaluation.",
    )
    parser.add_argument(
        "--diagnose_uma_direction", "--diagnose-uma-direction",
        dest="diagnose_uma_direction", action="store_true",
        help=(
            "At each guide step, measure whether the baseline endpoint's UMA H F "
            "direction points toward its nearest true saddle or away from its nearest false bridge."
        ),
    )
    parser.add_argument("--guide-steps", default="10,12,15,17")
    parser.add_argument("--force-tol", type=float, default=0.01, help="mobile force RMS, eV/A")
    parser.add_argument("--fd-eps", type=float, default=0.001, help="A; H F finite-difference step")
    parser.add_argument("--endpoint-step", type=float, default=0.02, help="A RMS; first correction trial")
    parser.add_argument("--max-endpoint-step", type=float, default=0.05, help="A RMS; correction cap")
    parser.add_argument("--alpha", type=float, default=0.5, help="endpoint-to-velocity correction weight")
    parser.add_argument("--max-backtracks", type=int, default=3)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    return parser.parse_args()


def parse_guide_steps(text: str, K: int) -> set[int]:
    """Parse a comma-separated step list and reject unstable t=1 guidance."""
    try:
        steps = {int(value.strip()) for value in text.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError("--guide-steps must be comma-separated integer steps.") from exc
    invalid = sorted(step for step in steps if step < 0 or step >= K - 1)
    if invalid:
        raise ValueError(
            f"Guide steps must satisfy 0 <= step < K-1 ({K - 1}); invalid: {invalid}. "
            "Guidance at t=1 is intentionally forbidden."
        )
    return steps


def load_flow_model(checkpoint: Path, device: str, use_ema: bool = True):
    """Rebuild the LiC architecture, including saved physics inputs, and EMA."""
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected {config_path} next to checkpoint directory.")
    config = json.loads(config_path.read_text())
    extras = config.get("extras", {})
    delta_channels = int(extras.get("delta_endpoint_channels", 0) or 0)
    if delta_channels != 0:
        raise SystemExit(
            "This LiC force-guidance runner currently targets the unconditioned "
            f"TS-denoiser.  Checkpoint has delta_endpoint_channels={delta_channels}."
        )

    backbone_name = str(extras.get("backbone", "uma-s-1p2"))
    raw_backbone = load_uma_backbone(backbone_name, device=device, freeze=True, eval_mode=True)
    blocks = extras.get("unfreeze_uma_blocks")
    if not blocks and extras.get("unfreeze_uma_last", False): blocks = [-1]
    if bool(extras.get("unfreeze_uma_all", False)) or blocks:
        trainable_blocks = raw_backbone.blocks if extras.get("unfreeze_uma_all", False) else [raw_backbone.blocks[i] for i in blocks]
        for block in trainable_blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

    backbone = raw_backbone
    if bool(extras.get("early_time_film", False)):
        block_indices = [int(value) for value in str(extras.get("early_time_film_blocks", "0,1,2,3")).split(",")]
        backbone = TimeFiLMBackbone(
            raw_backbone, inject_block_indices=block_indices, inject_force=False,
            inject_hint=bool(extras.get("inject_hint", False)),
        ).to(device)

    attention = GlobalAttn(
        sphere_channels=raw_backbone.sphere_channels,
        lmax=raw_backbone.lmax,
        num_heads=int(extras.get("attn_heads", 8)),
        num_layers=int(extras.get("attn_layers", 0)),
    ).to(device)
    try: from conditioning import head_options, make_conditioner
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import head_options, make_conditioner
    velocity_head = VelocityHead(
        sphere_channels=raw_backbone.sphere_channels,
        input_lmax=raw_backbone.lmax,
        depth=int(extras.get("head_depth", 1)),
        delta_endpoint_channels=0,
        **head_options(extras),
    ).to(device)

    # This checkpoint fine-tuned the backbone and added time-FiLM, so EMA
    # weights must be restored into all three modules in training-time order.
    modules = [backbone, attention, velocity_head]
    if not use_ema:
        if extras.get("early_time_film", False) or any(p.requires_grad for p in raw_backbone.parameters()):
            raise ValueError("raw loader cannot restore a trainable/time-FiLM backbone; use EMA")
        modules = [attention, velocity_head]
    load_ema_weights(str(checkpoint), modules, device=device, use_ema=use_ema)
    backbone.eval()
    attention.eval()
    velocity_head.eval()
    if extras.get("source_channels", 0):
        print(f"[model] source memory={extras.get('source_conditioning', False)}; "
              "same-size source branch; no target/parent input or extra PES queries", flush=True)
    if extras.get("inject_force", False) or extras.get("curvature_conditioning", False):
        frozen = load_uma_backbone(backbone_name, device=device, freeze=True, eval_mode=True)
        force_head, force_tasks = load_uma_force_head(backbone_name, device=device)
        velocity_head._physics_conditioner = make_conditioner(frozen, force_head, force_tasks,
            inject_force=bool(extras.get("inject_force", False)), curvature=bool(extras.get("curvature_conditioning", False)),
            fd_eps=float(extras.get("curvature_fd_eps", .002)), probe_batch_size=int(extras.get("physics_batch_size", 32)))
        print(f"[model] online frozen UMA conditioning: force={extras.get('inject_force',False)}, "
              f"scalar curvature={extras.get('curvature_conditioning',False)}; geometry/time only, no target supplied", flush=True)
    return config, backbone_name, backbone, attention, velocity_head


def make_force_function(sample: dict, backbone_name: str, device: str):
    """Load a separate frozen UMA energy model and return a plain force function."""
    print(f"[UMA] loading separate frozen {backbone_name!r} force model", flush=True)
    force_backbone = load_uma_backbone(backbone_name, device=device, freeze=True, eval_mode=True)
    force_head, force_tasks = load_uma_force_head(backbone_name, device=device)
    force_backbone.eval()
    force_head.eval()

    def force_function(positions: torch.Tensor) -> torch.Tensor:
        return uma_forces(
            positions,
            Z=sample["Z"],
            cell=sample["cell"],
            task_name=sample["task_name"],
            charge=int(sample["charge"]),
            spin=int(sample["spin"]),
            fixed=sample["fixed"],
            force_backbone=force_backbone,
            force_head=force_head,
            force_tasks=force_tasks,
            device=device,
        )

    return force_function


def model_velocity(
    positions: torch.Tensor,
    time_value: float,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    device: str,
    *,
    source_positions: torch.Tensor | None = None,
    hint: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate LiC velocity with any saved current-state physics inputs.

    hint: (B, N, 3) or (N, 3) hint field, required by a hint-injecting backbone; held fixed by the caller
    for the whole rollout."""
    batch_size = len(positions)
    data_list = [
        build_atomic_data(
            positions[index], sample["Z"], sample["cell"], sample["task_name"],
            int(sample["charge"]), int(sample["spin"]), sample["fixed"],
        )
        for index in range(batch_size)
    ]
    atomic_batch = data_list_collater(data_list, otf_graph=True).to(device)
    time_tensor = torch.full((batch_size,), time_value, dtype=torch.float32, device=device)

    inputs = {}
    if getattr(velocity_head, "source_conditioning", False):
        if source_positions is None or source_positions.shape != positions.shape:
            raise ValueError("sampling requires the original x0, retained unchanged at every step")
        source_delta = mic_displacement(source_positions, positions, sample["cell"])
        source_delta[:, sample["fixed"]] = 0.
        inputs["source_delta"] = source_delta.reshape(-1, 3).to(device)
    conditioner = getattr(velocity_head, "_physics_conditioner", None)
    if conditioner is not None:
        physics = conditioner(positions, [sample]*batch_size, device)
        if velocity_head.force_field_channels: inputs["force_field"] = physics["force_field"]
        if velocity_head.curvature_conditioning: inputs["curvature_scalars"] = physics["curvature_scalars"]

    with torch.no_grad():
        if isinstance(backbone, TimeFiLMBackbone):
            hint_field = None
            if getattr(backbone, "inject_hint", False):
                if hint is None:
                    raise ValueError("this checkpoint injects a hint; pass hint=(B, N, 3)")
                hint_field = (hint if hint.dim() == 3 else hint.expand(batch_size, -1, -1))
                hint_field = hint_field.reshape(-1, 3).to(device=device, dtype=torch.float32)
            features = backbone(atomic_batch, time_tensor, atomic_batch.batch, hint=hint_field)
        else:
            features = backbone(atomic_batch)
        hidden = attention(features["node_embedding"], atomic_batch.batch)
        velocity = velocity_head(hidden, time_tensor, atomic_batch.batch, **inputs)
        velocity = apply_output_projections(
            velocity,
            sample["fixed"].to(device).repeat(batch_size),
            atomic_batch.batch,
            num_systems=batch_size,
        )
    return velocity.view(batch_size, -1, 3).cpu()


def add_trace_records(trace: list[dict], info: dict[str, torch.Tensor], step: int,
                      time_value: float, first_trajectory: int) -> None:
    """Turn one tensor-valued guidance event into simple JSON-ready records."""
    keys = [
        "force_before", "fmax_before", "force_after", "fmax_after",
        "above_tolerance", "accepted", "backtracks", "shift_rms", "shift_max",
    ]
    values = {key: info[key].detach().cpu().numpy() for key in keys}
    for local_index in range(len(values["force_before"])):
        trace.append({
            "trajectory": first_trajectory + local_index,
            "step": step,
            "t": time_value,
            "force_before": float(values["force_before"][local_index]),
            "fmax_before": float(values["fmax_before"][local_index]),
            "force_after": float(values["force_after"][local_index]),
            "fmax_after": float(values["fmax_after"][local_index]),
            "above_tolerance": bool(values["above_tolerance"][local_index]),
            "accepted": bool(values["accepted"][local_index]),
            "n_backtracks": int(values["backtracks"][local_index]),
            "endpoint_shift_rms": float(values["shift_rms"][local_index]),
            "endpoint_shift_max": float(values["shift_max"][local_index]),
        })


def integrate_one_batch(
    starts: torch.Tensor,
    *,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    device: str,
    K: int,
    guide_steps: set[int],
    force_function=None,
    force_tol: float = 0.01,
    fd_eps: float = 0.001,
    endpoint_step: float = 0.02,
    max_endpoint_step: float = 0.05,
    alpha: float = 0.5,
    max_backtracks: int = 3,
    backtrack_factor: float = 0.5,
    first_trajectory: int = 0,
    capture_terminal_steps: set[int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[dict],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, torch.Tensor],
]:
    """Euler-integrate one batch, optionally retargeting at selected steps."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1].")
    x = starts.clone()
    source_kwargs = {"source_positions": starts.detach().clone()} if getattr(velocity_head, "source_conditioning", False) else {}
    cell = sample["cell"]
    mobile_mask = ~sample["fixed"]
    dt = 1.0 / K
    path = torch.empty(K + 1, len(x), x.shape[1], 3, dtype=x.dtype)
    path[0] = x
    trace: list[dict] = []
    total_shift_rms = torch.zeros(len(x), dtype=x.dtype)
    max_shift_rms = torch.zeros(len(x), dtype=x.dtype)
    accepted_count = torch.zeros(len(x), dtype=torch.int64)
    capture_terminal_steps = capture_terminal_steps or set()
    terminal_estimates: dict[int, torch.Tensor] = {}

    for step in range(K):
        time_value = step / K
        velocity = model_velocity(
            x, time_value, sample, backbone, attention, velocity_head, device, **source_kwargs
        )

        need_terminal_estimate = (
            step in capture_terminal_steps
            or (force_function is not None and step in guide_steps)
        )
        if need_terminal_estimate:
            # The algebraic terminal estimate for a straight rectified flow.
            # It is wrapped before UMA sees it or before saving a diagnostic.
            terminal_estimate = wrap_positions(x + (1.0 - time_value) * velocity, cell)
            if step in capture_terminal_steps:
                terminal_estimates[step] = terminal_estimate.clone()

        if force_function is not None and step in guide_steps:
            corrected_terminal, info = guide_terminal_estimates(
                terminal_estimate,
                mobile_mask=mobile_mask,
                cell=cell,
                force_function=force_function,
                force_tol=force_tol,
                fd_eps=fd_eps,
                endpoint_step=endpoint_step,
                max_endpoint_step=max_endpoint_step,
                max_backtracks=max_backtracks,
                backtrack_factor=backtrack_factor,
            )
            endpoint_delta = mic_displacement(corrected_terminal, terminal_estimate, cell)
            endpoint_delta[:, ~mobile_mask, :] = 0.0
            velocity = velocity + alpha * endpoint_delta / (1.0 - time_value)

            total_shift_rms += info["shift_rms"].cpu()
            max_shift_rms = torch.maximum(max_shift_rms, info["shift_rms"].cpu())
            accepted_count += info["accepted"].cpu().to(torch.int64)
            add_trace_records(trace, info, step, time_value, first_trajectory)

        x = wrap_positions(x + dt * velocity, cell)
        path[step + 1] = x

    return x, path, trace, total_shift_rms, max_shift_rms, accepted_count, terminal_estimates


def minimum_image_vector(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """PBC displacement for row-vector ASE cells."""
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def find_false_midpoints(train_traj: str, test_traj: str):
    """Load the 179 labels and reconstruct the ten LiC saddle-less bridges."""
    train_frames = list(read(train_traj, index=":"))
    test_frames = list(read(test_traj, index=":"))
    reference = train_frames[0]
    cell = np.asarray(reference.get_cell())
    carbon = reference.get_positions()[:N_CARBON]
    train_saddles = np.asarray([frame.get_positions()[LI] for frame in train_frames[1::3]])
    test_saddles = np.asarray([frame.get_positions()[LI] for frame in test_frames[1::3]])
    all_saddles = np.vstack([train_saddles, test_saddles])

    midpoints = []
    for carbon_i, carbon_j in itertools.combinations(range(N_CARBON), 2):
        bond = minimum_image_vector(carbon[carbon_j] - carbon[carbon_i], cell)
        if not 0.1 < np.linalg.norm(bond) < 1.8:
            continue
        midpoint = carbon[carbon_i] + 0.5 * bond
        fractional = midpoint @ np.linalg.inv(cell)
        midpoint = (fractional - np.floor(fractional)) @ cell
        midpoints.append(midpoint)
    midpoints = np.asarray(midpoints)
    distance_to_saddle = nearest_in_plane_distance(midpoints, all_saddles, cell)
    false_midpoints = midpoints[distance_to_saddle > FALSE_MIDPOINT_CUTOFF]
    if len(false_midpoints) != 10:
        raise ValueError(f"Expected 10 false LiC bridges, found {len(false_midpoints)}.")

    minima = np.asarray(
        [frame.get_positions()[LI] for frame in train_frames[0::3] + train_frames[2::3]
         + test_frames[0::3] + test_frames[2::3]]
    )
    return reference, train_saddles, test_saddles, false_midpoints, float(minima[:, 2].mean())


def nearest_in_plane_distance(points: np.ndarray, references: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Nearest in-plane reference distance, using neighbouring periodic images."""
    distance = np.full(len(points), np.inf)
    for shift_i, shift_j in itertools.product((-1, 0, 1), repeat=2):
        offset = shift_i * cell[0] + shift_j * cell[1]
        delta = points[:, None, :2] - (references[None, :, :2] + offset[:2])
        distance = np.minimum(distance, np.linalg.norm(delta, axis=2).min(axis=1))
    return distance


def nearest_3d_distance(points: np.ndarray, references: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Nearest three-dimensional reference distance, using in-plane PBC images."""
    distance = np.full(len(points), np.inf)
    for shift_i, shift_j in itertools.product((-1, 0, 1), repeat=2):
        offset = shift_i * cell[0] + shift_j * cell[1]
        delta = points[:, None, :] - (references[None, :, :] + offset)
        distance = np.minimum(distance, np.linalg.norm(delta, axis=2).min(axis=1))
    return distance


def closest_reference_vectors(
    points: np.ndarray,
    references: np.ndarray,
    cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MIC vectors from each point to its closest in-plane-PBC reference.

    The returned vector points *from* ``points`` *to* the closest reference.
    Only in-plane periodic images are considered, matching the LiC plotting and
    task metrics.  The z displacement remains ordinary Cartesian distance.
    """
    closest_distance = np.full(len(points), np.inf)
    closest_index = np.full(len(points), -1, dtype=np.int64)
    closest_vector = np.zeros_like(points)
    for shift_i, shift_j in itertools.product((-1, 0, 1), repeat=2):
        offset = shift_i * cell[0] + shift_j * cell[1]
        vectors = references[None, :, :] + offset - points[:, None, :]
        distances = np.linalg.norm(vectors, axis=2)
        index = distances.argmin(axis=1)
        distance = distances[np.arange(len(points)), index]
        improved = distance < closest_distance
        closest_distance[improved] = distance[improved]
        closest_index[improved] = index[improved]
        closest_vector[improved] = vectors[np.arange(len(points))[improved], index[improved]]
    return closest_vector, closest_distance, closest_index


def direction_cosine(direction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cosine between two per-trajectory Cartesian direction vectors.

    Rows with a zero or numerically negligible vector are returned as NaN;
    their direction is undefined rather than arbitrarily called aligned.
    """
    direction_norm = np.linalg.norm(direction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = np.full(len(direction), np.nan)
    valid = (direction_norm > 1e-12) & (target_norm > 1e-12)
    cosine[valid] = (
        (direction[valid] * target[valid]).sum(axis=1)
        / (direction_norm[valid] * target_norm[valid])
    )
    return cosine


def alignment_summary(records: list[dict]) -> dict[int, dict]:
    """Summarize H F directional alignment records one guide step at a time."""
    summary = {}
    for step in sorted({record["step"] for record in records}):
        at_step = [record for record in records if record["step"] == step]
        true_cosine = np.asarray(
            [record["hf_cosine_to_nearest_true"] for record in at_step if record["hf_cosine_to_nearest_true"] is not None]
        )
        away_cosine = np.asarray(
            [record["hf_cosine_away_from_nearest_false"] for record in at_step if record["hf_cosine_away_from_nearest_false"] is not None]
        )
        summary[step] = {
            "count": len(at_step),
            "above_tolerance": sum(record["above_tolerance"] for record in at_step),
            "valid_true_cosine": len(true_cosine),
            "median_true_cosine": float(np.median(true_cosine)) if len(true_cosine) else None,
            "positive_true_cosine_pct": float(100 * np.mean(true_cosine > 0)) if len(true_cosine) else None,
            "valid_away_cosine": len(away_cosine),
            "median_away_cosine": float(np.median(away_cosine)) if len(away_cosine) else None,
            "positive_away_cosine_pct": float(100 * np.mean(away_cosine > 0)) if len(away_cosine) else None,
        }
    return summary


def measure_uma_direction_alignment(
    *,
    terminal: torch.Tensor,
    step: int,
    K: int,
    force_function,
    mobile_mask: torch.Tensor,
    cell: torch.Tensor,
    all_saddles: np.ndarray,
    false_midpoints: np.ndarray,
    force_tol: float,
    fd_eps: float,
) -> list[dict]:
    """Measure whether baseline UMA H F points toward true LiC saddles.

    This is intentionally a diagnostic on the *unguided baseline* algebraic
    terminal estimate, not on the already-perturbed guided path.  It performs
    one regular UMA force call and, above threshold, the two finite-difference
    force calls needed to estimate H F.  It does not alter any trajectory.
    """
    forces = force_function(terminal)
    force_rms, _ = force_rms_and_max(forces, mobile_mask)
    above_tolerance = force_rms > force_tol
    hf = torch.zeros_like(terminal)
    if above_tolerance.any():
        indices = torch.where(above_tolerance)[0]
        hf[indices] = estimate_hf_direction(
            terminal[indices],
            forces[indices],
            mobile_mask,
            cell,
            force_function,
            fd_eps,
        )

    terminal_li = terminal[:, LI].detach().cpu().numpy()
    hf_li = hf[:, LI].detach().cpu().numpy()
    cell_np = cell.detach().cpu().numpy()
    toward_true, true_distance, true_index = closest_reference_vectors(
        terminal_li, all_saddles, cell_np
    )
    toward_false, false_distance, false_index = closest_reference_vectors(
        terminal_li, false_midpoints, cell_np
    )
    cosine_to_true = direction_cosine(hf_li, toward_true)
    cosine_away_from_false = direction_cosine(hf_li, -toward_false)

    records = []
    for index in range(len(terminal)):
        records.append({
            "trajectory": int(index),
            "step": int(step),
            "t": float(step / K),
            "force_rms": float(force_rms[index]),
            "above_tolerance": bool(above_tolerance[index]),
            "hf_norm": float(np.linalg.norm(hf_li[index])),
            "nearest_true_saddle_index": int(true_index[index]),
            "nearest_true_saddle_distance": float(true_distance[index]),
            "nearest_false_bridge_index": int(false_index[index]),
            "nearest_false_bridge_distance": float(false_distance[index]),
            "hf_cosine_to_nearest_true": (
                float(cosine_to_true[index]) if np.isfinite(cosine_to_true[index]) else None
            ),
            "hf_cosine_away_from_nearest_false": (
                float(cosine_away_from_false[index])
                if np.isfinite(cosine_away_from_false[index]) else None
            ),
        })

    summary = alignment_summary(records)[step]
    true_text = (
        f"median={summary['median_true_cosine']:.3f}, positive={summary['positive_true_cosine_pct']:.1f}%"
        if summary["median_true_cosine"] is not None else "undefined"
    )
    false_text = (
        f"median={summary['median_away_cosine']:.3f}, positive={summary['positive_away_cosine_pct']:.1f}%"
        if summary["median_away_cosine"] is not None else "undefined"
    )
    print(
        f"[alignment] baseline step {step:>2} t={step / K:.3f}; "
        f"above tolerance={summary['above_tolerance']}/{summary['count']}; "
        f"H F -> nearest true: {true_text}; H F away from nearest false: {false_text}",
        flush=True,
    )
    return records


def make_starts(reference, z_mean: float, n: int, grid: int, z_jitter: float, seed: int) -> np.ndarray:
    """Create the exact random/grid Li starts used for both baseline and guided runs."""
    cell = np.asarray(reference.get_cell())
    base = reference.get_positions()
    if grid > 0:
        fractions = (np.arange(grid) + 0.5) / grid
        grid_x, grid_y = np.meshgrid(fractions, fractions, indexing="ij")
        xy_fractional = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        z = np.full(len(xy_fractional), z_mean)
    else:
        generator = np.random.default_rng(seed)
        xy_fractional = generator.random((n, 2))
        z = z_mean + z_jitter * generator.standard_normal(n)
    starts = np.repeat(base[None], len(xy_fractional), axis=0)
    starts[:, LI, :2] = xy_fractional @ cell[:2, :2]
    starts[:, LI, 2] = z
    return starts


def make_false_saddle_starts(
    reference,
    false_midpoints: np.ndarray,
    saddle_z: float,
    *,
    noise_sigma: float,
    samples_per_bridge: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact or locally perturbed starts around each false bridge.

    The x-y coordinates are the ten C-C bridge midpoints for which the LiC
    dataset contains no independently verified saddle label.  The z coordinate
    is held at the mean true-saddle height before optional noise.  Noise is
    isotropic in Li's three Cartesian coordinates, matching the TS-denoising
    training-noise geometry; every carbon remains fixed.

    Returns both full ASE-style coordinate arrays and the false-bridge index
    that generated each row, so local-basin behavior remains auditable.
    """
    if noise_sigma < 0:
        raise ValueError("--false-saddle-noise-sigma must be non-negative")
    if samples_per_bridge <= 0:
        raise ValueError("--false-saddle-samples-per-bridge must be positive")

    bridge_index = np.repeat(
        np.arange(len(false_midpoints), dtype=np.int64), samples_per_bridge
    )
    centers = false_midpoints[bridge_index].copy()
    centers[:, 2] = saddle_z
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        centers += noise_sigma * rng.standard_normal(centers.shape)

    starts = np.repeat(reference.get_positions()[None], len(centers), axis=0)
    starts[:, LI] = centers
    return starts, bridge_index


def write_plot_compatible_throw(
    output_path: Path,
    paths: np.ndarray,
    train_saddles: np.ndarray,
    test_saddles: np.ndarray,
    false_midpoints: np.ndarray,
    cell: np.ndarray,
    false_bridge_index: np.ndarray,
    saddle_z: float,
) -> None:
    """Save one path set in the schema read directly by plot_throws.py.

    Extra false-bridge fields are ignored by the existing plotter but make the
    starting population explicit for later analysis.
    """
    np.savez_compressed(
        output_path,
        starts=paths[:, 0],
        finals=paths[:, -1],
        paths=paths,
        train_saddles=train_saddles,
        test_saddles=test_saddles,
        false_midpoints=false_midpoints,
        false_bridge_index=false_bridge_index,
        false_saddle_start_z=np.asarray(saddle_z),
        cell=cell,
    )


def run_baseline(
    starts: np.ndarray,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    args: argparse.Namespace,
):
    """Run the student's normal inputs, without endpoint force guidance."""
    final_positions, paths = [], []

    for start in range(0, len(starts), args.batch_size):
        stop = min(start + args.batch_size, len(starts))
        x0 = torch.tensor(starts[start:stop], dtype=torch.float32)
        x0 = wrap_positions(x0, sample["cell"])
        final, path, _, _, _, _, _ = integrate_one_batch(
            x0,
            sample=sample,
            backbone=backbone,
            attention=attention,
            velocity_head=velocity_head,
            device=args.device,
            K=args.K,
            guide_steps=set(),
            first_trajectory=start,
        )
        final_positions.append(final[:, LI].numpy())
        paths.append(path[:, :, LI].permute(1, 0, 2).numpy())
        print(f"[sample] {stop}/{len(starts)} baseline trajectories", flush=True)

    return {
        "baseline_final": np.concatenate(final_positions),
        "baseline_path": np.concatenate(paths),
    }


def run_baseline_and_guided(
    starts: np.ndarray,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    force_function,
    args: argparse.Namespace,
    guide_steps: set[int],
    all_saddles: np.ndarray,
    false_midpoints: np.ndarray,
):
    """Run paired baseline/guided integrations from identical starts."""
    baseline_final, guided_final = [], []
    baseline_path, guided_path = [], []
    baseline_rms, baseline_fmax = [], []
    guided_rms, guided_fmax = [], []
    total_shift, max_shift, accepted_events = [], [], []
    trace: list[dict] = []
    baseline_terminal_estimates: dict[int, list[np.ndarray]] = {
        step: [] for step in guide_steps
    }

    for start in range(0, len(starts), args.batch_size):
        stop = min(start + args.batch_size, len(starts))
        x0 = torch.tensor(starts[start:stop], dtype=torch.float32)
        x0 = wrap_positions(x0, sample["cell"])

        base_final, base_path, _, _, _, _, base_terminals = integrate_one_batch(
            x0, sample=sample, backbone=backbone, attention=attention,
            velocity_head=velocity_head, device=args.device, K=args.K,
            guide_steps=set(), first_trajectory=start,
            capture_terminal_steps=guide_steps if args.diagnose_uma_direction else set(),
        )
        guided, guided_path_one, batch_trace, batch_total, batch_max, batch_accepted, _ = integrate_one_batch(
            x0, sample=sample, backbone=backbone, attention=attention,
            velocity_head=velocity_head, device=args.device, K=args.K,
            guide_steps=guide_steps, force_function=force_function,
            force_tol=args.force_tol, fd_eps=args.fd_eps,
            endpoint_step=args.endpoint_step, max_endpoint_step=args.max_endpoint_step,
            alpha=args.alpha, max_backtracks=args.max_backtracks,
            backtrack_factor=args.backtrack_factor, first_trajectory=start,
        )
        base_forces = force_function(base_final)
        guided_forces = force_function(guided)
        base_rms, base_max = force_rms_and_max(base_forces, ~sample["fixed"])
        guide_rms, guide_max = force_rms_and_max(guided_forces, ~sample["fixed"])

        baseline_final.append(base_final[:, LI].numpy())
        guided_final.append(guided[:, LI].numpy())
        baseline_path.append(base_path[:, :, LI].permute(1, 0, 2).numpy())
        guided_path.append(guided_path_one[:, :, LI].permute(1, 0, 2).numpy())
        baseline_rms.append(base_rms.numpy())
        baseline_fmax.append(base_max.numpy())
        guided_rms.append(guide_rms.numpy())
        guided_fmax.append(guide_max.numpy())
        total_shift.append(batch_total.numpy())
        max_shift.append(batch_max.numpy())
        accepted_events.append(batch_accepted.numpy())
        trace.extend(batch_trace)
        for step, terminal in base_terminals.items():
            baseline_terminal_estimates[step].append(terminal.numpy())
        print(f"[sample] {stop}/{len(starts)} paired baseline/guided trajectories", flush=True)

    direction_alignment = []
    if args.diagnose_uma_direction:
        for step in sorted(guide_steps):
            terminal = torch.from_numpy(np.concatenate(baseline_terminal_estimates[step]))
            direction_alignment.extend(
                measure_uma_direction_alignment(
                    terminal=terminal,
                    step=step,
                    K=args.K,
                    force_function=force_function,
                    mobile_mask=~sample["fixed"],
                    cell=sample["cell"],
                    all_saddles=all_saddles,
                    false_midpoints=false_midpoints,
                    force_tol=args.force_tol,
                    fd_eps=args.fd_eps,
                )
            )

    return {
        "baseline_final": np.concatenate(baseline_final),
        "guided_final": np.concatenate(guided_final),
        "baseline_path": np.concatenate(baseline_path),
        "guided_path": np.concatenate(guided_path),
        "baseline_rms": np.concatenate(baseline_rms),
        "baseline_fmax": np.concatenate(baseline_fmax),
        "guided_rms": np.concatenate(guided_rms),
        "guided_fmax": np.concatenate(guided_fmax),
        "total_shift": np.concatenate(total_shift),
        "max_shift": np.concatenate(max_shift),
        "accepted_events": np.concatenate(accepted_events),
        "trace": trace,
        "direction_alignment": direction_alignment,
    }


def describe_endpoints(
    finals: np.ndarray,
    all_saddles: np.ndarray,
    false_midpoints: np.ndarray,
    cell: np.ndarray,
    force_rms: np.ndarray | None = None,
    force_fmax: np.ndarray | None = None,
) -> dict:
    """Compute LiC discovery metrics, and force metrics only when UMA was used."""
    saddle_distance = nearest_3d_distance(finals, all_saddles, cell)
    false_distance = nearest_in_plane_distance(finals, false_midpoints, cell)
    metrics = {
        "p50_distance": float(np.percentile(saddle_distance, 50)),
        "p90_distance": float(np.percentile(saddle_distance, 90)),
        "p99_distance": float(np.percentile(saddle_distance, 99)),
        "max_distance": float(saddle_distance.max()),
        "true_hit_pct": float(100 * np.mean(saddle_distance < TRUE_HIT_RADIUS)),
        "false_basin_pct": float(100 * np.mean(false_distance < FALSE_BASIN_RADIUS)),
    }
    if force_rms is not None and force_fmax is not None:
        metrics.update({
            "mean_force_rms": float(force_rms.mean()),
            "median_force_rms": float(np.median(force_rms)),
            "p90_force_rms": float(np.percentile(force_rms, 90)),
            "mean_fmax": float(force_fmax.mean()),
            "median_fmax": float(np.median(force_fmax)),
        })
    return metrics


def make_report(
    args: argparse.Namespace,
    guide_steps: set[int],
    output: dict,
    baseline: dict,
    guided: dict | None,
    cell: np.ndarray,
) -> str:
    """Create the compact report printed to stdout and written to report.txt."""
    task_header = "run                 p50_d(A)  p90_d(A)  hit_<0.10(%)  false_<0.30(%)"
    task_row = lambda name, metrics: (
        f"{name:<19} {metrics['p50_distance']:>8.4f}  {metrics['p90_distance']:>8.4f}  "
        f"{metrics['true_hit_pct']:>12.2f}  {metrics['false_basin_pct']:>15.2f}"
    )

    report_lines = [
        "LiC SaddleFlow sampling",
        "=======================",
        f"checkpoint: {Path(args.ckpt).resolve()}",
        (
            "starts: {} per each of 10 known false C-C bridge candidates; "
            "Gaussian Li noise sigma={:.3f} A".format(
                args.false_saddle_samples_per_bridge,
                args.false_saddle_noise_sigma,
            )
            if args.false_saddles_only
            else f"starts: {'grid ' + str(args.grid) + 'x' + str(args.grid) if args.grid else str(args.n) + ' random'}"
        ),
        f"Euler steps: {args.K}",
        "",
        "LiC task metrics",
        task_header,
        "-" * len(task_header),
        task_row("baseline", baseline),
    ]
    if guided is None:
        report_lines.extend([
            "",
            "Force guidance: disabled. Architectural physics inputs, if enabled, are still evaluated.",
            f"Online conditioning PES geometries: {output.get('conditioning_pes_calls', 0)}.",
            "Run again with --force_guided_sampling for the paired UMA-guided ablation.",
            "",
        ])
        return "\n".join(report_lines)

    final_delta = minimum_image_vector(output["guided_final"] - output["baseline_final"], cell)
    final_delta_norm = np.linalg.norm(final_delta, axis=1)
    trace = output["trace"]
    above_tolerance = sum(record["above_tolerance"] for record in trace)
    accepted = sum(record["accepted"] for record in trace)
    force_header = (
        "run                 median_fRMS  p90_fRMS  median_fmax"
    )
    force_row = lambda name, metrics: (
        f"{name:<19} {metrics['median_force_rms']:>11.6f}  "
        f"{metrics['p90_force_rms']:>8.6f}  {metrics['median_fmax']:>11.6f}"
    )
    report_lines.extend([
        task_row("guided", guided),
        "",
        f"Force guidance: enabled at steps {sorted(guide_steps)}; alpha: {args.alpha:g}",
        "guidance: force_tol={:.4f} eV/A, fd_eps={:.4f} A, endpoint_step={:.4f} A RMS, ".format(
            args.force_tol, args.fd_eps, args.endpoint_step
        ) + "max_endpoint_step={:.4f} A RMS".format(args.max_endpoint_step),
        "",
        "Final UMA force metrics",
        force_header,
        "-" * len(force_header),
        force_row("baseline", baseline),
        force_row("guided", guided),
        "",
        "Guidance-event audit",
        f"events checked: {len(trace)}; above tolerance: {above_tolerance}; accepted: {accepted}",
        "accepted-event rate among above-tolerance events: "
        f"{(100 * accepted / above_tolerance) if above_tolerance else 0.0:.2f}%",
        f"mean accepted corrections per trajectory: {output['accepted_events'].mean():.4f}",
        f"median total endpoint correction per trajectory: {np.median(output['total_shift']):.6f} A RMS",
        f"maximum one-event endpoint correction: {output['max_shift'].max():.6f} A RMS",
        "median guided-vs-baseline final displacement: "
        f"{np.median(final_delta_norm):.6f} A",
    ])

    direction_alignment = output.get("direction_alignment", [])
    if direction_alignment:
        report_lines.extend([
            "",
            "UMA H F direction alignment on baseline terminal estimates",
            "step    t     active     median cos(HF,true)  positive(%)  "
            "median cos(HF,away-false)  positive(%)",
            "--------------------------------------------------------------------------------",
        ])
        for step, row in alignment_summary(direction_alignment).items():
            true_cosine = "undefined" if row["median_true_cosine"] is None else f"{row['median_true_cosine']:.3f}"
            true_positive = "undefined" if row["positive_true_cosine_pct"] is None else f"{row['positive_true_cosine_pct']:.1f}"
            false_cosine = "undefined" if row["median_away_cosine"] is None else f"{row['median_away_cosine']:.3f}"
            false_positive = "undefined" if row["positive_away_cosine_pct"] is None else f"{row['positive_away_cosine_pct']:.1f}"
            report_lines.append(
                f"{step:>4}  {step / args.K:>5.2f}  {row['above_tolerance']:>3}/{row['count']:<3}  "
                f"{true_cosine:>20}  {true_positive:>11}  "
                f"{false_cosine:>26}  {false_positive:>11}"
            )
        report_lines.extend([
            "A positive cosine means the raw UMA H F direction points in the named direction.",
            "Negative or near-zero alignment means increasing endpoint correction size is unlikely",
            "to route the flow toward the desired true-saddle basin.",
        ])

    report_lines.extend([
        "",
        "Interpretation",
        "The target outcome is a lower false-bridge basin rate without reducing",
        "the known true-saddle hit rate.  Lower final UMA force alone is not enough",
        "to establish success; compare both LiC task rows above.",
        "",
    ])
    return "\n".join(report_lines)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.ckpt)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    try: from conditioning import catalog_paths
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import catalog_paths
    extras = json.loads((checkpoint.parent/"config.json").read_text()).get("extras", {})
    args.train_traj, args.test_traj = catalog_paths(extras, args.train_traj, args.test_traj)
    if args.false_saddles_only and args.grid:
        raise ValueError("--false_saddles_only cannot be combined with --grid; it supplies its own 10 starts.")
    if args.false_saddle_noise_sigma < 0:
        raise ValueError("--false-saddle-noise-sigma must be non-negative")
    if args.false_saddle_samples_per_bridge <= 0:
        raise ValueError("--false-saddle-samples-per-bridge must be positive")
    if args.diagnose_uma_direction and not args.force_guided_sampling:
        raise ValueError("--diagnose-uma-direction requires --force_guided_sampling to load UMA.")
    guide_steps = parse_guide_steps(args.guide_steps, args.K) if args.force_guided_sampling else set()
    if args.grid > 0:
        args.n = args.grid * args.grid
    output_dir = Path(args.out_dir) if args.out_dir else checkpoint.parent / "force_guided_sampling"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, train_saddles, test_saddles, false_midpoints, z_mean = find_false_midpoints(
        args.train_traj, args.test_traj
    )
    all_saddles = np.vstack([train_saddles, test_saddles])
    false_saddle_z = float(all_saddles[:, 2].mean())
    if args.false_saddles_only:
        starts, false_bridge_index = make_false_saddle_starts(
            reference,
            false_midpoints,
            false_saddle_z,
            noise_sigma=args.false_saddle_noise_sigma,
            samples_per_bridge=args.false_saddle_samples_per_bridge,
            seed=args.seed,
        )
        args.n = len(starts)
    else:
        starts = make_starts(reference, z_mean, args.n, args.grid, args.z_jitter, args.seed)
        false_bridge_index = np.full(len(starts), -1, dtype=np.int64)
    sample = atoms_to_sample_dict(reference)
    config, backbone_name, backbone, attention, velocity_head = load_flow_model(checkpoint, args.device)
    del config  # Architecture has been rebuilt; the remaining fields are not needed below.

    print(
        f"[setup] {len(starts)} starts; {len(all_saddles)} true saddles; "
        f"{len(false_midpoints)} false bridge midpoints; "
        f"force guidance={'on' if args.force_guided_sampling else 'off'}",
        flush=True,
    )
    if args.false_saddles_only:
        print(
            f"[setup] false-saddles-only: {args.false_saddle_samples_per_bridge} "
            f"start(s) per false bridge; common true-saddle z={false_saddle_z:.4f} A; "
            f"Li noise sigma={args.false_saddle_noise_sigma:.4f} A",
            flush=True,
        )
    cell = np.asarray(reference.get_cell())

    if args.force_guided_sampling:
        # Separate endpoint guidance; architectural inputs are handled by the
        # shared loader/model_velocity independently of this flag.
        force_function = make_force_function(sample, backbone_name, args.device)
        output = run_baseline_and_guided(
            starts, sample, backbone, attention, velocity_head, force_function, args, guide_steps,
            all_saddles, false_midpoints,
        )
        baseline = describe_endpoints(
            output["baseline_final"], all_saddles, false_midpoints, cell,
            force_rms=output["baseline_rms"], force_fmax=output["baseline_fmax"],
        )
        guided = describe_endpoints(
            output["guided_final"], all_saddles, false_midpoints, cell,
            force_rms=output["guided_rms"], force_fmax=output["guided_fmax"],
        )
    else:
        output = run_baseline(starts, sample, backbone, attention, velocity_head, args)
        baseline = describe_endpoints(
            output["baseline_final"], all_saddles, false_midpoints, cell,
        )
        guided = None

    try: from conditioning import conditioning_queries
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import conditioning_queries
    output["conditioning_pes_calls"] = conditioning_queries(velocity_head)
    result_arrays = {
        "conditioning_pes_calls": np.asarray(output["conditioning_pes_calls"]),
        "force_guidance_enabled": np.asarray(args.force_guided_sampling),
        "false_saddles_only": np.asarray(args.false_saddles_only),
        "false_saddle_noise_sigma": np.asarray(args.false_saddle_noise_sigma),
        "false_saddle_samples_per_bridge": np.asarray(args.false_saddle_samples_per_bridge),
        "starts": starts[:, LI],
        "baseline_finals": output["baseline_final"],
        "baseline_paths": output["baseline_path"],
        "train_saddles": train_saddles,
        "test_saddles": test_saddles,
        "false_midpoints": false_midpoints,
        "false_bridge_index": false_bridge_index,
        "false_saddle_start_z": np.asarray(false_saddle_z),
        "cell": cell,
    }
    if args.force_guided_sampling:
        result_arrays.update({
            "guided_finals": output["guided_final"],
            "guided_paths": output["guided_path"],
            "baseline_force_rms": output["baseline_rms"],
            "guided_force_rms": output["guided_rms"],
            "baseline_fmax": output["baseline_fmax"],
            "guided_fmax": output["guided_fmax"],
            "total_endpoint_correction": output["total_shift"],
            "max_endpoint_correction": output["max_shift"],
            "accepted_events": output["accepted_events"],
        })
    np.savez_compressed(output_dir / "results.npz", **result_arrays)
    if args.force_guided_sampling:
        (output_dir / "guidance_trace.json").write_text(json.dumps(output["trace"], indent=2))
    if args.diagnose_uma_direction:
        (output_dir / "direction_alignment.json").write_text(
            json.dumps(output["direction_alignment"], indent=2)
        )
    if args.false_saddles_only:
        write_plot_compatible_throw(
            output_dir / "false_saddles_baseline_throw.npz",
            output["baseline_path"],
            train_saddles,
            test_saddles,
            false_midpoints,
            cell,
            false_bridge_index,
            false_saddle_z,
        )
        if args.force_guided_sampling:
            write_plot_compatible_throw(
                output_dir / "false_saddles_guided_throw.npz",
                output["guided_path"],
                train_saddles,
                test_saddles,
                false_midpoints,
                cell,
                false_bridge_index,
                false_saddle_z,
            )

    report = make_report(args, guide_steps, output, baseline, guided, cell)
    (output_dir / "report.txt").write_text(report)
    print("\n" + report, flush=True)
    files = "results and report" if not args.force_guided_sampling else "results, trace, and report"
    if args.diagnose_uma_direction:
        files += ", plus UMA-direction alignment"
    if args.false_saddles_only:
        files += ", plus plotting-ready false-saddle throw file(s)"
    print(f"[done] wrote {files} to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
