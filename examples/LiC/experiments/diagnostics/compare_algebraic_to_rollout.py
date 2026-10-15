"""Compare LiC's one-shot algebraic endpoint with its full Euler rollout.

For an intermediate state ``x_t`` in a straight rectified flow, the current
velocity predicts an algebraic terminal estimate

    S_alg = wrap(x_t + (1 - t) v_theta(x_t, t)).

This script asks whether that one-shot estimate is actually a good proxy for
the terminal structure produced by the sampler.  At each requested Euler step
it saves the state reached by the ordinary K-step sampler, then branches from
that exact state into:

* ``S_alg``: the algebraic terminal estimate above;
* ``S_rollout``: an independent Euler integration for the remaining steps.

The supplied LiC data provide two useful, separately reported populations:

* 179 true-saddle starts, each with deterministic Li noise of the same scale
  as the TS-denoising training distribution;
* 10 verified saddle-less C--C bridge starts at the common saddle height.

The result is an inference diagnostic, not a training run.  It always loads
the checkpoint EMA weights, matching ``random_throw.py``.

Example from the repository root:

    CUDA_VISIBLE_DEVICES=0 python examples/LiC/experiments/diagnostics/compare_algebraic_to_rollout.py \
        --ckpt /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final \
        --K 20 --diagnostic-steps 0,5,10,15 --device cuda

It writes ``comparison.npz``, ``report.txt``, and ``comparison.pdf`` under
``<checkpoint-parent>/algebraic_vs_rollout`` unless ``--out-dir`` is supplied.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from saddleflow.constr_gen import force_rms_and_max
from saddleflow.data import atoms_to_sample_dict
from saddleflow.data.transforms import mic_displacement, wrap_positions

# Keep the model rebuild, EMA loading, LiC label discovery, and frozen UMA
# interface identical to the existing false-saddle inference experiment.  This
# script adds no model behavior; it only compares two terminal constructions.
from force_guided_sampling import (  # noqa: E402
    FALSE_BASIN_RADIUS,
    LI,
    TRUE_HIT_RADIUS,
    find_false_midpoints,
    load_flow_model,
    make_force_function,
    model_velocity,
    nearest_3d_distance,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[2]  # examples/LiC
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="LiC checkpoint_final directory")
    parser.add_argument("--train-traj", default=str(here / "train_set.traj"))
    parser.add_argument("--test-traj", default=str(here / "test_set.traj"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--K", type=int, default=20, help="Full Euler step count")
    parser.add_argument(
        "--diagnostic-steps", default="0,5,10,15",
        help="Comma-separated Euler steps where algebraic and remaining-rollout terminals are compared.",
    )
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of starts sharing a SaddleFlow forward.")
    parser.add_argument("--force-batch-size", type=int, default=16,
                        help="Number of endpoints per frozen-UMA force call.")
    parser.add_argument("--true-noise-sigma", type=float, default=0.25,
                        help="A; deterministic Li noise on each true-saddle start.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--true-saddles-only", action="store_true",
                        help="Evaluate only the 179 true-saddle-noise starts.")
    parser.add_argument("--false-bridges-only", action="store_true",
                        help="Evaluate only the ten verified false-bridge starts.")
    args = parser.parse_args()
    if args.K < 2:
        raise SystemExit("--K must be at least 2")
    if args.batch_size <= 0 or args.force_batch_size <= 0:
        raise SystemExit("--batch-size and --force-batch-size must be positive")
    if args.true_noise_sigma < 0:
        raise SystemExit("--true-noise-sigma must be non-negative")
    if args.true_saddles_only and args.false_bridges_only:
        raise SystemExit("Choose at most one of --true-saddles-only and --false-bridges-only")
    return args


def parse_steps(text: str, K: int) -> list[int]:
    """Parse diagnostic steps; t=1 has no velocity, so it is excluded."""
    try:
        steps = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    except ValueError as error:
        raise ValueError("--diagnostic-steps must be comma-separated integers") from error
    if not steps:
        raise ValueError("--diagnostic-steps must name at least one step")
    invalid = [step for step in steps if step < 0 or step >= K]
    if invalid:
        raise ValueError(f"diagnostic steps must satisfy 0 <= step < {K}; got {invalid}")
    return steps


def make_start_population(
    reference_atoms,
    true_saddles: np.ndarray,
    false_bridges: np.ndarray,
    *,
    true_noise_sigma: float,
    seed: int,
    include_true: bool,
    include_false: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    """Create the 179 noisy true starts and/or ten false-bridge starts.

    Carbon atoms stay exactly as in the input reference.  Only the mobile Li
    coordinate differs, which makes the starts conform to the LiC constraint.
    """
    if not include_true and not include_false:
        raise ValueError("at least one start population must be enabled")
    cell = torch.tensor(np.asarray(reference_atoms.cell), dtype=torch.float32)
    base = torch.tensor(reference_atoms.get_positions(), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts: list[torch.Tensor] = []
    population: list[str] = []

    if include_true:
        noise = torch.randn(len(true_saddles), 3, generator=generator) * true_noise_sigma
        for saddle, displacement in zip(true_saddles, noise):
            positions = base.clone()
            positions[LI] = torch.tensor(saddle, dtype=torch.float32) + displacement
            starts.append(wrap_positions(positions, cell))
            population.append("true_saddle_noise")

    if include_false:
        saddle_height = float(np.mean(true_saddles[:, 2]))
        for bridge in false_bridges:
            positions = base.clone()
            positions[LI, :2] = torch.tensor(bridge[:2], dtype=torch.float32)
            positions[LI, 2] = saddle_height
            starts.append(wrap_positions(positions, cell))
            population.append("false_bridge")

    return torch.stack(starts), np.asarray(population)


def remaining_euler_rollout(
    x: torch.Tensor,
    first_step: int,
    *,
    sample: dict,
    backbone,
    attention,
    velocity_head,
    device: str,
    K: int,
) -> torch.Tensor:
    """Finish the standard Euler sampler from a saved intermediate state."""
    endpoint = x.clone()
    dt = 1.0 / K
    for step in range(first_step, K):
        velocity = model_velocity(
            endpoint, step / K, sample, backbone, attention, velocity_head, device
        )
        endpoint = wrap_positions(endpoint + dt * velocity, sample["cell"])
    return endpoint


def force_rms_in_chunks(
    positions: torch.Tensor,
    force_function,
    mobile_mask: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Evaluate frozen-UMA mobile force RMS without making one oversized batch."""
    values = []
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        forces = force_function(chunk)
        rms, _ = force_rms_and_max(forces, mobile_mask)
        values.append(rms.cpu().numpy())
    return np.concatenate(values)


def compare_one_batch(
    starts: torch.Tensor,
    *,
    diagnostic_steps: list[int],
    sample: dict,
    backbone,
    attention,
    velocity_head,
    device: str,
    K: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Run one ordinary trajectory batch and branch at each requested state."""
    x = starts.clone()
    dt = 1.0 / K
    diagnostic_set = set(diagnostic_steps)
    algebraic: dict[int, torch.Tensor] = {}
    saved_state: dict[int, torch.Tensor] = {}

    # First run is exactly the ordinary sampler.  We save its states before
    # each requested velocity evaluation so the comparison uses real visited
    # inference states, not synthetic points on the labeled interpolation.
    for step in range(K):
        time_value = step / K
        velocity = model_velocity(x, time_value, sample, backbone, attention, velocity_head, device)
        if step in diagnostic_set:
            saved_state[step] = x.clone()
            algebraic[step] = wrap_positions(x + (1.0 - time_value) * velocity, sample["cell"])
        x = wrap_positions(x + dt * velocity, sample["cell"])
    ordinary_final = x

    rollout: dict[int, torch.Tensor] = {}
    for step in diagnostic_steps:
        # The t=0 remaining rollout is exactly the ordinary result already
        # computed above; avoid spending another K model evaluations on it.
        rollout[step] = (
            ordinary_final.clone()
            if step == 0
            else remaining_euler_rollout(
                saved_state[step], step,
                sample=sample,
                backbone=backbone,
                attention=attention,
                velocity_head=velocity_head,
                device=device,
                K=K,
            )
        )
    return algebraic, rollout


def li_pbc_distance(first: torch.Tensor, second: torch.Tensor, cell: torch.Tensor) -> np.ndarray:
    """Return the Li-only PBC distance between matching endpoint arrays."""
    displacement = mic_displacement(first, second, cell)
    return torch.linalg.vector_norm(displacement[:, LI], dim=-1).cpu().numpy()


def endpoint_summary(
    algebraic: np.ndarray,
    rollout: np.ndarray,
    force_algebraic: np.ndarray,
    force_rollout: np.ndarray,
    true_saddles: np.ndarray,
    false_bridges: np.ndarray,
    cell: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute task-level endpoint metrics using the established LiC labels."""
    algebraic_li = algebraic[:, LI]
    rollout_li = rollout[:, LI]
    return {
        "alg_to_rollout": li_pbc_distance(
            torch.tensor(algebraic), torch.tensor(rollout), torch.tensor(cell)
        ),
        "alg_to_true": nearest_3d_distance(algebraic_li, true_saddles, cell),
        "rollout_to_true": nearest_3d_distance(rollout_li, true_saddles, cell),
        "alg_to_false": nearest_3d_distance(algebraic_li, false_bridges, cell),
        "rollout_to_false": nearest_3d_distance(rollout_li, false_bridges, cell),
        "force_alg": force_algebraic,
        "force_rollout": force_rollout,
    }


def summary_row(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    """One transparent median/p90 summary for a chosen start population."""
    choose = lambda name, q: float(np.percentile(metrics[name][mask], q))
    return {
        "n": int(mask.sum()),
        "median_alg_rollout_A": choose("alg_to_rollout", 50),
        "p90_alg_rollout_A": choose("alg_to_rollout", 90),
        "median_alg_true_A": choose("alg_to_true", 50),
        "median_rollout_true_A": choose("rollout_to_true", 50),
        "median_force_alg_eVA": choose("force_alg", 50),
        "median_force_rollout_eVA": choose("force_rollout", 50),
        "alg_true_hit_pct": float(100 * (metrics["alg_to_true"][mask] < TRUE_HIT_RADIUS).mean()),
        "rollout_true_hit_pct": float(100 * (metrics["rollout_to_true"][mask] < TRUE_HIT_RADIUS).mean()),
        "alg_false_basin_pct": float(100 * (metrics["alg_to_false"][mask] < FALSE_BASIN_RADIUS).mean()),
        "rollout_false_basin_pct": float(100 * (metrics["rollout_to_false"][mask] < FALSE_BASIN_RADIUS).mean()),
    }


def write_report(
    output: Path,
    summaries: dict[str, dict[str, dict[str, float]]],
    args,
    n_starts: int,
) -> None:
    """Write a compact human-readable table, separated by population and time."""
    lines = [
        "LiC algebraic endpoint versus remaining Euler rollout",
        "=" * 58,
        f"checkpoint: {Path(args.ckpt).resolve()}",
        f"starts: {n_starts}",
        f"Euler steps: {args.K}; true-saddle noise sigma: {args.true_noise_sigma:.3f} A",
        "",
        "Each row branches from a state actually visited by the ordinary sampler.",
        "S_alg is x_t + (1-t) v(x_t,t); S_rollout is the remaining Euler integration.",
        "",
    ]
    for population, rows in summaries.items():
        lines += [population, "step    t      N   med d(alg,roll)  p90 d(alg,roll)  "
                  "med d_alg,true  med d_roll,true  med F_alg  med F_roll  "
                  "alg hit%  rollout hit%  alg false%  rollout false%"]
        for step_text, row in rows.items():
            step = int(step_text)
            lines.append(
                f"{step:4d}  {step / args.K:5.2f}  {row['n']:3d}  "
                f"{row['median_alg_rollout_A']:15.6f}  {row['p90_alg_rollout_A']:15.6f}  "
                f"{row['median_alg_true_A']:14.6f}  {row['median_rollout_true_A']:15.6f}  "
                f"{row['median_force_alg_eVA']:9.6f}  {row['median_force_rollout_eVA']:10.6f}  "
                f"{row['alg_true_hit_pct']:8.1f}  {row['rollout_true_hit_pct']:12.1f}  "
                f"{row['alg_false_basin_pct']:10.1f}  {row['rollout_false_basin_pct']:13.1f}"
            )
        lines.append("")
    output.write_text("\n".join(lines) + "\n")


def write_pdf(output: Path, diagnostic_steps: list[int], population: np.ndarray,
              metrics_by_step: dict[int, dict[str, np.ndarray]], K: int) -> None:
    """Plot the central comparison without hiding true and false populations."""
    groups = [
        ("true_saddle_noise", "true-saddle noise", "tab:blue"),
        ("false_bridge", "false bridge", "tab:red"),
    ]
    times = np.asarray(diagnostic_steps, dtype=float) / K
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        ("alg_to_rollout", "distance: algebraic vs rollout (Å)", True),
        ("force", "mobile UMA force RMS (eV/Å)", True),
        ("to_true", "distance to nearest true saddle (Å)", True),
    ]
    for axis, (kind, ylabel, logarithmic) in zip(axes, panels):
        any_group = False
        for group_key, group_label, color in groups:
            mask = population == group_key
            if not mask.any():
                continue
            any_group = True
            if kind == "alg_to_rollout":
                algebraic = [np.median(metrics_by_step[step]["alg_to_rollout"][mask])
                             for step in diagnostic_steps]
                axis.plot(times, algebraic, "o-", color=color, label=group_label)
            elif kind == "force":
                algebraic = [np.median(metrics_by_step[step]["force_alg"][mask])
                             for step in diagnostic_steps]
                rollout = [np.median(metrics_by_step[step]["force_rollout"][mask])
                           for step in diagnostic_steps]
                axis.plot(times, algebraic, "o--", color=color, label=f"{group_label}: algebraic")
                axis.plot(times, rollout, "o-", color=color, label=f"{group_label}: rollout")
            else:
                algebraic = [np.median(metrics_by_step[step]["alg_to_true"][mask])
                             for step in diagnostic_steps]
                rollout = [np.median(metrics_by_step[step]["rollout_to_true"][mask])
                           for step in diagnostic_steps]
                axis.plot(times, algebraic, "o--", color=color, label=f"{group_label}: algebraic")
                axis.plot(times, rollout, "o-", color=color, label=f"{group_label}: rollout")
        axis.set_xlabel("diagnostic flow time t")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        if logarithmic and any_group:
            axis.set_yscale("log")
        axis.legend(fontsize=7)
    fig.suptitle("LiC: one-shot algebraic endpoint versus full remaining Euler rollout", y=1.02)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    diagnostic_steps = parse_steps(args.diagnostic_steps, args.K)
    checkpoint = Path(args.ckpt)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    output_dir = Path(args.out_dir) if args.out_dir else checkpoint.parent / "algebraic_vs_rollout"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, train_saddles, test_saddles, false_bridges, _ = find_false_midpoints(
        args.train_traj, args.test_traj
    )
    true_saddles = np.vstack([train_saddles, test_saddles])
    # A bridge has a well-defined x/y C--C midpoint but no reference Li height.
    # Use the same common true-saddle height for its start and its basin metric.
    false_bridges = false_bridges.copy()
    false_bridges[:, 2] = float(np.mean(true_saddles[:, 2]))
    starts, population = make_start_population(
        reference,
        true_saddles,
        false_bridges,
        true_noise_sigma=args.true_noise_sigma,
        seed=args.seed,
        include_true=not args.false_bridges_only,
        include_false=not args.true_saddles_only,
    )
    sample = atoms_to_sample_dict(reference)
    config, backbone_name, backbone, attention, velocity_head = load_flow_model(checkpoint, args.device)
    del config
    force_function = make_force_function(sample, backbone_name, args.device)
    print(
        f"[compare] {len(starts)} starts: "
        f"{int((population == 'true_saddle_noise').sum())} true-saddle noise + "
        f"{int((population == 'false_bridge').sum())} false bridges; "
        f"K={args.K}; diagnostic steps={diagnostic_steps}",
        flush=True,
    )

    algebraic_by_step: dict[int, list[torch.Tensor]] = {step: [] for step in diagnostic_steps}
    rollout_by_step: dict[int, list[torch.Tensor]] = {step: [] for step in diagnostic_steps}
    for start in range(0, len(starts), args.batch_size):
        batch = starts[start:start + args.batch_size]
        algebraic, rollout = compare_one_batch(
            batch,
            diagnostic_steps=diagnostic_steps,
            sample=sample,
            backbone=backbone,
            attention=attention,
            velocity_head=velocity_head,
            device=args.device,
            K=args.K,
        )
        for step in diagnostic_steps:
            algebraic_by_step[step].append(algebraic[step])
            rollout_by_step[step].append(rollout[step])
        print(f"[compare] sampled {min(start + len(batch), len(starts))}/{len(starts)} starts", flush=True)

    algebraic = {step: torch.cat(parts) for step, parts in algebraic_by_step.items()}
    rollout = {step: torch.cat(parts) for step, parts in rollout_by_step.items()}
    metrics_by_step: dict[int, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, dict[str, float]]] = {
        "true_saddle_noise": {}, "false_bridge": {},
    }
    results: dict[str, np.ndarray] = {
        "starts": starts.numpy(),
        "population": population,
        "true_saddles": true_saddles,
        "false_bridges": false_bridges,
        "cell": np.asarray(reference.cell),
        "diagnostic_steps": np.asarray(diagnostic_steps, dtype=np.int64),
    }
    for step in diagnostic_steps:
        alg = algebraic[step]
        roll = rollout[step]
        force_alg = force_rms_in_chunks(
            alg, force_function, ~sample["fixed"], args.force_batch_size
        )
        force_roll = force_rms_in_chunks(
            roll, force_function, ~sample["fixed"], args.force_batch_size
        )
        metrics = endpoint_summary(
            alg.numpy(), roll.numpy(), force_alg, force_roll,
            true_saddles, false_bridges, np.asarray(reference.cell),
        )
        metrics_by_step[step] = metrics
        results[f"algebraic_step_{step:03d}"] = alg.numpy()
        results[f"rollout_step_{step:03d}"] = roll.numpy()
        for key, value in metrics.items():
            results[f"{key}_step_{step:03d}"] = value
        for group in summaries:
            mask = population == group
            if mask.any():
                summaries[group][str(step)] = summary_row(metrics, mask)

    np.savez_compressed(output_dir / "comparison.npz", **results)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    write_report(output_dir / "report.txt", summaries, args, n_starts=len(starts))
    write_pdf(output_dir / "comparison.pdf", diagnostic_steps, population, metrics_by_step, args.K)
    print(f"[done] wrote {output_dir / 'comparison.npz'}")
    print(f"[done] wrote {output_dir / 'report.txt'}")
    print(f"[done] wrote {output_dir / 'comparison.pdf'}")


if __name__ == "__main__":
    main()
