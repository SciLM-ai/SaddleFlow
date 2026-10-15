"""Visualise in-distribution recovery for an unconditioned TS-denoise model.

The TS-denoise objective is trained on ``x_0 = S + epsilon -> S``.  This
script uses the stored S frame from ``one_saddle.traj`` as the sampler input,
so ``sample_saddles`` creates exactly that type of noisy initial state.  It is
therefore a *local saddle-recovery* check, not a centre-of-hexagon discovery
test and not a product-conditioned Mode-1 plot.

Example:
    CUDA_VISIBLE_DEVICES=0 python examples/LiC_simpler/visualize_ts_recovery.py \\
        --ckpt-dir /path/to/checkpoint_final --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from ase.io import Trajectory

from saddleflow.data import atoms_to_sample_dict
from saddleflow.flow.sampler import sample_saddles
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import load_ema_weights, load_uma_backbone


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True,
                   help="Checkpoint directory, e.g. .../checkpoint_final.")
    p.add_argument("--traj", default=str(here / "one_saddle.traj"))
    p.add_argument("--sigma-inf", type=float, default=None,
                   help="Å. Default: ts_denoise_sigma recorded in config.json.")
    p.add_argument("--n-perturbations", type=int, default=64)
    p.add_argument("--K", type=int, default=50, help="Euler steps.")
    p.add_argument("--hit-radius", type=float, default=0.05,
                   help="Å. Final Li positions within this 3D PBC distance count as recovered.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ema", action="store_true",
                   help="Use raw checkpoint weights rather than the EMA shadow.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=None,
                   help="Default: <ckpt-dir>/viz_ts_recovery.")
    return p.parse_args()


def mic(vec: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image displacement for row-vector ASE cells."""
    frac = np.asarray(vec) @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def load_ts_model(ckpt_dir: Path, device: str, use_ema: bool):
    """Rebuild the checkpoint architecture, including fine-tuned UMA/FiLM."""
    cfg_path = ckpt_dir.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path}; cannot reconstruct checkpoint architecture")
    extras = json.loads(cfg_path.read_text()).get("extras", {})

    delta_channels = int(extras.get("delta_endpoint_channels", 0) or 0)
    if delta_channels != 0:
        raise SystemExit(
            "This is a TS-denoise recovery plot and requires an unconditioned "
            f"checkpoint (delta_endpoint_channels=0); config has {delta_channels}."
        )
    if float(extras.get("ts_denoise_sigma", 0.0) or 0.0) <= 0.0:
        raise SystemExit("checkpoint config is not a TS-denoise run (ts_denoise_sigma <= 0).")
    if int(extras.get("force_field_channels", 0) or 0) > 0 or bool(extras.get("force_film", False)):
        raise SystemExit("force-conditioned checkpoints are not supported by this TS recovery visualizer.")

    unfreeze_all = bool(extras.get("unfreeze_uma_all", False))
    unfreeze_last = bool(extras.get("unfreeze_uma_last", False))
    unfreeze_last2 = bool(extras.get("unfreeze_uma_last2", False))
    raw_backbone = load_uma_backbone(
        str(extras.get("backbone", "uma-s-1p2")), device=device,
        freeze=True, eval_mode=True, unfreeze_last_block=unfreeze_last,
    )
    if unfreeze_all:
        for block in raw_backbone.blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    elif unfreeze_last2:
        for parameter in raw_backbone.blocks[-2].parameters():
            parameter.requires_grad_(True)

    sphere_channels, lmax = raw_backbone.sphere_channels, raw_backbone.lmax
    use_tfilm = bool(extras.get("early_time_film", False))
    if use_tfilm:
        indices = [int(x) for x in str(extras.get("early_time_film_blocks", "0,1,2,3")).split(",")]
        backbone = TimeFiLMBackbone(raw_backbone, inject_block_indices=indices,
                                    inject_force=False).to(device)
    else:
        backbone = raw_backbone

    attn = GlobalAttn(
        sphere_channels=sphere_channels, lmax=lmax,
        num_heads=int(extras.get("attn_heads", 8)),
        num_layers=int(extras.get("attn_layers", 0)),
    ).to(device)
    head = VelocityHead(
        sphere_channels=sphere_channels, input_lmax=lmax,
        depth=int(extras.get("head_depth", 1)),
        delta_endpoint_channels=0,
    ).to(device)

    modules = ([backbone, attn, head]
               if use_tfilm or any(p.requires_grad for p in raw_backbone.parameters())
               else [attn, head])
    load_ema_weights(str(ckpt_dir), modules, device=device, use_ema=use_ema)
    for module in (backbone, attn, head):
        module.eval()

    print("[ts-recovery] checkpoint config: "
          f"ts_sigma={float(extras['ts_denoise_sigma']):.3f} "
          f"attn_layers={int(extras.get('attn_layers', 0))} "
          f"time_film={use_tfilm} "
          f"unfreeze_all={unfreeze_all}")
    return backbone, attn, head, float(extras["ts_denoise_sigma"])


def carbon_bonds(c_xy: np.ndarray, cutoff: float = 1.8):
    distances = np.linalg.norm(c_xy[:, None, :] - c_xy[None, :, :], axis=-1)
    i, j = np.where((distances < cutoff) & (distances > 0.1))
    return [(a, b) for a, b in zip(i, j) if a < b]


def plot_recovery(*, local_paths, c_xy, reference_xy, sigma, hit_radius,
                  initial_error, final_error, output: Path):
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    for i, j in carbon_bonds(c_xy):
        ax.plot([c_xy[i, 0], c_xy[j, 0]], [c_xy[i, 1], c_xy[j, 1]],
                color="0.55", linewidth=0.8, zorder=0)
    ax.scatter(c_xy[:, 0], c_xy[:, 1], s=8, c="0.35", zorder=1, label="C atoms")

    ax.add_collection(LineCollection(local_paths[:, :, :2], colors="lightskyblue",
                                     alpha=0.55, linewidths=0.75, zorder=2,
                                     label="denoising flow"))
    ax.scatter(local_paths[:, 0, 0], local_paths[:, 0, 1], s=16, c="red",
               alpha=0.65, edgecolors="none", zorder=3, label="noisy saddle input")
    ax.scatter(local_paths[:, -1, 0], local_paths[:, -1, 1], s=18, c="tab:blue",
               alpha=0.85, edgecolors="black", linewidths=0.25, zorder=4,
               label="predicted final Li")
    ax.scatter([reference_xy[0]], [reference_xy[1]], marker="*", s=190,
               c="black", zorder=5, label="stored reference saddle")

    recovered = 100.0 * float((final_error <= hit_radius).mean())
    title = (
        f"TS-denoise recovery: S + N(0, {sigma:.2f}² Å²) → S\n"
        f"n={len(local_paths)}, K={local_paths.shape[1] - 1}, hit≤{hit_radius:.3f} Å: "
        f"{recovered:.1f}%   initial median={np.median(initial_error):.3f} Å   "
        f"final median={np.median(final_error):.3f} Å"
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    pad = 2.6
    ax.set_xlim(reference_xy[0] - pad, reference_xy[0] + pad)
    ax.set_ylim(reference_xy[1] - pad, reference_xy[1] + pad)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not (ckpt_dir / "ema.pt").is_file():
        raise FileNotFoundError(f"{ckpt_dir} is not a complete checkpoint (missing ema.pt)")
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "viz_ts_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone, attn, head, train_sigma = load_ts_model(
        ckpt_dir, args.device, use_ema=not args.no_ema,
    )
    sigma = train_sigma if args.sigma_inf is None else args.sigma_inf

    frames = list(Trajectory(args.traj, "r"))
    if len(frames) != 3:
        raise ValueError(f"expected [R, S, P] in {args.traj}; found {len(frames)} frames")
    saddle = frames[1]
    cell = np.asarray(saddle.cell[:], dtype=np.float64)
    saddle_pos = saddle.get_positions()
    li_indices = np.flatnonzero(saddle.numbers == 3)
    if len(li_indices) != 1:
        raise ValueError(f"expected exactly one Li atom; found {len(li_indices)}")
    li_index = int(li_indices[0])
    reference = saddle_pos[li_index]

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.no_grad():
        # Passing the S frame as the sampler's input makes x_init = S + epsilon
        # for an unconditioned (delta_endpoint_channels=0) head.
        final, trajectory = sample_saddles(
            atoms_to_sample_dict(saddle), backbone, attn, head,
            sigma_inf=sigma, n_perturbations=args.n_perturbations, K=args.K,
            device=args.device, generator=generator, partner_pos=None,
            return_trajectory=True,
        )
    final = final.cpu().numpy()
    trajectory = trajectory.cpu().numpy()

    local_paths = np.stack([
        reference + mic(trajectory[:, i, li_index] - reference, cell)
        for i in range(trajectory.shape[1])
    ])
    initial_error = np.linalg.norm(local_paths[:, 0] - reference, axis=1)
    final_error = np.linalg.norm(local_paths[:, -1] - reference, axis=1)
    c_mask = saddle.numbers == 6
    c_xy = (reference + mic(saddle_pos[c_mask] - reference, cell))[:, :2]

    plot_path = out_dir / "ts_recovery.pdf"
    plot_recovery(
        local_paths=local_paths, c_xy=c_xy, reference_xy=reference[:2],
        sigma=sigma, hit_radius=args.hit_radius, initial_error=initial_error,
        final_error=final_error, output=plot_path,
    )
    np.savez_compressed(
        out_dir / "ts_recovery_data.npz",
        cell=cell, reference_saddle=reference, trajectory=trajectory, final=final,
        initial_error=initial_error, final_error=final_error,
        sigma_inf=sigma, hit_radius=args.hit_radius,
    )
    print(f"[ts-recovery] initial Li error: median={np.median(initial_error):.3f} Å "
          f"p90={np.percentile(initial_error, 90):.3f} Å")
    print(f"[ts-recovery] final Li error:   median={np.median(final_error):.3f} Å "
          f"p90={np.percentile(final_error, 90):.3f} Å "
          f"hit={100.0 * (final_error <= args.hit_radius).mean():.1f}%")
    print(f"[ts-recovery] wrote {plot_path}")


if __name__ == "__main__":
    main()
