"""
Per-checkpoint Li-trajectory plots for the flower-reproduction run.

For every `checkpoint_epoch_*` (and `checkpoint_final`) under --run-dir:
load the EMA shadow into a fresh attn+head, Euler-integrate
`--n-perturbations` trajectories from the perturbed reactant with the
SAME seed (so the perturbation draws are identical across checkpoints),
and plot the Li atom's xy path over the carbon sheet. Also emits a
montage figure (`flower_evolution.png`) with one panel per checkpoint so
the flower → sunburst transition is visible at a glance.

Run:
    CUDA_VISIBLE_DEVICES=0 python examples/LiC_simpler/viz_checkpoints.py \
        --run-dir examples/LiC_simpler/runs/flower_obj1_sig0p5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from saddleflow.data import TrajTripletDataset
from saddleflow.flow.sampler import sample_saddles
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import load_ema_weights, load_uma_backbone


LI_INDEX = 112     # single Li adatom
N_CARBON = 112     # C atoms occupy indices 0..111


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=str(here / "runs" / "flower_obj1_sig0p5"))
    p.add_argument("--traj", default=str(here / "one_saddle.traj"))
    p.add_argument("--sigma-inf", type=float, default=0.15,
                   help="Å. Inference-time Gaussian perturbation around r_R.")
    p.add_argument("--n-perturbations", type=int, default=48)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn-layers", type=int, default=1)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)
    p.add_argument("--unfreeze-last-block", action="store_true",
                   help="match a run trained with unfrozen blocks[-1] (EMA param order)")
    p.add_argument("--unfreeze-all-blocks", action="store_true")
    p.add_argument("--every", type=int, default=1,
                   help="plot every Nth epoch checkpoint (final always included).")
    p.add_argument("--out-dir", default=None, help="default: <run-dir>/viz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def mic(vec: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image displacement for row-vector ASE cells: frac = cart @ inv(cell)."""
    inv = np.linalg.inv(cell)
    frac = vec @ inv
    frac -= np.round(frac)
    return frac @ cell


def orbit_positions(li_r: np.ndarray, li_s: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """The 6 C6-rotated images (about z through the reactant Li) of the known saddle."""
    d = mic(li_s - li_r, cell)
    out = []
    for k in range(6):
        a = math.radians(60.0 * k)
        rot = np.array([[math.cos(a), -math.sin(a), 0.0],
                        [math.sin(a),  math.cos(a), 0.0],
                        [0.0,          0.0,         1.0]])
        out.append(li_r + rot @ d)
    return np.array(out)


def plot_panel(ax, C_xy, li_r, li_s, orbit, li_paths, title):
    ax.scatter(C_xy[:, 0], C_xy[:, 1], s=12, c="0.6", zorder=1)
    cmap = plt.get_cmap("tab20")
    for i, path in enumerate(li_paths):        # path: (K+1, 3), already MIC-unwrapped
        c = cmap(i % 20)
        ax.plot(path[:, 0], path[:, 1], "-", color=c, lw=0.9, alpha=0.65, zorder=2)
        ax.plot(path[-1, 0], path[-1, 1], "o", color=c, ms=4, zorder=3)
    ax.scatter(orbit[:, 0], orbit[:, 1], marker="*", s=170, facecolors="none",
               edgecolors="0.3", linewidths=1.2, zorder=4)
    # ring through the 6 nearest carbons = the ATOP directions (30 deg off the saddles)
    dC = np.linalg.norm(C_xy - li_r[:2], axis=1)
    for j in np.argsort(dC)[:6]:
        ax.plot([li_r[0], C_xy[j, 0]], [li_r[1], C_xy[j, 1]], "-", color="0.85", lw=1.0, zorder=0)
    ax.plot(li_s[0], li_s[1], "*", color="k", ms=15, zorder=5)
    ax.plot(li_r[0], li_r[1], "o", color="red", ms=10, mec="k", zorder=6)
    pad = 3.2
    ax.set_xlim(li_r[0] - pad, li_r[0] + pad)
    ax.set_ylim(li_r[1] - pad, li_r[1] + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TrajTripletDataset(args.traj, compute_stats=False)
    rec = ds[0]                                        # R→S record
    cell = rec["cell"].numpy()
    pos = rec["start_pos"].numpy()
    # Derive the mobile-Li index from the structure so any cell size works
    # (stock 112-C cell -> 112; small hexagonal n=4/5/6 -> 32/50/72).
    Z = rec["Z"].numpy()
    li_idx = int(np.where(Z == 3)[0][0])
    c_mask = (Z == 6)
    globals()["LI_INDEX"] = li_idx
    li_r = pos[li_idx]
    # Show carbons in the periodic image NEAREST the Li, otherwise in a small
    # cell most of the sheet falls outside the plot window and the lattice
    # (and hence bridge vs atop) is impossible to judge by eye.
    C_xy = (li_r + mic(pos[c_mask] - li_r, cell))[:, :2]
    li_s = li_r + mic(rec["saddle_un_pos"].numpy()[li_idx] - li_r, cell)
    orbit = orbit_positions(li_r, li_s, cell)

    device = args.device
    backbone = load_uma_backbone("uma-s-1p2", device=device, freeze=True, eval_mode=True,
                                 unfreeze_last_block=args.unfreeze_last_block)
    if args.unfreeze_all_blocks:
        # Match the trainer's convention (examples/MaterialsSaddles/train.py):
        # the loader exposes only unfreeze_last_block, so an all-blocks run
        # unfreezes in the caller. This must mirror training exactly -- the EMA
        # shadow is stored in trainable-parameter order, so a mismatch here
        # silently loads the wrong tensors into the wrong slots.
        for blk in backbone.blocks:
            for prm in blk.parameters():
                prm.requires_grad_(True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                      num_heads=args.attn_heads, num_layers=args.attn_layers).to(device)
    head = VelocityHead(sphere_channels=sc, input_lmax=lmax, depth=args.head_depth).to(device)

    ckpts = sorted(run_dir.glob("checkpoint_epoch_*"))[:: args.every]
    final = run_dir / "checkpoint_final"
    if final.is_dir() and final not in ckpts:
        ckpts.append(final)
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}")
    print(f"[viz] {len(ckpts)} checkpoints, {args.n_perturbations} trajectories each, "
          f"σ_inf={args.sigma_inf}, K={args.K}")

    panels = []
    for ckpt in ckpts:
        meta = json.loads((ckpt / "meta.json").read_text()) if (ckpt / "meta.json").exists() else {}
        epoch = meta.get("epoch", ckpt.name)
        ema_modules = ([backbone, attn, head] if (args.unfreeze_last_block or args.unfreeze_all_blocks)
                       else [attn, head])
        load_ema_weights(str(ckpt), ema_modules, device=device)
        attn.eval(); head.eval()
        gen = torch.Generator().manual_seed(args.seed)   # identical draws per checkpoint
        with torch.no_grad():
            _, traj = sample_saddles(
                # Unconditioned head -> partner_pos must be omitted, and the
                # sampler then starts each trajectory at the REACTANT plus the
                # sigma_inf perturbation (not the (R,P) midpoint). That is the
                # flower setup: a ring of starts around R fanning into the six
                # symmetry-equivalent saddles.
                rec, backbone, attn, head,
                sigma_inf=args.sigma_inf,
                n_perturbations=args.n_perturbations, K=args.K,
                device=device, generator=gen, return_trajectory=True,
            )
        li_traj = traj[:, :, li_idx, :].cpu().numpy()          # (K+1, n_pert, 3)
        li_paths = np.stack(
            [li_r + mic(li_traj[:, i, :] - li_r, cell) for i in range(li_traj.shape[1])]
        )                                                        # (n_pert, K+1, 3)
        np.savez(out_dir / f"li_paths_{epoch}.npz", li_paths=li_paths,
                 li_r=li_r, li_s=li_s, orbit=orbit)

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_panel(ax, C_xy, li_r, li_s, orbit, li_paths,
                   f"epoch {epoch} — σ_inf={args.sigma_inf}, n={args.n_perturbations}, K={args.K}")
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectories_epoch_{epoch}.png", dpi=160)
        plt.close(fig)
        panels.append((epoch, li_paths))
        print(f"[viz] epoch {epoch}: saved")

    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes)
    for slot, ax in enumerate(axes.flat):
        if slot < len(panels):
            epoch, li_paths = panels[slot]
            plot_panel(ax, C_xy, li_r, li_s, orbit, li_paths, f"epoch {epoch}")
        else:
            ax.axis("off")
    fig.suptitle("Flower → sunburst evolution (EMA, identical perturbation draws)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "flower_evolution.png", dpi=150)
    plt.close(fig)
    print(f"[viz] montage → {out_dir / 'flower_evolution.png'}")


if __name__ == "__main__":
    main()
