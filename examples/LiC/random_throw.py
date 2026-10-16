"""Throw random Li positions onto the frozen sheet and flow each to a saddle.

This is the open-ended test of an unconditioned (TS-denoise) model: it is given
no reactant, no product, and no hint of which saddle to aim at -- just a Li
somewhere on the sheet -- and should relax to whichever real saddle is nearest.

Sampling: by default uniform x, y over the full cell, z at the mean Li adsorption
height from the training minima plus a small Gaussian jitter. `--grid N` instead
puts the starts on a regular N x N grid at that height; `--grid 64` is what
README.md plots. Carbons are frozen and keep their reference positions throughout.

Scoring compares each endpoint against the union of TRAIN and TEST saddles under
PBC, and reports separately for the two, so you can see whether the model reaches
saddles it never trained on. It also reports how far the *start* already was, so
"flowed to a saddle" is not confused with "was thrown next to one".

Run from the repository root, after train.py:

    python examples/LiC/random_throw.py --grid 64 \\
        --ckpt examples/LiC/runs/tsdenoise_sigma0.25/checkpoint_final

The full Li path of every throw goes to <run>/random_throw/throw.npz; draw it
with plot_throws.py.
"""
import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parent  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import Trajectory

from saddleflow.data import atoms_to_sample_dict
from saddleflow.data.transforms import wrap_positions
try:
    from force_guided_sampling import load_flow_model, model_velocity
    from conditioning import conditioning_queries
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model, model_velocity
    from examples.LiC.rollout_eval.conditioning import conditioning_queries

LI = 126


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint dir (config.json read from its parent)")
    p.add_argument("--train-traj", default=None)
    p.add_argument("--test-traj", default=None)
    p.add_argument("--n", type=int, default=2000, help="number of random starts")
    p.add_argument("--grid", type=int, default=0,
                   help="If >0, ignore --n/--z-jitter and start from a regular GRIDxGRID "
                        "of cell-centred xy positions at the mean adsorption height.")
    p.add_argument("--K", type=int, default=20, help="Euler steps per trajectory")
    p.add_argument("--z-jitter", type=float, default=0.3, help="Gaussian sigma on z (A)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hit-tol", type=float, default=0.10,
                   help="A; endpoint counts as reaching a saddle (same radius as plot_throws.py)")
    p.add_argument("--seed", type=int, default=0)
    # EMA by default: load_ema_weights' safetensors path only knows the
    # "global_attn"/"velocity_head" prefixes, so it CANNOT restore an unfrozen
    # backbone -- it would leave the backbone at pretrained init. The EMA path
    # walks the module list by parameter order and handles any stack.
    p.add_argument("--no-ema", dest="use_ema", action="store_false",
                   help="Load raw weights instead of the EMA shadow. Not available "
                        "for runs with an unfrozen backbone (see above).")
    p.set_defaults(use_ema=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def li_sites(traj):
    """Return (minima, saddles) Li positions from a flat [R,S,P,...] trajectory."""
    frames = list(Trajectory(traj))
    mins = np.array([frames[i].get_positions()[LI] for i in range(0, len(frames), 3)]
                    + [frames[i + 2].get_positions()[LI] for i in range(0, len(frames), 3)])
    sad = np.array([frames[i + 1].get_positions()[LI] for i in range(0, len(frames), 3)])
    return mins, sad


def min_image_dist(points, refs, cell):
    """Per-point distance to the nearest ref under PBC (3x3 image search in a,b)."""
    A = np.asarray(cell)
    out = np.full(len(points), np.inf)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            off = i * A[0] + j * A[1]
            d = np.linalg.norm(points[:, None, :] - (refs[None, :, :] + off), axis=2)
            out = np.minimum(out, d.min(axis=1))
    return out


def main():
    args = parse_args()
    dev = args.device
    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    ex = cfg.get("extras", {})
    try: from conditioning import catalog_paths
    except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import catalog_paths
    args.train_traj, args.test_traj = catalog_paths(ex, args.train_traj, args.test_traj)
    attn_layers = int(ex.get("attn_layers", 0)); attn_heads = int(ex.get("attn_heads", 8))
    head_depth = int(ex.get("head_depth", 1)); delta_C = int(ex.get("delta_endpoint_channels", 0) or 0)
    tfilm = bool(ex.get("early_time_film", False))
    uma_unfrozen = bool(ex.get("unfreeze_uma_all", False) or ex.get("unfreeze_uma_last", False)
                        or ex.get("unfreeze_uma_blocks"))
    print(f"[throw] cfg: attn={attn_layers} depth={head_depth} delta_C={delta_C} "
          f"tfilm={tfilm} uma_unfrozen={uma_unfrozen}")
    if delta_C:
        raise SystemExit("this script is for UNCONDITIONED models (delta_endpoint_channels=0); "
                         "a conditioned model needs an (R, P) pair, not a random start.")

    ref = list(Trajectory(args.train_traj))[0]
    cell = np.array(ref.get_cell())
    tr_min, tr_sad = li_sites(args.train_traj)
    te_min, te_sad = li_sites(args.test_traj)
    all_sad = np.vstack([tr_sad, te_sad])
    z0 = float(np.mean(np.vstack([tr_min, te_min])[:, 2]))
    print(f"[throw] saddles: {len(tr_sad)} train + {len(te_sad)} test = {len(all_sad)}; "
          f"Li adsorption height z = {z0:.3f} A")

    _, _, backbone, attn, head = load_flow_model(ckpt, dev, use_ema=args.use_ema)

    rng = np.random.default_rng(args.seed)
    base = ref.get_positions().copy()
    if args.grid > 0:
        g = (np.arange(args.grid) + 0.5) / args.grid          # cell-centred fractions
        gx, gy = np.meshgrid(g, g, indexing="ij")
        frac = np.stack([gx.ravel(), gy.ravel()], 1)
        args.n = len(frac)
        z = np.full(args.n, z0)                                # no z variation
        print(f"[throw] regular {args.grid}x{args.grid} grid: {args.n} starts at z={z0:.3f}")
    else:
        frac = rng.random((args.n, 2))
        z = z0 + args.z_jitter * rng.standard_normal(args.n)
    xy = frac @ np.asarray(cell)[:2, :2]
    starts = np.repeat(base[None], args.n, axis=0)
    starts[:, LI, :2] = xy
    starts[:, LI, 2] = z

    sample = atoms_to_sample_dict(ref)
    cell_t = sample["cell"]
    finals = np.zeros((args.n, 3), dtype=np.float64)
    # Full Li path per throw, so trajectories can be drawn later without re-running.
    paths = np.zeros((args.n, args.K + 1, 3), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, args.n, args.batch_size):
            s1 = min(s0 + args.batch_size, args.n)
            x = torch.tensor(starts[s0:s1], dtype=torch.float32)
            x = torch.stack([wrap_positions(x[i], cell_t) for i in range(len(x))])
            paths[s0:s1, 0] = x[:, LI, :].numpy()
            source_kwargs = {"source_positions": x.clone()} if getattr(head, "source_conditioning", False) else {}
            for k in range(args.K):
                t = k / args.K
                v = model_velocity(x, t, sample, backbone, attn, head, dev, **source_kwargs)
                x = torch.stack([wrap_positions(x[i] + v[i].float() / args.K, cell_t)
                                 for i in range(len(x))])
                paths[s0:s1, k + 1] = x[:, LI, :].numpy()
            finals[s0:s1] = x[:, LI, :].numpy()
            print(f"  {s1}/{args.n}", flush=True)

    d_start = min_image_dist(starts[:, LI, :], all_sad, cell)
    d_end = min_image_dist(finals, all_sad, cell)
    d_end_tr = min_image_dist(finals, tr_sad, cell)
    d_end_te = min_image_dist(finals, te_sad, cell)
    hit = d_end < args.hit_tol
    out_dir = Path(args.out_dir or (ckpt.parent / "random_throw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "throw.npz", starts=starts[:, LI, :], finals=finals, paths=paths,
             d_start=d_start, d_end=d_end, d_end_train=d_end_tr, d_end_test=d_end_te,
             train_saddles=tr_sad, test_saddles=te_sad, cell=cell,
             conditioning_pes_calls=np.asarray(conditioning_queries(head)))
    print(f"[throw] online conditioning PES geometries={conditioning_queries(head)}; no force-guidance correction")
    qs = [0, 1, 5, 10, 50, 90, 95, 99, 100]
    print(f"\n[throw] {args.n} random starts, K={args.K}")
    print("  distance to the nearest dataset saddle (A):")
    hdr = "".join(f"{('min' if q==0 else 'max' if q==100 else f'p{q}'):>9s}" for q in qs)
    print(f"    {'':<7s}{hdr}")
    for nm, arr in (("START", d_start), ("END", d_end)):
        print(f"    {nm:<7s}" + "".join(f"{np.percentile(arr,q):>9.3f}" for q in qs))
    print(f"  reached a saddle (< {args.hit_tol} A): {100*hit.mean():.1f}%")
    if hit.any():
        print(f"  of those, nearest was a TEST saddle (unseen in training): "
              f"{100*np.mean(d_end_te[hit] <= d_end_tr[hit]):.1f}%")
    print(f"  wrote {out_dir/'throw.npz'}")


if __name__ == "__main__":
    main()
