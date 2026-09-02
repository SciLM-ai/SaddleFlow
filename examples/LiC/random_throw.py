"""Throw random Li positions onto the frozen sheet and flow each to a saddle.

This is the open-ended test of an unconditioned (TS-denoise) model: it is given
no reactant, no product, and no hint of which saddle to aim at -- just a Li
somewhere on the sheet -- and should relax to whichever real saddle is nearest.

Sampling: uniform x, y over the full cell, z at the mean Li adsorption height
from the training minima plus a small Gaussian jitter. Carbons are frozen and
keep their reference positions throughout.

Scoring compares each endpoint against the union of TRAIN and TEST saddles under
PBC, and reports separately for the two, so you can see whether the model reaches
saddles it never trained on. It also reports how far the *start* already was, so
"flowed to a saddle" is not confused with "was thrown next to one".
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import Trajectory
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.data import atoms_to_sample_dict
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.data.transforms import wrap_positions
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import load_ema_weights, load_uma_backbone

LI = 126


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint dir (config.json read from its parent)")
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))
    p.add_argument("--n", type=int, default=2000, help="number of random starts")
    p.add_argument("--K", type=int, default=20, help="Euler steps per trajectory")
    p.add_argument("--z-jitter", type=float, default=0.3, help="Gaussian sigma on z (A)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hit-tol", type=float, default=0.35, help="A; endpoint counts as reaching a saddle")
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

    backbone = load_uma_backbone("uma-s-1p2", device=dev, freeze=True, eval_mode=True)
    # The EMA loader matches trainable params by count AND order, so the
    # unfreeze pattern here must reproduce training exactly.
    unfreeze_blocks = ex.get("unfreeze_uma_blocks")
    if unfreeze_blocks:
        for i in unfreeze_blocks:
            for p_ in backbone.blocks[i].parameters():
                p_.requires_grad_(True)
        print(f"[throw] UMA blocks {sorted(unfreeze_blocks)} unfrozen (from config)")
    elif uma_unfrozen:
        for blk in backbone.blocks:
            for p_ in blk.parameters():
                p_.requires_grad_(True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    if tfilm:
        idx = [int(x) for x in str(ex.get("early_time_film_blocks", "0,1,2,3")).split(",")]
        backbone = TimeFiLMBackbone(backbone, inject_block_indices=idx, inject_force=False).to(dev)
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax, num_heads=attn_heads,
                      num_layers=attn_layers).to(dev)
    head = VelocityHead(sphere_channels=sc, input_lmax=lmax, depth=head_depth).to(dev)
    mods = [backbone, attn, head] if (uma_unfrozen or tfilm) else [attn, head]
    if not args.use_ema and (uma_unfrozen or tfilm):
        raise SystemExit("--no-ema cannot restore an unfrozen/time-FiLM backbone; "
                         "the raw-weights loader only handles global_attn and "
                         "velocity_head. Drop --no-ema.")
    load_ema_weights(str(ckpt), mods, device=dev, use_ema=args.use_ema)
    for m in mods:
        m.eval()

    rng = np.random.default_rng(args.seed)
    base = ref.get_positions().copy()
    frac = rng.random((args.n, 2))
    xy = frac @ np.asarray(cell)[:2, :2]
    z = z0 + args.z_jitter * rng.standard_normal(args.n)
    starts = np.repeat(base[None], args.n, axis=0)
    starts[:, LI, :2] = xy
    starts[:, LI, 2] = z

    sample = atoms_to_sample_dict(ref)
    fixed = sample["fixed"].to(dev)
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
            for k in range(args.K):
                t = k / args.K
                dl = [build_atomic_data(x[i], sample["Z"], cell_t, sample["task_name"],
                                        sample["charge"], sample["spin"], sample["fixed"])
                      for i in range(len(x))]
                b = data_list_collater(dl, otf_graph=True).to(dev)
                tt = torch.full((len(x),), float(t), dtype=torch.float32, device=dev)
                feat = (backbone(b, tt, b.batch)
                        if "TimeFiLM" in type(backbone).__name__ else backbone(b))
                h = attn(feat["node_embedding"], b.batch)
                v = head(h, tt, b.batch)
                v = apply_output_projections(v, fixed.repeat(len(x)), b.batch, len(x))
                v = v.view(len(x), -1, 3).cpu()
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
             train_saddles=tr_sad, test_saddles=te_sad, cell=cell)
    qs = [0, 1, 5, 10, 50, 90, 95, 99, 100]
    print(f"\n[throw] {args.n} random starts, K={args.K}")
    print("  distance to the nearest dataset saddle (A):")
    hdr = "".join(f"{('min' if q==0 else 'max' if q==100 else f'p{q}'):>9s}" for q in qs)
    print(f"    {'':<7s}{hdr}")
    for nm, arr in (("START", d_start), ("END", d_end)):
        print(f"    {nm:<7s}" + "".join(f"{np.percentile(arr,q):>9.3f}" for q in qs))
    print(f"  reached a saddle (< {args.hit_tol} A): {100*hit.mean():.1f}%")
    print(f"  of those, nearest was a TEST saddle (unseen in training): "
          f"{100*np.mean(d_end_te[hit] <= d_end_tr[hit]):.1f}%")
    print(f"  wrote {out_dir/'throw.npz'}")


if __name__ == "__main__":
    main()
