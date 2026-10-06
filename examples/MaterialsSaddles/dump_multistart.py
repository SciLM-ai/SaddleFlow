"""Multi-start dump for round-2 retargeting (the LiC recipe on MaterialsSaddles).

For every triplet of a split, integrate the flow (optionally a two-stage cascade)
from the (R+P)/2 midpoint AND from Gaussian-perturbed midpoints (one start per
value in --sigmas, mobile atoms only), and write every ENDPOINT as one frame with
info tid / k / sigma / src. Run `sella_eval.py --check-index` on the result, then
`build_pairs.py` turns (endpoint, Sella saddle) into a `--pair-override` npz.

  python dump_multistart.py --data-glob '<mp20bat>/*.aselmdb' --split train \
      --ckpt <stage1> [--ckpt2 <refiner>] --sigmas 0,0.15,0.30 --num-cases 8000 \
      --tag r2 --outdir <dir> --shard i --nshards n
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import load_official_splits                       # noqa: E402
from dump_predictions import run_flow                             # noqa: E402
from eval_full_testset_K10 import load_model                      # noqa: E402
from saddleflow.data.materials_saddles_dataset import MaterialsSaddlesDataset  # noqa: E402
from saddleflow.data.transforms import wrap_positions             # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", required=True); p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--subset", default="mp20bat")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--ckpt", required=True); p.add_argument("--ckpt2", default=None)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--sigmas", default="0,0.15,0.30", help="one start per value (A), Gaussian on mobile atoms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-cases", type=int, default=0, help="0 = all; else a seeded random subset")
    p.add_argument("--restrict-tids", default=None)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--use-ema", action="store_true")
    a = p.parse_args()
    device = "cuda"
    sigmas = [float(x) for x in a.sigmas.split(",")]

    tids = [int(t) for t in load_official_splits(a.subset)[{"train": 0, "val": 1, "test": 2}[a.split]]]
    if a.restrict_tids:
        keep = set(np.load(a.restrict_tids)["tids"].tolist()); tids = [t for t in tids if t in keep]
    if a.num_cases and a.num_cases < len(tids):
        rng = np.random.default_rng(a.seed); tids = sorted(rng.choice(tids, a.num_cases, replace=False).tolist())
    tids = tids[a.shard::a.nshards]

    ds = MaterialsSaddlesDataset(sorted(glob.glob(a.data_glob)))
    m1, cfg = load_model(Path(a.ckpt), device, use_ema=a.use_ema)
    dc = int(cfg["extras"].get("delta_endpoint_channels") or 0)
    m2 = load_model(Path(a.ckpt2), device, use_ema=a.use_ema)[0] if a.ckpt2 else None

    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    out = Trajectory(f"{a.outdir}/{a.tag}_{a.shard:02d}.traj", "w")
    g = torch.Generator().manual_seed(a.seed * 100003 + a.shard)
    for i, tid in enumerate(tids):
        rec = ds[int(2 * tid)]; cell = rec["cell"]
        xmid = wrap_positions(0.5 * (rec["start_pos"] + rec["partner_un_pos"]), cell)
        mobile = (~rec["fixed"]).float().unsqueeze(1)
        for k, sig in enumerate(sigmas):
            x = xmid if sig == 0 else wrap_positions(xmid + sig * torch.randn(xmid.shape, generator=g) * mobile, cell)
            x = run_flow(m1, x, rec, cell, a.K, dc, device)
            if m2 is not None:
                x = run_flow(m2, x, rec, cell, a.K, 0, device)
            at = Atoms(positions=x.numpy().astype(float), numbers=rec["Z"].numpy(), cell=cell.numpy().astype(float), pbc=True)
            fixed = torch.where(rec["fixed"])[0].tolist()
            if fixed: at.set_constraint(FixAtoms(indices=fixed))
            at.info.update(tid=int(tid), k=int(k), sigma=float(sig), src=f"{a.tag}:k{k}")
            out.write(at)
        if (i + 1) % 100 == 0: print(f"  {a.tag} shard{a.shard}: {i + 1}/{len(tids)} cases", flush=True)
    out.close()
    print(f"{a.tag} shard {a.shard}: wrote {len(tids)} cases x {len(sigmas)} starts")


if __name__ == "__main__":
    main()
