"""Dimer-style throws: start from a REACTANT (R and/or P of each triplet) plus Gaussian noise,
flow to a saddle with an unconditional model, and write every endpoint as one frame
(positions = endpoint, arrays['x0'] = the noisy start, info tid/side/sigma/src).
Feed the output to SaddleMill (method = Sella) and then to `sm_collect.py` / `build_pairs.py`.

  python dump_throws.py --data-glob '<mp20bat>/*.aselmdb' --split train --num-cases 5000 --seed 1 \
      --ckpt <D1 ckpt> --sides 0,1 --sigma 0.3 --K 20 --tag thr --outdir <dir> --shard i --nshards n
"""
from __future__ import annotations
import os
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import load_official_splits                       # noqa: E402
from dump_predictions import run_flow, run_flow_ho              # noqa: E402
from eval_full_testset_K10 import load_model                      # noqa: E402
from saddleflow.data.materials_saddles_dataset import MaterialsSaddlesDataset  # noqa: E402


def _parse_task_map(spec):
    """'oc22=oc20,oc25=oc20' -> {'oc22': 'oc20', 'oc25': 'oc20'}.  Needed because uma-m-1p1 carries only
    {omat, oc20, omol, odac, omc} and raises KeyError('oc22'), while uma-s-1p2 also has oc22/oc25."""
    if not spec: return {}
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part: continue
        k, _, v = part.partition("=")
        if not k or not v: raise ValueError(f"bad --task-name-map entry {part!r}, expected SRC=DST")
        out[k.strip()] = v.strip()
    return out
from saddleflow.data.transforms import wrap_positions             # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", required=True); p.add_argument("--outdir", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--subset", default="mp20bat"); p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--ckpt", required=True); p.add_argument("--ckpt2", default=None); p.add_argument("--K", type=int, default=20)
    p.add_argument("--sides", default="0,1", help="0 = start from R, 1 = start from P; '0,1' throws from both")
    p.add_argument("--sigma", type=float, default=0.3, help="Gaussian sigma (A) on every mobile-atom coordinate of the start")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--num-cases", type=int, default=0)
    p.add_argument("--restrict-tids", default=None); p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--task-name-map", default=None,
                   help="Remap UMA task names, e.g. 'oc22=oc20' (uma-m-1p1 has no oc22 expert).")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--integrator", default="euler", choices=["euler", "heun", "rk4"],
                   help="heun/rk4 cost 2x/4x field evals per step; compare at equal EVALS, not equal K")
    a = p.parse_args(); device = "cuda"; sides = [int(s) for s in a.sides.split(",")]

    tids = [int(t) for t in load_official_splits(a.subset)[{"train": 0, "val": 1, "test": 2}[a.split]]]
    if a.restrict_tids:
        keep = set(np.load(a.restrict_tids)["tids"].tolist()); tids = [t for t in tids if t in keep]
    if a.num_cases and a.num_cases < len(tids):
        rng = np.random.default_rng(a.seed); tids = sorted(rng.choice(tids, a.num_cases, replace=False).tolist())
    tids = tids[a.shard::a.nshards]
    ds = MaterialsSaddlesDataset(sorted(glob.glob(a.data_glob)),
                                 task_name_map=_parse_task_map(getattr(a, 'task_name_map', None)))
    m1, cfg = load_model(Path(a.ckpt), device, use_ema=a.use_ema); dc = int(cfg["extras"].get("delta_endpoint_channels") or 0)
    m2 = load_model(Path(a.ckpt2), device, use_ema=a.use_ema)[0] if a.ckpt2 else None
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    out = Trajectory(f"{a.outdir}/{a.tag}_{a.shard:02d}.traj", "w")
    g = torch.Generator().manual_seed(a.seed * 100003 + a.shard)
    X0T, X0S, X0P = [], [], []          # starts are also saved to <tag>_<shard>_x0.npz (ASE traj drops custom arrays)
    for i, tid in enumerate(tids):
        for side in sides:
            rec = ds[int(2 * tid + side)]; cell = rec["cell"]
            mobile = (~rec["fixed"]).float().unsqueeze(1)
            x0 = wrap_positions(rec["start_pos"] + a.sigma * torch.randn(rec["start_pos"].shape, generator=g) * mobile, cell)
            _np = int(os.environ.get("SF_NAPPLY", "1"))   # apply the stage-1 flow this many times
            x = x0.clone()
            for _i in range(_np):
                if a.integrator == "euler":
                    x = run_flow(m1, x, rec, cell, a.K, dc, device)
                else:
                    x = run_flow_ho(m1, x, rec, cell, a.K, dc, device, a.integrator)
            if m2 is not None:
                for _i in range(int(os.environ.get("SF_NAPPLY2", "1"))):
                    if a.integrator == "euler":
                        x = run_flow(m2, x, rec, cell, a.K, 0, device)
                    else:
                        x = run_flow_ho(m2, x, rec, cell, a.K, 0, device, a.integrator)
            at = Atoms(positions=x.numpy().astype(float), numbers=rec["Z"].numpy(), cell=cell.numpy().astype(float), pbc=True)
            X0T.append(int(tid)); X0S.append(int(side)); X0P.append(x0.numpy().astype(np.float32))
            fixed = torch.where(rec["fixed"])[0].tolist()
            if fixed: at.set_constraint(FixAtoms(indices=fixed))
            at.info.update(tid=int(tid), side=int(side), sigma=float(a.sigma), src=f"{a.tag}:s{side}")
            out.write(at)
        if (i + 1) % 100 == 0: print(f"  {a.tag} shard{a.shard}: {i + 1}/{len(tids)} triplets", flush=True)
    out.close()
    off = np.concatenate([[0], np.cumsum([len(x) for x in X0P])]) if X0P else np.zeros(1, int)
    np.savez(f"{a.outdir}/{a.tag}_{a.shard:02d}_x0.npz", tids=np.array(X0T), sides=np.array(X0S), offsets=off, x0=(np.concatenate(X0P, 0) if X0P else np.zeros((0, 3), np.float32)))
    print(f"{a.tag} shard {a.shard}: wrote {len(tids)} triplets x {len(sides)} sides")


if __name__ == "__main__":
    main()
