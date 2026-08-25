"""Integrate the flow and dump predicted saddle geometries as an ASE trajectory.

This is the front half of the Sella evaluation: it produces the candidate
geometries that sella_eval.py then optimises. Kept separate because the two
halves have different bottlenecks (GPU-bound batched inference vs many
independent CPU/GPU saddle searches) and shard differently.

Three prediction modes:
  --midpoint            the (R+P)/2 baseline, no model
  --ckpt A              single-stage: integrate A from the midpoint
  --ckpt A --ckpt2 B    two-stage cascade: integrate A from the midpoint, then
                        integrate B from A's output. B is a refiner trained with
                        `train.py --start-override` on A's own predictions, so its
                        input distribution matches what it sees here at test time.

Conditioning is read from the checkpoint's own config (`delta_endpoint_channels`):
conditioned models get per-atom Delta_R/Delta_P at every step, unconditioned models
get none. Stage 2 is always run unconditioned — the refiner never sees R/P.

Shard over N GPUs with --shard/--nshards; each shard writes its own trajectory.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import load_official_splits
from eval_full_testset_K10 import load_model

from saddleflow.data.materials_saddles_dataset import MaterialsSaddlesDataset
from saddleflow.data.transforms import mic_displacement, wrap_positions
from saddleflow.flow.matching import apply_output_projections, build_atomic_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-glob", required=True,
                   help="Glob for the subset's .aselmdb shards.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True, help="Names the output trajectory and info['src'].")
    p.add_argument("--subset", default="mp20bat")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ckpt", default=None, help="Stage-1 checkpoint dir.")
    p.add_argument("--ckpt2", default=None, help="Stage-2 refiner checkpoint dir.")
    p.add_argument("--midpoint", action="store_true", help="Emit (R+P)/2, ignore --ckpt.")
    p.add_argument("--K", type=int, default=10, help="Euler steps per stage.")
    p.add_argument("--use-ema", action="store_true",
                   help="Use the EMA shadow instead of the live weights.")
    p.add_argument("--restrict-tids", default=None,
                   help="npz with a 'tids' array; keep only these triplet ids. Use to "
                        "score every model on an identical case set.")
    p.add_argument("--num-cases", type=int, default=0, help="0 = all.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def run_flow(model, x, record, cell, n_steps, delta_channels, device):
    """Forward-Euler integration of the velocity field from x over n_steps."""
    with torch.no_grad():
        for step in range(n_steps):
            t = step / n_steps
            t_tensor = torch.tensor([t], device=device)
            data = build_atomic_data(x, record["Z"], cell, record["task_name"],
                                     record["charge"], record["spin"], record["fixed"])
            batch = data_list_collater([data], otf_graph=True).to(device)
            is_filmed = "TimeFiLM" in type(model.backbone).__name__
            feat = (model.backbone(batch, t_tensor, batch.batch) if is_filmed
                    else model.backbone(batch))
            h = feat["node_embedding"]
            if model.global_attn is not None:
                h = model.global_attn(h, batch.batch)
            if delta_channels > 0:
                delta = torch.stack([
                    mic_displacement(record["start_pos"], x, cell),
                    mic_displacement(record["partner_un_pos"], x, cell),
                ], dim=1).to(device)
                v = model.velocity_head(h, t_tensor, batch.batch, delta_endpoint=delta)
            else:
                v = model.velocity_head(h, t_tensor, batch.batch)
            v = apply_output_projections(v, record["fixed"].to(device), batch.batch, 1).cpu()
            x = wrap_positions(x + v.float() / n_steps, cell)
    return x


def main():
    args = parse_args()
    device = "cuda"

    tids = [int(t) for t in load_official_splits(args.subset)[
        {"train": 0, "val": 1, "test": 2}[args.split]]]
    if args.restrict_tids:
        keep = set(np.load(args.restrict_tids)["tids"].tolist())
        tids = [t for t in tids if t in keep]
    if args.num_cases:
        tids = tids[:args.num_cases]
    tids = tids[args.shard::args.nshards]

    dataset = MaterialsSaddlesDataset(sorted(glob.glob(args.data_glob)))

    model1 = model2 = None
    delta_channels = 0
    if not args.midpoint:
        model1, cfg = load_model(Path(args.ckpt), device, use_ema=args.use_ema)
        delta_channels = int(cfg["extras"].get("delta_endpoint_channels") or 0)
        if args.ckpt2:
            model2, _ = load_model(Path(args.ckpt2), device, use_ema=args.use_ema)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out = Trajectory(f"{args.outdir}/{args.tag}_{args.shard:02d}.traj", "w")
    for i, tid in enumerate(tids):
        # R->S and P->S doubling means record 2*tid is the R-anchored pair.
        record = dataset[int(2 * tid)]
        cell = record["cell"]
        x = wrap_positions(0.5 * (record["start_pos"] + record["partner_un_pos"]), cell)
        if not args.midpoint:
            x = run_flow(model1, x, record, cell, args.K, delta_channels, device)
            if model2 is not None:
                # The refiner is unconditioned by construction.
                x = run_flow(model2, x, record, cell, args.K, 0, device)
        atoms = Atoms(positions=x.numpy().astype(float), numbers=record["Z"].numpy(),
                      cell=cell.numpy().astype(float), pbc=True)
        fixed = torch.where(record["fixed"])[0].tolist()
        if fixed:
            atoms.set_constraint(FixAtoms(indices=fixed))
        atoms.info.update(tid=int(tid), src=args.tag)
        out.write(atoms)
        if (i + 1) % 100 == 0:
            print(f"  {args.tag} shard{args.shard}: {i + 1}/{len(tids)}", flush=True)
    out.close()
    print(f"{args.tag} shard {args.shard}: wrote {len(tids)}")


if __name__ == "__main__":
    main()
