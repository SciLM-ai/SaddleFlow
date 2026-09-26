"""Batched Dimer-style throws: the same starts, integrator and outputs as dump_throws2.py, but B throws
per UMA forward instead of one. The noise draws are made in the SAME order with the SAME generator as
dump_throws2.py, so for equal --seed/--shard the starts are bit-identical and the two scripts can be
compared endpoint-by-endpoint.

  SF_NAPPLY=5 python dump_throws_batched.py --data-glob '<sub>/*.aselmdb' --subset oc22 --split test \
      --restrict-tids <npz> --seed 0 --ckpt <ckpt> --sides 0 --sigma 0.3 --K 4 --tag ev --outdir <dir> \
      --shard 0 --nshards 1 --batch-size 32

Outputs <tag>_<shard>.traj (endpoints, one frame per throw, same order), <tag>_<shard>_x0.npz (starts) and,
with SF_SAVE_PASSES=1, <tag>_<shard>_passes.npz (endpoint after every pass). --limit N integrates only the
first N throws (validation against the unbatched script).
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import load_official_splits                                  # noqa: E402
from dump_throws3 import _parse_task_map                                    # noqa: E402
from eval_full_testset_K10 import load_model                                # noqa: E402
from saddleflow.data.materials_saddles_dataset import MaterialsSaddlesDataset  # noqa: E402
from saddleflow.data.transforms import mic_displacement, wrap_positions     # noqa: E402
from saddleflow.flow.matching import apply_output_projections, build_atomic_data  # noqa: E402


def flow_batch(model, xs, recs, K, dc, device, vscale=1.0):
    """K Euler steps for a list of throws (one system each) with ONE backbone/head call per step."""
    B = len(xs); cells = [r["cell"] for r in recs]
    fixed_cat = torch.cat([r["fixed"] for r in recs]).to(device)
    sizes = [int(r["Z"].shape[0]) for r in recs]
    is_filmed = "TimeFiLM" in type(model.backbone).__name__
    with torch.no_grad():
        for step in range(K):
            t = step / K
            tt = torch.full((B,), float(t), dtype=torch.float32, device=device)
            dl = [build_atomic_data(xs[i], recs[i]["Z"], cells[i], recs[i]["task_name"],
                                    recs[i]["charge"], recs[i]["spin"], recs[i]["fixed"]) for i in range(B)]
            b = data_list_collater(dl, otf_graph=True).to(device)
            feat = model.backbone(b, tt, b.batch) if is_filmed else model.backbone(b)
            h = feat["node_embedding"]
            if model.global_attn is not None:
                h = model.global_attn(h, b.batch)
            if dc > 0:
                delta = torch.cat([torch.stack([mic_displacement(recs[i]["start_pos"], xs[i], cells[i]),
                                                mic_displacement(recs[i]["partner_un_pos"], xs[i], cells[i])], dim=1)
                                   for i in range(B)], dim=0).to(device)
                v = model.velocity_head(h, tt, b.batch, delta_endpoint=delta)
            else:
                v = model.velocity_head(h, tt, b.batch)
            v = apply_output_projections(v, fixed_cat, b.batch, B).cpu().float()
            vs = torch.split(v, sizes)
            xs = [wrap_positions(xs[i] + vscale * vs[i] / K, cells[i]) for i in range(B)]
    return xs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", required=True); p.add_argument("--outdir", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--subset", default="mp20bat"); p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ckpt", required=True); p.add_argument("--ckpt2", default=None); p.add_argument("--K", type=int, default=20)
    p.add_argument("--sides", default="0,1"); p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--num-cases", type=int, default=0)
    p.add_argument("--restrict-tids", default=None); p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--task-name-map", default=None); p.add_argument("--use-ema", action="store_true")
    p.add_argument("--batch-size", type=int, default=32, help="throws per UMA forward")
    p.add_argument("--limit", type=int, default=0, help="integrate only the first N throws (validation)")
    a = p.parse_args(); device = "cuda"; sides = [int(s) for s in a.sides.split(",")]

    tids = [int(t) for t in load_official_splits(a.subset)[{"train": 0, "val": 1, "test": 2}[a.split]]]
    if a.restrict_tids:
        keep = set(np.load(a.restrict_tids)["tids"].tolist()); tids = [t for t in tids if t in keep]
    if a.num_cases and a.num_cases < len(tids):
        rng = np.random.default_rng(a.seed); tids = sorted(rng.choice(tids, a.num_cases, replace=False).tolist())
    tids = tids[a.shard::a.nshards]
    ds = MaterialsSaddlesDataset(sorted(glob.glob(a.data_glob)), task_name_map=_parse_task_map(a.task_name_map))
    m1, cfg = load_model(Path(a.ckpt), device, use_ema=a.use_ema); dc = int(cfg["extras"].get("delta_endpoint_channels") or 0)
    m2 = load_model(Path(a.ckpt2), device, use_ema=a.use_ema)[0] if a.ckpt2 else None
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    # 1. draw every start in dump_throws2.py's order with its generator -> identical starts
    g = torch.Generator().manual_seed(a.seed * 100003 + a.shard)
    T, Sd, X0, R = [], [], [], []
    for tid in tids:
        for side in sides:
            rec = ds[int(2 * tid + side)]; mobile = (~rec["fixed"]).float().unsqueeze(1)
            x0 = wrap_positions(rec["start_pos"] + a.sigma * torch.randn(rec["start_pos"].shape, generator=g) * mobile, rec["cell"])
            T.append(int(tid)); Sd.append(int(side)); X0.append(x0); R.append(rec)
    if a.limit: T, Sd, X0, R = T[:a.limit], Sd[:a.limit], X0[:a.limit], R[:a.limit]
    n_ap = int(os.environ.get("SF_NAPPLY", "1")); n_ap2 = int(os.environ.get("SF_NAPPLY2", "1"))
    vsc = float(os.environ.get("SF_VSCALE", "1.0")); save_p = os.environ.get("SF_SAVE_PASSES", "0") == "1"
    if float(os.environ.get("SF_RESTART_SIGMA", "0")) > 0:
        raise SystemExit("SF_RESTART_SIGMA is not supported in the batched dumper (it would change the draw order)")

    # 2. integrate in chunks of --batch-size throws, all passes
    out = Trajectory(f"{a.outdir}/{a.tag}_{a.shard:02d}.traj", "w"); PASS = []; t0 = time.time(); nfwd = 0
    for s0 in range(0, len(T), a.batch_size):
        idx = list(range(s0, min(s0 + a.batch_size, len(T)))); xs = [X0[i].clone() for i in idx]; recs = [R[i] for i in idx]
        trace = [[x.numpy().astype(np.float32)] for x in xs] if save_p else None
        for _ in range(n_ap):
            xs = flow_batch(m1, xs, recs, a.K, dc, device, vsc); nfwd += a.K
            if save_p:
                for j, x in enumerate(xs): trace[j].append(x.numpy().astype(np.float32))
        if m2 is not None:
            for _ in range(n_ap2):
                xs = flow_batch(m2, xs, recs, a.K, 0, device, vsc); nfwd += a.K
                if save_p:
                    for j, x in enumerate(xs): trace[j].append(x.numpy().astype(np.float32))
        for j, i in enumerate(idx):
            rec = R[i]; at = Atoms(positions=xs[j].numpy().astype(float), numbers=rec["Z"].numpy(), cell=rec["cell"].numpy().astype(float), pbc=True)
            fx = torch.where(rec["fixed"])[0].tolist()
            if fx: at.set_constraint(FixAtoms(indices=fx))
            at.info.update(tid=T[i], side=Sd[i], sigma=float(a.sigma), src=f"{a.tag}:s{Sd[i]}"); out.write(at)
            if save_p: PASS.append(np.stack(trace[j], 0))
        print(f"  {a.tag} shard{a.shard}: {idx[-1]+1}/{len(T)} throws  {time.time()-t0:.0f}s  "
              f"({nfwd*len(idx)/(time.time()-t0):.1f} system-forwards/s)", flush=True)
    out.close()
    off = np.concatenate([[0], np.cumsum([len(x) for x in X0])])
    np.savez(f"{a.outdir}/{a.tag}_{a.shard:02d}_x0.npz", tids=np.array(T), sides=np.array(Sd), offsets=off, x0=np.concatenate([x.numpy().astype(np.float32) for x in X0], 0))
    if PASS:
        np.savez_compressed(f"{a.outdir}/{a.tag}_{a.shard:02d}_passes.npz", tids=np.array(T), sides=np.array(Sd), offsets=off,
                            npass=np.array([t.shape[0] for t in PASS]), passes=np.concatenate([t.reshape(-1, 3) for t in PASS], 0))
    print(f"{a.tag} shard {a.shard}: wrote {len(T)} throws (batch {a.batch_size}, {nfwd} step-batches, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
