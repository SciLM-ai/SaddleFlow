"""random_throw.py for a release directory: throw Li starts onto the sheet and flow each with SaddleFlow.from_pretrained.

    python examples/LiC/fm2_recipe/throw_release.py --model <release> --grid 64 --out-dir <dir>

random_throw.py loads LiC train.py checkpoints; this loads what export_pretrained.py writes (FM2's pipeline: the flag
input, and conv-layer separation when the config asks for it), so the models of run_lic_conv_layer_separation.sh can be
drawn with plot_throws.py. The starts are random_throw.py's (--grid N: an N x N grid over the cell at the Li minima's mean
height, no noise), with the Li flagged as the moving atom. The flow is the model's own inference schedule (its config's
passes x steps, each pass restarting at t = 0 from the previous endpoint); every Euler step's Li position is saved, so a
path has passes x steps + 1 points. throw.npz has random_throw.py's keys, and the printed summary is the same.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import Trajectory
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

HERE = Path(__file__).resolve().parent.parent                # examples/LiC
sys.path.insert(0, str(HERE))
from random_throw import LI, li_sites, min_image_dist      # noqa: E402

from saddleflow import SaddleFlow                            # noqa: E402
from saddleflow.data.transforms import wrap_positions        # noqa: E402
from saddleflow.flow.matching import apply_output_projections, build_atomic_data  # noqa: E402
from saddleflow.pretrained import _no_tf32                   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="release directory (config.json, model.safetensors)")
    p.add_argument("--train-traj", default=str(HERE / "train_set.traj"))
    p.add_argument("--test-traj", default=str(HERE / "test_set.traj"))
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--passes", type=int, default=None, help="default: the model's config")
    p.add_argument("--steps", type=int, default=None, help="Euler steps per pass; default: the model's config")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hit-tol", type=float, default=0.10)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


@torch.no_grad()
def flow(model, starts, rec, passes, steps, batch_size):
    """Li path of every start, (n, passes * steps + 1, 3), and the endpoints (n, N, 3)."""
    dev = model.device
    n = len(starts)
    paths = np.zeros((n, passes * steps + 1, 3), dtype=np.float32)
    finals = np.zeros_like(starts)
    fixed = rec["fixed"]
    flag_one = (~fixed).float()
    for s0 in range(0, n, batch_size):
        xs = [wrap_positions(torch.tensor(x, dtype=torch.float32), rec["cell"]) for x in starts[s0:s0 + batch_size]]
        B = len(xs)
        flag = flag_one.repeat(B).to(dev) if model.backbone.inject_flag else None
        fixed_b = fixed.repeat(B).to(dev)
        paths[s0:s0 + B, 0] = np.stack([x[LI].numpy() for x in xs])
        i = 0
        for _ in range(passes):
            for k in range(steps):
                t = torch.full((B,), k / steps, dtype=torch.float32, device=dev)
                data = [build_atomic_data(x, rec["Z"], rec["cell"], rec["task_name"], rec["charge"], rec["spin"],
                                          fixed) for x in xs]
                batch = data_list_collater(data, otf_graph=True).to(dev)
                h = model.backbone(batch, t, batch.batch, flag=flag)["node_embedding"]
                v = model.velocity_head(h, t, batch.batch)
                v = apply_output_projections(v, fixed_b, batch.batch, B).detach().cpu().float()
                xs = [wrap_positions(x + dv / steps, rec["cell"]) for x, dv in zip(xs, torch.split(v, len(rec["Z"])))]
                i += 1
                paths[s0:s0 + B, i] = np.stack([x[LI].numpy() for x in xs])
        finals[s0:s0 + B] = np.stack([x.numpy() for x in xs])
        print(f"  {s0 + B}/{n}", flush=True)
    return paths, finals


def main():
    args = parse_args()
    model = SaddleFlow.from_pretrained(args.model).eval()
    inf = model.config["inference"]
    passes = args.passes or int(inf["passes"])
    steps = args.steps or int(inf["steps"])
    print(f"[throw] {args.model}: mode {model.mode}, {passes} passes x {steps} steps")

    ref = list(Trajectory(args.train_traj))[0]
    cell = np.array(ref.get_cell())
    tr_min, tr_sad = li_sites(args.train_traj)
    te_min, te_sad = li_sites(args.test_traj)
    all_sad = np.vstack([tr_sad, te_sad])
    z0 = float(np.mean(np.vstack([tr_min, te_min])[:, 2]))

    g = (np.arange(args.grid) + 0.5) / args.grid
    gx, gy = np.meshgrid(g, g, indexing="ij")
    frac = np.stack([gx.ravel(), gy.ravel()], 1)
    n = len(frac)
    starts = np.repeat(ref.get_positions()[None], n, axis=0)
    starts[:, LI, :2] = frac @ cell[:2, :2]
    starts[:, LI, 2] = z0
    print(f"[throw] {args.grid}x{args.grid} grid: {n} starts at z={z0:.3f}; saddles {len(tr_sad)} train + "
          f"{len(te_sad)} test")

    rec = model._record(ref, None)
    with _no_tf32():
        paths, finals = flow(model, starts, rec, passes, steps, args.batch_size)
    li_end = finals[:, LI, :]
    d_start = min_image_dist(starts[:, LI, :], all_sad, cell)
    d_end = min_image_dist(li_end, all_sad, cell)
    d_end_tr = min_image_dist(li_end, tr_sad, cell)
    d_end_te = min_image_dist(li_end, te_sad, cell)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "throw.npz", starts=starts[:, LI, :], finals=li_end, paths=paths, d_start=d_start, d_end=d_end,
             d_end_train=d_end_tr, d_end_test=d_end_te, train_saddles=tr_sad, test_saddles=te_sad, cell=cell,
             conditioning_pes_calls=np.asarray(0))
    (out / "model_mode.txt").write_text(model.mode + "\n")

    hit = d_end < args.hit_tol
    qs = [0, 1, 5, 10, 50, 90, 95, 99, 100]
    print(f"\n[throw] {n} starts, {passes} x {steps}")
    print("  distance to the nearest dataset saddle (A):")
    print("    " + " " * 7 + "".join(f"{('min' if q == 0 else 'max' if q == 100 else f'p{q}'):>9s}" for q in qs))
    for nm, arr in (("START", d_start), ("END", d_end)):
        print(f"    {nm:<7s}" + "".join(f"{np.percentile(arr, q):>9.3f}" for q in qs))
    print(f"  reached a saddle (< {args.hit_tol} A): {100 * hit.mean():.1f}%")
    if hit.any():
        print(f"  of those, nearest was a TEST saddle (unseen in training): "
              f"{100 * np.mean(d_end_te[hit] <= d_end_tr[hit]):.1f}%")
    print(f"  wrote {out / 'throw.npz'}")


if __name__ == "__main__":
    main()
