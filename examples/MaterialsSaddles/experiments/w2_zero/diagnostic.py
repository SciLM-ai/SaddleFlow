"""What does W2 = 0 change in a trained model, before any retraining? Forward passes only.

For the released model or a hint fine-tune checkpoint:
    weights    ||W2|| / ||W1|| in every m > 0 SO(2) layer of the backbone (Frobenius, over all experts)
    velocity   one network call per structure at flow times t in --times, on --n starts of each mirror-test set
               (mirror_test.start_sets: the 500 noise starts and the hint-grid starts, seeded subset), with the
               stock weights and with W2 zeroed:
                 inversion error  |v(-x) + v(x)| / |v(x)|, before and after (after should be float error)
                 rotation error   the same for a 180-degree rotation about z (control)
                 change           |v_W2=0(x) - v(x)| / |v(x)|, and the largest per-atom change in A per unit time
               norms over all atoms of a structure; the table gives the median and p90 over structures
    features   |h_W2=0 - h| / |h| per degree l of the backbone output (the head reads l = 0 and 1)

The hint is given on the hint starts for a fine-tune (as in the mirror test) and is zero on the noise starts; the
released model has no hint input.

    python examples/MaterialsSaddles/experiments/w2_zero/diagnostic.py --ckpt released --out-dir <dir>
    python examples/MaterialsSaddles/experiments/w2_zero/diagnostic.py --ckpt <checkpoint dir> --out-dir <dir>

Writes <out-dir>/diagnostic.json and report.txt.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mirror_test"))
import mirror_test as mt  # noqa: E402
from mirror_test import ev  # noqa: E402


def velocities(model, jobs, t, released, hinted, batch_size):
    """[(features (N, num_sph, C), velocity (N, 3))] per job, one network call at time t."""
    import torch
    from fairchem.core.datasets.collaters.simple_collater import data_list_collater
    from saddleflow.data.transforms import wrap_positions
    from saddleflow.flow.matching import apply_output_projections, build_atomic_data
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s0 in range(0, len(jobs), batch_size):
            chunk = jobs[s0:s0 + batch_size]
            data, sizes = [], [len(j[2]) for j in chunk]
            for _, _, m, pos, _, _ in chunk:
                cell = torch.as_tensor(m.cell.array, dtype=torch.float32)
                x = wrap_positions(torch.as_tensor(pos, dtype=torch.float32), cell)
                data.append(build_atomic_data(x, torch.as_tensor(m.numbers, dtype=torch.long), cell, "omat", 0, 0,
                                              torch.zeros(len(m), dtype=torch.bool)))
            batch = data_list_collater(data, otf_graph=True).to(dev)
            B = len(chunk)
            tt = torch.full((B,), t, dtype=torch.float32, device=dev)
            if released:
                h = model.backbone(batch, tt, batch.batch)["node_embedding"]
                feat = h
            else:
                hint = torch.cat([torch.as_tensor(j[4] if hinted and j[4] is not None else np.zeros((len(j[2]), 3)),
                                                  dtype=torch.float32) for j in chunk]).to(dev)
                h = model.backbone(batch, tt, batch.batch, hint=hint)["node_embedding"]
                feat = model.global_attn(h, batch.batch)
            v = model.velocity_head(feat, tt, batch.batch)
            v = apply_output_projections(v, torch.zeros(len(v), dtype=torch.bool, device=dev), batch.batch, B)
            out += list(zip(torch.split(h.cpu().double(), sizes), torch.split(v.cpu().double(), sizes)))
    return out


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="fine-tune checkpoint directory, or 'released'")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fm-run", default=mt.FM_RUN)
    p.add_argument("--per-minimum", type=int, default=64)
    p.add_argument("--n", type=int, default=128, help="starts per set (seeded subset)")
    p.add_argument("--times", default="0,0.5,0.875")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if os.environ.get("SF_PARITY_EVEN") == "1":
        raise SystemExit("unset SF_PARITY_EVEN: the diagnostic compares the stock weights with W2 = 0 itself")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    import torch
    from saddleflow.utils.parity import assert_parity_even, w2_report, zero_w2_

    sets, _ = mt.start_sets(SimpleNamespace(fm_run=args.fm_run, per_minimum=args.per_minimum))
    rng = np.random.default_rng(0)
    sets = {k: [v[i] for i in sorted(rng.choice(len(v), min(args.n, len(v)), replace=False))] for k, v in sets.items()}
    released = args.ckpt == "released"
    model = mt.load(SimpleNamespace(ckpt=args.ckpt, device=args.device))      # stock weights (SF_PARITY_EVEN unset)
    weights = w2_report(model.backbone)
    lmax = model.backbone.backbone.lmax
    maps = {"inversion": -np.eye(3), "rotation": mt.MAPS["rotation 180 about z"]}

    def run(jobs, t, hinted):
        base = velocities(model, jobs, t, released, hinted, args.batch_size)
        mapped = {k: velocities(model, mt.mapped(jobs, R), t, released, hinted, args.batch_size) for k, R in maps.items()}
        err = {k: [rel(v.numpy() @ R, b[1].numpy()) for (_, v), b in zip(mapped[k], base)] for k, R in maps.items()}
        return base, err

    times = [float(x) for x in args.times.split(",")]
    stock = {(k, t): run(jobs, t, k == "hint") for k, jobs in sets.items() for t in times}
    with torch.no_grad():
        zero_w2_(model.backbone)
    assert_parity_even(model.backbone)
    rows = []
    for (kind, t), (base, err) in stock.items():
        base0, err0 = run(sets[kind], t, kind == "hint")
        dv = [rel(b0[1].numpy(), b[1].numpy()) for b0, b in zip(base0, base)]
        dmax = [float(np.linalg.norm(b0[1].numpy() - b[1].numpy(), axis=1).max()) for b0, b in zip(base0, base)]
        dh = {l: [rel(b0[0][:, l * l:(l + 1) ** 2].numpy(), b[0][:, l * l:(l + 1) ** 2].numpy())
                  for b0, b in zip(base0, base)] for l in range(lmax + 1)}
        rows.append(dict(set=kind, t=t, n=len(base), inv_stock=err["inversion"], rot_stock=err["rotation"],
                         inv_w2zero=err0["inversion"], rot_w2zero=err0["rotation"], dv=dv, dv_max_atom=dmax,
                         dh={str(l): v for l, v in dh.items()},
                         v_norm=[float(np.linalg.norm(b[1].numpy())) for b in base]))
    (out / "diagnostic.json").write_text(json.dumps(dict(ckpt=args.ckpt, weights=weights, rows=rows)))

    q = lambda v: f"{np.median(v):9.2e} {np.quantile(v, 0.9):9.2e}"
    L = [f"W2 = 0 diagnostic: {args.ckpt}", "=" * 100, "",
         "||W2|| / ||W1|| per SO(2) layer (m > 0):"]
    by_m = {}
    for r in weights:
        by_m.setdefault(r["m"], []).append(r["ratio"])
        L.append(f"  {r['name']:70s} m={r['m']}  {r['ratio']:.3f}")
    L += ["  " + "; ".join(f"m={m}: median {np.median(v):.3f}, range {min(v):.3f}-{max(v):.3f}" for m, v in by_m.items()), "",
          "Velocity, relative errors and changes (median, p90 over structures):",
          f"  {'set':6s} {'t':>6s} {'n':>4s} | {'inversion stock':>19s} | {'inversion W2=0':>19s} | {'rotation W2=0':>19s} "
          f"| {'|dv|/|v|':>19s} | {'max atom |dv| A':>19s}"]
    for r in rows:
        L.append(f"  {r['set']:6s} {r['t']:6.3f} {r['n']:4d} | {q(r['inv_stock'])} | {q(r['inv_w2zero'])} | "
                 f"{q(r['rot_w2zero'])} | {q(r['dv'])} | {q(r['dv_max_atom'])}")
    L += ["", "Backbone features, |h_W2=0 - h| / |h| per degree l (median, p90):"]
    for r in rows:
        L.append(f"  {r['set']:6s} {r['t']:6.3f} | " + " | ".join(f"l={l}: {q(v)}" for l, v in r["dh"].items()))
    text = "\n".join(L)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
