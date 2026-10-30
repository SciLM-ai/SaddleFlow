"""Near-saddle frames for the hint fine-tune: the pilot's own landing points on the Tier B training minima.

Starts: 64 points of the hint grid per minimum of labels.json (eval_hints.grid: every atom x 16 directions,
subsampled), the minimum + 0.15 A noise, the hinted atom pushed 0.6 A (the evaluation's hint start). Landing:
the fine-tuned checkpoint, hint given, 1 pass x 8 Euler steps (the evaluation's ft_hint_1x8).

A landing is a frame of a region when that region's saddle is within RADIUS of it and every other region of the
minimum (held-out ones included) is farther than RADIUS; the region must be a training region with barrier
>= 0.05 eV. Per region at most PER_REGION frames, spread out (farthest-point). Each frame is stored unwrapped
next to its saddle, overall shift removed, with the region's hint rows (finetune.py --frames).

    python examples/MaterialsSaddles/experiments/hint_finetune/frames.py --out-dir <dir> --ckpt <checkpoint dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_hints as ev  # noqa: E402
from eval_hints import hl, write_rows  # noqa: E402

N_STARTS, RADIUS, PER_REGION, LOW = 64, 0.3, 5, 0.05


def spread(m, Y, n):
    """Indices of n points of Y, farthest-point from the one farthest from the saddle (Y[0] is the saddle)."""
    pick = [1 + int(np.argmax(hl.saddle_distance(m, Y[1:], Y[0])))]
    while len(pick) < min(n, len(Y) - 1):
        gap = np.min([hl.saddle_distance(m, Y[1:], Y[p]) for p in pick], axis=0)
        pick.append(1 + int(np.argmax(gap)))
    return [p - 1 for p in pick]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", default=f"{ev.DATA}/hint_finetune/runs/pilot_21872158/checkpoint_final")
    p.add_argument("--labels", default=ev.LABELS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    entries = json.loads(Path(args.labels).read_text())["minima"]
    jobs = []
    for e in entries:
        m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
        for a, k, d in ev.grid(e["tid"], m, N_STARTS):
            pos = m.positions + ev.HINT_NOISE * np.random.default_rng([e["tid"], 21, a, k]).normal(size=m.positions.shape)
            pos[a] += ev.PUSH * d
            h = np.zeros((len(m), 3)); h[a] = d
            jobs.append((f"{e['tid']}:{a}:{k}", e["tid"], m, pos, h, {}))

    path = out / "landings.json"
    if not path.exists():
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = ev.load_finetuned(args.ckpt, args.device)
        lands = ev.roll_finetuned(model, jobs, True, 1, 8, args.batch_size)
        write_rows(path, [dict(id=j[0], tid=j[1], landing=np.round(x, 5).tolist()) for j, x in zip(jobs, lands)])
    lands = {l["id"]: np.array(l["landing"]) for l in json.loads(path.read_text())}

    frames, near, n_regions = [], 0, 0
    for e in entries:
        m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
        regs = e["regions"]
        P = np.array([r["saddle"] for r in regs])
        found = {}
        for jid, tid, *_ in jobs:
            if tid != e["tid"] or not regs:
                continue
            d = hl.saddle_distance(m, P, lands[jid])
            k = int(d.argmin())
            near += d[k] <= RADIUS
            if d[k] <= RADIUS and np.sum(d <= RADIUS) == 1 and not regs[k]["held_out"] and regs[k]["barrier"] >= LOW:
                found.setdefault(k, []).append(lands[jid])
        for k, X in found.items():
            S = np.array(regs[k]["saddle"])
            Y = [S]
            for x in X:                                          # unwrapped next to S, overall shift removed
                u = hl.mic(m.cell.array, x - S); Y.append(S + u - u.mean(0))
            Y = np.array(Y)
            for i in spread(m, Y, PER_REGION):
                frames.append(dict(tid=e["tid"], region=regs[k]["region"], numbers=e["numbers"], cell=e["cell"],
                                   frame=np.round(Y[1 + i], 5).tolist(), saddle=regs[k]["saddle"], hints=regs[k]["hints"],
                                   distance=float(hl.saddle_distance(m, Y[[0]], Y[1 + i])[0])))
        n_regions += len(found)
    (out / "frames.json").write_text(json.dumps(frames))
    dist = [f["distance"] for f in frames]
    train = sum(not r["held_out"] and r["barrier"] >= LOW for e in entries for r in e["regions"])
    text = (f"{len(jobs)} landings from {args.ckpt} on {len(entries)} minima; {near} within {RADIUS} A of a region\n"
            f"frames: {len(frames)} on {n_regions} of {train} training regions (<= {PER_REGION} per region); "
            f"distance to the saddle p10 / p50 / p90 {np.percentile(dist, 10):.3f} / {np.median(dist):.3f} / "
            f"{np.percentile(dist, 90):.3f} A\n")
    (out / "frames.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
