"""Check that an exported model reproduces the flow endpoints of an evaluation dump from the same starts.

    python check_pretrained.py --model <export dir or SciLM/SaddleFlow> --dump <eval dir> --task omat --n 8

<eval dir> holds what dump_throws_batched.py wrote with SF_NAPPLY=5 --K 4: ev_00.traj (endpoints) and
ev_00_x0.npz (starts). The model is run from each start with no added noise and the maximum per-atom
displacement (minimum image) to the recorded endpoint is printed, for the smallest cells (--pick smallest)
or a seeded random sample (--pick random). Differences of order 1e-5 A are GPU-vs-CPU rounding; a few
sensitive cells amplify them over the passes.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from ase.io import read

from saddleflow.pretrained import SaddleFlow
from saddleflow.search import max_displacement


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--pick", choices=["smallest", "random"], default="smallest")
    a = p.parse_args()

    ends = read(f"{a.dump}/ev_00.traj", index=":")
    z = np.load(f"{a.dump}/ev_00_x0.npz")
    off, x0 = z["offsets"], z["x0"]
    if a.pick == "random":
        order = np.random.default_rng(0).choice(len(ends), min(a.n, len(ends)), replace=False)
    else:
        order = np.argsort([len(e) for e in ends], kind="stable")[: a.n]
    starts = []
    for i in order:
        s = ends[i].copy()
        s.set_positions(x0[off[i]:off[i + 1]].astype(float), apply_constraint=False)
        starts.append(s)

    model = SaddleFlow.from_pretrained(a.model, device=a.device)
    t0 = time.time()
    props = model.propose(starts, n=1, sigma=0.0, task_name=a.task)
    for i, (s, pr) in enumerate(zip(order, props)):
        print(f"throw {s}: {len(ends[s])} atoms, maxd to recorded endpoint {max_displacement(ends[s], pr[0]):.2e} A, "
              f"move from start {max_displacement(starts[i], pr[0]):.3f} A")
    print(f"{len(order)} throws in {time.time() - t0:.0f} s on {model.device}")


if __name__ == "__main__":
    main()
