"""Hint-conditioned training pairs for LiC, from the classical hint search (hint_pilot.py a3).

A hint is a unit in-plane direction d on the Li (the only mobile atom). Each training pair is one A3 run
of the chosen arm (default nudge0.8, first runs only) that Sella took to a catalog saddle connected to its
minimum, restricted to the training minima:

    x_0 = the minimum with the Li moved `step` A along d, plus N(0, jitter^2) on the Li (fresh each draw)
    x_1 = the catalog saddle that run reached, minimum-image unwrapped to x_0
    hint_field = d on the Li, zero on every other atom (N, 3)

With probability 1/2 a draw is inverted (r -> -r on every atom, frozen carbons included; the hint flips
too). The energy is unchanged by inversion and -I maps every lattice onto itself, so the inverted pair is
an exact training pair; it teaches the mirror behaviour that rotation-equivariance alone does not give.

The records are the plain dicts FlowMatchingLoss reads; `flow_source_pos` makes it use x_0 as given.
"""
import argparse
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hint_pilot"))
import hint_pilot as hp  # noqa: E402

from saddleflow.data.core import mic_unwrap  # noqa: E402

DEFAULT_A3 = "/scratch/gqs7zr/saddleflow/lic_hint_pilot/a3"


def lic_context(archive=hp.ARCHIVE):
    """The coverage archive's reference structure, minima, catalog saddles and connectivity."""
    return hp.context(SimpleNamespace(archive=archive, method="n12_uncond"))


def load_pairs(a3_dir=DEFAULT_A3, ctx=None, arm="nudge0.8", minima=None):
    """(minimum, direction index, direction vector, catalog saddle) for each connected first run of `arm`."""
    ctx = ctx or lic_context()
    rows = [r for p in sorted(glob.glob(str(Path(a3_dir) / "runs_*.json"))) for r in json.loads(Path(p).read_text())]
    pairs = []
    for r in rows:
        if r["arm"] != arm or r["repeat"] != 0 or (minima is not None and r["minimum"] not in minima):
            continue
        result, saddle = hp.a3_outcome(r, ctx)
        if result == "connected":
            pairs.append((int(r["minimum"]), int(r["direction"]), np.asarray(r["vector"], float), saddle))
    return sorted(pairs, key=lambda p: (p[0], p[1]))


class HintPairDataset(Dataset):
    def __init__(self, pairs, ctx, *, step=0.3, jitter=0.02, inversion=True, seed=0):
        if not pairs:
            raise ValueError("no hint pairs")
        self.pairs, self.step, self.jitter, self.inversion = pairs, step, jitter, inversion
        self.ref = ctx["reference"].positions.astype(np.float64)
        self.li, self.cell = ctx["li"], np.asarray(ctx["cell"], float)
        self.minima, self.saddles = ctx["minima"], ctx["saddles"]
        self.Z = torch.as_tensor(ctx["reference"].numbers, dtype=torch.long)
        self.fixed = torch.as_tensor(~ctx["data"]["free_mask"].astype(bool))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        m, j, d, s = self.pairs[i]
        x0 = self.ref.copy()
        x0[self.li] = self.minima[m] + self.step * d + self.rng.normal(0.0, self.jitter, 3)
        x1 = x0.copy()
        x1[self.li] = mic_unwrap(x0[self.li], self.saddles[s], self.cell)
        hint = np.zeros_like(x0); hint[self.li] = d
        inverted = bool(self.inversion and self.rng.random() < 0.5)
        if inverted:
            x0, x1, hint = -x0, -x1, -hint
        x0, x1 = torch.as_tensor(x0, dtype=torch.float32), torch.as_tensor(x1, dtype=torch.float32)
        return {"start_pos": x0.clone(), "flow_source_pos": x0.clone(), "saddle_un_pos": x1,
                "partner_un_pos": x0.clone(), "Z": self.Z.clone(),
                "cell": torch.as_tensor(self.cell, dtype=torch.float32), "fixed": self.fixed.clone(),
                "task_name": "omat", "charge": 0, "spin": 0, "delta_norm": torch.linalg.vector_norm(x1 - x0),
                "role": "hint_to_saddle", "triplet_id": s,
                "metadata": {"minimum": m, "direction": j, "saddle": s, "inverted": inverted},
                "hint_field": torch.as_tensor(hint, dtype=torch.float32)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a3-dir", default=DEFAULT_A3)
    p.add_argument("--arm", default="nudge0.8")
    args = p.parse_args()
    ctx = lic_context()
    train_minima = set(json.loads((Path(args.a3_dir) / "training_minima.json").read_text()))
    pairs = load_pairs(args.a3_dir, ctx, args.arm, train_minima)
    saddles = {s for *_, s in pairs}
    print(f"{len(pairs)} pairs on {len({p[0] for p in pairs})} training minima, {len(saddles)} distinct saddles")
    ds = HintPairDataset(pairs, ctx)
    r = ds[0]
    li = ctx["li"]
    print("first record: minimum", r["metadata"], "Li start", r["flow_source_pos"][li].numpy().round(3),
          "target", r["saddle_un_pos"][li].numpy().round(3), "hint", r["hint_field"][li].numpy().round(3),
          "|x1 - x0| on the Li %.3f A" % float(torch.linalg.norm(r["saddle_un_pos"][li] - r["flow_source_pos"][li])))


if __name__ == "__main__":
    main()
