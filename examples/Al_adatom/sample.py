"""
Run a SaddleFlow model from the Al(100) + adatom minimum and save the endpoints for score.py. The model is a released one
(SciLM/SaddleFlow, --revision fm2) or train.py's <run>/release. The noised (and, for FM2, flagged) atoms are:
  movers  for every saddle in al_triplets.traj, its own moving atoms (more than --thresh from the minimum, plus the
          most-moving one), --n draws each: does the model reach that saddle when told which atoms react?
  all     every mobile atom, --n draws
  ball    the mobile atoms within --radius of an atom of the top layer or the adatom (z within 2.5 A of the highest atom),
          a random one per draw, --n draws: an atom of interest and a cutoff, as a user would choose

    python examples/Al_adatom/sample.py --model examples/Al_adatom/runs/fm2_ft/release --mode movers --n 10 \\
        --out examples/Al_adatom/runs/fm2_ft/movers
    python examples/Al_adatom/sample.py --model SciLM/SaddleFlow --revision fm2 --mode ball --n 2000 \\
        --out examples/Al_adatom/runs/fm2/ball
"""

import argparse
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write

from saddleflow import SaddleFlow


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hub repo or a directory with config.json and model.safetensors")
    p.add_argument("--revision", default=None)
    p.add_argument("--mode", choices=["movers", "all", "ball"], required=True)
    p.add_argument("--n", type=int, default=10, help="draws (per saddle for --mode movers)")
    p.add_argument("--thresh", type=float, default=0.5, help="--mode movers: displacement threshold (A)")
    p.add_argument("--radius", type=float, default=4.0, help="--mode ball: radius (A)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--traj", default=str(here / "al_triplets.traj"))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    frames = read(a.traj, ":")
    R, saddles = frames[0], frames[1::3]
    mobile = np.ones(len(R), bool); mobile[R.constraints[0].index] = False
    model = SaddleFlow.from_pretrained(a.model, revision=a.revision)
    if a.mode == "movers":
        sets = []
        for S in saddles:
            d = np.linalg.norm(find_mic(S.positions - R.positions, R.cell, R.pbc)[0], axis=1) * mobile
            sets.append(sorted(set(np.flatnonzero(d > a.thresh).tolist()) | {int(d.argmax())}))
        out = model.propose([R] * len(sets), n=a.n, seed=a.seed, task_name="oc20", batch_size=a.batch_size, atoms_to_move=sets)
        ends = [(e, {"saddle": k, "draw": j}) for k, row in enumerate(out) for j, e in enumerate(row)]
    elif a.mode == "all":
        ends = [(e, {"draw": j}) for j, e in enumerate(model.propose(R, n=a.n, seed=a.seed, task_name="oc20",
                                                                       batch_size=a.batch_size))]
    else:
        top = np.flatnonzero(mobile & (R.positions[:, 2] > R.positions[:, 2].max() - 2.5))
        centres = np.random.default_rng(a.seed).choice(top, a.n)
        out = model.propose([R] * a.n, n=1, seed=a.seed, task_name="oc20", batch_size=a.batch_size,
                            atoms_to_move=[[int(c)] for c in centres], radius=a.radius)
        ends = [(row[0], {"centre": int(c), "draw": j}) for j, (row, c) in enumerate(zip(out, centres))]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for e, info in ends:
        e.info = {"mode": a.mode, **info}
    write(Path(a.out) / "endpoints.traj", [e for e, _ in ends])
    print(f"{len(ends)} endpoints ({a.mode}) -> {a.out}/endpoints.traj")


if __name__ == "__main__":
    main()
