"""Build the 2-frame [R, P] input for the SaddleMill NEB from the LiC_simpler
training triplet ``../one_saddle.traj`` (= [R, S, P]).

The saddle frame S is NOT written -- SaddleMill interpolates the band between R
and P itself.  S's Li position and barrier are stashed as scalars in ``.info``
so the NEB output (which carries the input ``.info`` under ``orig_info``) is
self-contained for a later comparison.
"""
from pathlib import Path

import numpy as np
from ase.io import Trajectory, read

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "one_saddle.traj"          # examples/LiC_simpler/one_saddle.traj
OUT = HERE / "data" / "LiC_simpler_RP.traj"    # config.ini: dir_path = data
LI = 112  # the only mobile atom; C 0-111 are FixAtoms

R, S, P = read(SRC, index=":")
assert len(R) == len(S) == len(P) == 113
assert R.constraints and set(R.constraints[0].get_indices()) == set(range(112))

ref = {
    "ref_saddle_li_pos": S.positions[LI].tolist(),
    "ref_barrier_eV": float(S.info["barrier"]),
    "ref_source": "examples/LiC_simpler/one_saddle.traj",
}
OUT.parent.mkdir(exist_ok=True)
with Trajectory(OUT, "w") as w:
    for a in (R, P):
        a = a.copy()
        a.info.update(ref)
        w.write(a)

frames = read(OUT, index=":")
print(f"wrote {OUT.relative_to(HERE)}: {len(frames)} frames")
for tag, a in zip("RP", frames):
    print(f"  {tag}: Li = {np.round(a.positions[LI], 4)}  fixed = {len(a.constraints[0].get_indices())}"
          f"  task_name = {a.info['task_name']}")
print(f"  |P-R| (Li) = {np.linalg.norm(P.positions[LI] - R.positions[LI]):.4f} A;"
      f"  ref saddle Li = {np.round(S.positions[LI], 4)}  barrier = {S.info['barrier']:.4f} eV")
