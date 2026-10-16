"""Roll the released unconditional model out from test-set minima, one landing point per minimum.

The model is never shown a product or a saddle. Each start is a minimum (a reactant or a
product frame of the test split) plus the model's own training noise, integrated with the released
inference (5 passes x 4 Euler steps). The stored saddle is written alongside only so the analysis can
report whether Sella's saddle is the NEB one; it plays no part in the rollout or in the labels.

    python examples/MaterialsSaddles/failure_mode_analysis/make_proposals.py \
        --out-dir <dir> --n-starts 500 --seed 0

Writes, aligned frame by frame (frame k has info['tid'] = k):
    proposals.traj        landing points, the input of sella_eval.py
    minima.traj           the starting minima
    stored_saddles.traj   the dataset saddle of the triplet each minimum came from
    selection.json        provenance of every start, plus the run settings
--select-only stops before the model (CPU check of the data reading and selection).
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import Trajectory

DEFAULT_DATA = "/scratch/gqs7zr/saddleflow/MaterialsSaddles/mp20bat/test/mp20bat_test_0000.aselmdb"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA, help="One MaterialsSaddles .aselmdb split file.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-starts", type=int, default=500, help="Random minima to roll out; 0 = all unique minima.")
    p.add_argument("--seed", type=int, default=0, help="Seeds both the selection and the start noise.")
    p.add_argument("--dedup-tol", type=float, default=0.05, help="Max per-atom MIC distance (A) for duplicate minima.")
    p.add_argument("--model", default="SciLM/SaddleFlow", help="Hub repo or local dir with config.json + model.safetensors.")
    p.add_argument("--task-name", default="omat")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--select-only", action="store_true", help="Write the selection and minima; skip the model.")
    return p.parse_args()


def clean(atoms, **info):
    """Positions, numbers, cell, PBC and constraints only, with a small explicit info dict."""
    out = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    out.set_constraint([c.copy() for c in atoms.constraints])
    out.info.update(info)
    return out


def is_saddle(info):
    return info.get("side") == 0 or info.get("image_type") == "climbing"


def read_minima(path):
    """Every endpoint of every [R, S, P] triplet, with the triplet's saddle."""
    from fairchem.core.datasets import AseDBDataset
    ds = AseDBDataset({"src": path})
    if len(ds) % 3:
        raise ValueError(f"{path} has {len(ds)} frames, not a whole number of triplets")
    minima = []
    for first in range(0, len(ds), 3):
        frames = [ds.get_atoms(first + j) for j in range(3)]
        infos = [f.info.get("info", f.info) for f in frames]
        if not is_saddle(infos[1]) or is_saddle(infos[0]) or is_saddle(infos[2]):
            raise ValueError(f"frames {first}-{first + 2} are not ordered [R, S, P]")
        for j, role in [(0, "R"), (2, "P")]:
            minima.append(dict(atoms=frames[j], saddle=frames[1], role=role, frame=first + j,
                               ms_id=int(infos[j].get("ms_id", -1)), saddle_ms_id=int(infos[1].get("ms_id", -1)),
                               task_name=infos[j].get("task_name")))
    return minima


def deduplicate(minima, tol):
    """Keep the first of any minima that match atom for atom within tol (same numbers and cell)."""
    groups = defaultdict(list)
    for m in minima:
        a = m["atoms"]
        groups[(a.get_chemical_formula(), tuple(np.round(a.cell.cellpar(), 3)))].append(m)
    unique = []
    for items in groups.values():
        kept = []
        for m in items:
            a = m["atoms"]
            if not any(np.array_equal(a.numbers, k["atoms"].numbers)
                       and find_mic(a.positions - k["atoms"].positions, a.cell, a.pbc)[1].max() < tol
                       for k in kept):
                kept.append(m)
        unique += kept
    return sorted(unique, key=lambda m: m["frame"])


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    minima = read_minima(args.data)
    unique = deduplicate(minima, args.dedup_tol)
    rng = np.random.default_rng(args.seed)
    n = len(unique) if args.n_starts <= 0 else min(args.n_starts, len(unique))
    chosen = [unique[i] for i in np.sort(rng.choice(len(unique), n, replace=False))]
    tasks = {m["task_name"] for m in chosen}
    if tasks - {args.task_name, None}:
        raise ValueError(f"data task names {tasks} do not match --task-name {args.task_name}")
    print(f"[select] {len(minima)} endpoints -> {len(unique)} unique minima -> {n} chosen (seed {args.seed})",
          flush=True)

    rows = [dict(tid=k, ms_id=m["ms_id"], role=m["role"], frame=m["frame"], saddle_ms_id=m["saddle_ms_id"],
                 formula=m["atoms"].get_chemical_formula(), natoms=len(m["atoms"])) for k, m in enumerate(chosen)]
    with Trajectory(out / "minima.traj", "w") as traj:
        for row, m in zip(rows, chosen):
            traj.write(clean(m["atoms"], tid=row["tid"], src=f"{row['ms_id']}:{row['role']}"))
    with Trajectory(out / "stored_saddles.traj", "w") as traj:
        for row, m in zip(rows, chosen):
            traj.write(clean(m["saddle"], tid=row["tid"], src=f"{row['saddle_ms_id']}:S"))
    settings = dict(data=args.data, seed=args.seed, dedup_tol=args.dedup_tol, task_name=args.task_name,
                    n_endpoints=len(minima), n_unique_minima=len(unique), n_chosen=n, model=args.model)
    if args.select_only:
        (out / "selection.json").write_text(json.dumps(dict(settings=settings, starts=rows), indent=1))
        print(f"[select] --select-only; wrote {out}/minima.traj and selection.json")
        return

    from saddleflow import SaddleFlow
    sf = SaddleFlow.from_pretrained(args.model, device=args.device)
    settings["inference"] = dict(sf.config["inference"])
    start = time.perf_counter()
    proposals = sf.propose([m["atoms"] for m in chosen], n=1, seed=args.seed, task_name=args.task_name,
                           batch_size=args.batch_size)
    seconds = time.perf_counter() - start
    settings["propose_seconds"] = seconds
    with Trajectory(out / "proposals.traj", "w") as traj:
        for row, (landing,) in zip(rows, proposals):
            traj.write(clean(landing, tid=row["tid"], src=f"{row['ms_id']}:{row['role']}"))
    (out / "selection.json").write_text(json.dumps(dict(settings=settings, starts=rows), indent=1))
    print(f"[propose] {n} landing points in {seconds:.1f} s -> {out}/proposals.traj", flush=True)


if __name__ == "__main__":
    main()
