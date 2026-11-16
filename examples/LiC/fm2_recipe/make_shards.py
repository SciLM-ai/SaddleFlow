"""LiC's triplets as MaterialsSaddles shards, so that examples/MaterialsSaddles/train.py (FM2's pipeline) can train on them.

    python examples/LiC/fm2_recipe/make_shards.py --out <dir>

Writes to <dir>:
    lic_000_train.aselmdb    the 12 triplets of examples/LiC/train_set.traj, rows R, S, P
    lic_001_test.aselmdb     the 167 triplets of examples/LiC/test_set.traj
    split_manifest.csv       ms_id_R, split: the 12 train, the 167 test, and the 167 again as val
                             (validation loss is monitored only; nothing is selected on it)

Each row keeps the frame's positions, cell, pbc and FixAtoms (the 126 carbons), and in data["info"]: ms_id (the row's
position, from 0), side (-1 reactant, 0 saddle, 1 product), task_name, charge, spin, and the saddle's eigenmode. Energies
and forces are dropped. The script then reads the shards back with MaterialsSaddlesDataset (validating every triplet) and
checks every record against TrajTripletDataset on the .traj, and the manifest's split sizes.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.db import connect
from ase.io import read

import fairchem.core.datasets  # noqa: F401  (registers the aselmdb backend)
from saddleflow.data import MaterialsSaddlesDataset, TrajTripletDataset

HERE = Path(__file__).resolve().parent.parent           # examples/LiC
SOURCES = (("train", HERE / "train_set.traj"), ("test", HERE / "test_set.traj"))
SIDES = (-1, 0, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    return p.parse_args()


def write_shard(path, frames, first_ms_id):
    if len(frames) % 3:
        raise SystemExit(f"{path}: {len(frames)} frames is not a whole number of [R, S, P] triplets")
    if path.exists():
        raise SystemExit(f"{path} exists; remove it first")
    db = connect(str(path), type="aselmdb", use_lock_file=False)
    try:
        for i, frame in enumerate(frames):
            atoms = frame.copy()
            atoms.calc = None
            fixed = [c for c in frame.constraints if isinstance(c, FixAtoms)]
            atoms.set_constraint(fixed)
            info = dict(ms_id=first_ms_id + i, side=SIDES[i % 3], task_name=frame.info["task_name"],
                        charge=int(frame.info.get("charge", 0)), spin=int(frame.info.get("spin", 0)))
            if i % 3 == 1 and "eigenmode" in frame.info:
                info["eigenmode"] = np.asarray(frame.info["eigenmode"], dtype=float)
            atoms.info = {}
            db.write(atoms, data={"info": info})
    finally:
        db.close()


def same_records(a, b):
    for k in ("start_pos", "saddle_un_pos", "partner_un_pos", "cell"):
        if not np.allclose(np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float), atol=1e-6):
            return k
    for k in ("Z", "fixed"):
        if not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
            return k
    for k in ("task_name", "charge", "spin", "role"):
        if a[k] != b[k]:
            return k
    return None


def main():
    out = Path(parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rows, ms_id, n = [], 0, {}
    for k, (split, traj) in enumerate(SOURCES):
        frames = read(str(traj), ":")
        write_shard(out / f"lic_{k:03d}_{split}.aselmdb", frames, ms_id)
        n[split] = len(frames) // 3
        for j in range(n[split]):
            rows.append((ms_id + 3 * j, split))
            if split == "test":
                rows.append((ms_id + 3 * j, "val"))
        ms_id += len(frames)
    with open(out / "split_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ms_id_R", "split"])
        w.writerows(rows)
    print(f"wrote {out}: train {n['train']} triplets, test {n['test']} (also val)")

    # read back
    import sys
    sys.path.insert(0, str(HERE.parent / "MaterialsSaddles"))
    from data_prep import load_local_triplet_splits

    ds = MaterialsSaddlesDataset(str(out), validate=True)
    train, val, test = load_local_triplet_splits(str(out), str(out / "split_manifest.csv"))
    if (len(train), len(val), len(test)) != (n["train"], n["test"], n["test"]) or ds.num_triplets != sum(n.values()):
        raise SystemExit(f"split sizes {len(train)}, {len(val)}, {len(test)} of {ds.num_triplets} triplets")
    offset = 0
    for split, traj in SOURCES:
        ref = TrajTripletDataset([str(traj)])
        for i in range(len(ref)):
            bad = same_records(ds[2 * offset + i], ref[i])
            if bad:
                raise SystemExit(f"{split} record {i}: {bad} differs from {traj}")
        offset += len(ref) // 2
    print(f"checked: {len(ds)} records equal to the .traj records; splits train {len(train)}, val {len(val)}, "
          f"test {len(test)}")


if __name__ == "__main__":
    main()
