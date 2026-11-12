"""Is a SaddleFlow model mirror-equivariant?

    python examples/MaterialsSaddles/mirror_test.py --model fm1 --dataset <test dir> --n_minima 500

--model is fm1, fm2 or a release directory (config.json, model.safetensors). --dataset is a directory of .aselmdb shards
or a .traj of [R, S, P] triplets. Minima are the dataset's unique endpoints, picked at random with the seed.

Each minimum gets one noised start (the model's own noise), rolled out three ways with the model's defaults: unchanged,
rotated 180 degrees about z (the control: the backbone is rotation-equivariant by construction), and inverted (r -> -r;
every mirror is an inversion times a rotation, and inversion maps every lattice onto itself). The landing is mapped back
and compared with the unchanged one: the largest per-atom distance, rigid shift removed for cells with no fixed atoms.
An equivariant model gives zero up to float error.

SF_PARITY_EVEN=1 zeroes W2 in the loaded model first (saddleflow.utils.parity), with no retraining.

Writes <out-dir>/report.txt and mirror.json (one row per minimum).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_connected_uncond import distance, load_model, pick_minima, read_dataset
from saddleflow.data.transforms import transform_structure
from saddleflow.utils.parity import parity_from_env

MAPS = {"rotation 180 about z": np.diag([-1.0, -1.0, 1.0]), "inversion": -np.eye(3)}
NEAR = 0.2              # A: the cutoff a saddle has to meet in eval_connected_uncond.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="fm1, fm2 or a release directory")
    p.add_argument("--dataset", required=True, help="directory of .aselmdb shards, or a .traj of triplets")
    p.add_argument("--n_minima", type=int, default=500, help="minima to test, picked with the seed (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", help="default mirror_test/<model>_<dataset>_<n_minima>_seed<seed>")
    return p.parse_args()


def noised_start(minimum, sigma, rng):
    """The minimum plus noise on its mobile atoms, with no rigid shift when no atom is fixed."""
    mobile = np.ones(len(minimum), bool)
    for constraint in minimum.constraints:
        mobile[constraint.get_indices()] = False

    noise = sigma * rng.normal(size=minimum.positions.shape) * mobile[:, None]
    if mobile.all():
        noise = noise - noise.mean(0)

    start = minimum.copy()
    start.set_positions(minimum.positions + noise, apply_constraint=False)
    return start


def roll_out(model, structures, batch_size):
    """Landing positions; sigma 0 because the noise is already in the starts."""
    proposals = model.propose(structures, n=1, sigma=0.0, batch_size=batch_size)
    return [draws[0].positions for draws in proposals]


def main():
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    name = f"{Path(args.model).name}_{dataset.parent.name}_{dataset.stem}_{args.n_minima}_seed{args.seed}"
    out = Path(args.out_dir or f"mirror_test/{name}")
    out.mkdir(parents=True, exist_ok=True)

    minima = pick_minima(read_dataset(args.dataset), args.n_minima, args.seed)
    model = load_model(args.model)
    w2_zero = parity_from_env(model.backbone)

    rng = np.random.default_rng(args.seed)
    starts = [noised_start(m, model.config["inference"]["sigma"], rng) for m in minima]
    landings = roll_out(model, starts, args.batch_size)
    print(f"{len(starts)} starts rolled out", flush=True)

    rows = [dict(minimum=i, formula=m.get_chemical_formula(), n_atoms=len(m), task=m.info["task_name"])
            for i, m in enumerate(minima)]
    mode = model.mode + (", W2 zeroed at load" if w2_zero else "")
    print(f"mode: {mode}", flush=True)
    lines = [f"Mirror test: {args.model} on {args.dataset}", f"Model mode: {mode}", "=" * 100, "",
             f"{len(starts)} minima, one noised start each. Largest per-atom distance (A) between the mapped start's "
             "landing, mapped back, and the unchanged landing.", "",
             f"  map                        p50      p90      p99      max |  > {NEAR} A    > 1 A"]

    for map_name, R in MAPS.items():
        mapped = roll_out(model, [transform_structure(s, R) for s in starts], args.batch_size)

        distances = []
        for row, start, unchanged, landing in zip(rows, starts, landings, mapped):
            a, b = start.copy(), start.copy()
            a.set_positions(unchanged, apply_constraint=False)
            b.set_positions(landing @ R, apply_constraint=False)        # back to the original frame
            row[map_name] = distance(a, b)
            distances.append(row[map_name])

        d = np.array(distances)
        q = "  ".join(f"{x:7.4f}" for x in np.quantile(d, [0.5, 0.9, 0.99, 1.0]))
        lines.append(f"  {map_name:22s} {q} | {100 * np.mean(d > NEAR):6.1f}%  {100 * np.mean(d > 1):6.1f}%")
        print(f"{map_name} done", flush=True)

    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    (out / "mirror.json").write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")
    print(text)


if __name__ == "__main__":
    main()
