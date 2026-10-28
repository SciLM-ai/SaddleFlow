"""Fine-tune the released SaddleFlow model (UMA-M) to follow single-atom hints, on the Tier B labels plus replay.

Starts from the released weights (SciLM/SaddleFlow, EMA) with a zero-initialised HintFiLM before every
time-FiLM block, so at step 0 the model is the released model exactly and ignores the hint.

Every batch is half Tier B and half replay, exactly: each dataset item is one (Tier B, replay) pair and the
loss flattens the pairs. All rows carry a hint field (N, 3): a unit direction on one atom, zero elsewhere, or
all zero for "no hint".

    Tier B rows (tierb.py labels.json, training regions only; one row per region and labelled atom; regions
                 with a barrier below --min-barrier eV are left out of training and validation alike, since
                 they are mostly wobbles on a flat surface next to the minimum rather than real hops)
        hint       the labelled atom and its direction, tilted uniformly within --hint-tilt degrees
        start      with probability --hint-path-prob a path start E + u (S - E) + noise, u ~ U(0,1)^--path-u-power
                   (the restart passes at inference begin near this line with the hint still given); otherwise
                   a hint start: E with the hinted atom moved --hint-push A along the tilted direction, + noise.
                   Noise: N(0, --hint-noise^2) on every atom.
        path       the straight line x_t = (1 - t) x_0 + t S to the region's representative saddle S
    Replay rows (mp20bat train split, the released recipe's path starts: sigma 0.3, u ~ U(0,1)^2)
        hint       none with probability 1 - --replay-hint-prob; otherwise one hint built from the stored saddle
                   (hint_labels.make_hints, one labelled atom drawn at random, tilted as above), so "a hint is
                   present" is not tied to the Tier B chemistries

Options (both off by default, which is the pilot):
    --hint-rows dataset  the hinted half comes from the dataset's own saddles instead of Tier B: every
                 (minimum, saddle) pair of the --dataset-file triplets (reactant and product sides, barrier
                 --min-barrier to 1 eV), one row per labelled atom (hint_labels.make_hints), built into a record
                 exactly as a Tier B row is; triplets in a Tier B element set are left out (the steering test's
                 minima). Validation stays on the held-out Tier B rows.
    --inversion-prob p   each training record (both halves) is inverted with probability p: every position
                 field and the hint negated, the cell unchanged (every lattice maps onto itself under
                 inversion), so the target stays exact. Teaches the mirror behaviour that rotation-equivariance
                 alone does not give. Validation draws are never inverted.

Objective: MSE (not the released Huber), so several saddles sharing a hint stay separate distributions and the
start noise chooses among them; convergent target, x_t perturbation 0.05 A, CoM-symmetric loss, as released.

Learning rates (cosine with warmup, as in train.py): backbone blocks --uma-lr (3e-4, 10x below train.py's
3e-3), velocity head and time-FiLMs --learning-rate (1e-4, 10x below 1e-3), hint layers --hint-lr (1e-3, not
lowered: they start at zero and must move). fp32. One process per GPU (run_finetune.sh: srun SPMD).

    python examples/MaterialsSaddles/experiments/hint_finetune/finetune.py --labels <tierb dir>/labels.json \\
        --output-dir <dir>
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hint_labels as hl  # noqa: E402

from saddleflow.data import MaterialsSaddlesDataset  # noqa: E402
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss  # noqa: E402
from saddleflow.models import GlobalAttn, VelocityHead  # noqa: E402
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone  # noqa: E402
from saddleflow.utils import TrainingConfig, load_uma_backbone, train  # noqa: E402

MP20BAT = "/scratch/gqs7zr/saddleflow/MaterialsSaddles/mp20bat"
DATASET_FILE = "/scratch/gqs7zr/saddleflow/practicality/hf/mp20bat/train/mp20bat_train_0000.aselmdb"   # current release, main


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="tierb.py labels.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--replay-dir", default=MP20BAT, help="MaterialsSaddles subset directory (train/ and val/)")
    p.add_argument("--init", default="SciLM/SaddleFlow", help="released model: Hub repo or local directory")
    p.add_argument("--backbone", default="uma-m-1p1")
    # Hint rows.
    p.add_argument("--min-barrier", type=float, default=0.05, help="eV; regions below it are not used")
    p.add_argument("--hint-tilt", type=float, default=30.0, help="degrees")
    p.add_argument("--hint-push", type=float, default=0.6, help="A")
    p.add_argument("--hint-noise", type=float, default=0.15, help="A, on every atom of a Tier B start")
    p.add_argument("--hint-path-prob", type=float, default=0.5)
    p.add_argument("--replay-hint-prob", type=float, default=0.5)
    p.add_argument("--hint-rows", default="tierb", choices=["tierb", "dataset"], help="source of the hinted half")
    p.add_argument("--dataset-file", default=DATASET_FILE, help="--hint-rows dataset: MaterialsSaddles .aselmdb file")
    p.add_argument("--inversion-prob", type=float, default=0.0, help="share of training records inverted (r -> -r)")
    p.add_argument("--path-u-power", type=float, default=2.0)
    # Objective (released values unless noted).
    p.add_argument("--loss-type", default="mse", choices=["mse", "huber"])
    p.add_argument("--huber-delta", type=float, default=0.05)
    p.add_argument("--path-noise-sigma", type=float, default=0.3)
    p.add_argument("--xt-perturb-sigma", type=float, default=0.05)
    # Optimisation.
    p.add_argument("--uma-lr", type=float, default=3e-4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--hint-lr", type=float, default=1e-3)
    p.add_argument("--batch-pairs", type=int, default=4, help="pairs per GPU (records per GPU = 2x)")
    p.add_argument("--max-steps", type=int, default=5000, help="optimizer steps")
    p.add_argument("--warmup-steps", type=int, default=100, help="optimizer steps (converted to scheduler ticks)")
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--save-every-steps", type=int, default=1000)
    p.add_argument("--val-every-steps", type=int, default=500)
    p.add_argument("--max-val-pairs", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", default=None)
    return p.parse_args()


# ---------------------------------------------------------------- data
def unit_field(n, atom, u):
    h = np.zeros((n, 3)); h[atom] = u
    return torch.as_tensor(h, dtype=torch.float32)


class TierBRows:
    """One row per (region, labelled atom); builds a training record with a fresh tilt, start and noise."""

    def __init__(self, labels_path, held_out, args):
        data = json.loads(Path(labels_path).read_text())["minima"]
        self.args, self.minima, self.rows = args, [], []
        for e in data:
            m = Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True)
            k = len(self.minima)
            self.minima.append((e["tid"], m))
            for r in e["regions"]:
                if r["held_out"] == held_out and r["barrier"] >= args.min_barrier:
                    S = np.array(r["saddle"])
                    for h in r["hints"]:
                        self.rows.append((k, r["region"], S, h["atom"], np.array(h["direction"])))

    def __len__(self):
        return len(self.rows)

    def record(self, i, rng):
        a = self.args
        k, region, S, atom, u = self.rows[i]
        tid, m = self.minima[k]
        E, n = m.positions, len(m)
        d = hl.tilt(u, a.hint_tilt, rng)
        noise = a.hint_noise * rng.normal(size=(n, 3))
        path = rng.random() < a.hint_path_prob
        if path:
            x0 = E + rng.random() ** a.path_u_power * (S - E) + noise
        else:
            x0 = E + noise
            x0[atom] += a.hint_push * d
        x1 = x0 + hl.mic(m.cell.array, S - x0)          # the saddle in the minimum-image frame of the start
        x0t, x1t = torch.as_tensor(x0, dtype=torch.float32), torch.as_tensor(x1, dtype=torch.float32)
        return {"start_pos": x0t.clone(), "flow_source_pos": x0t.clone(), "saddle_un_pos": x1t,
                "partner_un_pos": x0t.clone(), "Z": torch.as_tensor(m.numbers, dtype=torch.long),
                "cell": torch.as_tensor(m.cell.array, dtype=torch.float32), "fixed": torch.zeros(n, dtype=torch.bool),
                "task_name": "omat", "charge": 0, "spin": 0, "delta_norm": torch.linalg.vector_norm(x1t - x0t),
                "role": "hint_path" if path else "hint_start", "triplet_id": int(tid),
                "metadata": {"tid": int(tid), "region": int(region), "atom": int(atom)},
                "hint_field": unit_field(n, atom, d)}


class DatasetRows(TierBRows):
    """TierBRows built from the dataset's own saddles: one row per (minimum, saddle) pair and labelled atom."""

    def __init__(self, path, labels_path, args, cutoff=1.0):
        from ase.db import connect
        excluded = {tuple(sorted(set(Atoms(numbers=e["numbers"]).get_chemical_symbols())))
                    for e in json.loads(Path(labels_path).read_text())["minima"]}
        self.args, self.minima, self.rows = args, [], []
        self.skipped, trip, n = 0, [], 0
        with connect(path) as db:
            for row in db.select():
                trip.append(row)
                if len(trip) < 3:
                    continue
                (R, S, P), (eR, eS, eP) = (r.toatoms() for r in trip), (r.energy for r in trip)
                trip = []
                if tuple(sorted(set(R.get_chemical_symbols()))) in excluded:
                    self.skipped += 1; n += 1; continue
                for side, E, e in ((0, R, eR), (1, P, eP)):
                    if args.min_barrier <= eS - e <= cutoff:
                        m = Atoms(numbers=E.numbers, positions=E.positions, cell=E.cell.array, pbc=True)
                        k = len(self.minima)
                        self.minima.append((n, m))
                        for atom, u, _ in hl.make_hints(m, S.positions):
                            self.rows.append((k, 2 * n + side, S.positions, atom, u))
                n += 1


def invert(record):
    """r -> -r on every position field and the hint (and the eigenmode); the cell is unchanged."""
    for key in ("start_pos", "flow_source_pos", "saddle_un_pos", "partner_un_pos", "hint_field", "eigenmode"):
        if key in record and record[key] is not None:
            record[key] = -record[key]
    record["metadata"] = dict(record.get("metadata") or {}, inverted=True)
    return record


class Replay:
    """mp20bat records as released, with a hint from the stored saddle on a share of them."""

    def __init__(self, split, args):
        self.args = args
        self.ds = MaterialsSaddlesDataset(sorted(glob.glob(f"{args.replay_dir}/{split}/*.aselmdb")))

    def __len__(self):
        return len(self.ds)

    def record(self, i, rng):
        r = self.ds[i]
        n = int(r["Z"].shape[0])
        r["hint_field"] = torch.zeros(n, 3)
        if rng.random() < self.args.replay_hint_prob:
            E = r["start_pos"].double().numpy()
            m = Atoms(numbers=r["Z"].numpy(), positions=E, cell=r["cell"].double().numpy(), pbc=True)
            hints = hl.make_hints(m, r["saddle_un_pos"].double().numpy())
            atom, u, _ = hints[int(rng.integers(len(hints)))]
            r["hint_field"] = unit_field(n, atom, hl.tilt(u, self.args.hint_tilt, rng))
        return r


class Pairs(Dataset):
    """Item i = (Tier B row i, a replay record). Training draws are random; validation draws are fixed."""

    def __init__(self, tierb, replay, fixed_seed=None, size=None, inversion=0.0):
        self.tierb, self.replay, self.fixed_seed, self.inversion = tierb, replay, fixed_seed, inversion
        self.size = min(size or len(tierb), len(tierb))
        self.order = (np.random.default_rng(fixed_seed).permutation(len(tierb))[:self.size]
                      if fixed_seed is not None else None)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.fixed_seed is None:
            rng = np.random.default_rng(int(torch.randint(2 ** 62, (1,)).item()))
            j = i
        else:
            rng = np.random.default_rng([self.fixed_seed, i])
            j = int(self.order[i])
        pair = (self.tierb.record(j, rng), self.replay.record(int(rng.integers(len(self.replay))), rng))
        if self.fixed_seed is None and self.inversion:          # training draws only
            pair = tuple(invert(r) if rng.random() < self.inversion else r for r in pair)
        return pair


class PairLoss(FlowMatchingLoss):
    """FlowMatchingLoss over a batch of (Tier B, replay) pairs: flatten, then the usual loss."""

    def forward(self, batch, *a, **k):
        flat = [r for item in batch for r in (item if isinstance(item, (tuple, list)) else (item,))]
        return super().forward(flat, *a, **k)


# ---------------------------------------------------------------- model
def released_weights(init):
    from safetensors.torch import load_file
    path = Path(init)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(init, allow_patterns=["config.json", "model.safetensors"]))
    return json.loads((path / "config.json").read_text()), load_file(str(path / "model.safetensors"))


def build(args, device):
    config, weights = released_weights(args.init)
    # fairchem accepts only "cpu" or "cuda"; "cuda" is this rank's GPU (torch.cuda.set_device in main).
    raw = load_uma_backbone(args.backbone, device="cuda" if device.startswith("cuda") else "cpu", freeze=True,
                            eval_mode=True, unfreeze_last_block=True).to(device)
    for block in raw.blocks:                         # --unfreeze-uma-all, as in the paper model
        for p in block.parameters():
            p.requires_grad_(True)
    backbone = TimeFiLMBackbone(raw, inject_block_indices=list(config["film_blocks"]), inject_force=False,
                                inject_hint=True).to(device)
    attn = GlobalAttn(sphere_channels=raw.sphere_channels, lmax=raw.lmax, num_heads=8, num_layers=0).to(device)
    head = VelocityHead(sphere_channels=raw.sphere_channels, input_lmax=raw.lmax, depth=int(config["head_depth"])).to(device)
    loss = PairLoss(FlowMatchingConfig(
        mode=1, xt_perturb_sigma=args.xt_perturb_sigma, com_symmetric_loss=True, xt_target_correction=True,
        xt_target_correction_t_floor=0.1, path_start_prob=1.0, path_noise_sigma=args.path_noise_sigma,
        path_u_power=args.path_u_power, loss_type=args.loss_type, huber_delta=args.huber_delta),
        backbone, attn, head)
    missing, unexpected = loss.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.startswith(("backbone.hint_films.", "backbone.film.", "global_attn."))]
    if unexpected or missing:
        raise RuntimeError(f"released weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    hint = [p for p in backbone.hint_films.parameters()]
    uma = [p for p in raw.parameters() if p.requires_grad]
    taken = {id(p) for p in hint + uma}
    rest = [p for p in loss.parameters() if p.requires_grad and id(p) not in taken]
    groups = [{"name": "head_film", "params": rest, "lr": args.learning_rate},
              {"name": "uma_blocks", "params": uma, "lr": args.uma_lr},
              {"name": "hint_films", "params": hint, "lr": args.hint_lr}]
    return loss, groups, config


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tierb_train = (TierBRows(args.labels, False, args) if args.hint_rows == "tierb"
                   else DatasetRows(args.dataset_file, args.labels, args))
    tierb_val = TierBRows(args.labels, True, args)
    replay_train, replay_val = Replay("train", args), Replay("val", args)
    train_ds = Pairs(tierb_train, replay_train, inversion=args.inversion_prob)
    val_ds = Pairs(tierb_val, replay_val, fixed_seed=args.seed, size=args.max_val_pairs) if len(tierb_val) else None
    loss, groups, config = build(args, device)
    if local_rank == 0:
        n = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}
        print(f"[finetune] hinted rows ({args.hint_rows}): train {len(tierb_train)} ({len(tierb_train.minima)} minima"
              + (f"; {tierb_train.skipped} triplets in Tier B element sets left out" if args.hint_rows == "dataset" else "")
              + f"), Tier B held out {len(tierb_val)}; replay train {len(replay_train)}, val {len(replay_val)}; "
              f"inversion {args.inversion_prob}")
        print(f"[finetune] world {world}, {args.batch_pairs} pairs ({2 * args.batch_pairs} records) per GPU; "
              f"trainable parameters {n}")

    cfg = TrainingConfig(
        output_dir=args.output_dir, num_epochs=10 ** 6, batch_size=args.batch_pairs, max_steps=args.max_steps,
        num_workers=args.num_workers, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps * world,          # scheduler ticks once per rank per optimizer step
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay, mixed_precision="no", seed=args.seed,
        log_every=10, save_every_epochs=10 ** 6, save_every_steps=args.save_every_steps,
        val_every_steps=args.val_every_steps if val_ds is not None else 0, resume_from=args.resume_from,
        extras={"finetune": vars(args), "released_config": config, "inject_hint": True, "hint_training": args.hint_rows,
                "film_blocks": config["film_blocks"], "head_depth": config["head_depth"], "backbone": args.backbone,
                "early_time_film": True, "early_time_film_blocks": ",".join(map(str, config["film_blocks"])),
                "loss_type": args.loss_type, "delta_endpoint_channels": 0, "attn_layers": 0,
                "unfreeze_uma_all": True, "xt_target_correction": True, "com_symmetric_loss": True},
    )
    train(loss, train_ds, cfg, val_dataset=val_ds, val_every_epochs=10 ** 6, param_groups=groups)


if __name__ == "__main__":
    main()
