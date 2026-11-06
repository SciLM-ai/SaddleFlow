"""
Train the atom-set selector (saddleflow.selector) on MaterialsSaddles: the official train split of each subset, both
sides of every triplet (start = reactant or product), data from $MATERIALSSADDLES_ROOT as train.py. A set is the atoms
moving more than --thresh from the start to the saddle, plus the most-moving one. The defaults are the selector used with
FM2 (sel1): UMA-S-1.2 with time-FiLM on blocks 0-3, every block trained (heads 1e-3, backbone 1e-2), 30 epochs on oc22;
sample its sets at temperature 1.5.

    python examples/MaterialsSaddles/train_selector.py --subsets oc22 --output-dir runs/selector
    accelerate launch --num_processes 32 examples/MaterialsSaddles/train_selector.py --subsets all \\
        --subset-repeat oc22=8,mp20bat=8,oc20=2 --output-dir runs/selector_all
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import ensure_subsets, load_official_splits  # noqa: E402

from saddleflow.data import MaterialsSaddlesDataset  # noqa: E402
from saddleflow.selector import SelectorLoss, build_selector  # noqa: E402
from saddleflow.utils import TrainingConfig, train  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subsets", default="oc22", help="comma list of MaterialsSaddles subsets, or 'all'")
    p.add_argument("--subset-repeat", default="", help="repeats per subset, e.g. oc22=8,mp20bat=8,oc20=2; unlisted once")
    p.add_argument("--task-name-map", default="", help="e.g. oc22=oc20 (UMA-M has no oc22 expert)")
    p.add_argument("--default-task-name", default="omat", help="task name of records without one")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--thresh", type=float, default=0.5, help="an atom is in the set if it moves more than this (A)")
    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--early-time-film-blocks", default="0,1,2,3")
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8, help="per GPU")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="heads and FiLM/flag layers")
    p.add_argument("--uma-lr", type=float, default=1e-2, help="UMA blocks")
    p.add_argument("--warmup-steps", type=int, default=100, help="scheduler ticks: one per GPU per optimizer step")
    p.add_argument("--ema-decay", type=float, default=None, help="default 1 - 50 / total steps")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every-epochs", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    subs = ["oc20", "oc22", "lemat", "mp20bat"] if a.subsets == "all" else a.subsets.split(",")
    rep = {k: int(v) for k, v in (x.split("=") for x in a.subset_repeat.split(",") if x)}
    tmap = dict(x.split("=") for x in a.task_name_map.split(",") if x)
    staged = ensure_subsets(subs); parts = []
    for sub in subs:
        d = MaterialsSaddlesDataset(str(staged[sub]), default_task_name=a.default_task_name, task_name_map=tmap or None,
                                    stats_cache=str(out / f"dataset_stats_{sub}.json"))
        train_tids, _, _ = load_official_splits(sub)
        idx = sorted([2 * t for t in train_tids] + [2 * t + 1 for t in train_tids])
        parts += [torch.utils.data.Subset(d, idx)] * rep.get(sub, 1)
        print(f"[train] {sub}: {len(idx)} records x {rep.get(sub, 1)}")
    ds = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
    steps = a.num_epochs * -(-len(ds) // a.batch_size)
    ema = a.ema_decay if a.ema_decay is not None else 1.0 - 50.0 / steps
    print(f"[train] selector: {len(ds)} records, sets = movers > {a.thresh} A, {steps} steps on one GPU, EMA {ema:.5f}")

    backbone, atom_head, stop_head = build_selector(a.backbone, [int(x) for x in a.early_time_film_blocks.split(",")], a.device)
    new = list(atom_head.parameters()) + list(stop_head.parameters()) + \
        [q for f in list(backbone.films) + list(backbone.flag_films) for q in f.parameters()]
    groups = [{"name": "heads_film", "params": new, "lr": a.learning_rate},
              {"name": "uma_unfrozen", "params": [q for q in backbone.backbone.parameters() if q.requires_grad], "lr": a.uma_lr}]
    cfg = TrainingConfig(output_dir=str(out), num_epochs=a.num_epochs, batch_size=a.batch_size, num_workers=a.num_workers,
                         learning_rate=a.learning_rate, warmup_steps=a.warmup_steps, ema_decay=ema, seed=a.seed,
                         log_every=a.log_every, save_every_epochs=a.save_every_epochs,
                         extras={"selector": "ar", "thresh": a.thresh, "backbone": a.backbone,
                                 "early_time_film_blocks": a.early_time_film_blocks, "uma_lr": a.uma_lr,
                                 "subsets": a.subsets, "subset_repeat": a.subset_repeat, "task_name_map": a.task_name_map})
    train(SelectorLoss(backbone, atom_head, stop_head, a.thresh), ds, cfg, param_groups=groups)


if __name__ == "__main__":
    main()
