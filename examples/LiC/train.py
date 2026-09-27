"""
Train SaddleFlow on the Li-on-defective-graphene example.

The model is unconditioned: it is never shown a reactant or a product. Each
training sample starts at one of the training saddles with Gaussian noise on the
Li atom, x_0 = saddle + N(0, sigma^2), and the flow learns to carry it back to
x_1 = saddle. At inference any Li position on the sheet is a valid start, and
the flow carries it to the nearest saddle -- see random_throw.py and README.md.

Data: `train_set.traj`, flat [R1, S1, P1, R2, S2, P2, ...] frames. Only the
saddles S enter the objective.

Run from the repository root. The defaults are the reference recipe in
README.md (16000 epochs, fp32, about 4 h on one GPU):

    python examples/LiC/train.py
"""

import argparse
from pathlib import Path

import torch

from saddleflow.data import TrajTripletDataset
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import TrainingConfig, load_uma_backbone, train


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--output-dir", default=None,
                   help="Default: runs/tsdenoise_sigma<sigma> next to this script, with a "
                        "suffix for any architecture flag changed from its default.")
    p.add_argument("--ts-denoise-sigma", type=float, default=0.25,
                   help="Std (A) of the Gaussian noise added to the saddle to make x_0.")

    # 12 triplets -> 24 records (R->S and P->S) -> 6 optimizer steps per epoch.
    p.add_argument("--num-epochs", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="LR of the velocity head and the time-FiLM layers.")
    p.add_argument("--uma-lr", type=float, default=1e-2,
                   help="LR of the unfrozen UMA backbone blocks.")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--mixed-precision", default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--save-every-epochs", type=int, default=4000,
                   help="Also save a checkpoint every N epochs; each one is ~4.6 GB "
                        "(weights, EMA, optimizer). 0 = only the final checkpoint.")

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--unfreeze-uma-all", action=argparse.BooleanOptionalAction, default=True,
                   help="Train all UMA backbone blocks at --uma-lr. "
                        "--no-unfreeze-uma-all trains the head only.")
    p.add_argument("--early-time-film", action=argparse.BooleanOptionalAction, default=True,
                   help="Let the flow time t modulate the backbone blocks in "
                        "--early-time-film-blocks (equivariant FiLM).")
    p.add_argument("--early-time-film-blocks", default="-2,-1")
    p.add_argument("--head-depth", type=int, default=1)
    p.add_argument("--attn-layers", type=int, default=0)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.output_dir is None:
        name = "tsdenoise_sigma%g" % args.ts_denoise_sigma
        if not args.unfreeze_uma_all:
            name += "_frozen"
        if not args.early_time_film:
            name += "_notfilm"
        if args.head_depth != 1:
            name += f"_depth{args.head_depth}"
        if args.attn_layers != 0:
            name += f"_attn{args.attn_layers}"
        args.output_dir = str(here / "runs" / name)
    return args


def main():
    args = parse_args()
    if args.ts_denoise_sigma <= 0:
        raise SystemExit("--ts-denoise-sigma must be positive")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] dataset: {args.train_traj}")
    dataset = TrajTripletDataset(args.train_traj,
                                 stats_cache=str(out_dir / "dataset_stats.json"))
    print(f"[train] {len(dataset)} records ({dataset.num_triplets} triplets x 2 sides)")
    print(f"[train] TS-denoise: x_0 = saddle + N(0, {args.ts_denoise_sigma}^2), x_1 = saddle")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    raw_backbone = load_uma_backbone(args.backbone, device=args.device,
                                     freeze=True, eval_mode=True)
    if args.unfreeze_uma_all:
        for blk in raw_backbone.blocks:
            for prm in blk.parameters():
                prm.requires_grad_(True)
        n_uma = sum(p.numel() for p in raw_backbone.parameters() if p.requires_grad)
        print(f"[train] UMA unfrozen: all {len(raw_backbone.blocks)} blocks "
              f"({n_uma:,} params) at uma_lr={args.uma_lr:g}")
    backbone = raw_backbone
    if args.early_time_film:
        idx = [int(x) for x in args.early_time_film_blocks.split(",")]
        backbone = TimeFiLMBackbone(raw_backbone, inject_block_indices=idx,
                                    inject_force=False).to(args.device)
        print(f"[train] time-FiLM inside the backbone at blocks {idx}")
    sc, lmax = raw_backbone.sphere_channels, raw_backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                      num_heads=args.attn_heads, num_layers=args.attn_layers).to(args.device)
    head = VelocityHead(sphere_channels=sc, input_lmax=lmax,
                        depth=args.head_depth).to(args.device)

    # The head, attention and FiLM layers are new; the backbone is pretrained, so
    # it gets its own learning rate in a separate parameter group.
    head_params = list(attn.parameters()) + list(head.parameters())
    if args.early_time_film:
        for film in backbone.films:
            head_params += list(film.parameters())
    print(f"[train] attn_layers={args.attn_layers}  head_depth={args.head_depth}  "
          f"head+attn+FiLM params={sum(p.numel() for p in head_params):,}")
    param_groups = None
    if args.unfreeze_uma_all:
        param_groups = [
            {"name": "head_attn_film",
             "params": [p for p in head_params if p.requires_grad],
             "lr": args.learning_rate},
            {"name": "uma_unfrozen",
             "params": [p for p in raw_backbone.parameters() if p.requires_grad],
             "lr": args.uma_lr},
        ]

    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(mode=1, ts_denoise_sigma=args.ts_denoise_sigma),
        backbone, attn, head,
    )

    train_cfg = TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay,
        mixed_precision=args.mixed_precision, seed=args.seed,
        log_every=args.log_every,
        save_every_epochs=args.save_every_epochs or args.num_epochs + 1,
        # random_throw.py rebuilds the network from these keys.
        extras={
            "mode": 1,
            "ts_denoise_sigma": args.ts_denoise_sigma,
            "delta_endpoint_channels": 0,
            "backbone": args.backbone,
            "attn_layers": args.attn_layers, "attn_heads": args.attn_heads,
            "head_depth": args.head_depth,
            "unfreeze_uma_all": bool(args.unfreeze_uma_all),
            "early_time_film": bool(args.early_time_film),
            "early_time_film_blocks": args.early_time_film_blocks,
            "uma_lr": args.uma_lr if args.unfreeze_uma_all else None,
        },
    )
    train(loss_module, dataset, train_cfg, param_groups=param_groups)


if __name__ == "__main__":
    main()
