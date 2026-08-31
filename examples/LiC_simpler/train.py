"""
Train SaddleFlow on the symmetric Li-on-pristine-graphene test case.

Data: `one_saddle.traj` — a single `[R, S, P]` triplet for one Li-hop on a
defect-free carbon sheet (112 C + 1 Li, Li at index 112). The graphene
lattice is 6-fold symmetric around a Li adsorption site, so there are 6
equivalent saddles in the full symmetry orbit; we train on only one of
them and rely on Mode 1 (product-conditional, midpoint-of-(R,P) start)
plus UMA's SO(3) equivariance to propagate the learned field to all 6
wedges at inference.

Two objectives are available:

  mode 1 (default) — product-conditional, x_0 = (R+P)/2 -> x_1 = saddle.

  TS-denoise (--ts-denoise-sigma S) — x_0 = saddle + N(0, S^2) -> x_1 = saddle,
    with no R/P conditioning. This is the "flower" setup: integrate from a ring
    of perturbed starts and the trajectories should fan out into the six
    symmetry-equivalent saddles of the C6 orbit. Pair it with
    --delta-endpoint-channels 0, since the head must not see the endpoints.

Launch (~20 min on one GPU either way), then visualise:

    # (a) default: mode-1, product-conditional
    python examples/LiC_simpler/train.py
    python examples/LiC_simpler/viz_checkpoints.py --run-dir examples/LiC_simpler/runs/mode1

    # (b) the "flower": unconditioned TS-denoise
    python examples/LiC_simpler/train.py --ts-denoise-sigma 0.5 \
        --delta-endpoint-channels 0 --attn-layers 1 --ema-decay 0.99
    python examples/LiC_simpler/viz_checkpoints.py \
        --run-dir examples/LiC_simpler/runs/tsdenoise_sigma0.5

The visualiser reads the architecture from the run's config.json, so you never
pass matching --attn-layers/--head-depth by hand. See README.md in this folder.
"""

import argparse
from pathlib import Path

import torch

from saddleflow.data import TrajTripletDataset
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import TrainingConfig, load_uma_backbone, train


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "one_saddle.traj"))
    # Note: the default depends on the objective (set after parsing) so a
    # TS-denoise run does not silently overwrite a mode-1 run's checkpoints.
    p.add_argument("--output-dir", default=None)

    # 1 triplet → 2 records after R/P doubling; with batch_size=2, 1 step/epoch.
    # 10k steps gives a comfortable convergence margin.
    p.add_argument("--num-epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    # ~10k steps falls into the small-scale EMA rule; 0.99 ≈ 100-step window.
    p.add_argument("--ema-decay", type=float, default=0.99)
    # fp32 by default. bf16 is safe as of the autocast fix (the coordinate
    # round trip in data/transforms.py is now pinned to fp32), but this example
    # is the one where that bug showed up, so it stays on the conservative
    # setting. See CLAUDE.md latent-bug log for the measurement.
    p.add_argument("--mixed-precision", default="no",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every-epochs", type=int, default=500)

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--attn-layers", type=int, default=0,
                   help="Number of GlobalAttn layers. Default 0 (no attention) — "
                        "Mode 1's partner direction already breaks the symmetry that "
                        "GlobalAttn was originally introduced to handle.")
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)

    # Training mode (only mode=1 is implemented; reserved for future modes).
    p.add_argument("--mode", type=int, default=1)
    p.add_argument("--ts-denoise-sigma", type=float, default=0.0,
                   help="Single-ended TS-denoise objective: x_0 = saddle + "
                        "N(0, sigma^2) on mobile atoms, x_1 = saddle, no R/P "
                        "conditioning. 0 = off (use mode-1 midpoint start). "
                        "0.5 A reproduces the flower run.")
    p.add_argument("--delta-endpoint-channels", type=int, default=32,
                   help="Channel count for the partner-displacement feature in "
                        "VelocityHead. Default 32 — analogue of time_embed_dim.")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.output_dir is None:
        name = ("tsdenoise_sigma%g" % args.ts_denoise_sigma) if args.ts_denoise_sigma > 0 else "mode1"
        args.output_dir = str(here / "runs" / name)
    return args


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] dataset: {args.train_traj}")
    dataset = TrajTripletDataset(
        args.train_traj,
        stats_cache=str(out_dir / "dataset_stats.json"),
    )
    print(f"[train] {len(dataset)} records ({dataset.num_triplets} triplets × 2 sides), "
          f"<||Δ||> = {dataset.delta_norm_mean:.4f} Å")

    M = int((~dataset[0]["fixed"]).sum().item())
    if args.ts_denoise_sigma > 0:
        print(f"[train] TS-denoise — x_0 = saddle + N(0, {args.ts_denoise_sigma}^2), "
              f"x_1 = saddle (unconditioned)")
    else:
        print(f"[train] mode 1 — product-conditional (no noise on x_0)")
    print(f"[train] mobile atoms M={M}")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    backbone = load_uma_backbone(args.backbone, device=args.device, freeze=True, eval_mode=True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                      num_heads=args.attn_heads, num_layers=args.attn_layers).to(args.device)
    # The TS-denoise objective never supplies endpoint deltas, so the head must
    # be built without them or the forward will be handed an input it rejects.
    head_delta_C = (0 if args.ts_denoise_sigma > 0
                    else (args.delta_endpoint_channels if args.mode == 1 else 0))
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=args.head_depth,
        delta_endpoint_channels=head_delta_C,
    ).to(args.device)
    if head_delta_C != args.delta_endpoint_channels:
        print(f"[train] delta_endpoint_channels forced {args.delta_endpoint_channels} -> "
              f"{head_delta_C} (the TS-denoise objective supplies no endpoints)")
    print(f"[train] delta_endpoint_channels={head_delta_C} (effective)")
    print(f"[train] backbone K{backbone.num_layers}L{lmax} (sphere_channels={sc}), frozen")
    print(f"[train] attn_layers={args.attn_layers}  head_depth={args.head_depth}")
    print(f"[train] trainable params: "
          f"{sum(p.numel() for p in list(attn.parameters()) + list(head.parameters())):,}")

    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(
            mode=args.mode,
            ts_denoise_sigma=args.ts_denoise_sigma,
        ),
        backbone, attn, head,
    )

    train_cfg = TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay,
        mixed_precision=args.mixed_precision, seed=args.seed,
        log_every=args.log_every, save_every_epochs=args.save_every_epochs,
        extras={
            "mode": args.mode,
            "delta_endpoint_channels": head_delta_C,
            "backbone": args.backbone,
            "attn_layers": args.attn_layers, "attn_heads": args.attn_heads,
            "head_depth": args.head_depth,
            "ts_denoise_sigma": args.ts_denoise_sigma,
            "mixed_precision": args.mixed_precision,
        },
    )
    train(loss_module, dataset, train_cfg)


if __name__ == "__main__":
    main()
