"""
Fine-tune SaddleFlow FM2 on the Al(100) + adatom saddles with FM2's own recipe, then export the result in the format
`SaddleFlow.from_pretrained` reads (<output-dir>/release), so sample.py can run it exactly like the released model.

Recipe (as FM2): each training sample is the minimum (or the product) plus N(0, 0.3^2) noise on a set of mobile atoms,
on the line to its saddle with probability 0.5 (else at the endpoint), flagged to the network; x_1 = saddle. The set is
every mobile atom (20 %), the mobile atoms within U(3, 5) A of the most-moving atom (20 %), or the atoms moving more than
a threshold drawn from U(0, 1) A, plus the most-moving one (60 %). Huber loss (delta 0.05), x_t perturbation 0.05 A with
the convergent target, every UMA-M block trained (head 1e-3, backbone 3e-3), EMA decay 1 - 50 / steps.

Data: al_triplets.traj, [R, S, P] per triplet (R is the same minimum in all), both directions R -> S and P -> S. The
starting weights are the released FM2 (Hugging Face SciLM/SaddleFlow, revision fm2; gated: accept its terms and log in).

    python examples/Al_adatom/train.py --output-dir examples/Al_adatom/runs/fm2_ft
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from saddleflow.data import TrajTripletDataset
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import TrainingConfig, load_ema_weights, load_uma_backbone, train


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "al_triplets.traj"))
    p.add_argument("--output-dir", default=str(here / "runs" / "fm2_ft"))
    p.add_argument("--init", default="SciLM/SaddleFlow", help="released model to start from (Hub repo or local directory)")
    p.add_argument("--revision", default="fm2")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3, help="head and FiLM layers")
    p.add_argument("--uma-lr", type=float, default=3e-3, help="UMA backbone blocks")
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    init = Path(args.init) if Path(args.init).is_dir() else Path(snapshot_download(
        args.init, revision=args.revision, allow_patterns=["config.json", "model.safetensors"]))
    release_cfg = json.loads((init / "config.json").read_text())
    if not release_cfg.get("flag"):
        raise SystemExit(f"{args.init}@{args.revision} has no per-atom flag input; FM2's recipe needs one")

    dataset = TrajTripletDataset(args.train_traj, default_task_name="oc20", stats_cache=str(out / "dataset_stats.json"))
    steps = args.num_epochs * -(-len(dataset) // args.batch_size)
    print(f"[train] {len(dataset)} records ({len(dataset) // 2} triplets x 2 directions), {steps} steps, init {init}")

    raw = load_uma_backbone(release_cfg["backbone_name"], device=args.device, freeze=True, eval_mode=True)
    for blk in raw.blocks:
        for prm in blk.parameters():
            prm.requires_grad_(True)
    backbone = TimeFiLMBackbone(raw, inject_block_indices=release_cfg["film_blocks"], inject_flag=True).to(args.device)
    attn = GlobalAttn(sphere_channels=raw.sphere_channels, lmax=raw.lmax, num_heads=8, num_layers=0).to(args.device)
    head = VelocityHead(sphere_channels=raw.sphere_channels, input_lmax=raw.lmax,
                        depth=release_cfg["head_depth"]).to(args.device)
    head_params = list(attn.parameters()) + list(head.parameters())
    for m in list(backbone.films) + list(backbone.flag_films):
        head_params += list(m.parameters())
    param_groups = [{"name": "head_attn_film", "params": head_params, "lr": args.learning_rate},
                    {"name": "uma_unfrozen", "params": [p for p in raw.parameters() if p.requires_grad], "lr": args.uma_lr}]

    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(mode=1, mover_start_prob=1.0, mover_sigma=release_cfg["inference"]["sigma"],
                           mover_thresh_max=1.0, mover_all_prob=0.2, mover_ball_prob=0.2, mover_ball_rmin=3.0,
                           mover_ball_rmax=5.0, mover_u0_prob=0.5, mover_flag=True, xt_perturb_sigma=0.05,
                           xt_target_correction=True, loss_type="huber", huber_delta=0.05, com_symmetric_loss=True),
        backbone, attn, head,
    )
    cfg = TrainingConfig(output_dir=str(out), num_epochs=args.num_epochs, batch_size=args.batch_size,
                         num_workers=args.num_workers, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                         ema_decay=1.0 - 50.0 / steps, seed=args.seed, log_every=args.log_every,
                         save_every_epochs=args.num_epochs + 1, init_weights=str(init),
                         extras={"init": f"{args.init}@{args.revision}", "train_traj": args.train_traj})
    train(loss_module, dataset, cfg, param_groups=param_groups)

    # Export the EMA weights like the release: backbone (with its FiLM and flag layers) and velocity head.
    load_ema_weights(str(out / "checkpoint_final"), [loss_module], args.device)
    state = {k: v.detach().cpu().contiguous() for k, v in loss_module.state_dict().items()
             if k.startswith(("backbone.", "velocity_head.")) and not k.startswith("backbone.film.")}
    rel = out / "release"; rel.mkdir(exist_ok=True)
    save_file(state, str(rel / "model.safetensors"), metadata={"format": "pt"})
    release_cfg["training_run"] = f"FM2 fine-tuned on {Path(args.train_traj).name} ({args.num_epochs} epochs)"
    (rel / "config.json").write_text(json.dumps(release_cfg, indent=2) + "\n")
    print(f"[train] wrote {rel}: run it with  python examples/Al_adatom/sample.py --model {rel}")


if __name__ == "__main__":
    main()
