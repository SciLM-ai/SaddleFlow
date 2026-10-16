"""
Train SaddleFlow on the Li-on-defective-graphene example.

By default the model is unconditioned: it is never shown a reactant or a product. Each
training sample starts at one of the training saddles with Gaussian noise on the
Li atom, x_0 = saddle + N(0, sigma^2), and the flow learns to carry it back to
x_1 = saddle. At inference any Li position on the sheet is a valid start, and
the flow aims to reach a nearby saddle -- see random_throw.py and README.md.
--source-conditioning optionally retains the actual noisy x_0, not
the saddle-relative perturbation. experiments/source_study/README.md describes the matched 12/48 study.

Data: `train_set.traj`, flat [R1, S1, P1, R2, S2, P2, ...] frames. Only the
saddles S enter the objective.

Run from the repository root. The defaults are the reference recipe in
README.md (16000 epochs, fp32, about 4 h on one GPU):

    python examples/LiC/train.py
"""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/experiments/source_study, examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parent  # examples/LiC
sys.path[:0] = [str(_LIC / 'experiments' / 'source_study'), str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import Trajectory

from saddleflow.constr_gen import force_rms_and_max, uma_forces
from saddleflow.data import TrajTripletDataset, atoms_to_sample_dict
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import TrainingConfig, load_uma_backbone, train
from saddleflow.utils.forces import load_uma_force_head

try:  # Works when invoked as `python examples/LiC/train.py ...`.
    from recovery_pairs_dataset import (  # noqa: E402
        RecoveryMixedDataset,
        build_recovery_records,
        write_recovery_artifacts,
    )
except ModuleNotFoundError:  # Also support module-style invocation/imports.
    from examples.LiC.recovery_pairs_dataset import (  # noqa: E402
        RecoveryMixedDataset,
        build_recovery_records,
        write_recovery_artifacts,
    )


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--study-split", help="immutable source-study split directory, prepared by source_study.py")
    p.add_argument("--train-saddles", type=int, choices=[12, 48], default=12)
    p.add_argument("--source-conditioning", action=argparse.BooleanOptionalAction, default=False,
                   help="retain the actual noisy initial x0 and inject MIC(x0-x_t); no saddle/perturbation target supplied")
    p.add_argument("--source-channels", type=int, default=32, help="study arms use the same branch width, with zeros in unconditioned arms")
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
    p.add_argument("--save-every-steps", type=int, default=0,
                   help="Also save a checkpoint every N optimizer steps. 0 = disable. "
                        "Use this rather than epochs when a run must be resumable after "
                        "a bounded wall-time allocation.")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after this many optimizer updates and use it as the LR-schedule "
                        "horizon. 0 = num_epochs × batches/epoch. Useful when an augmented "
                        "dataset makes epoch count less intuitive.")

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--unfreeze-uma-all", action=argparse.BooleanOptionalAction, default=True,
                   help="Train all UMA backbone blocks at --uma-lr. "
                        "--no-unfreeze-uma-all trains the head only.")
    p.add_argument("--early-time-film", action=argparse.BooleanOptionalAction, default=True,
                   help="Let the flow time t modulate the backbone blocks in "
                        "--early-time-film-blocks (equivariant FiLM).")
    p.add_argument("--early-time-film-blocks", default="-2,-1")
    p.add_argument("--head-depth", type=int, default=1)
    p.add_argument("--inject-force", action=argparse.BooleanOptionalAction, default=False,
                   help="feed detached frozen-UMA current-state forces to the velocity head; also needed at inference.")
    p.add_argument("--force-field-channels", type=int, default=32)
    p.add_argument("--curvature-conditioning", action=argparse.BooleanOptionalAction, default=False,
                   help="scalar FiLM from current free-Hessian lambda1, lambda2, eigengap; NOT a curvature loss.")
    p.add_argument("--curvature-mlp-hidden", type=int, default=64)
    p.add_argument("--curvature-fd-eps", type=float, default=.002, help="Li-only central FD displacement, A.")
    p.add_argument("--physics-batch-size", type=int, default=32, help="frozen UMA geometry chunk size, including FD probes.")
    p.add_argument("--attn-layers", type=int, default=0)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init-weights", default=None,
                   help="Optional checkpoint directory whose EMA weights are loaded before "
                        "a fresh fine-tuning optimizer/schedule begins. This matches LiC "
                        "inference, which also uses EMA weights.")
    p.add_argument("--resume-from", default=None,
                   help="Resume optimizer, LR scheduler, model, and EMA state from a prior "
                        "checkpoint directory. Use the same --output-dir and --num-epochs "
                        "as the interrupted run; cannot be combined with --init-weights.")

    # Optional data aggregation from a completed robustness-shell experiment.
    # The shell archive already contains every source, its parent saddle, and
    # frozen-UMA force at the baseline rollout endpoint.  Selecting high-force
    # sources creates ordinary supervised x0 -> parent-saddle FM pairs; it does
    # not add a new physics loss to this initial mechanistic experiment.
    p.add_argument("--recovery-shell-archive", default=None,
                   help="Optional robustness_shells.npz archive. Rows whose saved baseline "
                        "endpoint force exceeds --recovery-force-epsilon become recovery pairs.")
    p.add_argument("--recovery-force-epsilon", type=float, default=0.010,
                   help="eV/A. Mine a shell source when its saved baseline endpoint free-atom "
                        "UMA RMS force is strictly greater than this value. Default 0.010.")
    p.add_argument("--recovery-original-fraction", type=float, default=0.75,
                   help="Fraction of original stochastic TS-denoising records in each augmented "
                        "epoch. Remaining records are exact high-force recovery pairs.")

    # Optional frozen-UMA training signals.  The two modes are intentionally
    # exclusive: direct force loss differentiates through UMA, whereas the
    # stop-gradient mode uses UMA only to reweight normal supervised FM loss.
    force_mode = p.add_mutually_exclusive_group()
    force_mode.add_argument("--force-constrained", dest="force_constrained", action="store_true",
                            help="Add a differentiable UMA force dead-band loss at the algebraic "
                                 "endpoint implied by the current velocity prediction.")
    force_mode.add_argument("--force-constraint-stopgrad", action="store_true",
                            help="Use frozen UMA only as a detached force gate: high-force "
                                 "late-time predictions receive more normal supervised FM loss, "
                                 "without backpropagating through UMA forces.")
    p.add_argument("--force-constraint-weight", type=float, default=0.1,
                   help="Beta multiplying the direct force loss or the detached gated-FM term.")
    p.add_argument("--force-epsilon", type=float, default=0.018,
                   help="UMA free-atom RMS-force dead band in eV/A. Default 0.018 is "
                        "the observed median full-rollout force of successful LiC "
                        "true-saddle-noise samples, not the stricter static-label force.")
    p.add_argument("--force-constraint-min-t", type=float, default=0.75,
                   help="Apply the one-shot UMA loss only when sampled flow time t is at "
                        "least this value. Ordinary FM loss still uses every t.")
    p.add_argument("--force-epsilon-from-static-saddles", action="store_true",
                   help="Use the old static-label percentile calibration instead of the "
                        "model-rollout-oriented --force-epsilon value.")
    p.add_argument("--force-epsilon-percentile", type=float, default=99.0,
                   help="Percentile of static true-saddle forces when --force-epsilon-from-static-saddles "
                        "is explicitly requested (default: 99).")
    p.add_argument("--force-rollout-samples-per-batch", type=int, default=1,
                   help="Number of random batch structures receiving UMA force evaluation. "
                        "Recommended LiC first ablation: 1.")
    p.add_argument("--force-rollout-every-n-batches", type=int, default=4,
                   help="Evaluate UMA force every N optimizer batches. "
                       "Recommended LiC first ablation: 4.")
    p.add_argument("--force-constraint-stopgrad-max-gate", type=float, default=10.0,
                   help="Cap on relu(f_RMS / epsilon - 1) in --force-constraint-stopgrad "
                        "mode. Prevents one high-force prediction dominating a batch. "
                        "Default: 10; use a negative value to disable the cap.")
    p.add_argument("--force-calibration-trajs", nargs="+",
                   default=[str(here / "train_set.traj"), str(here / "test_set.traj")],
                   help="LiC trajectories whose stored S frames calibrate an omitted "
                        "--force-epsilon. Defaults to all 12 train + 167 test saddles.")
    p.add_argument("--force-calibration-batch-size", type=int, default=4,
                   help="Frozen-UMA batch size used only while calibrating an omitted "
                        "--force-epsilon from stored saddle labels.")
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
        if args.force_constrained:
            name += "_lateforceconstraint"
        if args.force_constraint_stopgrad:
            name += "_lateforcegate"
        if args.inject_force: name += "_forceinput"
        if args.curvature_conditioning: name += "_curvatureinput"
        if args.source_conditioning: name += "_sourceinput"
        if args.study_split: name += f"_n{args.train_saddles}"
        args.output_dir = str(here / "runs" / name)
    return args


def calibrate_force_epsilon(
    traj_paths: list[str],
    *,
    percentile: float,
    force_backbone,
    force_head,
    force_tasks: dict,
    device: str,
    batch_size: int,
) -> tuple[float, dict]:
    """Calibrate LiC's dead band from stored, independently known saddles.

    This is deliberately a calibration step, not a training target.  It reads
    the middle ``S`` frame of every ``[R, S, P]`` triplet in the supplied LiC
    trajectories, evaluates frozen UMA forces, and returns the requested
    percentile of the same free-atom RMS metric used by the new loss.
    """
    if not 0 < percentile <= 100:
        raise ValueError("--force-epsilon-percentile must lie in (0, 100]")
    if batch_size <= 0:
        raise ValueError("--force-calibration-batch-size must be positive")

    saddle_samples = []
    for traj_path in traj_paths:
        trajectory = Trajectory(traj_path, "r")
        try:
            if len(trajectory) % 3:
                raise ValueError(f"{traj_path} does not contain [R, S, P] triplets")
            saddle_samples.extend(
                atoms_to_sample_dict(trajectory[frame + 1])
                for frame in range(0, len(trajectory), 3)
            )
        finally:
            trajectory.close()
    if not saddle_samples:
        raise ValueError("force-epsilon calibration found no saddle frames")

    template = saddle_samples[0]
    for sample in saddle_samples[1:]:
        same_layout = (
            torch.equal(sample["Z"], template["Z"])
            and torch.equal(sample["fixed"], template["fixed"])
            and torch.allclose(sample["cell"], template["cell"])
            and sample["task_name"] == template["task_name"]
            and sample["charge"] == template["charge"]
            and sample["spin"] == template["spin"]
        )
        if not same_layout:
            raise ValueError(
                "LiC epsilon calibration batches require a common atom layout, cell, "
                "constraints, and UMA task metadata. Supply --force-epsilon explicitly "
                "for a heterogeneous dataset."
            )

    values: list[float] = []
    mobile = ~template["fixed"]
    for start in range(0, len(saddle_samples), batch_size):
        chunk = saddle_samples[start:start + batch_size]
        positions = torch.stack([sample["start_pos"] for sample in chunk])
        forces = uma_forces(
            positions,
            Z=template["Z"],
            cell=template["cell"],
            task_name=template["task_name"],
            charge=template["charge"],
            spin=template["spin"],
            fixed=template["fixed"],
            force_backbone=force_backbone,
            force_head=force_head,
            force_tasks=force_tasks,
            device=device,
        )
        force_rms, _ = force_rms_and_max(forces, mobile)
        values.extend(force_rms.cpu().tolist())

    array = np.asarray(values, dtype=float)
    epsilon = float(np.percentile(array, percentile))
    summary = {
        "num_true_saddles": len(values),
        "metric": "free-atom RMS force (eV/A)",
        "percentile": float(percentile),
        "epsilon": epsilon,
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "trajectory_paths": [str(path) for path in traj_paths],
    }
    return epsilon, summary


def main():
    args = parse_args()
    force_constraint_enabled = bool(
        args.force_constrained or args.force_constraint_stopgrad
    )
    physics_enabled = args.inject_force or args.curvature_conditioning
    source_channels = args.source_channels if args.study_split or args.source_conditioning else 0
    split_sha256 = None
    if args.study_split:
        try: from source_study import load_split, file_hash
        except ModuleNotFoundError: from examples.LiC.experiments.source_study.source_study import load_split, file_hash
        split_dir = Path(args.study_split).resolve()
        load_split(split_dir)
        split_sha256 = file_hash(split_dir/"split.json")
        args.train_traj = str(split_dir/f"train{args.train_saddles}.traj")
        if not args.max_steps: args.max_steps = 96000
        if physics_enabled or force_constraint_enabled or args.recovery_shell_archive:
            raise SystemExit("the matched source study uses plain FM only; no physics inputs/losses or recovery buffer")
    elif args.train_saddles != 12:
        raise SystemExit("--train-saddles 48 requires --study-split")
    if source_channels < 0 or ((args.study_split or args.source_conditioning) and source_channels < 1):
        raise SystemExit("--source-channels must be positive for source-study models")
    if args.force_field_channels < 1 or args.curvature_mlp_hidden < 1 or args.physics_batch_size < 1 or args.curvature_fd_eps <= 0:
        raise SystemExit("conditioning widths, physics batch size, and FD epsilon must be positive")
    # EMA is positional: added input layers cannot silently inherit an old
    # unconditioned EMA. The first matched ablation uses the baseline recipe
    # from scratch; continuation/resume is supported for matching architectures.
    for checkpoint in [args.init_weights, args.resume_from]:
        if not checkpoint: continue
        old = json.loads((Path(checkpoint).parent / "config.json").read_text()).get("extras", {})
        if (int(old.get("force_field_channels", 0)) != (args.force_field_channels if args.inject_force else 0)
                or bool(old.get("curvature_conditioning", False)) != args.curvature_conditioning
                or (args.curvature_conditioning and int(old.get("curvature_mlp_hidden", 64)) != args.curvature_mlp_hidden)
                or int(old.get("source_channels", 0)) != source_channels
                or bool(old.get("source_conditioning", False)) != args.source_conditioning):
            raise SystemExit("checkpoint conditioning architecture differs; train the new input branches from scratch first")
        if args.study_split and (old.get("study_split_sha256") != split_sha256 or old.get("train_saddle_count") != args.train_saddles):
            raise SystemExit("cannot initialize/resume this study arm from a different data split")
    if args.force_constraint_stopgrad and args.force_constraint_stopgrad_max_gate < 0:
        args.force_constraint_stopgrad_max_gate = None
    if args.ts_denoise_sigma <= 0:
        raise SystemExit("--ts-denoise-sigma must be positive")
    if args.init_weights and args.resume_from:
        raise SystemExit("--init-weights and --resume-from are mutually exclusive")
    if args.save_every_epochs < 0:
        raise SystemExit("--save-every-epochs must be non-negative")
    if args.save_every_steps < 0:
        raise SystemExit("--save-every-steps must be non-negative")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative")
    if args.recovery_shell_archive:
        if args.recovery_force_epsilon <= 0:
            raise SystemExit("--recovery-force-epsilon must be positive")
        if not 0 < args.recovery_original_fraction < 1:
            raise SystemExit("--recovery-original-fraction must lie strictly between 0 and 1")
    if force_constraint_enabled:
        if args.force_constraint_weight < 0:
            raise SystemExit("--force-constraint-weight must be non-negative")
        if args.force_rollout_samples_per_batch <= 0:
            raise SystemExit("--force-rollout-samples-per-batch must be positive")
        if args.force_rollout_every_n_batches <= 0:
            raise SystemExit("--force-rollout-every-n-batches must be positive")
        if not 0 <= args.force_constraint_min_t < 1:
            raise SystemExit("--force-constraint-min-t must lie in [0, 1)")
        if args.force_epsilon <= 0:
            raise SystemExit("--force-epsilon must be positive")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] dataset: {args.train_traj}")
    original_dataset = TrajTripletDataset(
        args.train_traj, stats_cache=str(out_dir / "dataset_stats.json")
    )
    dataset = original_dataset
    if args.study_split:
        batches_per_epoch = (len(dataset)+args.batch_size-1)//args.batch_size
        if args.num_epochs*batches_per_epoch < args.max_steps:
            raise SystemExit("--num-epochs is too small for the requested matched --max-steps budget")
        print(f"[study] N={args.train_saddles}; source conditioning={args.source_conditioning}; "
              f"identical {source_channels}-channel branch in both arms; budget={args.max_steps} updates; "
              f"common held-out set=131; split SHA256={split_sha256}", flush=True)
    print(f"[train] {len(original_dataset)} original records "
          f"({original_dataset.num_triplets} triplets x 2 sides)")
    recovery_summary = None
    if args.recovery_shell_archive:
        recovery_records, recovery_summary, recovery_arrays = build_recovery_records(
            args.recovery_shell_archive,
            original_dataset,
            force_epsilon=args.recovery_force_epsilon,
        )
        summary_path, buffer_path = write_recovery_artifacts(
            out_dir, recovery_summary, recovery_arrays
        )
        dataset = RecoveryMixedDataset(
            original_dataset,
            recovery_records,
            original_fraction=args.recovery_original_fraction,
        )
        print(
            "[train] recovery data aggregation: "
            f"{len(recovery_records)} shell sources selected by saved UMA endpoint "
            f"force > {args.recovery_force_epsilon:.6f} eV/A"
        )
        print(
            "[train] recovery selection by radius: "
            + ", ".join(
                f"{radius} A: {count}"
                for radius, count in recovery_summary["selected_count_by_radius_angstrom"].items()
            )
        )
        print(
            "[train] mixed epoch: "
            f"{dataset.n_original_per_epoch} original stochastic records + "
            f"{len(recovery_records)} exact recovery records = {len(dataset)} total "
            f"({100 * args.recovery_original_fraction:.1f}% original target fraction)"
        )
        print(f"[train] wrote recovery provenance -> {summary_path}")
        print(f"[train] wrote selected recovery buffer -> {buffer_path}")
    print(f"[train] TS-denoise: x_0 = saddle + N(0, {args.ts_denoise_sigma}^2), x_1 = saddle")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    # Seed BEFORE module construction, not only inside the generic train loop.
    # Optional input branches preserve the common head's initialization/RNG.
    torch.manual_seed(args.seed)
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
                        depth=args.head_depth,
                        force_field_channels=args.force_field_channels if args.inject_force else 0,
                        curvature_conditioning=args.curvature_conditioning,
                        curvature_mlp_hidden=args.curvature_mlp_hidden,
                        identity_force_init=args.inject_force,
                        source_channels=source_channels,
                        source_conditioning=args.source_conditioning).to(args.device)

    force_backbone = None
    force_head = None
    force_tasks = None
    force_epsilon = None
    force_calibration = None
    if force_constraint_enabled or physics_enabled:
        # This is intentionally a second, fully frozen UMA copy.  The flow
        # backbone may be fine-tuned; the physical stationarity detector must
        # remain the original pretrained UMA throughout the run.
        print(f"[train] loading separate frozen UMA force model {args.backbone!r}")
        force_backbone = load_uma_backbone(
            args.backbone, device=args.device, freeze=True, eval_mode=True,
        )
        force_head, force_tasks = load_uma_force_head(args.backbone, device=args.device)
    if force_constraint_enabled:
        if args.force_epsilon_from_static_saddles:
            print("[train] calibrating force epsilon from stored true LiC saddles …")
            force_epsilon, force_calibration = calibrate_force_epsilon(
                args.force_calibration_trajs,
                percentile=args.force_epsilon_percentile,
                force_backbone=force_backbone,
                force_head=force_head,
                force_tasks=force_tasks,
                device=args.device,
                batch_size=args.force_calibration_batch_size,
            )
            (out_dir / "force_epsilon_calibration.json").write_text(
                json.dumps(force_calibration, indent=2) + "\n"
            )
            print(
                "[train] force epsilon: "
                f"p{args.force_epsilon_percentile:g} of "
                f"{force_calibration['num_true_saddles']} true saddles = "
                f"{force_epsilon:.6f} eV/A "
                f"(p95={force_calibration['p95']:.6f}, "
                f"p99={force_calibration['p99']:.6f})"
            )
        else:
            force_epsilon = float(args.force_epsilon)
            print(
                "[train] force constraint: "
                f"late-time rollout-oriented epsilon={force_epsilon:.6f} eV/A"
            )
        mode = (
            "stop-gradient force-gated FM"
            if args.force_constraint_stopgrad else "direct force penalty"
        )
        print(
            f"[train] force constraint ({mode}): "
            f"weight={args.force_constraint_weight:g}  "
            f"samples/batch={args.force_rollout_samples_per_batch}  "
            f"every={args.force_rollout_every_n_batches} batches  "
            f"min_t={args.force_constraint_min_t:g}  metric=free-atom RMS"
        )
        if args.force_constraint_stopgrad:
            print(
                "[train] force gate: "
                f"max_gate={args.force_constraint_stopgrad_max_gate}  "
                "UMA force evaluation is detached; only the supervised FM gradient is used"
            )

    physics_conditioner = None
    if physics_enabled:
        try: from conditioning import make_conditioner
        except ModuleNotFoundError: from examples.LiC.rollout_eval.conditioning import make_conditioner
        physics_conditioner = make_conditioner(force_backbone, force_head, force_tasks,
            inject_force=args.inject_force, curvature=args.curvature_conditioning,
            fd_eps=args.curvature_fd_eps, probe_batch_size=args.physics_batch_size)
        queries = int(args.inject_force) + 6*int(args.curvature_conditioning)
        print(f"[train] force input={args.inject_force}; scalar curvature={args.curvature_conditioning}; "
              f"{queries} UMA geometries/state, about {queries*args.batch_size}/batch; also queried at inference.", flush=True)
        print("[train] ordinary FM targets/data/time sampling unchanged; no MMF, eigenmode, or extra force loss unless separately requested.", flush=True)

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

    flow_config = FlowMatchingConfig(
        mode=1,
        ts_denoise_sigma=args.ts_denoise_sigma,
        force_constraint_enabled=force_constraint_enabled,
        force_constraint_weight=args.force_constraint_weight,
        force_constraint_epsilon=force_epsilon,
        force_constraint_samples_per_batch=args.force_rollout_samples_per_batch,
        force_constraint_every_n_batches=args.force_rollout_every_n_batches,
        force_constraint_min_time=args.force_constraint_min_t,
        force_constraint_stopgrad=bool(args.force_constraint_stopgrad),
        force_constraint_stopgrad_max_gate=args.force_constraint_stopgrad_max_gate,
    )
    loss_module = FlowMatchingLoss(
        flow_config,
        backbone,
        attn,
        head,
        force_head=force_head,
        force_tasks=force_tasks,
        frozen_force_backbone=force_backbone,
        physics_conditioner=physics_conditioner,
    )

    train_cfg = TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay,
        mixed_precision=args.mixed_precision, seed=args.seed,
        log_every=args.log_every,
        save_every_epochs=args.save_every_epochs or args.num_epochs + 1,
        save_every_steps=args.save_every_steps,
        init_weights=args.init_weights,
        # LiC inference scripts load checkpoint EMA weights by default.  Keep
        # fine-tuning on exactly that same model rather than the raw final
        # optimizer iterate in model.safetensors.
        init_weights_use_ema=bool(args.init_weights),
        resume_from=args.resume_from,
        # random_throw.py rebuilds the network from these keys.
        extras={
            "mode": 1,
            "ts_denoise_sigma": args.ts_denoise_sigma,
            "delta_endpoint_channels": 0,
            "backbone": args.backbone,
            "attn_layers": args.attn_layers, "attn_heads": args.attn_heads,
            "head_depth": args.head_depth,
            "train_traj": str(Path(args.train_traj).resolve()),
            "source_conditioning": args.source_conditioning,
            "source_channels": source_channels,
            "study_split": str(Path(args.study_split).resolve()) if args.study_split else None,
            "study_split_sha256": split_sha256,
            "train_saddle_count": args.train_saddles if args.study_split else original_dataset.num_triplets,
            "evaluation_train_traj": str(Path(args.train_traj).resolve()) if args.study_split else None,
            "evaluation_test_traj": str(Path(args.study_split).resolve()/
                ("heldout131.traj" if args.train_saddles == 48 else "catalog_rest.traj")) if args.study_split else None,
            "inject_force": bool(args.inject_force),
            "force_field_channels": args.force_field_channels if args.inject_force else 0,
            "curvature_conditioning": bool(args.curvature_conditioning),
            "curvature_mlp_hidden": args.curvature_mlp_hidden,
            "curvature_fd_eps": args.curvature_fd_eps,
            "physics_batch_size": args.physics_batch_size,
            "unfreeze_uma_all": bool(args.unfreeze_uma_all),
            "early_time_film": bool(args.early_time_film),
            "early_time_film_blocks": args.early_time_film_blocks,
            "uma_lr": args.uma_lr if args.unfreeze_uma_all else None,
            "force_constraint_enabled": force_constraint_enabled,
            "force_constraint_mode": (
                "stopgrad_gate" if args.force_constraint_stopgrad
                else "direct_force" if args.force_constrained else None
            ),
            "force_constraint_weight": args.force_constraint_weight if force_constraint_enabled else None,
            "force_constraint_epsilon": force_epsilon,
            "force_constraint_epsilon_source": (
                "static_true_saddle_percentile"
                if force_constraint_enabled and args.force_epsilon_from_static_saddles
                else "late_time_rollout_reference"
                if force_constraint_enabled else None
            ),
            "force_constraint_epsilon_percentile": (
                args.force_epsilon_percentile
                if force_constraint_enabled and args.force_epsilon_from_static_saddles else None
            ),
            "force_constraint_min_time": (
                args.force_constraint_min_t if force_constraint_enabled else None
            ),
            "force_constraint_samples_per_batch": (
                args.force_rollout_samples_per_batch if force_constraint_enabled else None
            ),
            "force_constraint_every_n_batches": (
                args.force_rollout_every_n_batches if force_constraint_enabled else None
            ),
            "force_constraint_stopgrad_max_gate": (
                args.force_constraint_stopgrad_max_gate
                if args.force_constraint_stopgrad else None
            ),
            "recovery_shell_archive": (
                str(Path(args.recovery_shell_archive).resolve())
                if args.recovery_shell_archive else None
            ),
            "recovery_force_epsilon": (
                args.recovery_force_epsilon if args.recovery_shell_archive else None
            ),
            "recovery_original_fraction": (
                args.recovery_original_fraction if args.recovery_shell_archive else None
            ),
            "recovery_pairs_selected": (
                recovery_summary["recovery_pairs_selected"] if recovery_summary else None
            ),
        },
    )
    train(loss_module, dataset, train_cfg, param_groups=param_groups)


if __name__ == "__main__":
    main()
