"""Train global/state-dependent MMF mobilities on noisy TRAINING saddles only.

Michael: this is a new small solver policy, not FM-checkpoint fine-tuning.
The frozen UMA potential provides forces and numerical minimum modes. The only
objective is actual-rollout endpoint MSE. No test labels, false bridges, recovery
sources, force penalty, or learned orientation enter training.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from ase.io import read

from saddleflow.classical_saddle_search import FreeDOFs, PESBackend, make_uma_calculator
from learned_mobility import MobilityPolicy, endpoint_loss, run_mobility_flow


def parse_args():
    here = Path(__file__).resolve().parents[2]  # examples/LiC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--policies", default="global,state", help="Train both controls with matched noise draws.")
    p.add_argument("--num-updates", type=int, default=200, help="Optimizer updates PER policy, not epochs.")
    p.add_argument("--batch-size", type=int, default=1, help="Start at 1: each source needs many UMA queries.")
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--noise-sigma", type=float, default=0.25)
    p.add_argument("--max-steps", type=int, default=100, help="Shared training/evaluation horizon.")
    p.add_argument("--step-size", type=float, default=0.2, help="Base mobility, A^2/eV; NOT 1/K FM time.")
    p.add_argument("--max-step", type=float, default=0.1, help="Per-free-atom RMS trust radius, A.")
    p.add_argument("--fd-eps", type=float, default=0.002)
    p.add_argument("--fmax", type=float, default=0.005)
    p.add_argument("--eigenvalue-tol", type=float, default=0.01)
    p.add_argument("--mobility-min", type=float, default=0.05)
    p.add_argument("--mobility-max", type=float, default=5.0)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--task-name", default="omat")
    p.add_argument("--device", default="cuda", help="UMA device. The tiny policy/rollout graph stays on CPU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--resume", action="store_true", help="Restore policy, Adam state, and source RNG from latest.")
    return p.parse_args()


def load_training_saddles(path):
    """Read only train_set.traj; the R/P frames never enter the objective."""
    frames = read(path, index=":")
    if not frames or len(frames) % 3:
        raise ValueError("Expected flat R,S,P triplets in the training trajectory.")
    saddles = frames[1::3]
    reference = saddles[0].copy()
    dofs = FreeDOFs(reference)
    if dofs.dimension != 3 or reference.numbers[dofs.free].tolist() != [3]:
        raise ValueError("This training experiment expects one free Li; the solver itself is generic.")
    for frame in saddles:
        if (not np.array_equal(frame.numbers, reference.numbers)
                or not np.allclose(frame.cell, reference.cell)
                or not np.array_equal(frame.pbc, reference.pbc)
                or not np.array_equal(FreeDOFs(frame).free, dofs.free)
                or not np.allclose(frame.positions[~dofs.free], reference.positions[~dofs.free])):
            raise ValueError("Training saddles must share atom ordering, constraints, cell, and fixed scaffold.")
    return reference, np.asarray([dofs.pack(frame.positions) for frame in saddles])


def rollout_options(config):
    """One set of physical solver settings shared by raw/global/state policies."""
    return dict(max_steps=config["max_steps"], step_size=config["step_size"],
                max_step=config["max_step"], fd_eps=config["fd_eps"],
                fmax_tol=config["fmax"], eigenvalue_tol=config["eigenvalue_tol"])


def write_checkpoint(path, policy, optimizer, rng, update, config, history):
    """Tiny resumable checkpoint: UMA weights are never copied into this file."""
    state = dict(policy_kind=policy.kind, policy_state=policy.state_dict(),
                 optimizer_state=optimizer.state_dict(), source_rng=rng.bit_generator.state,
                 update=update, config=config, history=history)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_policy(path):
    """Restore one of our small policy checkpoints, without loading UMA."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    policy = MobilityPolicy(state["policy_kind"], cfg["mobility_min"],
                            cfg["mobility_max"], cfg["hidden"])
    policy.load_state_dict(state["policy_state"])
    return policy, state


def train_policy(kind, args, reference, saddles, calculator, batch_evaluate):
    directory = Path(args.out_dir) / kind
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "checkpoint_latest.pt"
    torch.manual_seed(args.seed)
    policy = MobilityPolicy(kind, args.mobility_min, args.mobility_max, args.hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    # Both policies visit identical parents/noise, independent of MLP initialization
    # and of the other policy's convergence decisions.
    rng = np.random.default_rng(args.seed)
    first, history = 0, []
    if args.resume and latest.is_file():
        policy, state = load_policy(latest)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(state["optimizer_state"])
        rng.bit_generator.state = state["source_rng"]
        first, history = state["update"], state["history"]
        if first > args.num_updates:
            raise ValueError("--num-updates cannot be smaller than the completed checkpoint update.")
        print(f"[resume] {kind}: update {first}/{args.num_updates}", flush=True)
    elif latest.exists() or (directory / "checkpoint_final.pt").exists():
        raise FileExistsError(f"Use --resume or a new output directory: {directory}")
    policy.train()
    n_parameters = sum(p.numel() for p in policy.parameters())
    print(f"[train] {kind}: {n_parameters} trainable parameters; initial a=b=1 for a fresh run", flush=True)
    options = rollout_options(vars(args))
    for update in range(first, args.num_updates):
        optimizer.zero_grad(set_to_none=True)
        losses, stats, scales = [], [], []
        began = perf_counter()
        parents = rng.integers(len(saddles), size=args.batch_size)
        starts = saddles[parents] + rng.normal(0, args.noise_sigma, (args.batch_size, saddles.shape[1]))
        print(f"[train] {kind} update {update+1}/{args.num_updates} begin; parents={parents.tolist()}", flush=True)
        for index, (parent, source) in enumerate(zip(parents, starts)):
            pes = PESBackend(reference, calculator, batch_evaluate)

            def log_step(step, row):
                if step % args.log_every_steps == 0:
                    print(f"  sample {index+1}/{args.batch_size} step {step:3d}/{args.max_steps}"
                          f" PES={row['query_count']:4d} fmax={row['force_max']:.6f}"
                          f" a={row['mobility_a']:.3f} b={row['mobility_b']:.3f}", flush=True)

            result = run_mobility_flow(pes, source, policy, callback=log_step, **options)
            losses.append(endpoint_loss(result["terminal_tensor"], saddles[parent], pes.dofs, policy))
            evaluated = [row for row in result["trace"] if row["q"] is not None]
            scales.extend((row["mobility_a"], row["mobility_b"]) for row in evaluated)
            stats.append((result["steps"], result["force_calls"], result["status"] == "converged",
                          evaluated[-1]["force_max"], np.mean([row["clipped"] for row in evaluated])))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite endpoint loss; latest completed checkpoint is preserved.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        stats, scales = np.asarray(stats), np.asarray(scales)
        row = dict(update=update+1, endpoint_mse=float(loss.detach()), grad_norm=float(grad_norm),
                   mean_steps=float(stats[:, 0].mean()), force_calls=int(stats[:, 1].sum()),
                   converged_fraction=float(stats[:, 2].mean()), mean_fmax=float(stats[:, 3].mean()),
                   clipped_fraction=float(stats[:, 4].mean()), mean_a=float(scales[:, 0].mean()),
                   mean_b=float(scales[:, 1].mean()), seconds=perf_counter()-began)
        history.append(row)
        if (update+1) % args.log_every == 0 or update == first:
            print(f"[train] {kind} update {update+1:5d}/{args.num_updates} loss={row['endpoint_mse']:.8f}"
                  f" grad={row['grad_norm']:.4g} steps={row['mean_steps']:.1f}"
                  f" phys={100*row['converged_fraction']:.1f}% a={row['mean_a']:.3f} b={row['mean_b']:.3f}"
                  f" PES={row['force_calls']} seconds={row['seconds']:.1f}", flush=True)
        if (update+1) % args.save_every == 0 or update+1 == args.num_updates:
            write_checkpoint(latest, policy, optimizer, rng, update+1, vars(args), history)
            (directory / "training_log.json").write_text(json.dumps(history, indent=2))
            print(f"[save] {latest}", flush=True)
    write_checkpoint(directory / "checkpoint_final.pt", policy, optimizer, rng,
                     max(first, args.num_updates), vars(args), history)
    return history


def main():
    args = parse_args()
    kinds = args.policies.split(",")
    if not kinds or not set(kinds) <= {"global", "state"} or len(set(kinds)) != len(kinds):
        raise ValueError("--policies must be global,state or one of those names.")
    if min(args.num_updates, args.batch_size, args.save_every, args.log_every, args.log_every_steps,
           args.max_steps, args.hidden) < 1 or min(args.noise_sigma, args.learning_rate, args.grad_clip) <= 0:
        raise ValueError("Positive training counts, noise, LR, and clipping required.")
    MobilityPolicy("raw", args.mobility_min, args.mobility_max, args.hidden)  # validate bounds before loading UMA
    torch.set_num_threads(1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if not args.resume:
            raise FileExistsError(f"Use --resume or a new directory: {out}")
        mutable = {"num_updates", "resume", "out_dir", "log_every", "log_every_steps", "save_every", "device"}
        for key in set(vars(args)) - mutable:
            if old[key] != getattr(args, key):
                raise ValueError(f"Resume changed {key}: {old[key]} -> {getattr(args, key)}")
    reference, saddles = load_training_saddles(args.train_traj)
    config_path.write_text(json.dumps(vars(args), indent=2))
    print(f"[setup] Michael: {len(saddles)} TRAINING saddles; sigma={args.noise_sigma:g} A;"
          f" Kmax={args.max_steps}; policies={kinds}", flush=True)
    print(f"[setup] h={args.step_size:g} A^2/eV; trust={args.max_step:g} A;"
          f" fmax={args.fmax:g}; eigenvalue margin={args.eigenvalue_tol:g}", flush=True)
    print("[setup] Endpoint MSE only; local backward dF/dx=-H, dq/dx=0. No FM/force/recovery loss.", flush=True)
    print("[UMA] loading frozen potential; checkpoint files contain only the tiny policy and optimizer", flush=True)
    calculator, batch_evaluate = make_uma_calculator(args.model, args.device, args.task_name)
    warm = PESBackend(reference, calculator, batch_evaluate)
    warm.forces(saddles[0])
    warm.free_hessian(saddles[0], args.fd_eps)
    print("[UMA] ready", flush=True)
    histories = {}
    for kind in kinds:
        histories[kind] = train_policy(kind, args, reference, saddles, calculator, batch_evaluate)
    lines = ["Michael: LiC oracle-MMF mobility training", "=" * 85,
             f"Training parents: {len(saddles)}; noise sigma={args.noise_sigma:g} A; Kmax={args.max_steps}",
             "policy       updates   last MSE(A^2)  mean steps  phys(%)  mean a  mean b  total PES  seconds",
             "-" * 95]
    for kind, history in histories.items():
        row = history[-1]
        lines.append(f"{kind:12s} {row['update']:7d} {row['endpoint_mse']:15.8f} {row['mean_steps']:11.1f}"
                     f" {100*row['converged_fraction']:8.1f} {row['mean_a']:7.3f} {row['mean_b']:7.3f}"
                     f" {sum(r['force_calls'] for r in history):10d} {sum(r['seconds'] for r in history):8.1f}")
    lines += ["", "Training metrics above are stochastic-batch diagnostics, NOT test accuracy.",
              "Early stopping is physical, not based on target distance. The discrete stop is not differentiated.",
              "Endpoint loss alone does not explicitly reward faster convergence."]
    report = "\n".join(lines) + "\n"
    (out / "training_report.txt").write_text(report)
    print("\n" + report, flush=True)
    print(f"[done] evaluate A/B/C: python examples/LiC/experiments/mmf_mobility/compare_mmf_mobility.py --run-dir {out} --out-dir {out / 'comparison'}", flush=True)


if __name__ == "__main__":
    main()
