"""Matched E0/E1/E2 continuation, not minimum-action or MMF rollout training.

First run --calibrate-only. Then train E0, E1, E2 in separate invocations with
the SAME calibration directory. Only the auxiliary direction changes.
No held-out labels, false bridges, recovery buffer, or endpoint penalties enter
this file. Checkpoints remain compatible with the existing LiC EMA samplers.
"""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from ase.io import read
from safetensors.torch import load_model, save_model

from pes_direction import alignment, directional_loss, query_teacher
from saddleflow.data import TrajTripletDataset
from saddleflow.data.transforms import gaussian_perturbation, wrap_positions
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.classical_saddle_search import PESBackend, make_uma_calculator
from saddleflow.utils.training import EMA, _lr_lambda

try:
    from force_guided_sampling import load_flow_model
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model


DEFAULT_CHECKPOINT = "/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-weights", default=DEFAULT_CHECKPOINT)
    p.add_argument("--train-traj", default=str((Path(__file__).resolve().parents[2] / "train_set.traj")))
    p.add_argument("--calibration-dir", required=True)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--experiment", choices=["E0", "E1", "E2"], default="E0")
    p.add_argument("--output-dir")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--uma-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--noise-sigma", type=float, default=0.25)
    p.add_argument("--fd-eps", type=float, default=0.002)
    p.add_argument("--velocity-floor", type=float, default=1e-4)
    p.add_argument("--force-floor", type=float, default=1e-6)
    p.add_argument("--gradient-ratio", type=float, default=0.1)
    p.add_argument("--calibration-batches", type=int, default=16)
    p.add_argument("--audit-samples-per-saddle", type=int, default=16)
    p.add_argument("--save-steps", default="0,250,500,1000,2000")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume-from", help="Our checkpoint directory; restores raw weights, Adam, EMA and RNG.")
    return p.parse_args()


def scientific_settings(args):
    """Guard against accidentally using calibration from another experiment."""
    checkpoint = Path(args.init_weights).resolve()
    ema = checkpoint / "ema.pt"
    return dict(init_checkpoint=str(checkpoint), ema_size=ema.stat().st_size,
                ema_mtime_ns=ema.stat().st_mtime_ns,
                training_data_sha256=hashlib.sha256(Path(args.train_traj).read_bytes()).hexdigest(),
                batch_size=args.batch_size, noise_sigma=args.noise_sigma, fd_eps=args.fd_eps,
                velocity_floor=args.velocity_floor, force_floor=args.force_floor,
                gradient_ratio=args.gradient_ratio, seed=args.seed)


def build_student(args):
    """Reuse exact EMA reconstruction; teacher modules are NEVER student attributes."""
    cfg, name, backbone, attention, head = load_flow_model(Path(args.init_weights), args.device)
    ex = cfg["extras"]
    if ex.get("inject_force") or ex.get("endpoint_features") or ex.get("dimer_residual"):
        raise ValueError("This first ablation requires the unguided, unconditioned LiC architecture")
    model = FlowMatchingLoss(FlowMatchingConfig(ts_denoise_sigma=args.noise_sigma),
                            backbone, attention, head)
    raw = getattr(backbone, "backbone", backbone)
    # TimeFiLMBackbone calls its wrapped module `backbone` in this checkout.
    backbone_ids = {id(p) for p in raw.parameters() if p.requires_grad}
    groups = [dict(name="head_film", params=[p for p in model.parameters()
                    if p.requires_grad and id(p) not in backbone_ids], lr=args.learning_rate),
              dict(name="backbone", params=[p for p in model.parameters()
                    if p.requires_grad and id(p) in backbone_ids], lr=args.uma_lr)]
    groups = [group for group in groups if group["params"]]
    return model, groups, cfg, name


def training_data(args):
    frames = read(args.train_traj, index=":")
    saddles = frames[1::3]
    if not saddles or len(frames) % 3:
        raise ValueError("Training trajectory must contain R,S,P triplets")
    dataset = TrajTripletDataset(args.train_traj, compute_stats=False)
    return dataset, saddles


def batch_indices(size, update, batch_size, seed):
    """Shuffle each epoch independently; exact batch order is reproducible on resume."""
    batches_per_epoch = math.ceil(size/batch_size)
    epoch, offset = divmod(update, batches_per_epoch)
    rng = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=rng).tolist()
    return order[offset*batch_size:(offset+1)*batch_size]


def observe(pes, position, kind, args, deadband):
    # Even a teacher implementation that consumes torch randomness must not
    # change subsequent student forwards relative to the E0 control.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        teacher = query_teacher(pes, position, kind, args.fd_eps, args.seed)
    teacher["active"] = teacher["force_rms"] > deadband
    return teacher


def prior_loss(state, j, batch, teacher, args, deadband):
    return directional_loss(state["velocities"][j], teacher["vector"], ~batch[j]["fixed"],
                            force_rms=teacher["force_rms"], teacher_deadband=deadband,
                            velocity_floor=args.velocity_floor, force_floor=args.force_floor)


def parameter_gradient_norm(loss, parameters, retain_graph=True):
    """Actual parameter-gradient norm, not a velocity-space proxy."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    squared = sum(g.detach().double().square().sum() for g in gradients if g is not None)
    return float(torch.sqrt(squared))


def alignment_audit(saddles, backends, args, deadband, directory):
    """Training-only Gaussian bridges; no student predictions or held-out geometry."""
    rng = torch.Generator().manual_seed(args.seed + 4000)
    rows = []
    for parent, (atoms, pes) in enumerate(zip(saddles, backends)):
        for _ in range(args.audit_samples_per_saddle):
            eps = gaussian_perturbation(torch.as_tensor(pes.dofs.free), args.noise_sigma, rng)
            t = float(torch.rand((), generator=rng))
            target = -eps[pes.dofs.free].numpy().reshape(-1)
            positions = torch.as_tensor(atoms.positions, dtype=torch.float32) + (1-t)*eps
            positions = wrap_positions(positions, torch.as_tensor(np.asarray(atoms.cell), dtype=torch.float32))
            teacher = observe(pes, positions, "mmf", args, deadband)
            rows.append(dict(parent=parent, t=t, force_rms=teacher["force_rms"],
                active=teacher["active"], cosine_force=alignment(target, teacher["force"]),
                cosine_mmf=alignment(target, teacher["vector"][pes.dofs.free].cpu().numpy()),
                force_to_target_rms=teacher["force_rms"]/(np.linalg.norm(target)/pes.dofs.n_free**0.5),
                lambda1=float(teacher["eigs"][0]), eigengap=float(np.diff(teacher["eigs"])[0])))
        print(f"[audit] parent {parent+1}/{len(saddles)}; {len(rows)} bridge states", flush=True)
    data = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(directory / "alignment_audit.npz", **data)
    lines = ["training-only PES teacher alignment", "="*110,
             "Cosines are undefined for zero vectors. Active uses the empirical training-saddle maximum.",
             "time bin    N  active%  F cos50  F cos10  F opposed%  MMF cos50  MMF cos10  MMF opposed%  F_RMS50  lambda1_50  gap50"]
    for lo in np.arange(0, 1, 0.25):
        use = (data["t"] >= lo) & (data["t"] < lo+0.25)
        if not use.any():
            continue
        values = []
        for key in ["cosine_force", "cosine_mmf"]:
            cos = data[key][use]
            cos = cos[np.isfinite(cos)]
            values.extend([np.median(cos), np.percentile(cos, 10), 100*np.mean(cos < 0)])
        lines.append(f"{lo:.2f}-{lo+.25:.2f} {use.sum():4d} {100*data['active'][use].mean():8.1f} "
                     + " ".join(f"{v:10.4f}" for v in values)
                     + " " + " ".join(f"{np.median(data[k][use]):10.5f}" for k in ["force_rms", "lambda1", "eigengap"]))
    report = "\n".join(lines) + "\n"
    (directory / "alignment_audit.txt").write_text(report)
    print(report, flush=True)


def calibrate(args, model, name, dataset, saddles):
    directory = Path(args.calibration_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "calibration.json").exists():
        raise ValueError("Calibration already exists; reuse it or choose a new directory")
    print(f"[UMA] loading separate frozen teacher {name}", flush=True)
    calculator, batch_evaluate = make_uma_calculator(name, args.device, saddles[0].info.get("task_name", "omat"))
    backends = [PESBackend(atoms, calculator, batch_evaluate) for atoms in saddles]
    with torch.enable_grad():
        forces = [pes.dofs.force_metrics(pes.forces(pes.dofs.pack(atoms.positions)))[0]
                  for atoms, pes in zip(saddles, backends)]
    deadband = max(forces)
    print(f"[calibrate] {len(saddles)} training saddles ONLY: max F_RMS={deadband:.7f} eV/A", flush=True)
    alignment_audit(saddles, backends, args, deadband, directory)
    parameters = [p for p in model.parameters() if p.requires_grad]
    flow_rng = torch.Generator().manual_seed(args.seed + 3000)
    select_rng = torch.Generator().manual_seed(args.seed + 3500)
    records = []
    model.train()
    for update in range(args.calibration_batches):
        ids = batch_indices(len(dataset), update, args.batch_size, args.seed+5000)
        batch = [dataset[i] for i in ids]
        state = model(batch, generator=flow_rng, return_training_state=True)["training_state"]
        j = int(torch.randint(len(batch), (), generator=select_rng))
        pes = backends[int(batch[j]["triplet_id"])]
        # Query once with MMF: the central force is the E1 teacher as well.
        mmf = observe(pes, state["positions"][j], "mmf", args, deadband)
        force_vector = torch.zeros_like(mmf["vector"])
        force_vector[pes.dofs.free] = torch.as_tensor(mmf["force"].reshape(-1,3), dtype=force_vector.dtype)
        force = dict(mmf, vector=force_vector)
        loss_f = prior_loss(state, j, batch, force, args, deadband)
        loss_mmf = prior_loss(state, j, batch, mmf, args, deadband)
        g_fm = parameter_gradient_norm(state["fm_loss"], parameters)
        g_f = parameter_gradient_norm(loss_f, parameters)
        g_mmf = parameter_gradient_norm(loss_mmf, parameters, retain_graph=False)
        records.append(dict(batch=update, t=state["times"][j], active=mmf["active"],
                            grad_fm=g_fm, grad_force=g_f, grad_mmf=g_mmf))
        print(f"[calibrate] {update+1}/{args.calibration_batches}: grad FM={g_fm:.5g} "
              f"F={g_f:.5g} MMF={g_mmf:.5g} active={mmf['active']}", flush=True)
        del state, loss_f, loss_mmf
    means = {key: float(np.mean([row[key] for row in records])) for key in ["grad_fm", "grad_force", "grad_mmf"]}
    pooled = (means["grad_force"] + means["grad_mmf"])/2
    if pooled <= 0 or means["grad_fm"] <= 0:
        raise ValueError("Zero calibration gradients: inspect audit/mask instead of silently choosing lambda")
    weight = args.gradient_ratio * means["grad_fm"]/pooled
    calibration = dict(settings=scientific_settings(args), force_deadband=deadband,
        training_saddle_forces=forces, lambda_pes=weight, gradient_means=means,
        actual_initial_ratios=dict(E1=weight*means["grad_force"]/means["grad_fm"],
                                   E2=weight*means["grad_mmf"]/means["grad_fm"]),
        records=records, lambda_rule="target_ratio * mean(gFM) / mean(mean(gF), mean(gMMF))")
    (directory / "calibration.json").write_text(json.dumps(calibration, indent=2))
    report = (f"shared PES calibration\ntraining saddles: {len(saddles)}\n"
              f"epsilon_teacher (max): {deadband:.8f} eV/A\nlambda_PES: {weight:.8g}\n"
              f"initial E1 gradient ratio: {calibration['actual_initial_ratios']['E1']:.5f}\n"
              f"initial E2 gradient ratio: {calibration['actual_initial_ratios']['E2']:.5f}\n"
              "Ratios use means of actual parameter-gradient norms, including masked-zero batches.\n")
    (directory / "calibration_report.txt").write_text(report)
    print(report, flush=True)


def save_checkpoint(model, optimizer, scheduler, ema, directory, update, flow_rng, select_rng):
    directory.mkdir(parents=True, exist_ok=True)
    save_model(model, str(directory / "model.safetensors"), force_contiguous=True)
    torch.save(ema.state_dict(), directory / "ema.pt")
    torch.save(dict(update=update, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    flow_rng=flow_rng.get_state(), select_rng=select_rng.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
               directory / "training_state.pt")
    (directory / "meta.json").write_text(json.dumps(dict(global_step=update)))
    print(f"[save] update {update} -> {directory}", flush=True)


def finetune(args, model, groups, source_config, name, dataset, saddles):
    calibration = json.loads((Path(args.calibration_dir) / "calibration.json").read_text())
    if calibration["settings"] != scientific_settings(args):
        raise ValueError("Checkpoint/training-data/physics settings differ from calibration")
    weight, deadband = calibration["lambda_pes"], calibration["force_deadband"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["extras"] = dict(source_config["extras"], pes_experiment=args.experiment,
        lambda_pes=weight, teacher_deadband=deadband, inject_force=False, endpoint_features=False)
    if (out / "config.json").exists():
        if not args.resume_from:
            raise ValueError("Output run exists; use --resume-from or a new directory")
        old = json.loads((out / "config.json").read_text())
        for key in ["experiment", "updates", "batch_size", "learning_rate", "uma_lr", "warmup_steps", "ema_decay", "seed", "extras"]:
            if old[key] != config[key]:
                raise ValueError(f"Resume changed {key}")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    backends = None
    if args.experiment != "E0":
        print(f"[UMA] separate frozen teacher for {args.experiment}", flush=True)
        calculator, batch_evaluate = make_uma_calculator(name, args.device, saddles[0].info.get("task_name", "omat"))
        backends = [PESBackend(atoms, calculator, batch_evaluate) for atoms in saddles]
    # Teacher initialization must not change student RNG relative to E0.
    torch.manual_seed(args.seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, weight_decay=source_config.get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda k: _lr_lambda(k, args.warmup_steps, args.updates, source_config.get("min_lr_ratio", .01)))
    ema = EMA(parameters, args.ema_decay)  # fresh shadow of the LOADED EMA initializer
    flow_rng = torch.Generator().manual_seed(args.seed + 1000)
    select_rng = torch.Generator().manual_seed(args.seed + 2000)
    start = 0
    if args.resume_from:
        checkpoint = Path(args.resume_from)
        if checkpoint.parent.resolve() != out.resolve():
            raise ValueError("Resume checkpoint must belong to this output run")
        load_model(model, str(checkpoint / "model.safetensors"))
        ema.load_state_dict(torch.load(checkpoint / "ema.pt", map_location="cpu", weights_only=False))
        ema.shadow = [shadow.to(parameter) for shadow, parameter in zip(ema.shadow, parameters)]
        saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        flow_rng.set_state(saved["flow_rng"]); select_rng.set_state(saved["select_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["update"]
        # Drop log rows from work after this checkpoint before replaying them.
        log_path = out / "training_steps.jsonl"
        if log_path.exists():
            rows = [line for line in log_path.read_text().splitlines()
                    if json.loads(line)["update"] <= start]
            log_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    save_steps = {int(s) for s in args.save_steps.split(",")}
    if start == 0 and not args.resume_from:
        save_checkpoint(model, optimizer, scheduler, ema, out / "checkpoint_step_00000000", 0, flow_rng, select_rng)
    print(f"[train] {args.experiment} {start}/{args.updates}; EMA initializer; fresh Adam unless resumed", flush=True)
    print(f"[train] head/FiLM LR={args.learning_rate:g}, backbone LR={args.uma_lr:g}; "
          f"lambda={weight:.6g}; teacher deadband={deadband:.6g}; one teacher/batch; no time taper", flush=True)
    began = perf_counter()
    for update in range(start, args.updates):
        model.train()
        ids = batch_indices(len(dataset), update, args.batch_size, args.seed)
        batch = [dataset[i] for i in ids]
        j = int(torch.randint(len(batch), (), generator=select_rng))  # consumed in E0 too
        optimizer.zero_grad(set_to_none=True)
        state = model(batch, generator=flow_rng, return_training_state=True)["training_state"]
        fm = state["fm_loss"]
        prior = fm * 0
        teacher = None
        if backends is not None:
            teacher = observe(backends[int(batch[j]["triplet_id"])], state["positions"][j],
                              "force" if args.experiment == "E1" else "mmf", args, deadband)
            prior = prior_loss(state, j, batch, teacher, args, deadband)
        loss = fm + weight*prior
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step(); scheduler.step(); ema.update()
        # Hash the full bridge batch to make matching verifiable, not just asserted.
        signature = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes())
        signature.update(np.asarray(state["times"]).tobytes())
        for position in state["positions"]:
            signature.update(position.detach().cpu().numpy().tobytes())
        record = dict(update=update+1, loss_fm=float(fm.detach()), loss_prior=float(prior.detach()),
                      loss_total=float(loss.detach()), grad_norm=float(gradient), selected_example=j,
                      selected_parent=int(batch[j]["triplet_id"]), selected_t=state["times"][j],
                      batch_sha256=signature.hexdigest(), active=teacher["active"] if teacher else False,
                      force_rms=teacher["force_rms"] if teacher else None,
                      teacher_calls=teacher["force_calls"] if teacher else 0)
        with (out / "training_steps.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if (update+1) % args.log_every == 0 or update == start:
            print(f"[train] {args.experiment} update {update+1}/{args.updates} FM={record['loss_fm']:.6g} "
                  f"prior={record['loss_prior']:.5f} active={int(record['active'])} "
                  f"t={record['selected_t']:.3f} F_RMS={record['force_rms']} "
                  f"PES={record['teacher_calls']} elapsed={(perf_counter()-began)/60:.1f}min", flush=True)
        del state, fm, prior, loss
        if update+1 in save_steps and update+1 < args.updates:
            save_checkpoint(model, optimizer, scheduler, ema, out / f"checkpoint_step_{update+1:08d}",
                            update+1, flow_rng, select_rng)
    save_checkpoint(model, optimizer, scheduler, ema, out / "checkpoint_final", args.updates, flow_rng, select_rng)
    print(f"[done] {args.experiment}, exactly {args.updates} continuation updates", flush=True)


def main():
    args = parse_args()
    if min(args.updates, args.batch_size, args.calibration_batches, args.audit_samples_per_saddle, args.log_every) <= 0:
        raise ValueError("Counts must be positive")
    if min(args.noise_sigma, args.fd_eps, args.velocity_floor, args.force_floor,
           args.gradient_ratio, args.learning_rate, args.uma_lr, args.grad_clip) <= 0:
        raise ValueError("Noise, numerical floors, gradient ratio and learning rates must be positive")
    if not args.calibrate_only and not args.output_dir:
        raise ValueError("Training needs --output-dir")
    torch.manual_seed(args.seed)
    dataset, saddles = training_data(args)
    print(f"[setup] {len(saddles)} training saddles; {len(dataset)} records; sigma={args.noise_sigma}", flush=True)
    print("[setup] Student executes its UMA backbone, NOT a PES force/Hessian oracle at inference.", flush=True)
    model, groups, config, name = build_student(args)
    if args.calibrate_only:
        calibrate(args, model, name, dataset, saddles)
    else:
        finetune(args, model, groups, config, name, dataset, saddles)


if __name__ == "__main__":
    main()
