"""R0 continuation, R1 recovery FM, R2 recovery FM + detached MMF.

Run --stage prepare, then calibrate, then train each experiment. All students
start from the SAME EMA initializer; only R2 queries physics during updates.
This is experimental LiC code, not a new general training framework.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import load_model

from saddleflow.classical_saddle_search import PESBackend, make_uma_calculator
from saddleflow.utils.training import EMA, _lr_lambda

try:
    from recovery_basins import DEFAULT_ARCHIVE, file_hash, load_split, mixed_batch, prepare, read_npz
    from train_pes_directions import (DEFAULT_CHECKPOINT, build_student, observe, parameter_gradient_norm,
                                     prior_loss, save_checkpoint, training_data)
    from diagnose_pes_adoption import array_hash
except ModuleNotFoundError:
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import DEFAULT_ARCHIVE, file_hash, load_split, mixed_batch, prepare, read_npz
    from examples.LiC.experiments.pes_and_recovery.train_pes_directions import (DEFAULT_CHECKPOINT, build_student, observe,
        parameter_gradient_norm, prior_loss, save_checkpoint, training_data)
    from examples.LiC.experiments.pes_and_recovery.diagnose_pes_adoption import array_hash


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", choices=["prepare","calibrate","train"], default="train")
    p.add_argument("--experiment", choices=["R0","R1","R2"], default="R0")
    p.add_argument("--shell-archive", default=DEFAULT_ARCHIVE)
    p.add_argument("--init-weights", default=DEFAULT_CHECKPOINT)
    p.add_argument("--train-traj", default=str((Path(__file__).resolve().parents[2] / "train_set.traj")))
    p.add_argument("--train-basins", default="0,1,2", help="Zero-based original archive catalog IDs")
    p.add_argument("--fold", type=int, choices=[0,1], default=0, help="1 swaps observed train/holdout basins")
    p.add_argument("--holdout-fraction", type=float, default=.2)
    p.add_argument("--force-epsilon", type=float, default=.010, help="Mining: saved free fmax, eV/A")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--uma-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=.999)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--noise-sigma", type=float, default=.25)
    p.add_argument("--fd-eps", type=float, default=.002)
    p.add_argument("--velocity-floor", type=float, default=1e-4)
    p.add_argument("--force-floor", type=float, default=1e-6)
    p.add_argument("--gradient-ratio", type=float, default=.1)
    p.add_argument("--lambda-pes", type=float, help="Optional predeclared weight; otherwise calibrate on TRAIN recovery batches")
    p.add_argument("--calibration-batches", type=int, default=16)
    p.add_argument("--audit-sources", type=int, default=48)
    p.add_argument("--save-steps", default="0,250,500,1000,2000")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume-from", help="This experiment's checkpoint; restore Adam, EMA, scheduler and RNG")
    args = p.parse_args()
    args.batch_size = 4
    args.train_basins = [int(i) for i in args.train_basins.split(",")]
    if args.fold:
        a = read_npz(args.shell_archive)
        observed = np.unique(a["nearest_false_index"][(a["force_max"]>args.force_epsilon)&(a["outcome_code"]==2)])
        args.train_basins = np.setdiff1d(observed,args.train_basins).tolist()
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    if min(args.updates,args.log_every,args.audit_sources,args.calibration_batches) <= 0:
        raise ValueError("counts must be positive")
    if min(args.force_epsilon,args.noise_sigma,args.fd_eps,args.learning_rate,args.uma_lr,
           args.velocity_floor,args.force_floor,args.grad_clip,args.gradient_ratio) <= 0:
        raise ValueError("scales must be positive")
    if args.lambda_pes is not None and args.lambda_pes <= 0:
        raise ValueError("lambda must be positive")
    return args


def settings(args):
    return dict(split_sha256=file_hash(Path(args.run_dir)/"split.npz"),
        init_weights=str(Path(args.init_weights).resolve()), init_ema_sha256=file_hash(Path(args.init_weights)/"ema.pt"),
        train_sha256=file_hash(args.train_traj), seed=args.seed, noise_sigma=args.noise_sigma,
        fd_eps=args.fd_eps, velocity_floor=args.velocity_floor, force_floor=args.force_floor,
        gradient_ratio=args.gradient_ratio, lambda_override=args.lambda_pes,
        calibration_batches=args.calibration_batches)


def make_backends(args, name, saddles):
    print(f"[UMA] separate frozen {name} teacher", flush=True)
    calculator,batch_evaluate = make_uma_calculator(name,args.device,saddles[0].info.get("task_name","omat"))
    return [PESBackend(atoms,calculator,batch_evaluate) for atoms in saddles]


def cache_probe_teachers(args, backends, deadband):
    """One immutable cache, queried only on fixed TRAINING recovery probes."""
    root = Path(args.run_dir)
    path = root/"recovery_teacher.npz"
    probes = read_npz(root/"recovery_probes.npz")
    context = dict(probes_sha256=file_hash(root/"recovery_probes.npz"), fd_eps=args.fd_eps,
                   deadband=deadband, settings=settings(args))
    payload = list(probes)+["force","q","teacher","eigenvalues","force_rms","active"]
    if path.exists():
        data = read_npz(path)
        if json.loads(str(data["context_json"])) != context:
            raise ValueError("recovery teacher cache belongs to different settings")
        if any(not np.array_equal(data[k],v) for k,v in probes.items()):
            raise ValueError("cached recovery probes changed")
        if str(data["checksum"]) != array_hash(*[data[k] for k in payload]):
            raise ValueError("recovery teacher checksum failed")
        print("[audit cache] reusing frozen teachers; no repeated queries",flush=True)
        return data
    force,mode,vector,eigs,score = [],[],[],[],[]
    for i,(x,parent) in enumerate(zip(probes["positions"],probes["parent"])):
        pes = backends[int(parent)]
        teacher = observe(pes,torch.as_tensor(x),"mmf",args,deadband)
        full_f,full_q = np.zeros_like(x),np.zeros_like(x)
        full_f[pes.dofs.free] = teacher["force"].reshape(-1,3)
        full_q[pes.dofs.free] = teacher["q"].reshape(-1,3)
        force.append(full_f); mode.append(full_q); vector.append(teacher["vector"].cpu().numpy())
        eigs.append(teacher["eigs"]); score.append(teacher["force_rms"])
        if (i+1)%16 == 0 or i+1 == len(probes["t"]):
            print(f"[audit teacher] {i+1}/{len(probes['t'])}",flush=True)
    data = dict(probes,force=np.asarray(force),q=np.asarray(mode),teacher=np.asarray(vector),
                eigenvalues=np.asarray(eigs),force_rms=np.asarray(score),context_json=json.dumps(context))
    data["active"] = data["force_rms"] > deadband
    data["checksum"] = array_hash(*[data[k] for k in payload])
    with path.open("xb") as handle: np.savez_compressed(handle,**data)
    return data


def calibrate(args, model, name, dataset, recovery, saddles):
    path = Path(args.run_dir)/"calibration.json"
    if path.exists():
        if json.loads(path.read_text())["settings"] != settings(args):
            raise ValueError("calibration settings changed")
        print(f"[calibrate] reusing {path}",flush=True)
        return
    backends = make_backends(args,name,saddles)
    deadband = max(pes.dofs.force_metrics(pes.forces(pes.dofs.pack(atoms.positions)))[0]
                   for pes,atoms in zip(backends,saddles))
    cache_probe_teachers(args,backends,deadband)
    parameters = [p for p in model.parameters() if p.requires_grad]
    torch.manual_seed(args.seed+3000)
    rows = []
    model.train()
    for k in range(args.calibration_batches):
        batch = mixed_batch(dataset,recovery,k,args.seed+5000,"R2")
        rng = torch.Generator().manual_seed(args.seed+3000+k)
        state = model(batch,generator=rng,return_training_state=True)["training_state"]
        teacher = observe(backends[int(batch[3]["triplet_id"])],state["positions"][3],"mmf",args,deadband)
        prior = prior_loss(state,3,batch,teacher,args,deadband)
        fm_norm = parameter_gradient_norm(state["fm_loss"],parameters)
        prior_norm = parameter_gradient_norm(prior,parameters,retain_graph=False)
        rows.append(dict(fm=fm_norm,prior=prior_norm,active=teacher["active"],archive_row=batch[3]["archive_row"]))
        print(f"[calibrate] {k+1}/{args.calibration_batches}; gFM={fm_norm:.5g}, gMMF={prior_norm:.5g}",flush=True)
    fm_mean,prior_mean = [np.mean([r[key] for r in rows]) for key in ["fm","prior"]]
    if prior_mean <= 0 or fm_mean <= 0:
        raise ValueError("zero calibration gradients; inspect training recovery probes")
    weight = args.lambda_pes if args.lambda_pes is not None else args.gradient_ratio*fm_mean/prior_mean
    data = dict(settings=settings(args),lambda_pes=float(weight),teacher_deadband=deadband,
                initial_ratio=float(weight*prior_mean/fm_mean),records=rows,
                rule="fixed lambda from TRAIN recovery batches only; ratio of mean parameter-gradient norms")
    path.write_text(json.dumps(data,indent=2))
    print(f"[calibrate] lambda={weight:.7g}; initial ratio={data['initial_ratio']:.5f}; deadband={deadband:.7g}",flush=True)


def finetune(args, model, groups, source_config, name, dataset, recovery, saddles):
    calibration = json.loads((Path(args.run_dir)/"calibration.json").read_text())
    if calibration["settings"] != settings(args):
        raise ValueError("changed initialization/split/physics settings")
    weight,deadband = calibration["lambda_pes"],calibration["teacher_deadband"]
    out = Path(args.run_dir)/args.experiment
    out.mkdir(parents=True,exist_ok=True)
    config = dict(vars(args),extras=dict(source_config["extras"], recovery_experiment=args.experiment,
        lambda_pes=weight,teacher_deadband=deadband,inject_force=False,endpoint_features=False),
        split_sha256=calibration["settings"]["split_sha256"])
    if (out/"config.json").exists():
        old = json.loads((out/"config.json").read_text())
        if not args.resume_from:
            raise ValueError("run exists; use --resume-from or another directory")
        for key in ["updates","seed","learning_rate","uma_lr","warmup_steps","ema_decay","noise_sigma","grad_clip","extras","split_sha256"]:
            if old[key] != config[key]: raise ValueError(f"resume changed {key}")
    (out/"config.json").write_text(json.dumps(config,indent=2))
    backends = make_backends(args,name,saddles) if args.experiment == "R2" else None
    torch.manual_seed(args.seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(groups,weight_decay=source_config.get("weight_decay",1e-5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda k:_lr_lambda(k,args.warmup_steps,args.updates,source_config.get("min_lr_ratio",.01)))
    ema = EMA(parameters,args.ema_decay)
    flow_rng = torch.Generator(); select_rng = torch.Generator().manual_seed(args.seed+2000)
    start = 0
    if args.resume_from:
        checkpoint = Path(args.resume_from)
        if checkpoint.parent.resolve() != out.resolve(): raise ValueError("resume must belong to this experiment")
        load_model(model,str(checkpoint/"model.safetensors"))
        ema.load_state_dict(torch.load(checkpoint/"ema.pt",map_location="cpu",weights_only=False))
        ema.shadow = [s.to(p) for s,p in zip(ema.shadow,parameters)]
        saved = torch.load(checkpoint/"training_state.pt",map_location="cpu",weights_only=False)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"]: torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["update"]
        log = out/"training_steps.jsonl"
        if log.exists():
            lines = [s for s in log.read_text().splitlines() if json.loads(s)["update"] <= start]
            log.write_text("\n".join(lines)+("\n" if lines else ""))
    if start == 0 and not args.resume_from:
        save_checkpoint(model,optimizer,scheduler,ema,out/"checkpoint_step_00000000",0,flow_rng,select_rng)
    save_steps = {int(s) for s in args.save_steps.split(",")}
    print(f"[train] {args.experiment}, {start}/{args.updates}; 3 original + "
          f"{'1 original' if args.experiment == 'R0' else '1 recovery'}; lambda={weight:.6g}",flush=True)
    began = perf_counter()
    for update in range(start,args.updates):
        model.train(); optimizer.zero_grad(set_to_none=True)
        batch = mixed_batch(dataset,recovery,update,args.seed,args.experiment)
        flow_rng.manual_seed(args.seed+1000+update)
        state = model(batch,generator=flow_rng,return_training_state=True)["training_state"]
        fm = state["fm_loss"]; prior = fm*0; teacher = None
        if backends is not None:
            teacher = observe(backends[int(batch[3]["triplet_id"])],state["positions"][3],"mmf",args,deadband)
            prior = prior_loss(state,3,batch,teacher,args,deadband)
        loss = fm+weight*prior
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters,args.grad_clip)
        optimizer.step(); scheduler.step(); ema.update()
        original_hash = array_hash(np.asarray(state["times"][:3]),*[x.detach().cpu().numpy() for x in state["positions"][:3]])
        full_hash = array_hash(np.asarray(state["times"]),*[x.detach().cpu().numpy() for x in state["positions"]])
        row = dict(update=update+1,original_sha256=original_hash,batch_sha256=full_hash,
            recovery_row=batch[3].get("archive_row",-1),loss_fm=float(fm.detach()),loss_prior=float(prior.detach()),
            loss_total=float(loss.detach()),grad_norm=float(gradient),t=state["times"][3],
            active=teacher["active"] if teacher else False,teacher_calls=teacher["force_calls"] if teacher else 0,
            force_rms=teacher["force_rms"] if teacher else None)
        with (out/"training_steps.jsonl").open("a") as handle: handle.write(json.dumps(row)+"\n")
        if (update+1)%args.log_every == 0 or update == start:
            print(f"[train] {args.experiment} {update+1}/{args.updates} FM={row['loss_fm']:.6g} "
                  f"MMF={row['loss_prior']:.5f} active={int(row['active'])} t={row['t']:.3f} "
                  f"row={row['recovery_row']} PES={row['teacher_calls']} elapsed={(perf_counter()-began)/60:.1f}min",flush=True)
        del state,fm,prior,loss
        if update+1 in save_steps and update+1 < args.updates:
            save_checkpoint(model,optimizer,scheduler,ema,out/f"checkpoint_step_{update+1:08d}",update+1,flow_rng,select_rng)
    save_checkpoint(model,optimizer,scheduler,ema,out/"checkpoint_final",args.updates,flow_rng,select_rng)


def main():
    args = parse_args()
    dataset,saddles = training_data(args)
    if args.stage == "prepare":
        prepare(args,dataset); return
    split,recovery = load_split(args.run_dir,dataset)
    manifest = json.loads((Path(args.run_dir)/"split.json").read_text())["settings"]
    if (manifest["init_weights"] != str(Path(args.init_weights).resolve())
            or manifest["train_sha256"] != file_hash(args.train_traj)
            or args.train_basins != split["train_basins"].tolist()):
        raise ValueError("use the same initializer, training labels and fold as preparation")
    print(f"[setup] {len(recovery)} recovery TRAIN pairs; holdout never queried during optimization",flush=True)
    torch.manual_seed(args.seed)
    model,groups,config,name = build_student(args)
    if args.stage == "calibrate": calibrate(args,model,name,dataset,recovery,saddles)
    else: finetune(args,model,groups,config,name,dataset,recovery,saddles)


if __name__ == "__main__":
    main()
