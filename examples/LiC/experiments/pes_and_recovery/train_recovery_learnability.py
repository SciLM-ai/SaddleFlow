"""Deliberately fit a few seen recovery pairs, with early-time FM.

Plain recovery FM by default; --mmf adds one detached minimum-mode directional
teacher per batch. No new mining or inference guidance. This is a learnability
control, NOT a basin-generalization benchmark. Shared training is unchanged.
"""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from ase.geometry import find_mic
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from safetensors.torch import load_model

from saddleflow.data.transforms import wrap_positions
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils.training import EMA, _lr_lambda
from pes_direction import alignment, directional_loss

try:
    from recovery_basins import file_hash, load_split, read_npz, recovery_record
    from train_pes_directions import (DEFAULT_CHECKPOINT, build_student, observe,
                                     parameter_gradient_norm, save_checkpoint, training_data)
    from train_recovery_mmf import make_backends
    from robustness_shells import sample_points
    from saddle_search_data import classify, load_geometry
except ModuleNotFoundError:
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import file_hash, load_split, read_npz, recovery_record
    from examples.LiC.experiments.pes_and_recovery.train_pes_directions import (DEFAULT_CHECKPOINT, build_student, observe,
        parameter_gradient_norm, save_checkpoint, training_data)
    from examples.LiC.experiments.pes_and_recovery.train_recovery_mmf import make_backends
    from examples.LiC.rollout_eval.robustness_shells import sample_points
    from examples.LiC.rollout_eval.saddle_search_data import classify, load_geometry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--source-run",default="/scratch/gqs7zr/saddleflow/lic_recovery_mmf_21289950")
    p.add_argument("--init-weights",default=DEFAULT_CHECKPOINT)
    p.add_argument("--train-traj",default=str((Path(__file__).resolve().parents[2] / "train_set.traj")))
    p.add_argument("--test-traj",default=str((Path(__file__).resolve().parents[2] / "test_set.traj")))
    p.add_argument("--basin",type=int,default=0); p.add_argument("--parent",type=int,default=0)
    p.add_argument("--num-sources",type=int,default=6); p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--updates",type=int,default=6000)
    p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--uma-lr",type=float,default=1e-4)
    p.add_argument("--warmup-steps",type=int,default=100)
    p.add_argument("--lr-schedule",choices=["constant","cosine"],default="constant")
    p.add_argument("--ema-decay",type=float,default=.999); p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--noise-sigma",type=float,default=.25,help="Fixed clean evaluation starts ONLY")
    p.add_argument("--save-steps",default="0,250,500,1000,2000,4000,6000")
    p.add_argument("--log-every",type=int,default=25); p.add_argument("--eval-batch-size",type=int,default=32)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="cuda")
    p.add_argument("--mmf",action="store_true",help="R2: same recovery FM plus one frozen MMF directional teacher per batch")
    p.add_argument("--lambda-pes",type=float,help="Fixed MMF weight; default calibrates on the selected TRAIN recovery pairs")
    p.add_argument("--gradient-ratio",type=float,default=.1)
    p.add_argument("--calibration-batches",type=int,default=16)
    p.add_argument("--fd-eps",type=float,default=.002)
    p.add_argument("--velocity-floor",type=float,default=1e-4)
    p.add_argument("--force-floor",type=float,default=1e-6)
    p.add_argument("--resume-from"); p.add_argument("--dry-run",action="store_true")
    args = p.parse_args()
    if min(args.num_sources,args.batch_size,args.updates,args.log_every,args.eval_batch_size) <= 0:
        raise ValueError("counts must be positive")
    if min(args.learning_rate,args.uma_lr,args.grad_clip,args.noise_sigma) <= 0 or args.warmup_steps < 0:
        raise ValueError("invalid optimization scales")
    if args.calibration_batches <= 0 or not all(np.isfinite(v) and v > 0 for v in
        [args.gradient_ratio,args.fd_eps,args.velocity_floor,args.force_floor]):
        raise ValueError("invalid MMF calibration scales")
    if args.lambda_pes is not None and (not args.mmf or not np.isfinite(args.lambda_pes) or args.lambda_pes <= 0):
        raise ValueError("a positive --lambda-pes requires --mmf")
    return args


def select_rows(split, basin, parent, count, seed):
    # Only the previous fold's TRAIN rows, with one common valid target.
    rows = np.flatnonzero((split["partition"] == 0)&(split["baseline_outcome"] == 2)
                         &(split["baseline_basin"] == basin)&(split["parent"] == parent))
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} TRAIN sources for basin {basin}, parent {parent}; requested {count}")
    return np.sort(np.random.default_rng(seed).choice(rows,count,replace=False))


def sample_times(count, generator):
    """25% t=0; 50% U(0,.25); 25% U(0,1). No importance correction."""
    branch = torch.rand(count,generator=generator)
    uniform = torch.rand(count,generator=generator)
    return torch.where(branch < .25,0.,torch.where(branch < .75,.25*uniform,uniform))


def recovery_batch(records, update, args):
    """Identical R1/R2 draws; teacher selection has a separate RNG stream."""
    select_rng=torch.Generator().manual_seed(args.seed+2000+update)
    flow_rng=torch.Generator().manual_seed(args.seed+1000+update)
    ids=torch.randint(len(records),(args.batch_size,),generator=select_rng).tolist()
    samples=[records[i] for i in ids]; times=sample_times(len(samples),flow_rng)
    targets=[s["saddle_un_pos"]-s["flow_source_pos"] for s in samples]
    positions=[wrap_positions(s["flow_source_pos"]+t*v,s["cell"]) for s,t,v in zip(samples,times,targets)]
    return samples,times,targets,positions,flow_rng,select_rng


def mmf_loss(velocity, sample, teacher, args, deadband):
    return directional_loss(velocity,teacher["vector"],~sample["fixed"],
        force_rms=teacher["force_rms"],teacher_deadband=deadband,
        velocity_floor=args.velocity_floor,force_floor=args.force_floor)


def prepare_mmf(args, model, name, records, saddles, inputs):
    """Training-only calibration and fixed teachers, reused on resume.

    The force model is separate from the student/optimizer. Calibration makes
    no weight updates and cannot consume the subsequent training RNG stream.
    """
    out=Path(args.run_dir); path=out/"mmf_calibration.json"; cache_path=out/"mmf_probes.npz"
    selection=read_npz(out/"selection.npz")
    context=dict(version=1,selection_sha256=file_hash(out/"selection.npz"),
        init_ema_sha256=str(selection["init_ema_sha256"]),train_sha256=file_hash(args.train_traj),
        seed=args.seed,batch_size=args.batch_size,fd_eps=args.fd_eps,velocity_floor=args.velocity_floor,
        force_floor=args.force_floor,gradient_ratio=args.gradient_ratio,
        calibration_batches=args.calibration_batches,lambda_override=args.lambda_pes,model=name)
    saved=json.loads(path.read_text()) if path.exists() else None
    if saved and (saved["settings"] != context or file_hash(cache_path) != saved["probes_sha256"]):
        raise ValueError("MMF calibration or fixed teachers changed")
    if args.resume_from and saved is None:
        raise ValueError("MMF resume requires its saved calibration")
    # UMA loading and oracle queries must not perturb student randomness.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        backends=make_backends(args,name,saddles)
        if saved is None:
            deadband=max(pes.dofs.force_metrics(pes.forces(pes.dofs.pack(atoms.positions)))[0]
                         for pes,atoms in zip(backends,saddles))
            parameters=[p for p in model.parameters() if p.requires_grad]
            rows=[]; model.train()
            for k in range(args.calibration_batches):
                samples,times,targets,positions,_,_=recovery_batch(records,k+100000,args)
                j=int(torch.randint(len(samples),(),generator=torch.Generator().manual_seed(args.seed+8000+k)))
                v=velocity_forward(model,positions,times,samples,args.device)
                teacher=observe(backends[int(samples[j]["triplet_id"])],positions[j],"mmf",args,deadband)
                fm=fm_loss(v,targets,samples); prior=mmf_loss(v[j],samples[j],teacher,args,deadband)
                g_fm=parameter_gradient_norm(fm,parameters)
                g_mmf=parameter_gradient_norm(prior,parameters,retain_graph=False)
                rows.append(dict(grad_fm=g_fm,grad_mmf=g_mmf,active=teacher["active"],
                    archive_row=samples[j]["archive_row"],time=float(times[j]),force_calls=teacher["force_calls"]))
                print(f"[calibrate] MMF {k+1}/{args.calibration_batches}; "
                      f"gFM={g_fm:.5g} gMMF={g_mmf:.5g} active={teacher['active']}",flush=True)
                del v,fm,prior
            fm_mean,mmf_mean=[float(np.mean([r[key] for r in rows])) for key in ["grad_fm","grad_mmf"]]
            if min(fm_mean,mmf_mean) <= 0 or not np.isfinite([fm_mean,mmf_mean]).all():
                raise ValueError("MMF calibration has no usable gradient; inspect the mask/teachers")
            weight=args.lambda_pes if args.lambda_pes is not None else args.gradient_ratio*fm_mean/mmf_mean
            # Cache teachers on the existing fixed recovery probes: diagnostics
            # can track adoption without any new PES queries during sampling.
            teachers=[observe(backends[int(s["triplet_id"])],x,"mmf",args,deadband)
                      for s,x in zip(inputs["samples"],inputs["positions"])]
            np.savez_compressed(cache_path,positions=np.asarray(inputs["positions"]),t=inputs["t"],
                vector=np.asarray([t["vector"].cpu().numpy() for t in teachers]),
                active=np.asarray([t["active"] for t in teachers]))
            saved=dict(settings=context,lambda_pes=float(weight),teacher_deadband=float(deadband),
                initial_gradient_ratio=float(weight*mmf_mean/fm_mean),records=rows,
                probes_sha256=file_hash(cache_path),
                rule="fixed lambda from selected TRAIN recovery pairs; gradient ratio uses means of parameter-gradient norms")
            path.write_text(json.dumps(saved,indent=2))
    cache=read_npz(cache_path)
    if not np.array_equal(cache["positions"],np.asarray(inputs["positions"])) or not np.array_equal(cache["t"],inputs["t"]):
        raise ValueError("cached MMF probe states differ")
    inputs["teacher_vectors"],inputs["teacher_active"]=cache["vector"],cache["active"]
    print(f"[calibrate] frozen lambda={saved['lambda_pes']:.7g}; "
          f"initial gradient ratio={saved['initial_gradient_ratio']:.4f}; "
          f"training-saddle force deadband={saved['teacher_deadband']:.7g} eV/A",flush=True)
    return saved,backends


def velocity_forward(model, positions, times, samples, device):
    """The existing unconditioned velocity forward, with weight gradients ON.

    Positions are ordinary fixed training inputs. This does not differentiate
    through a PES or introduce a different student architecture.
    """
    dl = [build_atomic_data(x,s["Z"],s["cell"],s["task_name"],s["charge"],s["spin"],s["fixed"])
          for x,s in zip(positions,samples)]
    batch = data_list_collater(dl,otf_graph=True).to(device)
    t = torch.as_tensor(times,dtype=torch.float32,device=device)
    features = (model.backbone(batch,t,batch.batch) if isinstance(model.backbone,TimeFiLMBackbone)
                else model.backbone(batch))
    hidden = model.global_attn(features["node_embedding"],batch.batch)
    v = model.velocity_head(hidden,t,batch.batch)
    fixed = torch.cat([s["fixed"] for s in samples]).to(device)
    v = apply_output_projections(v,fixed,batch.batch,len(samples))
    return list(v.split([len(s["Z"]) for s in samples]))


def fm_loss(predictions, targets, samples):
    # LiC has one mobile atom per structure: exactly the usual atom-vector FM MSE.
    return torch.stack([(v[~s["fixed"].to(v.device)]-target.to(v)[~s["fixed"].to(v.device)])
                        .square().sum(-1).mean() for v,target,s in zip(predictions,targets,samples)]).mean()


def fixed_inputs(records, geometry, seed, sigma):
    times = np.asarray([0.,.05,.125,.25,.5,.75],dtype=np.float32)
    probe_samples,positions,targets,probe_time = [],[],[],[]
    for s in records:
        delta = s["saddle_un_pos"]-s["flow_source_pos"]
        for t in times:
            positions.append(wrap_positions(s["flow_source_pos"]+float(t)*delta,s["cell"]))
            targets.append(delta); probe_time.append(t); probe_samples.append(s)
    li = int(geometry["li_index"])
    starts = [s["flow_source_pos"][li].numpy() for s in records]
    population = ["recovery_train"]*len(starts)
    rng = np.random.default_rng(seed+9000)
    for group in ["train","test"]:
        starts.extend(geometry[group+"_saddles"]+rng.normal(0,sigma,(len(geometry[group+"_saddles"]),3)))
        population.extend(["true_noise_"+group]*len(geometry[group+"_saddles"]))
    return dict(positions=positions,targets=targets,t=np.asarray(probe_time),samples=probe_samples,
                starts=np.asarray(starts),population=np.asarray(population))


@torch.no_grad()
def diagnose(model, inputs, geometry, update, weights, device, batch_size):
    model.eval(); errors = []; cosines=[]
    for first in range(0,len(inputs["t"]),batch_size):
        end = first+batch_size; samples = inputs["samples"][first:end]
        v = velocity_forward(model,inputs["positions"][first:end],inputs["t"][first:end],samples,device)
        errors.extend(float(fm_loss([vi],[target],[s]))
                      for vi,target,s in zip(v,inputs["targets"][first:end],samples))
        if "teacher_vectors" in inputs:
            cosines.extend(alignment(vi[~s["fixed"]].cpu().numpy(),teacher[~s["fixed"]])
                for vi,s,teacher in zip(v,samples,inputs["teacher_vectors"][first:end]))
    paths=sample_points(inputs["starts"],geometry["positions"],sample=inputs["samples"][0],
        backbone=model.backbone,attention=model.global_attn,velocity_head=model.velocity_head,
        device=device,K=20,batch_size=batch_size,li_index=int(geometry["li_index"]),label=f"{weights} update={update}")
    outcome,d_true,d_false = classify(paths[:,-1],geometry)
    displacement,_ = find_mic(paths[:,1]-inputs["starts"],geometry["cell"],geometry["pbc"])
    result=dict(update=update,weights=weights,probe_error=np.asarray(errors),path=paths,outcome=outcome,
                distance_true=d_true,distance_false=d_false,first_step_displacement=np.linalg.norm(displacement,axis=1))
    if cosines: result["probe_teacher_cosine"]=np.asarray(cosines)
    return result


def write_diagnostics(out, snapshots, inputs, geometry, args, selected_rows, mmf=None):
    data = dict(**geometry,config_json=json.dumps(vars(args)),selected_rows=selected_rows,
        starts=inputs["starts"],population=inputs["population"],probe_t=inputs["t"],
        probe_positions=np.asarray(inputs["positions"]),probe_targets=np.asarray(inputs["targets"]))
    for key in snapshots[0]: data[key] = np.asarray([r[key] for r in snapshots])
    if mmf:
        data.update(probe_teacher_vector=inputs["teacher_vectors"],probe_teacher_active=inputs["teacher_active"],
                    mmf_calibration_json=json.dumps(mmf))
    temporary = out/"learnability.tmp.npz"
    np.savez_compressed(temporary,**data); temporary.replace(out/"learnability.npz")
    lines = ["" + ("recovery FM + MMF learnability control" if mmf else "recovery FM learnability control"), "="*115,
        f"Seen TRAIN sources only: basin={args.basin}, parent={args.parent}, archive rows={selected_rows.tolist()}.",
        (f"Recovery-only FM + detached MMF direction on one example/batch; lambda={mmf['lambda_pes']:.7g}. "
         "No endpoint force loss or inference guidance." if mmf else
         "Recovery-only batches; no MMF/teacher/force loss. The student still runs its UMA-derived neural backbone."),
        "Time mixture: 25% exact zero, 50% U(0,.25), 25% U(0,1); no importance correction.",
        "20-step autonomous Euler; any catalog true hit <.10 A; false occupancy <.30 A with true precedence.",
        "Geometry only: NO force/index-1 certification. Clean populations are diagnostics, never training/checkpoint selection.",
        "Velocity errors are free-atom VECTOR RMSE (A/FM-time); dx1 is first Euler displacement (A).", "",
        "step   weights population             N   source_v_RMSE  bridge_v_RMSE   true%   false%   med d_true(A)  med dx1(A)","-"*115]
    for s in snapshots:
        e=s["probe_error"]; zero=inputs["t"]==0
        for group in dict.fromkeys(inputs["population"]):
            use=inputs["population"]==group; recovery=group=="recovery_train"
            lines.append(f"{s['update']:5d}  {s['weights']:7s} {group:20s} {use.sum():4d} "
                f"{np.sqrt(e[zero].mean()) if recovery else np.nan:14.6f} {np.sqrt(e.mean()) if recovery else np.nan:14.6f} "
                f"{100*np.mean(s['outcome'][use]==0):7.2f} {100*np.mean(s['outcome'][use]==1):8.2f} "
                f"{np.median(s['distance_true'][use]):15.6f} {np.median(s['first_step_displacement'][use]):12.6f}")
    display=lines[:11]+lines[-6:]
    if mmf:
        active=inputs["teacher_active"]
        lines += ["", "Fixed TRAIN recovery probes: detached MMF teacher adoption (raw and EMA)",
                  "step  weights  N_active  mean_cosine  median_cosine"]
        for s in snapshots:
            c=s["probe_teacher_cosine"][active]; c=c[np.isfinite(c)]
            lines.append(f"{s['update']:5d} {s['weights']:7s} {active.sum():8d} "
                         f"{np.mean(c) if len(c) else np.nan:12.6f} {np.median(c) if len(c) else np.nan:14.6f}")
        display += ["", "Latest fixed-probe MMF adoption: step / weights / active / mean cosine / median cosine"]+lines[-2:]
    report="\n".join(lines)+"\n"; (out/"report.txt").write_text(report)
    print("\n".join(display),flush=True)
    print(f"[save] {out/'learnability.npz'} and {out/'report.txt'}",flush=True)


def train_control(args, model, groups, cfg, records, inputs, geometry, selected_rows, mmf=None, backends=None):
    out=Path(args.run_dir); out.mkdir(parents=True,exist_ok=True)
    arguments=dict(vars(args))
    if mmf is None:
        # The existing plain-FM mode and its saved resume configuration stay unchanged.
        for key in ["mmf","lambda_pes","gradient_ratio","calibration_batches","fd_eps","velocity_floor","force_floor"]:
            arguments.pop(key,None)
    config=dict(arguments,selection_sha256=file_hash(out/"selection.npz"),
        extras=dict(cfg["extras"],uma_lr=args.uma_lr,
                    recovery_experiment="learnability_mmf" if mmf else "learnability",
                    lambda_pes=mmf["lambda_pes"] if mmf else 0.,inject_force=False,endpoint_features=False))
    if mmf: config["mmf_calibration_sha256"]=file_hash(out/"mmf_calibration.json")
    if (out/"config.json").exists():
        old=json.loads((out/"config.json").read_text())
        if not args.resume_from: raise ValueError("run exists; resume or use a new directory")
        if {k:v for k,v in old.items() if k not in ["resume_from","dry_run"]} != {
            k:v for k,v in config.items() if k not in ["resume_from","dry_run"]}:
            raise ValueError("resume changed learnability settings")
    (out/"config.json").write_text(json.dumps(config,indent=2))
    torch.manual_seed(args.seed)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg.get("weight_decay",1e-5))
    def schedule(k):
        if args.lr_schedule=="cosine": return _lr_lambda(k,args.warmup_steps,args.updates,.01)
        return min(1.,k/max(1,args.warmup_steps)) if args.warmup_steps else 1.
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    ema=EMA(parameters,args.ema_decay); flow_rng=torch.Generator(); select_rng=torch.Generator()
    snapshots=[]; start=0
    if args.resume_from:
        checkpoint=Path(args.resume_from)
        if checkpoint.parent.resolve()!=out.resolve(): raise ValueError("resume must belong to this run")
        load_model(model,str(checkpoint/"model.safetensors"))
        ema.load_state_dict(torch.load(checkpoint/"ema.pt",map_location="cpu",weights_only=False))
        ema.shadow=[s.to(p) for s,p in zip(ema.shadow,parameters)]
        state=torch.load(checkpoint/"training_state.pt",map_location="cpu",weights_only=False)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]: torch.cuda.set_rng_state_all(state["cuda_rng"])
        start=state["update"]
        if (out/"learnability.npz").exists():
            saved=read_npz(out/"learnability.npz")
            keys=["update","weights","probe_error","path","outcome","distance_true","distance_false","first_step_displacement"]
            if mmf: keys.append("probe_teacher_cosine")
            snapshots=[{k:saved[k][i] for k in keys} for i,u in enumerate(saved["update"]) if u<=start]
        log=out/"training_steps.jsonl"
        if log.exists(): log.write_text("\n".join(l for l in log.read_text().splitlines() if json.loads(l)["update"]<=start)+"\n")
    def evaluate(update):
        for weights in ["raw","ema"]:
            if weights=="ema": ema.swap_in()
            try: snapshots.append(diagnose(model,inputs,geometry,update,weights,args.device,args.eval_batch_size))
            finally:
                if weights=="ema": ema.swap_out()
        write_diagnostics(out,snapshots,inputs,geometry,args,selected_rows,mmf)
    save_steps={int(s) for s in args.save_steps.split(",")}|{0,args.updates}
    if start==0 and not args.resume_from:
        save_checkpoint(model,optimizer,scheduler,ema,out/"checkpoint_step_00000000",0,flow_rng,select_rng)
        evaluate(0)
    elif not any(int(s["update"])==start for s in snapshots): evaluate(start)
    began=perf_counter()
    for update in range(start,args.updates):
        model.train(); optimizer.zero_grad(set_to_none=True)
        samples,times,targets,positions,flow_rng,select_rng=recovery_batch(records,update,args)
        v=velocity_forward(model,positions,times,samples,args.device)
        fm=fm_loss(v,targets,samples); prior=fm*0.; teacher=None; j=-1
        if mmf:
            j=int(torch.randint(len(samples),(),generator=torch.Generator().manual_seed(args.seed+8000+update)))
            teacher=observe(backends[int(samples[j]["triplet_id"])],positions[j],"mmf",args,mmf["teacher_deadband"])
            prior=mmf_loss(v[j],samples[j],teacher,args,mmf["teacher_deadband"])
        loss=fm+(mmf["lambda_pes"]*prior if mmf else 0.); loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(parameters,args.grad_clip)
        optimizer.step(); scheduler.step(); ema.update()
        row=dict(update=update+1,loss_fm=float(fm.detach()),loss_mmf=float(prior.detach()),
            loss_total=float(loss.detach()),grad_norm=float(gradient),
            archive_rows=[s["archive_row"] for s in samples],times=times.tolist(),
            teacher_calls=teacher["force_calls"] if teacher else 0)
        if teacher:
            free=~samples[j]["fixed"]
            row.update(teacher_index=j,teacher_active=teacher["active"],teacher_time=float(times[j]),
                teacher_force_rms=teacher["force_rms"],
                teacher_cosine=alignment(v[j][free].detach().cpu().numpy(),teacher["vector"][free].cpu().numpy()))
        with (out/"training_steps.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        if update==start or (update+1)%args.log_every==0:
            print(f"[train] learnability {update+1}/{args.updates} FM={row['loss_fm']:.6g} "
                  f"t0={int((times==0).sum())}/{len(times)} early={int((times<.25).sum())}/{len(times)} "
                  f"MMF={row['loss_mmf']:.5g} total={row['loss_total']:.6g} "
                  f"PES={row['teacher_calls']} elapsed={(perf_counter()-began)/60:.1f}min",flush=True)
        del v,loss,fm,prior
        if update+1 in save_steps:
            checkpoint=out/("checkpoint_final" if update+1==args.updates else f"checkpoint_step_{update+1:08d}")
            save_checkpoint(model,optimizer,scheduler,ema,checkpoint,update+1,flow_rng,select_rng)
            evaluate(update+1)
    print((out/"report.txt").read_text(),flush=True)


def main():
    args=parse_args(); dataset,saddles=training_data(args)
    split,_=load_split(args.source_run,dataset)
    manifest=json.loads((Path(args.source_run)/"split.json").read_text())["settings"]
    if Path(args.init_weights).resolve()!=Path(manifest["init_weights"]).resolve():
        raise ValueError("use the original EMA initializer, not the failed continuation")
    rows=select_rows(split,args.basin,args.parent,args.num_sources,args.seed)
    records=[recovery_record(split,int(row),dataset) for row in rows]
    _,geometry=load_geometry(args.train_traj,args.test_traj)
    for key in ["cell","train_saddles","test_saddles","false_bridges"]:
        if not np.allclose(split[key],geometry[key]): raise ValueError(f"source split differs in {key}")
    inputs=fixed_inputs(records,geometry,args.seed,args.noise_sigma)
    print(f"[setup] {len(rows)} seen TRAIN sources; basin={args.basin}, parent={args.parent}; rows={rows.tolist()}",flush=True)
    print(f"[setup] {args.updates} updates; recovery-only batches; head LR={args.learning_rate:g}, "
          f"backbone LR={args.uma_lr:g}; {args.lr_schedule} after warmup; "
          + ("R2: one detached MMF teacher/batch" if args.mmf else "R1: no physics teacher"),flush=True)
    if args.dry_run: return
    if file_hash(Path(args.init_weights)/"ema.pt")!=manifest["init_ema_sha256"]:
        raise ValueError("original initializer EMA changed")
    out=Path(args.run_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/"config.json").exists():
        old=json.loads((out/"config.json").read_text())
        if not args.resume_from or bool(old.get("mmf",False)) != args.mmf:
            raise ValueError("use a new run directory, or resume the same FM/MMF experiment")
    selection=dict(rows=rows,source_split_sha256=file_hash(Path(args.source_run)/"split.npz"),
                   init_ema_sha256=manifest["init_ema_sha256"])
    if (out/"selection.npz").exists():
        old=read_npz(out/"selection.npz")
        if any(not np.array_equal(old[k],v) for k,v in selection.items()): raise ValueError("selection changed")
    else: np.savez_compressed(out/"selection.npz",**selection)
    torch.manual_seed(args.seed); model,groups,cfg,name=build_student(args)
    mmf,backends=prepare_mmf(args,model,name,records,saddles,inputs) if args.mmf else (None,None)
    train_control(args,model,groups,cfg,records,inputs,geometry,rows,mmf,backends)
    for trajectory in dataset._traj_cache.values(): trajectory.close()
    print(f"[done] CPU replot: python examples/LiC/experiments/pes_and_recovery/plot_recovery_learnability.py {out/'learnability.npz'}",flush=True)


if __name__=="__main__": main()
