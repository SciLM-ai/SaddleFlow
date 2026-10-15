"""Michael: restart matched recovery students on saved failures at ORIGINAL times.

Inference only: no training, MMF teacher, or online PES guidance. The baseline
archive supplies x_k; each student takes K-k Euler steps with the original 1/K.
"""

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.geometry import find_mic

from saddleflow.data import atoms_to_sample_dict

try:
    from force_guided_sampling import load_flow_model
    from recovery_basins import file_hash, read_npz, load_learnability_run
    from robustness_shells import sample_points
    from saddle_search_data import classify, load_geometry
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import file_hash, read_npz, load_learnability_run
    from examples.LiC.rollout_eval.robustness_shells import sample_points
    from examples.LiC.rollout_eval.saddle_search_data import classify, load_geometry


def matched_runs(r1, r2):
    c1,s1,split=load_learnability_run(r1)
    c2,s2,_=load_learnability_run(r2)
    if c1.get("mmf",False) or not c2.get("mmf",False) or c2["extras"]["lambda_pes"] <= 0:
        raise ValueError("Michael: use plain six-source R1 and its MMF-enabled R2 counterpart")
    keys=["basin","parent","num_sources","batch_size","updates","learning_rate","uma_lr",
          "warmup_steps","lr_schedule","ema_decay","grad_clip","noise_sigma","seed"]
    if any(c1[k] != c2[k] for k in keys):
        raise ValueError("Michael: R1/R2 training settings are not matched")
    for key in ["init_weights","source_run","train_traj","test_traj"]:
        if Path(c1[key]).resolve() != Path(c2[key]).resolve():
            raise ValueError(f"Michael: R1/R2 differ in {key}")
    for key in ["rows","source_split_sha256","init_ema_sha256"]:
        if not np.array_equal(s1[key],s2[key]):
            raise ValueError(f"Michael: R1/R2 recovery provenance differs in {key}")
    steps=[json.loads((Path(p)/"meta.json").read_text())["global_step"] for p in [r1,r2]]
    if steps[0] != steps[1]: raise ValueError("Michael: compare R1/R2 at the same saved optimizer update")
    # Check actual source/time streams, not just matching config files.
    logs=[]
    for p in [r1,r2]:
        rows=[json.loads(line) for line in (Path(p).parent/"training_steps.jsonl").read_text().splitlines()]
        rows=[row for row in rows if row["update"] <= steps[0]]
        if [row["update"] for row in rows] != list(range(1,steps[0]+1)):
            raise ValueError("Michael: R1/R2 training logs are incomplete")
        logs.append(rows)
    if [(r["archive_rows"],r["times"]) for r in logs[0]] != [(r["archive_rows"],r["times"]) for r in logs[1]]:
        raise ValueError("Michael: R1/R2 source/time streams differ")
    if any(r["teacher_calls"] for r in logs[0]) or (steps[0] and not any(r["teacher_calls"] for r in logs[1])):
        raise ValueError("Michael: R1 should be FM-only; R2 must have actually queried its teacher")
    root=Path(r2).parent
    if file_hash(root/"mmf_calibration.json") != c2["mmf_calibration_sha256"]:
        raise ValueError("Michael: R2 calibration changed")
    return c1,s1,split,steps[0]


def select_episodes(split, selected, seen_basin, kind, basins, limit, seed):
    # The larger fold's partition=0 is NOT the six-source training set.
    # Every unselected episode is held out from this small R1/R2 experiment.
    trained=np.isin(np.arange(len(split["starts"])),selected)
    use=(split["partition"] >= 0)&(split["baseline_outcome"] == 2)
    if kind == "heldout": use &= ~trained
    elif kind == "trained": use &= trained
    if basins == "unseen": use &= split["baseline_basin"] != seen_basin
    elif basins != "all": use &= np.isin(split["baseline_basin"],[int(b) for b in basins.split(",")])
    rows=np.flatnonzero(use)
    if limit is not None:
        rows=np.sort(np.random.default_rng(seed).choice(rows,min(limit,len(rows)),replace=False))
    if not len(rows): raise ValueError("Michael: no saved known-false episodes match this selection")
    population=np.where(trained[rows],"trained",np.where(split["baseline_basin"][rows] == seen_basin,
                                                      "seen_basin_holdout","unseen_basins"))
    return rows,population


def report_groups(data):
    groups={"all":np.ones(len(data["archive_row"]),dtype=bool)}
    groups.update({str(p):data["population"] == p for p in np.unique(data["population"])})
    groups.update({f"basin_{b}":data["baseline_basin"] == b for b in np.unique(data["baseline_basin"])})
    return groups


def make_report(data):
    K=int(data["K"])
    lines=["Michael: trajectory-time recovery horizon (EMA)","="*130,
        f"Selected episodes={len(data['archive_row'])}; trained source rows={data['selected_rows'].tolist()}; "
        f"R1/R2 checkpoint update={int(data['training_update'])}.",
        "Heldout means NOT one of the six supervised sources, regardless of membership in the older larger fold.",
        "R0 is the untouched initializer that generated the saved failures. R1/R2 have matched source/time training streams.",
        f"Original K={K}, dt={1/K:g}; restart steps={data['restart_step'].tolist()}. Never reset t or rescale dt.",
        "The SAME baseline state/time is given to every student. Parent/episode IDs are metadata, not student conditioning.",
        "No updates or online energy/force/Hessian/MMF queries. The usual UMA-derived student backbone still executes.",
        "Geometry only: any catalog true <.10 A; any false bridge <.30 A, with true precedence. No physical certification.",
        "Late restarts have fewer steps and less correction time; failure does not prove a fixed point or a capacity limit.",
        "Rates need not be monotonic with restart time. Report the whole curve, not a guaranteed basin-wide cutoff.","",
        "method population                  step     t   remaining    N   true%  false%  other%  med d_true  med d_false  med dx1", "-"*130]
    for group,use in report_groups(data).items():
        for i,method in enumerate(data["method"]):
            for j,step in enumerate(data["restart_step"]):
                y=data["outcome"][i,j,use]
                lines.append(f"{method:6s} {group:27s} {step:4d} {step/K:5.2f} {K-step:11d} {use.sum():4d} "
                    f"{100*np.mean(y==0):7.2f} {100*np.mean(y==1):7.2f} {100*np.mean(y==2):7.2f} "
                    f"{np.median(data['distance_true'][i,j,use]):11.6f} "
                    f"{np.median(data['distance_false'][i,j,use]):12.6f} "
                    f"{np.median(data['first_step_displacement'][i,j,use]):9.6f}")
    lines += ["",f"R0 replay error to saved original endpoint: max={np.max(data['replay_error']):.7g} A, "
              f"median={np.median(data['replay_error']):.7g} A."]
    if np.max(data["replay_error"]) > float(data["replay_tol"]):
        lines.append("WARNING: baseline replay exceeds tolerance. Inspect sampler/checkpoint provenance before interpreting R1/R2.")
    if "R2" in data["method"]:
        i1,i2=[list(data["method"]).index(m) for m in ["R1","R2"]]
        for group,use in report_groups(data).items():
            for j,step in enumerate(data["restart_step"]):
                y1,y2=data["outcome"][i1,j,use] == 0,data["outcome"][i2,j,use] == 0
                lines.append(f"Paired R2 vs R1, {group}, t={step/K:.2f}: "
                             f"new true hits={np.sum(y2&~y1)}, lost true hits={np.sum(y1&~y2)}, N={use.sum()}.")
    return "\n".join(lines)+"\n"


def plot_results(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try: from plot_saddle_search import draw_endpoint_paths, path_segments
    except ModuleNotFoundError: from examples.LiC.rollout_eval.plot_saddle_search import draw_endpoint_paths, path_segments
    path=Path(path); data=read_npz(path); times=data["restart_step"]/int(data["K"])
    groups=report_groups(data)
    with PdfPages(path.with_suffix(".pdf")) as pdf:
        for group,use in groups.items():
            fig,axes=plt.subplots(2,2,figsize=(12,9))
            for i,method in enumerate(data["method"]):
                y=data["outcome"][i][:,use]
                axes[0,0].plot(times,100*np.mean(y == 0,axis=1),"o-",label=method)
                axes[0,1].plot(times,100*np.mean(y == 1,axis=1),"o-",label=method)
                axes[1,0].plot(times,np.median(data["distance_true"][i][:,use],axis=1),"o-",label=method)
                axes[1,1].plot(times,np.median(data["distance_false"][i][:,use],axis=1),"o-",label=method)
            for ax,title in zip(axes.flat,["Any-true hit (%)","False occupancy (%)","Median distance to any true (A)","Median distance to any false (A)"]):
                ax.set_title(title); ax.set_xlabel("Original restart time"); ax.grid(alpha=.2); ax.legend()
            for ax in axes[0]: ax.set_ylim(-2,102)
            fig.suptitle(f"Michael: {group}, N={use.sum()}; matched states/times, no PES guidance")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # Whole-population maps by restart time. Gray prefix is the saved failed
        # approach; blue suffix is the student's NEW remaining integration.
        indices=np.arange(len(data["archive_row"]))
        chosen=indices if len(indices)<=100 else np.random.default_rng(0).choice(indices,100,replace=False)
        for j,step in enumerate(data["restart_step"]):
            fig,axes=plt.subplots(1,len(data["method"]),figsize=(7*len(data["method"]),7),squeeze=False)
            for i,(ax,method) in enumerate(zip(axes.flat,data["method"])):
                draw_endpoint_paths(ax,data["path"][i,j],data["outcome"][i,j],data)
                for n in chosen:
                    for segment in path_segments(data["baseline_path"][n,:step+1],data["cell"],data["pbc"]):
                        ax.plot(segment[:,0],segment[:,1],c=".7",lw=.6,alpha=.35,zorder=1)
                ax.set_title(f"{method}: true={100*np.mean(data['outcome'][i,j]==0):.1f}%; "
                             f"false={100*np.mean(data['outcome'][i,j]==1):.1f}%")
            fig.suptitle(f"Michael: restart at t={step/int(data['K']):.2f}; remaining steps={int(data['K'])-step}\n"
                         "Gray: saved baseline prefix. Blue: new student suffix. Endpoint colors: true/false/other. No start dots.")
            fig.tight_layout(rect=(0,0,1,.92)); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] Michael: {path.with_suffix('.pdf')}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r1-ckpt",default="/scratch/gqs7zr/saddleflow/lic_recovery_mmf_21302150/checkpoint_final")
    p.add_argument("--r2-ckpt",help="Matched six-source MMF checkpoint; never substitute the older larger-fold R2")
    p.add_argument("--out-dir"); p.add_argument("--restart-steps",default="0,5,10,15,18,19")
    p.add_argument("--episodes",choices=["heldout","all","trained"],default="heldout")
    p.add_argument("--basins",default="all",help="all, unseen, or catalog IDs such as 0 or 1,2")
    p.add_argument("--limit-episodes",type=int); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--device",default="cuda")
    p.add_argument("--replay-tol",type=float,default=.001)
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--no-plot",action="store_true")
    p.add_argument("--plot-only",metavar="NPZ")
    args=p.parse_args()
    if args.plot_only: plot_results(args.plot_only); return
    if not args.r2_ckpt: p.error("Michael: --r2-ckpt is required for the matched R1/R2 test")
    if args.batch_size <= 0 or (args.limit_episodes is not None and args.limit_episodes <= 0):
        p.error("Michael: counts must be positive")
    if not np.isfinite(args.replay_tol) or args.replay_tol <= 0: p.error("Michael: replay tolerance must be positive")
    config,selection,split,update=matched_runs(args.r1_ckpt,args.r2_ckpt)
    manifest=json.loads((Path(config["source_run"])/"split.json").read_text())["settings"]
    archive_path=Path(manifest["shell_archive"])
    if file_hash(archive_path) != manifest["shell_sha256"] or file_hash(Path(config["init_weights"])/"ema.pt") != str(selection["init_ema_sha256"]):
        raise ValueError("Michael: original baseline archive or initializer EMA changed")
    archive=read_npz(archive_path); original=json.loads(str(archive["configuration_json"]))
    K=int(original["K"])
    if Path(original["ckpt"]).resolve() != Path(config["init_weights"]).resolve() or np.any(archive["start_step"] != 0):
        raise ValueError("Michael: use saved complete trajectories from the original initializer")
    steps=np.asarray(sorted({int(s) for s in args.restart_steps.split(",")}))
    if not len(steps) or np.any((steps < 0)|(steps >= K)):
        raise ValueError(f"Michael: restart steps must be between 0 and {K-1}")
    rows,population=select_episodes(split,selection["rows"],config["basin"],args.episodes,args.basins,args.limit_episodes,args.seed)
    baseline_path=archive["paths"][rows]
    if baseline_path.shape != (len(rows),K+1,3) or not np.isfinite(baseline_path).all():
        raise ValueError("Michael: selected baseline paths are incomplete")
    reference,geometry=load_geometry(config["train_traj"],config["test_traj"])
    for key in ["cell","train_saddles","test_saddles","false_bridges"]:
        if not np.allclose(archive[key],geometry[key]) or not np.allclose(split[key],geometry[key]):
            raise ValueError(f"Michael: saved geometry differs in {key}")
    if not np.allclose(archive["starts"][rows],split["starts"][rows]) or not np.all(classify(baseline_path[:,-1],geometry)[0] == 1):
        raise ValueError("Michael: archive rows do not reproduce the selected known-false episodes")
    groups={str(name):int(count) for name,count in zip(*np.unique(population,return_counts=True))}
    print(f"[setup] Michael: {len(rows)} saved failure episodes; "
          f"groups={groups}; basins={np.unique(split['baseline_basin'][rows]).tolist()}",flush=True)
    print(f"[setup] Michael: matched R1/R2 update={update}; K={K}; steps={steps.tolist()}; "
          "original t/dt preserved; no training or PES oracle",flush=True)
    if args.dry_run: return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Michael: run GPU inference in an allocation; --plot-only uses CPU")
    out=Path(args.out_dir) if args.out_dir else Path(args.r2_ckpt).parent/"recovery_horizon"
    out.mkdir(parents=True,exist_ok=True)
    sample=atoms_to_sample_dict(reference); names=[]; paths=[]; outcomes=[]; d_true=[]; d_false=[]; dx1=[]
    checkpoints=[("R0",config["init_weights"]),("R1",args.r1_ckpt),("R2",args.r2_ckpt)]
    for method,checkpoint in checkpoints:
        _,_,backbone,attention,head=load_flow_model(Path(checkpoint),args.device)
        branches=[]; shifts=[]
        for step in steps:
            branch=sample_points(baseline_path[:,step],geometry["positions"],sample=sample,
                backbone=backbone,attention=attention,velocity_head=head,device=args.device,K=K,
                batch_size=args.batch_size,start_step=int(step),li_index=int(geometry["li_index"]),label=f"{method} k={step}")
            delta,_=find_mic(branch[:,step+1]-baseline_path[:,step],geometry["cell"],geometry["pbc"])
            shifts.append(np.linalg.norm(delta,axis=1)); branches.append(branch)
        names.append(method); paths.append(np.asarray(branches)); dx1.append(np.asarray(shifts))
        scores=[classify(b[:,-1],geometry) for b in branches]
        outcomes.append(np.asarray([s[0] for s in scores])); d_true.append(np.asarray([s[1] for s in scores])); d_false.append(np.asarray([s[2] for s in scores]))
        del backbone,attention,head
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Persist complete methods before rendering, so even a partial run can
        # be inspected on CPU. Prefix padding remains NaN, not invented motion.
        delta,_=find_mic((paths[0][:,:,-1]-baseline_path[None,:,-1]).reshape(-1,3),geometry["cell"],geometry["pbc"])
        data=dict(**geometry,method=np.asarray(names),path=np.asarray(paths),outcome=np.asarray(outcomes),
            distance_true=np.asarray(d_true),distance_false=np.asarray(d_false),first_step_displacement=np.asarray(dx1),
            baseline_path=baseline_path,restart_step=steps,K=np.asarray(K),replay_error=np.linalg.norm(delta,axis=1).reshape(len(steps),len(rows)),
            replay_tol=np.asarray(args.replay_tol),archive_row=rows,population=population,
            parent=split["parent"][rows],baseline_basin=split["baseline_basin"][rows],selected_rows=selection["rows"],
            training_update=np.asarray(update),checkpoint_paths=np.asarray([p for _,p in checkpoints[:len(names)]]),
            source_archive=str(archive_path),source_archive_sha256=manifest["shell_sha256"],
            r1_config_json=json.dumps(config),config_json=json.dumps(vars(args)))
        numerical=out/"recovery_horizon.npz"; temporary=out/"recovery_horizon.tmp.npz"
        np.savez_compressed(temporary,**data); temporary.replace(numerical)
        report=make_report(data); (out/"report.txt").write_text(report); print(report,flush=True)
        print(f"[save] Michael: {numerical} and {out/'report.txt'}",flush=True)
    print(f"[done] Michael: CPU replot: python examples/LiC/experiments/pes_and_recovery/recovery_horizon.py --plot-only {numerical}",flush=True)
    if not args.no_plot: plot_results(numerical)


if __name__ == "__main__": main()
