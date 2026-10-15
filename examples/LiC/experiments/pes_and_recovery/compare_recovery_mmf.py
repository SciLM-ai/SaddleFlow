"""Michael: evaluate basin-disjoint R0/R1/R2, with autonomous SaddleFlow only.

All source sets and checkpoints are fixed before evaluation. UMA is used only
AFTER sampling for force/index certification. Results precede PDF generation.
"""

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from saddleflow.data import atoms_to_sample_dict
from saddleflow.classical_saddle_search import FreeDOFs, MinimumModeEstimator, PESBackend, evaluate_endpoint, make_uma_calculator

try:
    from compare_pes_directions import checkpoint_at
    from compare_saddle_search import save_archive
    from force_guided_sampling import load_flow_model
    from recovery_basins import file_hash, read_npz
    from robustness_shells import rollout_from_step
    from saddle_search_data import load_geometry
except ModuleNotFoundError:
    from examples.LiC.experiments.pes_and_recovery.compare_pes_directions import checkpoint_at
    from examples.LiC.rollout_eval.compare_saddle_search import save_archive
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import file_hash, read_npz
    from examples.LiC.rollout_eval.robustness_shells import rollout_from_step
    from examples.LiC.rollout_eval.saddle_search_data import load_geometry


def check_matching(root):
    configs = [json.loads((root/e/"config.json").read_text()) for e in ["R0","R1","R2"]]
    keys = ["init_weights","updates","learning_rate","uma_lr","warmup_steps","ema_decay","noise_sigma","seed","split_sha256"]
    for c in configs[1:]:
        if any(c[key] != configs[0][key] for key in keys): raise ValueError("Michael: R0/R1/R2 settings differ")
        if c["extras"]["lambda_pes"] != configs[0]["extras"]["lambda_pes"]: raise ValueError("Michael: calibration differs")
    logs = [[json.loads(s) for s in (root/e/"training_steps.jsonl").read_text().splitlines()] for e in ["R0","R1","R2"]]
    if any([r["update"] for r in rows] != list(range(1,configs[0]["updates"]+1)) for rows in logs):
        raise ValueError("Michael: not all matched continuation updates have completed")
    for rows in logs[1:]:
        if [r["original_sha256"] for r in rows] != [r["original_sha256"] for r in logs[0]]:
            raise ValueError("Michael: original-data streams differ")
    if [(r["batch_sha256"],r["recovery_row"]) for r in logs[1]] != [(r["batch_sha256"],r["recovery_row"]) for r in logs[2]]:
        raise ValueError("Michael: R1/R2 recovery bridges differ")
    allowed = np.flatnonzero(read_npz(root/"split.npz")["partition"] == 0)
    if not np.isin([r["recovery_row"] for r in logs[1]],allowed).all():
        raise ValueError("Michael: recovery logs contain non-training rows")
    if any(r["teacher_calls"] != 0 for rows in logs[:2] for r in rows):
        raise ValueError("Michael: R0/R1 should never query a teacher during updates")
    return configs


def shared_sources(split, geometry, seed, sigma, fresh_archive=None):
    """All mined rows plus clean train/test noise and exact/noisy false bridges."""
    rows = np.flatnonzero(split["partition"] >= 0)
    starts = list(split["starts"][rows]); parent = list(split["parent"][rows]); radius = list(split["radius"][rows])
    partition = list(split["partition"][rows]); archive_row = list(rows)
    basin = [int(split["baseline_basin"][r]) if split["baseline_outcome"][r] == 2 else -1 for r in rows]
    names = ["recovery_train","seen_basin_holdout","withheld_basin","generic_holdout"]
    population = [names[i] for i in partition]
    rng = np.random.default_rng(seed+9000)

    def append(x, group, p=-1, b=-1, r=np.nan):
        starts.append(x); population.append(group); parent.append(p); basin.append(b)
        radius.append(r); partition.append(-1); archive_row.append(-1)

    for group in ["train","test"]:
        for i,x in enumerate(geometry[group+"_saddles"]):
            append(x+rng.normal(0,sigma,3),"true_noise_"+group,i if group == "train" else -1)
    for i,x in enumerate(geometry["false_bridges"]):
        append(x.copy(),"false_exact",b=i)
        append(x+rng.normal(0,.1,3),"false_noise",b=i)
    if fresh_archive:
        a = read_npz(fresh_archive)
        if np.any(a["start_step"] != 0) or not np.allclose(a["cell"],geometry["cell"]):
            raise ValueError("Michael: fresh shell archive must use matching cell and t=0")
        for i,x in enumerate(a["starts"]): append(x,"fresh_shell",int(a["parent_index"][i]),r=float(a["radius"][i]))
    return dict(starts=np.asarray(starts),population=np.asarray(population),parent=np.asarray(parent),radius=np.asarray(radius),
                partition=np.asarray(partition),archive_row=np.asarray(archive_row),baseline_basin=np.asarray(basin))


def report_groups(data):
    p,b = data["source_partition"],data["source_baseline_basin"]
    groups = dict(recovery_train=p == 0,heldout_high_force=p > 0,
                  seen_basin_new_sources=p == 1,withheld_false_basins=p == 2,generic_holdout=p == 3)
    groups.update(all_mined_high_force=p >= 0,all_known_false_failures=(p >= 0)&(b >= 0),
                  recovery_known_false_train=(p == 0)&(b >= 0),recovery_generic_train=(p == 0)&(b < 0))
    for name in ["true_noise_train","true_noise_test","fresh_shell"]:
        groups[name] = data["source_population"] == name
    for name in ["false_exact","false_noise"]:
        for label,ids in [("seen",data["train_basins"]),("withheld",data["holdout_basins"])]:
            groups[name+"_"+label] = (data["source_population"] == name)&np.isin(b,ids)
        groups[name+"_unobserved"] = ((data["source_population"] == name)
            &~np.isin(b,np.r_[data["train_basins"],data["holdout_basins"]]))
    for basin in np.r_[data["train_basins"],data["holdout_basins"]]:
        groups[f"mined_basin_{basin}"] = (p >= 0)&(b == basin)
    return groups


def make_report(data):
    config = json.loads(str(data["config_json"]))
    lines = ["Michael: basin-disjoint recovery FM with/without MMF", "="*110,
        f"Train basins={data['train_basins'].tolist()}; withheld basins={data['holdout_basins'].tolist()} (zero-based catalog IDs)",
        "R0: original-only continuation; R1: 3 original + 1 recovery; R2: same pairs + detached recovery MMF.",
        "Reference is the untouched EMA initializer. All methods use identical autonomous 20-step Euler starts.",
        "No online force/Hessian/minimum-mode queries; student UMA-derived backbone still executes.",
        f"Mining is saved free fmax >{config['force_epsilon']} eV/A, INCLUDING geometric near-misses at true saddles.",
        "True hit: ANY catalog saddle <0.10 A; false occupancy: <0.30 A with true precedence.",
        f"Physical: free fmax <= {config['fmax']} AND lambda1 < -{config['eigenvalue_tol']} AND lambda2 > {config['eigenvalue_tol']}.",
        "The split excludes held-basin recovery SOURCE episodes, not every spatial point on training bridges.",
        "One fold is mechanistic evidence, not a definitive generalization claim. Repeat the complementary fold.","",
        "method         population                      N   true%  false%   phys%  med fmax", "-"*95]
    groups = report_groups(data)
    for group,source_use in groups.items():
        for method in dict.fromkeys(data["method"]):
            use = (data["method"] == method)&source_use[data["source_index"]]
            if not use.any(): continue
            lines.append(f"{method:14s} {group:30s} {use.sum():4d} {100*np.mean(data['outcome'][use]==0):7.2f}"
                f" {100*np.mean(data['outcome'][use]==1):7.2f} {100*data['physical_success'][use].mean():7.2f}"
                f" {np.median(data['force_max'][use]):10.6f}")
    # Paired successes and basin-macro averages prevent one large basin from
    # silently determining the scientific conclusion.
    methods = set(data["method"])
    for m1 in sorted(m for m in methods if m.startswith("R1_")):
        m2 = m1.replace("R1_","R2_",1)
        if m2 not in methods: continue
        i1,i2 = np.flatnonzero(data["method"] == m1),np.flatnonzero(data["method"] == m2)
        if not np.array_equal(data["source_index"][i1],data["source_index"][i2]): raise ValueError("Michael: unpaired evaluation sources")
        for group in ["heldout_high_force","withheld_false_basins"]:
            use = groups[group][data["source_index"][i1]]
            y1,y2 = data["outcome"][i1][use] == 0,data["outcome"][i2][use] == 0
            if use.any(): lines.append(f"Paired {m2} vs {m1}, {group}: new true hits={np.sum(y2&~y1)}, lost true hits={np.sum(y1&~y2)}, N={use.sum()}")
        for method in [m1,m2]:
            rates = []
            for basin in data["holdout_basins"]:
                use = (data["method"] == method)&((data["source_partition"] == 2)&(data["source_baseline_basin"] == basin))[data["source_index"]]
                if use.any(): rates.append(np.mean(data["outcome"][use] == 0))
            if rates: lines.append(f"{method}: withheld-basin macro true hit={100*np.mean(rates):.2f}% over {len(rates)} basins")
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--test-traj",default=str((Path(__file__).resolve().parents[2] / "test_set.traj")))
    p.add_argument("--updates",default="2000",help="Predeclared saved steps; initializer reference always included")
    p.add_argument("--fresh-shell-archive",help="Optional additional unseen shell sources, not their saved endpoints")
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--fmax",type=float,default=.005)
    p.add_argument("--eigenvalue-tol",type=float,default=.01)
    p.add_argument("--device",default="cuda")
    p.add_argument("--no-plot",action="store_true")
    args = p.parse_args(); root = Path(args.run_dir); out = root/"comparison"; out.mkdir(exist_ok=True)
    configs = check_matching(root); split = read_npz(root/"split.npz")
    manifest = json.loads((root/"split.json").read_text())
    if file_hash(root/"split.npz") != manifest["split_sha256"] or configs[0]["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("Michael: evaluation split differs from training")
    reference,geometry = load_geometry(configs[0]["train_traj"],args.test_traj)
    for key in ["cell","train_saddles","test_saddles","false_bridges"]:
        if not np.allclose(split[key],geometry[key]): raise ValueError(f"Michael: evaluation geometry differs in {key}")
    sources = shared_sources(split,geometry,configs[0]["seed"],configs[0]["noise_sigma"],args.fresh_shell_archive)
    args.force_epsilon = manifest["settings"]["force_epsilon"]
    if args.fresh_shell_archive and file_hash(args.fresh_shell_archive) == manifest["settings"]["shell_sha256"]:
        raise ValueError("Michael: the mining archive cannot also be called fresh evaluation data")
    args.K,args.max_steps,args.step_budgets,args.restarts = 20,20,"20",1
    args.model,args.fd_eps = configs[0]["extras"]["backbone"],configs[0]["fd_eps"]
    args.checkpoints = {}
    dofs,sample = FreeDOFs(reference),atoms_to_sample_dict(reference)
    calculator,batch_evaluate = make_uma_calculator(args.model,args.device,sample["task_name"])
    checkpoints = [("reference",root/"R0/checkpoint_step_00000000")]
    checkpoints += [(f"{e}_s{u:04d}",checkpoint_at(root/e,u,configs[0]["updates"]))
                    for e in ["R0","R1","R2"] for u in [int(s) for s in args.updates.split(",")]]
    results = []
    for method,checkpoint in checkpoints:
        args.checkpoints[method] = str(checkpoint)
        _,_,backbone,attention,head = load_flow_model(checkpoint,args.device)
        paths,elapsed = [],[]
        for first in range(0,len(sources["starts"]),args.batch_size):
            points = sources["starts"][first:first+args.batch_size]
            x = torch.as_tensor(np.asarray([dofs.expand(v) for v in points]),dtype=torch.float32)
            began = perf_counter()
            _,path = rollout_from_step(x,0,sample=sample,backbone=backbone,attention=attention,
                velocity_head=head,device=args.device,K=20)
            paths.extend(path.numpy()); elapsed.extend([(perf_counter()-began)/len(points)]*len(points))
            print(f"[sample] Michael: {method} {len(paths)}/{len(sources['starts'])}; no PES guidance",flush=True)
        del backbone,attention,head
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        for i,path in enumerate(paths):
            pes = PESBackend(reference,calculator,batch_evaluate); began = perf_counter()
            evaluation = evaluate_endpoint(pes,path[-1],MinimumModeEstimator(eps=args.fd_eps),args.fmax,args.eigenvalue_tol)
            trace = [dict(x=x,q=None,eigs=None,force_rms=np.nan,force_max=np.nan,query_count=0,
                force_parallel=np.nan,force_perpendicular=np.nan,shift_rms=0,hessian_asymmetry=np.nan) for x in path]
            results.append(dict(method=method,source_index=i,restart=0,final=path[-1],trace=trace,
                status="fixed_K",error="",steps=20,force_calls=0,wall_seconds=elapsed[i],model_forward_calls=20,
                diagnostic_seconds=perf_counter()-began,diagnostic_force_calls=evaluation["diagnostic_force_calls"],
                evaluation=evaluation,budget_results=[dict(x=path[-1],**evaluation)]))
            if (i+1)%32 == 0 or i+1 == len(paths): print(f"[verify] Michael: {method} {i+1}/{len(paths)} post-hoc endpoints",flush=True)
        archive = out/"saddle_search.npz"
        data = save_archive(archive,args,geometry,sources,results,[20])
        data.update(source_partition=sources["partition"],source_archive_row=sources["archive_row"],
            source_baseline_basin=sources["baseline_basin"],train_basins=split["train_basins"],holdout_basins=split["holdout_basins"],
            split_sha256=manifest["split_sha256"])
        np.savez_compressed(archive,**data)
        (out/"report.txt").write_text(make_report(data))
        print(f"[save] Michael: {archive}",flush=True)
    print(make_report(data),flush=True)
    if not args.no_plot:
        try: from plot_recovery_mmf import render_pdf
        except ModuleNotFoundError: from examples.LiC.experiments.pes_and_recovery.plot_recovery_mmf import render_pdf
        render_pdf(archive)


if __name__ == "__main__": main()
