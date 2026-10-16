"""Recovery generalization evaluation, not additional training.

Compare initializer and fine-tuned EMA on identical starts. In shell-source
mode, basin membership means the ORIGINAL baseline endpoint, not the saddle
origin of a shell start.
--starts false-bridges instead probes the catalog bridges themselves, exactly
and with fresh Li noise; --basins unseen excludes the recovery-trained basin.
No force/Hessian oracle is loaded. Geometric catalog hits are not certification.
"""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from saddleflow.data import atoms_to_sample_dict

try:
    from recovery_basins import read_npz, load_learnability_run
    from force_guided_sampling import load_flow_model
    from robustness_shells import sample_points
    from saddle_search_data import classify, load_geometry
except ModuleNotFoundError:
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import read_npz, load_learnability_run
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
    from examples.LiC.rollout_eval.robustness_shells import sample_points
    from examples.LiC.rollout_eval.saddle_search_data import classify, load_geometry


def build_starts(split, selected, basin, parent, sigmas, draws, seed):
    known=(split["baseline_outcome"]==2)&(split["baseline_basin"]==basin)&(split["parent"]==parent)
    if not np.all(known[selected]&(split["partition"][selected]==0)):
        raise ValueError("selected sources are not the intended training episodes")
    held=np.flatnonzero(known&(split["partition"]==1))
    if not len(held): raise ValueError("no withheld sources from this seen basin/parent")
    rng=np.random.default_rng(seed)
    starts,population,source_row,sigma_values,draw_ids=[],[],[],[],[]
    for kind,rows in [("trained",selected),("withheld",held)]:
        for row in rows:
            center=split["starts"][row]
            starts.append(center.copy()); population.append(kind+"_exact")
            source_row.append(row); sigma_values.append(0.); draw_ids.append(-1)
            # Same seeded directions across sigma and both models. Gaussian
            # sigma is per Cartesian component, NOT a fixed shell radius.
            noise=rng.normal(size=(draws,3))
            for sigma in sigmas:
                for j,z in enumerate(noise):
                    starts.append(center+sigma*z); population.append(kind+"_jitter")
                    source_row.append(row); sigma_values.append(sigma); draw_ids.append(j)
    return dict(starts=np.asarray(starts),population=np.asarray(population),
                source_row=np.asarray(source_row),sigma=np.asarray(sigma_values),draw=np.asarray(draw_ids)),held


def build_false_bridge_starts(bridges, seen_basin, basin_choice, sigmas, draws, seed):
    """Starts have a bridge ID, but no parent saddle or recovery target.

    Draw noise for every catalog bridge before subsetting, so --basins unseen
    uses exactly the same points as that subset of --basins all.
    """
    if basin_choice == "all":
        basins = np.arange(len(bridges))
    elif basin_choice == "unseen":
        basins = np.flatnonzero(np.arange(len(bridges)) != seen_basin)
    else:
        basins = np.asarray(sorted({int(s) for s in basin_choice.split(",")}))
    if not len(basins) or np.any((basins < 0) | (basins >= len(bridges))):
        raise ValueError(f"--basins must select IDs from 0 to {len(bridges)-1}")
    rng = np.random.default_rng(seed)
    starts, population, source_basin, sigma_values, draw_ids = [], [], [], [], []
    for basin, center in enumerate(bridges):
        noise = rng.normal(size=(draws, 3))
        if basin not in basins:
            continue
        starts.append(center.copy())
        population.append("false_exact")
        source_basin.append(basin)
        sigma_values.append(0.)
        draw_ids.append(-1)
        for sigma in sigmas:
            for j, z in enumerate(noise):
                starts.append(center + sigma*z)
                population.append("false_noise")
                source_basin.append(basin)
                sigma_values.append(sigma)
                draw_ids.append(j)
    return dict(starts=np.asarray(starts), population=np.asarray(population),
                source_basin=np.asarray(source_basin), sigma=np.asarray(sigma_values),
                draw=np.asarray(draw_ids))


def make_bridge_report(data):
    seen = int(data["seen_basin"])
    basins = np.unique(data["source_basin"])
    sigmas = np.unique(data["sigma"])
    lines = ["unchanged checkpoint, cross-basin false-bridge evaluation", "="*115,
        f"Catalog has {len(data['false_bridges'])} false bridges total; selected IDs={basins.tolist()}.",
        f"Explicit recovery supervision covered baseline-endpoint basin {seen} only, using shell sources, not exact bridge starts.",
        "Other basins received no explicit recovery pairs; this does not claim their entire spatial regions were absent from training.",
        "Exact/noisy starts are centered on FALSE BRIDGES, not true saddles or saved recovery shell sources. No parent target is supplied.",
        "Gaussian sigma is per Cartesian component. All noise directions and starts are identical between the two EMA models.",
        "Autonomous 20-step Euler; no updates, no force guidance, no online PES oracle. Student uses its normal UMA-derived backbone.",
        "Geometry only: ANY stored true saddle <.10 A; ANY false bridge <.30 A, with true precedence; otherwise other.",
        "No force/index-1 certification. Neighborhood draws are correlated; the transfer units are basins, not every jitter sample.",
        "", "method     basins      exposure sigma(A)    N   true%   false%   other%   med d_true(A)   baseline-failed rescued", "-"*115]
    # Overall, seen/unseen aggregates, then each source basin. Nothing is
    # filtered according to a model's new outcome or used to choose weights.
    groups = [("all", "all", np.ones(len(data["starts"]), dtype=bool)),
              ("seen", "seen", data["source_basin"] == seen),
              ("unseen", "unseen", data["source_basin"] != seen)]
    groups += [(str(b), "seen" if b == seen else "unseen", data["source_basin"] == b) for b in basins]
    baseline = data["outcome"][0] == 0
    for i, method in enumerate(data["method"]):
        for group, exposure, selection in groups:
            for sigma in sigmas:
                use = selection & (data["sigma"] == sigma)
                if not use.any():
                    continue
                failed = use & ~baseline
                outcome = data["outcome"][i, use]
                rescued = int(np.sum((data["outcome"][i] == 0) & failed))
                lines.append(f"{method:10s} {group:10s} {exposure:7s} {sigma:8.3f} {use.sum():5d} "
                    f"{100*np.mean(outcome==0):7.2f} {100*np.mean(outcome==1):8.2f} {100*np.mean(outcome==2):8.2f} "
                    f"{np.median(data['distance_true'][i,use]):15.6f} {rescued:5d}/{failed.sum():<5d}")
    for group, _, use in groups[:3]:
        if not use.any():
            continue
        gained = (data["outcome"][1] == 0) & ~baseline & use
        lost = (data["outcome"][1] != 0) & baseline & use
        lines.append(f"Paired {group}: new true hits={gained.sum()}, lost true hits={lost.sum()}, N={use.sum()}.")
    return "\n".join(lines)+"\n"


def make_report(data):
    if str(data.get("start_kind", "shell-sources")) == "false-bridges":
        return make_bridge_report(data)
    groups=list(dict.fromkeys(zip(data["population"],data["sigma"])))
    lines=["recovery learnability generalization check", "="*115,
        f"Trained source rows={data['selected_rows'].tolist()}; withheld rows={data['withheld_rows'].tolist()}.",
        "Both sets originate from the SAME original training saddle; withheld exact source states were never fine-tuned.",
        "Basin IDs refer to original baseline ENDPOINTS. This is within-basin/source generalization, not unseen-basin transfer.",
        "Gaussian Li jitter only; sigma is per-coordinate standard deviation, not a shell radius. Fixed carbons never change.",
        "All starts are shared between models; no filtering by their new outcomes. Jitter samples share centers and are correlated.",
        "Autonomous 20-step Euler; no online force/Hessian oracle. Geometry only: true <.10 A; false <.30 A with true precedence.",
        "No force/index-1 certification and no training/checkpoint selection occurs.", "",
        "method     population       sigma(A)    N   true%   false%   other%   med d_true(A)   baseline-failed rescued", "-"*115]
    baseline=data["outcome"][0]==0
    for i,method in enumerate(data["method"]):
        for group,sigma in groups:
            use=(data["population"]==group)&(data["sigma"]==sigma); failed=use&~baseline
            outcome=data["outcome"][i,use]
            rescued=int(np.sum((data["outcome"][i]==0)&failed))
            lines.append(f"{method:10s} {group:16s} {sigma:8.3f} {use.sum():5d} "
                f"{100*np.mean(outcome==0):7.2f} {100*np.mean(outcome==1):8.2f} {100*np.mean(outcome==2):8.2f} "
                f"{np.median(data['distance_true'][i,use]):15.6f} {rescued:5d}/{failed.sum():<5d}")
    gained=(data["outcome"][1]==0)&~baseline; lost=(data["outcome"][1]!=0)&baseline
    lines += [f"Paired overall: new true hits={gained.sum()}, lost true hits={lost.sum()}, N={len(baseline)}."]
    return "\n".join(lines)+"\n"


def plot_results(archive):
    # CPU-only entry point, including when the experiment was interrupted after
    # numerical output. Never reconstruct a model or repeat sampling here.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try: from plot_saddle_search import draw_endpoint_paths, OUTCOME_COLORS
    except ModuleNotFoundError: from examples.LiC.rollout_eval.plot_saddle_search import draw_endpoint_paths, OUTCOME_COLORS
    archive=Path(archive); data=read_npz(archive)
    groups=list(dict.fromkeys(zip(data["population"],data["sigma"])))
    bridges = str(data.get("start_kind", "shell-sources")) == "false-bridges"
    with PdfPages(archive.with_suffix(".pdf")) as pdf:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for i,method in enumerate(data["method"]):
            for ax,code in zip(axes,[0,1]):
                values=[100*np.mean(data["outcome"][i,(data["population"]==g)&(data["sigma"]==s)]==code) for g,s in groups]
                ax.bar(np.arange(len(groups))+.35*i,values,.35,label=method)
        for ax,title in zip(axes,["Any-true hit (%)","False occupancy (%)"]):
            ax.set_xticks(np.arange(len(groups))+.175,[f"{g}\nsigma={s:g}" for g,s in groups],rotation=35,ha="right",fontsize=8)
            ax.set_ylim(0,105); ax.set_title(title); ax.legend(); ax.grid(axis="y",alpha=.2)
        fig.suptitle("" + ("all selected false bridges and fresh jitter" if bridges else "unseen sources and fresh jitter")
                     + "; geometric evaluation only")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # For the bridge test, show every basin's outcome at each noise level.
        # Basin 0 remains a seen-basin diagnostic, separate from transfer.
        if bridges:
            basins = np.unique(data["source_basin"])
            sigmas = np.unique(data["sigma"])
            fig, axes = plt.subplots(len(sigmas), 2, figsize=(14, 3*len(sigmas)), squeeze=False)
            for row, sigma in enumerate(sigmas):
                for i, method in enumerate(data["method"]):
                    ax = axes[row, i]
                    bottom = np.zeros(len(basins))
                    for code, name in enumerate(["any true", "false region", "other"]):
                        rates = np.asarray([100*np.mean(data["outcome"][i,
                            (data["source_basin"] == b) & (data["sigma"] == sigma)] == code) for b in basins])
                        ax.bar(np.arange(len(basins)), rates, bottom=bottom, color=OUTCOME_COLORS[code], label=name)
                        bottom += rates
                    labels = [str(b) + ("\nseen" if b == int(data["seen_basin"]) else "") for b in basins]
                    ax.set_xticks(np.arange(len(basins)), labels)
                    ax.set_ylim(0, 105); ax.set_title(f"{method}; sigma={sigma:g} A")
                    ax.set_xlabel("Starting false-bridge basin ID"); ax.set_ylabel("Outcomes (%)")
                    if row == 0: ax.legend(fontsize=8)
            fig.suptitle("per-basin geometric outcomes, unchanged checkpoints")
            fig.tight_layout(rect=(0, 0, 1, .96)); pdf.savefig(fig); plt.close(fig)
        views = [(f"{group}, sigma={sigma:g} A", (data["population"] == group) & (data["sigma"] == sigma))
                 for group, sigma in groups]
        if bridges:
            views += [(f"false bridge {b}; " + ("seen" if b == int(data["seen_basin"]) else "unseen") + "; all sigmas",
                       data["source_basin"] == b) for b in np.unique(data["source_basin"])]
        for title, selection in views:
            use=np.flatnonzero(selection)
            fig,axes=plt.subplots(1,2,figsize=(14,7))
            for i,(ax,method) in enumerate(zip(axes,data["method"])):
                draw_endpoint_paths(ax,data["path"][i],data["outcome"][i],data,use)
                outcome=data["outcome"][i,use]
                ax.set_title(f"{method}; N={len(use)}\ntrue={100*np.mean(outcome==0):.1f}%; false={100*np.mean(outcome==1):.1f}%")
            fig.suptitle(f"{title}; identical starts; no start dots")
            fig.tight_layout(rect=(0,0,1,.94)); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] saved {archive.with_suffix('.pdf')}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",default="/scratch/gqs7zr/saddleflow/lic_recovery_mmf_21302150/checkpoint_final")
    p.add_argument("--out-dir")
    p.add_argument("--sigmas",default="0.02,0.05,0.10",
                   help="Positive Gaussian Li noise scales in A, or none for exact starts only")
    p.add_argument("--starts", choices=["shell-sources", "false-bridges"], default="shell-sources",
                   help="Existing within-basin shell-source check, or exact/noisy catalog false bridges")
    p.add_argument("--basins", default="all",
                   help="For false-bridges only: all, unseen (exclude the recovery-trained basin), or IDs such as 1,2,3")
    p.add_argument("--draws-per-source",type=int,default=32); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--seed",type=int,default=2026); p.add_argument("--device",default="cuda")
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--no-plot",action="store_true")
    p.add_argument("--plot-only",metavar="NPZ",help="CPU-only replot of an existing numerical archive")
    args=p.parse_args()
    if args.plot_only: plot_results(args.plot_only); return
    sigmas=[] if args.sigmas == "none" else sorted({float(s) for s in args.sigmas.split(",")})
    if min(args.draws_per_source,args.batch_size)<=0 or any(not np.isfinite(s) or s<=0 for s in sigmas):
        raise ValueError("counts and jitter sigmas must be positive")
    if args.starts == "shell-sources" and args.basins != "all":
        raise ValueError("--basins requires --starts false-bridges")
    checkpoint=Path(args.ckpt); root=checkpoint.parent
    config,selection,split=load_learnability_run(checkpoint)
    reference,geometry=load_geometry(config["train_traj"],config["test_traj"])
    for key in ["cell","train_saddles","test_saddles","false_bridges"]:
        if not np.allclose(split[key],geometry[key]): raise ValueError(f"geometry differs in {key}")
    if args.starts == "false-bridges":
        sources = build_false_bridge_starts(geometry["false_bridges"],config["basin"],args.basins,
                                            sigmas,args.draws_per_source,args.seed)
        held = np.asarray([], dtype=int)
        print(f"[setup] {len(geometry['false_bridges'])} catalog false bridges; "
              f"selected basins={np.unique(sources['source_basin']).tolist()}; "
              f"recovery-trained basin={config['basin']}; exact + noise sigmas={sigmas}",flush=True)
    else:
        sources,held=build_starts(split,selection["rows"],config["basin"],config["parent"],sigmas,args.draws_per_source,args.seed)
        print(f"[setup] {len(selection['rows'])} trained centers; {len(held)} withheld centers {held.tolist()}",flush=True)
    print(f"[setup] {len(sources['starts'])} shared starts; two EMA models, K=20, "
          "no training or PES oracle",flush=True)
    if args.dry_run: return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("no CUDA device here; run this evaluation in a GPU allocation")
    output_name = "false_bridges" if args.starts == "false-bridges" else "generalization"
    out=Path(args.out_dir) if args.out_dir else root/output_name; out.mkdir(parents=True,exist_ok=True)
    sample=atoms_to_sample_dict(reference); li=int(geometry["li_index"])
    paths=[]
    for name,path in [("baseline",Path(config["init_weights"])),("finetuned",checkpoint)]:
        _,_,backbone,attention,head=load_flow_model(path,args.device)
        paths.append(sample_points(sources["starts"],geometry["positions"],sample=sample,
            backbone=backbone,attention=attention,velocity_head=head,device=args.device,
            K=20,batch_size=args.batch_size,li_index=li,label=name))
        del backbone,attention,head
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    outcomes=[classify(path[:,-1],geometry) for path in paths]
    data=dict(**geometry,**sources,method=np.asarray(["baseline","finetuned"]),path=np.asarray(paths),
              outcome=np.asarray([r[0] for r in outcomes]),distance_true=np.asarray([r[1] for r in outcomes]),
              distance_false=np.asarray([r[2] for r in outcomes]),selected_rows=selection["rows"],withheld_rows=held,
              start_kind=np.asarray(args.starts),seen_basin=np.asarray(config["basin"]),
              checkpoint_paths=np.asarray([config["init_weights"],str(checkpoint)]),
              config_json=json.dumps(vars(args)),training_config_json=json.dumps(config))
    archive=out/f"{output_name}.npz"; np.savez_compressed(archive,**data)
    report=make_report(data); (out/"report.txt").write_text(report); print(report,flush=True)
    print(f"[save] {archive} and {out/'report.txt'}",flush=True)
    print(f"[done] CPU replot: python examples/LiC/experiments/pes_and_recovery/evaluate_recovery_learnability.py --plot-only {archive}",flush=True)
    if not args.no_plot: plot_results(archive)


if __name__=="__main__": main()
