"""Michael: plot the cached kappa audit. No UMA, GPU, or rollout is needed."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from ase.geometry import find_mic

try:
    from kappa_audit import OUTCOME_NAMES, read_npz
except ModuleNotFoundError:
    from examples.LiC.experiments.kappa_audit.kappa_audit import OUTCOME_NAMES, read_npz


COLORS = ["tab:green","darkorange","tab:red","tab:purple"]


def local_positions(x, center, data):
    delta,_ = find_mic(np.asarray(x)-center,data["cell"],data["pbc"])
    return center+delta


def history_page(pdf, data, use, title, path_limit):
    fig,axes = plt.subplots(1,3,figsize=(15,5))
    rng = np.random.default_rng(0)
    for outcome,name in enumerate(OUTCOME_NAMES):
        rows = np.flatnonzero(use & (data["outcome"] == outcome))
        if len(rows) > path_limit: rows = rng.choice(rows,path_limit,replace=False)
        for i in rows:
            n = data["trace_length"][i]; x = data["progress"][i,:n]
            for ax,key in zip(axes,["nu1","kappa","force_norm"]):
                y = data["history_"+key][i,:n]
                if key == "force_norm": y = np.where(y>0,y,np.nan)
                ax.plot(x,y,color=COLORS[outcome],lw=.7,alpha=.25)
    config = json.loads(str(data["config_json"]))
    axes[0].axhline(0,color="black",lw=.8)
    if config["curvature_tol"]: axes[0].axhline(-config["curvature_tol"],ls="--",c="0.4",lw=.7)
    axes[1].axhline(0,color="black",lw=.8); axes[1].set_yscale("symlog",linthresh=1.)
    axes[2].set_yscale("log"); axes[2].axhline(config["force_floor"],ls="--",c="black",lw=.8)
    for ax,ylabel in zip(axes,["nu1 (eV/A^2)","kappa (1/A)","Free force norm (eV/A)"]):
        ax.set_xlabel("Normalized stored-trajectory progress (not a shared solver clock)")
        ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    legend = [Line2D([],[],color=c,label=f"{name}: N={np.sum(use&(data['outcome']==j))}")
              for j,(c,name) in enumerate(zip(COLORS,OUTCOME_NAMES))]
    fig.legend(handles=legend,loc="lower center",ncol=4,fontsize=9)
    fig.suptitle(title+f"\nActual full-3D path values; nu1/kappa gaps are masked. At most {path_limit} curves/category drawn.")
    fig.tight_layout(rect=(0,.09,1,.91)); pdf.savefig(fig); plt.close(fig)


def map_page(pdf, data, use, title, path_limit):
    grid = data["grid_point_index"]
    if not grid.size: return
    config = json.loads(str(data["config_json"]))
    shape = grid.shape; x = data["grid_positions"][:,0].reshape(shape); y = data["grid_positions"][:,1].reshape(shape)
    available = grid >= 0; safe = np.maximum(grid,0)
    energy = np.where(available,data["point_energy"][safe],np.nan)
    if available.any(): energy = energy-np.nanmin(energy)
    nu = np.where(available & data["point_valid"][safe],data["point_nu1"][safe],np.nan)
    sign = np.ma.masked_invalid(np.sign(-nu))  # negative kappa dark; positive light
    cmap = ListedColormap(["#285a93","#a9d8ef"]).with_extremes(bad="#dedede")
    center = data["map_center"]
    fig,axes = plt.subplots(1,2,figsize=(13,6))
    rng = np.random.default_rng(0)
    for ax,successful in zip(axes,[True,False]):
        ax.pcolormesh(x,y,sign,cmap=cmap,vmin=-1,vmax=1,shading="auto",zorder=0)
        if available.any() and np.ptp(energy[available]) > 1e-12:
            contours = ax.contour(x,y,np.ma.masked_invalid(energy),levels=12,colors="black",linewidths=.45,alpha=.65)
            ax.clabel(contours,levels=contours.levels[::3],fontsize=6,fmt="%.2f")
        finite = nu[np.isfinite(nu)]
        if len(finite) and finite.min() < 0 < finite.max():
            ax.contour(x,y,np.ma.masked_invalid(nu),levels=[0],colors="white",linewidths=1.8)
        else: ax.text(.02,.98,"No resolved kappa=0 contour",transform=ax.transAxes,va="top",fontsize=8)
        if not available.all():
            ax.text(.02,.93,f"PARTIAL PES map: {available.sum()}/{grid.size} points; missing points blank",
                    transform=ax.transAxes,va="top",fontsize=8)
        rows = np.flatnonzero(use & ((data["outcome"] < 2) if successful else (data["outcome"] == 2)))
        visible = []
        for i in rows:
            p = local_positions(data["path"][i,:data["trace_length"][i]],center,data)
            inside = (p[:,0]>=x.min())&(p[:,0]<=x.max())&(p[:,1]>=y.min())&(p[:,1]<=y.max())
            if inside.any(): visible.append(i)
        chosen = rng.choice(visible,min(path_limit,len(visible)),replace=False) if visible else []
        z_excursions = []
        for i in chosen:
            p = local_positions(data["path"][i,:data["trace_length"][i]],center,data)
            # Break periodic jumps; do not invent cross-cell trajectory lines.
            cuts = np.flatnonzero(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
                > .5*min(np.linalg.norm(data["cell"][0]),np.linalg.norm(data["cell"][1])))+1
            for segment in np.split(p,cuts):
                ax.plot(segment[:,0],segment[:,1],color="limegreen" if successful else "crimson",lw=1.,alpha=.7,zorder=4)
            ax.scatter(p[-1,0],p[-1,1],s=15,c=COLORS[data["outcome"][i]],zorder=5)
            z_excursions.append(np.max(np.abs(p[:,2]-center[2])))
        for key,color,marker in [("train_saddles","black","*"),("test_saddles","black","*"),("false_bridges","red","x")]:
            points = local_positions(data[key],center,data)
            ax.scatter(points[:,0],points[:,1],c=color,marker=marker,s=35,zorder=6)
        ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max()); ax.set_aspect("equal")
        ax.set_xlabel("Li x (A)"); ax.set_ylabel("Li y (A)")
        dz = np.median(z_excursions) if z_excursions else np.nan
        ax.set_title(("True geometric endpoints" if successful else "False-bridge endpoints")+
            f"\n{len(visible)} paths intersect patch; {len(chosen)} drawn; median max |z-z0|={dz:.3f} A")
    legend = [Line2D([],[],c="black",label="Relative UMA energy contours (eV)"),
        Line2D([],[],c="white",lw=2,label="kappa=0 (from valid nu1=0)"),
        Line2D([],[],c="#285a93",lw=8,label="kappa<0"),Line2D([],[],c="#a9d8ef",lw=8,label="kappa>0"),
        Line2D([],[],c="#dedede",lw=8,label="Undefined / low-force / not yet evaluated"),
        Line2D([],[],c="black",marker="*",ls="",label="Stored true saddles"),
        Line2D([],[],c="red",marker="x",ls="",label="Known false bridges")]
    fig.legend(handles=legend,loc="lower center",ncol=4,fontsize=8,facecolor="0.85")
    fig.suptitle(title+f"\nFixed z={center[2]:.4f} A slice; curvature uses FULL 3D F/H. Projected paths need not stay in this plane.")
    fig.tight_layout(rect=(0,.14,1,.91)); pdf.savefig(fig); plt.close(fig)


def render_pdf(path, output=None, path_limit=30):
    path = Path(path); data = read_npz(path)
    output = Path(output) if output else path.with_suffix(".pdf")
    groups = list(dict.fromkeys(zip(data["collection"],data["method"])))
    partial = "audit_complete" in data and not bool(data["audit_complete"])
    coverage = (f"PARTIAL: {int(data['num_complete_trajectories'])}/{int(data['num_planned_trajectories'])} trajectories; "
                f"{int(data['num_evaluated_points'])}/{int(data['num_planned_points'])} physics points"
                if partial else "")
    with PdfPages(output) as pdf:
        fig,axes = plt.subplots(1,2,figsize=(12,5))
        for j,(collection,method) in enumerate(groups):
            use = (data["collection"] == collection)&(data["method"] == method)
            eligible = use & data["lead_observable"]
            lead = data["lead_steps"][eligible]; lead = lead[np.isfinite(lead)]
            if len(lead): axes[0].plot(np.sort(lead),np.arange(1,len(lead)+1)/len(lead),label=f"{collection}/{method}, N={len(lead)}")
            true = use & (data["outcome"] == 0)&(data["valid_fraction"]>0)
            axes[1].bar(j,100*data["any_warning"][true].mean() if true.any() else np.nan)
        axes[0].axvline(0,c="black",lw=.8); axes[0].set_xlabel("False-entry step minus first-warning step")
        axes[0].set_ylabel("CDF among trajectories with a defined warning")
        if axes[0].get_legend_handles_labels()[0]: axes[0].legend(fontsize=8)
        if not groups:
            axes[0].text(.5,.5,"No fully audited trajectories yet",ha="center",transform=axes[0].transAxes)
        axes[0].set_title("Lead (positive=advance warning); missing warnings excluded here")
        axes[1].set_xticks(np.arange(len(groups)),[f"{c}\n{m}" for c,m in groups],fontsize=8)
        axes[1].set_ylabel("Any-warning false-positive rate on observable certified true paths (%)")
        axes[1].set_ylim(0,105)
        fig.suptitle("Michael: candidate kappa warning, NOT a failure label\n"+coverage+
                     "\nRead report.txt for complete lead denominators, missing warnings and masked coverage")
        fig.tight_layout(rect=(0,0,1,.86)); pdf.savefig(fig); plt.close(fig)
        for collection,method in groups:
            use = (data["collection"] == collection)&(data["method"] == method)
            title = f"Michael: {collection} / {method}"+(f" — {coverage}" if partial else "")
            history_page(pdf,data,use,title,path_limit)
            map_page(pdf,data,use,title,path_limit)
    print(f"[plot] Michael: saved {output}",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive"); p.add_argument("--out"); p.add_argument("--path-limit",type=int,default=30)
    args = p.parse_args(); render_pdf(args.archive,args.out,args.path_limit)
