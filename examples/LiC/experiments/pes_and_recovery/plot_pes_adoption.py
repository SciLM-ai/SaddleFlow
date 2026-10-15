"""Michael: fixed-probe teacher-adoption plots from an existing NPZ only."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def render_pdf(path, output=None):
    path = Path(path)
    with np.load(path,allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    output = Path(output) if output else path.with_suffix(".pdf")
    first,second = data.get("comparison_names",np.asarray(["E0","E2"]))
    with PdfPages(output) as pdf:
        fig,axes = plt.subplots(2,2,figsize=(12,9),sharey="row")
        for i,weights in enumerate(data["weights"]):
            use = data["active"] & np.isfinite(data["delta_cosine"][i])
            scatter = axes[0,i].scatter(data["teacher_disagreement"][use],data["delta_cosine"][i,use],
                                       c=data["t"][use],cmap="Blues",vmin=0,vmax=1,s=22,alpha=.8)
            axes[0,i].set_title(weights+ (" — primary" if weights=="ema" else " — diagnostic"))
            axes[0,i].set_xlabel("Teacher/FM disagreement: 1 - cos(target, teacher)")
            axes[0,i].set_ylabel(f"Teacher adoption: cos({second},teacher) - cos({first},teacher)")
            axes[1,i].scatter(data["delta_velocity_rms"][i,use],data["delta_cosine"][i,use],
                              c=data["t"][use],cmap="Blues",vmin=0,vmax=1,s=22,alpha=.8)
            axes[1,i].set_xlabel(f"RMS({second} velocity - {first} velocity), A/FM-time")
            axes[1,i].set_ylabel("Teacher adoption (delta cosine)")
            for ax in axes[:,i]: ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.2)
        fig.colorbar(scatter,ax=axes.ravel().tolist(),label="Flow time t",fraction=.025,pad=.025)
        fig.suptitle(f"Michael: fixed probes, cached MMF teacher; {second} versus {first}\nPositive delta cosine is better teacher alignment")
        fig.subplots_adjust(top=.9,hspace=.3,wspace=.25,right=.86)
        pdf.savefig(fig); plt.close(fig)
        fig,axes = plt.subplots(1,2,figsize=(12,5))
        edges = np.array([0,.1,.25,.5,1,2.000001])
        labels = [f"{a:g}-{min(b,2):g}" for a,b in zip(edges[:-1],edges[1:])]
        for i,weights in enumerate(data["weights"]):
            means,errors,counts = [],[],[]
            for a,b in zip(edges[:-1],edges[1:]):
                use = data["active"] & (data["teacher_disagreement"]>=a) & (data["teacher_disagreement"]<b)
                use &= np.isfinite(data["delta_cosine"][i])
                counts.append(use.sum())
                means.append(np.mean(data["delta_cosine"][i,use]) if use.any() else np.nan)
                errors.append(np.mean(data["delta_fm_error"][i,use]) if use.any() else np.nan)
            axes[0].plot(labels,means,"o-",label=weights)
            axes[1].plot(labels,errors,"o-",label=weights)
        axes[0].set_ylabel("Mean delta cosine (positive = adoption)")
        axes[1].set_ylabel("Mean delta FM error (positive = worse)")
        for ax in axes:
            ax.set_xlabel("Teacher/FM disagreement bin"); ax.axhline(0,color="black",lw=.8)
            ax.grid(alpha=.2); ax.legend()
        fig.suptitle("Teacher disagreement versus adoption and supervised tradeoff\nPer-bin counts and parent/time breakdowns are in report.txt")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] Michael: saved {output}",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive"); p.add_argument("--out")
    args = p.parse_args()
    render_pdf(args.archive,args.out)
