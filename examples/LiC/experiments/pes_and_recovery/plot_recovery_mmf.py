"""Replot recovery comparisons without any model or UMA calls."""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    from compare_recovery_mmf import report_groups
    from plot_saddle_search import trajectory_page
    from recovery_basins import read_npz
except ModuleNotFoundError:
    from examples.LiC.experiments.pes_and_recovery.compare_recovery_mmf import report_groups
    from examples.LiC.rollout_eval.plot_saddle_search import trajectory_page
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import read_npz


def render_pdf(path, output=None):
    path = Path(path); data = read_npz(path)
    output = Path(output) if output else path.with_suffix(".pdf")
    methods = list(dict.fromkeys(data["method"].tolist()))
    groups = report_groups(data)
    with PdfPages(output) as pdf:
        fig,axes = plt.subplots(1,3,figsize=(15,5))
        names = ["recovery_train","seen_basin_new_sources","withheld_false_basins","generic_holdout","true_noise_test"]
        for ax,key in zip(axes,["true","false","physical"]):
            width = .8/len(methods)
            for j,method in enumerate(methods):
                values = []
                for group in names:
                    use = (data["method"] == method)&groups[group][data["source_index"]]
                    metric = data["physical_success"][use] if key == "physical" else data["outcome"][use] == (0 if key == "true" else 1)
                    values.append(100*metric.mean() if use.any() else np.nan)
                ax.bar(np.arange(len(names))+j*width,values,width,label=method)
            ax.set_xticks(np.arange(len(names))+.4-width/2,[s.replace("_","\n") for s in names],fontsize=8)
            ax.set_title(key+" (%)"); ax.set_ylim(0,105); ax.grid(axis="y",alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("endpoint-basin-disjoint recovery, autonomous 20-step sampling")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for group,use_sources in groups.items():
            if not use_sources.any(): continue
            use = use_sources[data["source_index"]]
            trajectory_page(pdf,data,methods,use,group,show_starts=False,path_limit=150)
    print(f"[plot] saved {output}",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive"); p.add_argument("--out")
    args = p.parse_args(); render_pdf(args.archive,args.out)
