"""CPU-only learnability curves and trajectories from saved NPZ."""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    from plot_saddle_search import draw_endpoint_paths
    from recovery_basins import read_npz
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.plot_saddle_search import draw_endpoint_paths
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import read_npz


def render_pdf(path, output=None):
    path=Path(path); data=read_npz(path)
    output=Path(output) if output else path.with_suffix(".pdf")
    recovery=data["population"]=="recovery_train"
    mmf=json.loads(str(data["config_json"])).get("mmf",False)
    with PdfPages(output) as pdf:
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for weights,style in [("raw","--"),("ema","-")]:
            rows=np.flatnonzero(data["weights"]==weights); steps=data["update"][rows]
            errors=data["probe_error"][rows]
            axes[0].plot(steps,np.sqrt(errors[:,data["probe_t"]==0].mean(axis=1)),style,label=weights+" source t=0")
            axes[0].plot(steps,np.sqrt(errors.mean(axis=1)),style,alpha=.6,label=weights+" bridge")
            axes[1].plot(steps,100*(data["outcome"][rows][:,recovery]==0).mean(axis=1),style,label=weights)
            for group in ["true_noise_train","true_noise_test"]:
                use=data["population"]==group
                axes[2].plot(steps,100*(data["outcome"][rows][:,use]==0).mean(axis=1),style,label=weights+" "+group)
        for ax in axes: ax.set_xlabel("Optimizer update"); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0].set_ylabel("Free-atom velocity vector RMSE (A/FM-time)")
        axes[0].set_title("Fixed recovery probes")
        axes[1].set_title("Seen recovery sources: any-true hit (%)"); axes[1].set_ylim(-2,102)
        axes[2].set_title("Fixed clean-noise regression check (%)"); axes[2].set_ylim(-2,102)
        fig.suptitle("" + ("recovery FM + detached MMF directional teacher" if mmf else "plain-FM learnability control")
                     + "; no force/index certification")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        if "probe_teacher_cosine" in data:
            fig,ax=plt.subplots(figsize=(9,5))
            active=data["probe_teacher_active"]
            for weights,style in [("raw","--"),("ema","-")]:
                rows=np.flatnonzero(data["weights"]==weights)
                values=data["probe_teacher_cosine"][rows][:,active]
                means=[np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in values]
                ax.plot(data["update"][rows],means,style,label=weights)
            ax.set_xlabel("Optimizer update"); ax.set_ylabel("Mean student/MMF cosine on active fixed TRAIN probes")
            ax.set_ylim(-1.05,1.05); ax.grid(alpha=.2); ax.legend()
            ax.set_title("teacher adoption; cached oracle, no extra PES evaluation")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # EMA is primary; raw remains available in curves/NPZ/report.
        rows=np.flatnonzero(data["weights"]=="ema")
        for group in dict.fromkeys(data["population"]):
            use=np.flatnonzero(data["population"]==group)
            fig,axes=plt.subplots(1,2,figsize=(14,7))
            for ax,row in zip(axes,[rows[0],rows[-1]]):
                draw_endpoint_paths(ax,data["path"][row],data["outcome"][row],data,use,path_limit=150)
                outcome=data["outcome"][row,use]
                ax.set_title(f"EMA update {int(data['update'][row])}; N={len(use)}\n"
                             f"any true={100*np.mean(outcome==0):.1f}%; false={100*np.mean(outcome==1):.1f}%")
            fig.suptitle(f"{group}; baseline versus latest saved checkpoint\n"
                         "Blue lines: rollouts; green/red/purple endpoints: true/false/other. No start dots.")
            fig.tight_layout(rect=(0,0,1,.93)); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] saved {output}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive"); p.add_argument("--out")
    args=p.parse_args(); render_pdf(args.archive,args.out)
