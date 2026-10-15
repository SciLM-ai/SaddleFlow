"""Michael: replot E0/E1/E2 archives without models, UMA, or repeated sampling."""

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
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
    from plot_saddle_search import trajectory_page
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.plot_saddle_search import trajectory_page


def render_pdf(archive_path, output_path=None):
    archive_path = Path(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    output_path = Path(output_path) if output_path else archive_path.with_suffix(".pdf")
    methods = list(dict.fromkeys(data["method"].tolist()))
    updates = sorted({int(m.split("_s")[1]) for m in methods})
    population = data["source_population"][data["source_index"]]
    print(f"[plot] Michael: {output_path}", flush=True)
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for experiment in ["E0", "E1", "E2"]:
            present = [u for u in updates if f"{experiment}_s{u:04d}" in methods]
            for axis, key in zip(axes, ["true", "false", "physical"]):
                values = []
                for u in present:
                    use = data["method"] == f"{experiment}_s{u:04d}"
                    metric = (data["physical_success"][use] if key == "physical"
                              else data["outcome"][use] == (0 if key == "true" else 1))
                    values.append(100*metric.mean())
                axis.plot(present, values, "o-", label=experiment)
        for axis, title in zip(axes, ["Any catalogued saddle", "False-region occupancy", "Force + index-1 certificate"]):
            axis.set_title(title); axis.set_xlabel("Fine-tuning updates")
            axis.set_ylabel("Trajectories (%)"); axis.set_ylim(-2, 102)
            axis.grid(alpha=.2); axis.legend()
        fig.suptitle("Matched E0/E1/E2 continuation; autonomous 20-step Euler sampling")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for update in updates:
            selected = [m for m in methods if m.endswith(f"_s{update:04d}")]
            for group in [None] + np.unique(population).tolist():
                use = np.ones(len(population), dtype=bool) if group is None else population == group
                title = f"Fine-tuning update {update}: " + ("all identical starts" if group is None else group)
                trajectory_page(pdf, data, selected, use, title, show_starts=False, path_limit=300)
    print(f"[plot] Michael: saved {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--out")
    args = parser.parse_args()
    render_pdf(args.archive, args.out)
