"""Fixed TRAINING-recovery bridge probes, R2 versus matched R1.

Reuses the write-once oracle cache created during calibration. No new UMA
queries, no optimization, no rollouts. EMA is primary and raw is diagnostic.
"""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load_model

from pes_direction import alignment
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.utils import load_ema_weights

try:
    from compare_recovery_mmf import check_matching
    from diagnose_pes_adoption import compare_metrics, evaluate_velocities, make_report
    from force_guided_sampling import load_flow_model
    from plot_pes_adoption import render_pdf
    from recovery_basins import load_split
    from train_pes_directions import training_data
    from train_recovery_mmf import cache_probe_teachers
except ModuleNotFoundError:
    from examples.LiC.experiments.pes_and_recovery.compare_recovery_mmf import check_matching
    from examples.LiC.experiments.pes_and_recovery.diagnose_pes_adoption import compare_metrics, evaluate_velocities, make_report
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
    from examples.LiC.experiments.pes_and_recovery.plot_pes_adoption import render_pdf
    from examples.LiC.experiments.pes_and_recovery.recovery_basins import load_split
    from examples.LiC.experiments.pes_and_recovery.train_pes_directions import training_data
    from examples.LiC.experiments.pes_and_recovery.train_recovery_mmf import cache_probe_teachers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True); p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--no-plot",action="store_true")
    cli = p.parse_args(); root = Path(cli.run_dir)
    configs = check_matching(root)
    args = SimpleNamespace(**configs[1]); args.device = cli.device
    dataset,saddles = training_data(args)
    split,_ = load_split(root,dataset)
    if not (root/"recovery_teacher.npz").exists(): raise ValueError("calibration must create the frozen audit teacher cache first")
    calibration = json.loads((root/"calibration.json").read_text())
    cache = cache_probe_teachers(args,None,calibration["teacher_deadband"])
    if not np.all(split["partition"][cache["archive_row"]] == 0): raise ValueError("audit contains non-training recovery rows")
    free = cache["free_mask"]
    cache["teacher_disagreement"] = np.clip(1-np.asarray([alignment(v[free],b[free])
        for v,b in zip(cache["targets"],cache["teacher"])]),0,2)
    _,_,backbone,attention,head = load_flow_model(root/"R0/checkpoint_step_00000000",cli.device)
    model = FlowMatchingLoss(FlowMatchingConfig(),backbone,attention,head)
    cases = [("initializer",root/"R0/checkpoint_step_00000000","ema")]
    cases += [(f"E{i}_{w}",root/e/"checkpoint_final",w)
              for w in ["ema","raw"] for i,e in [(0,"R1"),(2,"R2")]]
    velocities = {}
    for name,path,weights in cases:
        print(f"[probe student] {name.replace('E0','R1').replace('E2','R2')} {weights}",flush=True)
        if weights == "ema": load_ema_weights(str(path),[backbone,attention,head],cli.device,use_ema=True)
        else: load_model(model,str(path/"model.safetensors"),strict=True,device=cli.device)
        velocities[name] = evaluate_velocities(model,cache,saddles,cli.device,cli.batch_size)
    results = compare_metrics(velocities,cache)
    results.update(comparison_names=np.asarray(["R1","R2"]),source_archive_row=cache["archive_row"],
        source_known_false=split["baseline_outcome"][cache["archive_row"]] == 2,
        source_baseline_basin=split["baseline_basin"][cache["archive_row"]],
        teacher_cache_checksum=cache["checksum"],models_json=json.dumps({n:dict(path=str(p),weights=w) for n,p,w in cases}))
    out = root/"recovery_adoption"; out.mkdir(exist_ok=True)
    np.savez_compressed(out/"adoption.npz",**results)
    report = make_report(results).replace("E0","R1").replace("E2","R2")
    report = report.replace("Probes are the original training-distribution audit, not exact samples used in optimizer updates.",
        "Fixed TRAINING recovery bridges; cached t=0,0.25,0.5,0.75. Not exact optimizer draws. No held-out teacher queries.")
    report += "\nRecovery-source categories (correlated times from each source; descriptive only):\n"
    report += "weights source_group        N_cos    mean dc   med dv_RMS   mean delta FM\n"
    for i,w in enumerate(results["weights"]):
        for name,selected in [("known_false",results["source_known_false"]),("generic_high_force",~results["source_known_false"])]:
            use = selected & results["active"] & np.isfinite(results["delta_cosine"][i])
            if use.any():
                report += (f"{w:7s} {name:19s} {use.sum():5d} {np.mean(results['delta_cosine'][i,use]):10.6f} "
                    f"{np.median(results['delta_velocity_rms'][i,use]):12.6g} {np.mean(results['delta_fm_error'][i,use]):14.6g}\n")
    (out/"report.txt").write_text(report); print(report,flush=True)
    if not cli.no_plot: render_pdf(out/"adoption.npz")
    for trajectory in dataset._traj_cache.values(): trajectory.close()


if __name__ == "__main__": main()
