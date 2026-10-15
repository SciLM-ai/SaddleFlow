# Matched 100-step oracle-MMF mobility experiment

Michael, this experiment learns a translation policy, not a new flow-matching
velocity field. It does not initialize from the TS-denoise checkpoint. UMA stays
pretrained/frozen; only two global parameters or a small eight-input MLP are
trained. The TS-denoise checkpoint remains an evaluation comparator.

| Method | Translation | Learned parameters |
|---|---|---|
| A: raw MMF | `F_perp - F_parallel` | none |
| B: global MMF | `a F_perp - b F_parallel` | two global scalars |
| C: learned MMF | `a(z) F_perp - b(z) F_parallel` | small MLP |

All three use the same numerical minimum mode, base mobility `h=0.2 A^2/eV`,
trust radius `0.1 A` per free atom, FD displacement `0.002 A`, and maximum 100
translations. Both learned policies start at exactly `a=b=1`; bounded positive
coefficients default to `[0.05,5]`. No unrestricted velocity residual is added.

Physical early stopping requires **both** `fmax <= 0.005 eV/A` and
`lambda1 < -0.01`, `lambda2 > 0.01 eV/A^2`, matching the existing benchmark.
Stopping is not based on target distance. The iteration coordinate is always
`k/100`; the 20/40/60/80/100 snapshots do not change or rescale the policy clock.
An early-stopped endpoint is retained for subsequent anytime budgets.

## Train and compare

The initial recipe is deliberately bounded: **200 optimizer updates per policy**,
batch size 1, learning rate 0.003. Only the 12 training saddles are read. Each
visit draws fresh Li noise with sigma 0.25 A. B and C receive the same sequence
of parent/noise draws. R/P frames, held-out saddles, false bridge starts, shell
failures, force losses, and curvature losses do not enter training.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/experiments/mmf_mobility/train_mmf_mobility.py \
  --out-dir /scratch/gqs7zr/saddleflow/lic_mmf_mobility_k100 \
  --max-steps 100 --num-updates 200 --batch-size 1
```

The sole objective is actual-rollout free-coordinate MIC endpoint MSE to the
training parent. This is NOT the rectified-flow algebraic endpoint approximation.
Physical stopping is discrete and its derivative is omitted. Endpoint loss
alone does not explicitly reward faster convergence; anytime tables measure it.

Reuse the previous **completed** benchmark's exact 68 sources, plus its Sella
and ordinary SaddleFlow final results. A/B/C are newly evaluated, not substituted
from old oracle trajectories:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/experiments/mmf_mobility/compare_mmf_mobility.py \
  --run-dir /scratch/gqs7zr/saddleflow/lic_mmf_mobility_k100 \
  --out-dir /scratch/gqs7zr/saddleflow/lic_mmf_mobility_k100/comparison \
  --baseline-archive /scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_21107961/saddle_search.npz
```

Without `--baseline-archive`, the runner generates the same default mixed
population with seed 1 (false exact/noisy, training-saddle noise, and shells).
It includes only A/B/C rows. `--starts-npz` accepts the existing shell archive;
its t>0 starts are excluded. The previous fixed-force query budgets are distinct
from this experiment's **translation-step** budgets.

The baseline reuse checks geometry, constraint masks, true/false catalogs, UMA
model/task, and force/eigenvalue/FD tolerances. Historical Sella/FM timings are
clearly identified; they are not newly measured timings. The imported rows do
not invent MMF-step anytime results.

## Batch job and resume

```bash
sbatch examples/LiC/experiments/mmf_mobility/run_mmf_mobility.sh
```

This trains both policies and then compares A/B/C on the previous 68 starts.
It accepts one A100/A40/B200/A6000/RTXPRO6000 GPU and requests 24 hours, 64 GB
host memory, and 4 CPUs. Slurm chooses the GPU; its assignment is not overwritten.
Default results go to `/scratch/gqs7zr/saddleflow/lic_mmf_mobility_<JOBID>/`.
Stdout, including progress and final tables, goes to
`/scratch/gqs7zr/saddleflow/lic_mmf_mobility_<JOBID>.out`; stderr goes to `.err`.

Override `LIC_MMF_OUT_DIR`, `LIC_MMF_PYTHON`, or `LIC_MMF_BASELINE` if needed.
An empty `LIC_MMF_BASELINE` requests fresh mixed starts. Extra launcher arguments
are passed to the trainer, for example `--num-updates 400`.

Every 25 optimizer updates (configurable), each policy saves a tiny
`global/checkpoint_latest.pt` or `state/checkpoint_latest.pt`. The file contains
policy weights, Adam state, source RNG, configuration, and training history;
**UMA weights are not copied**. Completed training also saves `checkpoint_final.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/experiments/mmf_mobility/train_mmf_mobility.py \
  --out-dir /scratch/gqs7zr/saddleflow/lic_mmf_mobility_k100 \
  --max-steps 100 --num-updates 200 --batch-size 1 --resume
```

Resume uses the saved source RNG and optimizer. Interrupted updates after the
last checkpoint are repeated. Increase `--num-updates` to extend training. Use
a fresh comparison output directory; comparison does not implement resume.

## Outputs and replotting

Training: `config.json`, `training_report.txt`, per-policy checkpoints and
`training_log.json`. Evaluation: `saddle_search.npz`, `report.txt`, and
`saddle_search.pdf`. The numerical archive/report are updated after every trial,
before plotting. PDF failures cannot discard sampling results.

```bash
python examples/LiC/rollout_eval/plot_saddle_search.py \
  /scratch/gqs7zr/saddleflow/lic_mmf_mobility_k100/comparison/saddle_search.npz
```

The PDF preserves the C-C network, all stored saddle stars, light-blue paths,
outcome-colored endpoints, no start dots by default, and cumulative shell pages.
New summary curves use 20/40/60/80/100 steps. A separate page shows a/b histories.
NPZ saves per-step mobilities, forces, modes, eigenvalues, clipping, coordinates,
stop status, query counts, and all final/anytime metrics. Geometric success is a
hit on **any** catalogued saddle, independently of physical certification.

## Gradient and cost limitations

The forward force is UMA's actual force. Backward uses the measured local
`dF/dx = -H` via `F_observed - H @ (x-x.detach())`. Mode, Hessian, and eigenvalues
are detached. This avoids retaining UMA graphs and differentiating eigenvectors,
but it is an approximate gradient when the minimum mode/eigenvalues vary.
Force-norm features retain this local coordinate derivative. Trust clipping and
the coordinate recurrence remain graph-connected; MIC image shifts are constants.

There are at most 100 translations and 101 evaluated states, including the
initial and terminal checks: about **707 UMA geometry evaluations per source**
before independent terminal verification. Six FD probes are batched. Training
examples and evaluation sources run serially in this first implementation. The
100-step budget is not comparable in cost to 100 learned FM velocity calls.

The generic rollout supports arbitrary small free-coordinate spaces with the
existing FixAtoms/unconstrained mask. This dense-Hessian prototype is not a
scalable full-system learned implementation; a future HVP backend is separate.

CPU tests (no model download):

```bash
python -m unittest discover -s tests -p 'test_*mobility.py'
python -m unittest discover -s tests -p test_saddle_search.py
```
