# Michael: matched PES-direction fine-tuning

The question is whether a frozen physical teacher can reshape an already trained
FM field. This is not minimum-action optimization, a learned MMF solver, or a
nearest-saddle guarantee. Only the auxiliary signal changes:

| Run | Objective |
| --- | --- |
| E0 | ordinary FM continuation |
| E1 | FM + lambda × masked smooth cosine to ordinary UMA force |
| E2 | FM + the same lambda × masked smooth cosine to minimum-mode-reflected force |

All three load the same initial **EMA weights as trainable parameters**, reset
Adam and EMA, and take exactly 2,000 updates. Batch size is 4. Head/FiLM LR is
1e-4; the previously unfrozen backbone blocks use 1e-5. The LR schedule is
100-step warmup + cosine with a 1% floor; EMA decay remains 0.999. Early EMA
checkpoints therefore deliberately smooth strongly toward the initializer.

Training uses only the 12 training saddles (the usual 24 direction records),
fresh sigma=0.25 A Gaussian noise on free atoms, and the existing straight FM
bridge. One uniformly selected structure receives the teacher every update.
There is no M/B multiplier, time taper, force gate reweighting, algebraic endpoint
penalty, recovery data, or false/test-saddle input to training/calibration.

## Run

From the repository root, Michael:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_pes_directions.sh
```

No job is submitted just by creating these files. The batch script requests one
A100/A40/B200/A6000/RTX Pro 6000 and runs calibration, E0/E1/E2 sequentially,
then evaluation of **all five predeclared checkpoints**.

```text
/scratch/gqs7zr/saddleflow/lic_pes_directions_<JOBID>/
    calibration/  alignment_audit.npz, alignment_audit.txt,
                  calibration.json, calibration_report.txt
    E0/, E1/, E2/ config.json, training_steps.jsonl,
                  checkpoint_step_00000000/
                  checkpoint_step_00000250/
                  checkpoint_step_00000500/
                  checkpoint_step_00001000/
                  checkpoint_final/              # update 2000
    comparison/   saddle_search.npz, report.txt, saddle_search.pdf
```

Slurm logs:

```text
/scratch/gqs7zr/saddleflow/lic_pes_directions_<JOBID>.out
/scratch/gqs7zr/saddleflow/lic_pes_directions_<JOBID>.err
```

Override `LIC_PES_OUT`, `LIC_PES_INIT`, or `LIC_PES_STARTS` before `sbatch` if
needed. Each full checkpoint contains raw weights, EMA, optimizer/scheduler and
RNG state; allow roughly 60–75 GB for the three runs. Teachers are not saved.

Standalone stages (use the SAME options for calibration and training):

```bash
python examples/LiC/experiments/pes_and_recovery/train_pes_directions.py \
  --calibrate-only --calibration-dir /scratch/gqs7zr/saddleflow/lic_pes/calibration

python examples/LiC/experiments/pes_and_recovery/train_pes_directions.py \
  --experiment E2 --calibration-dir /scratch/gqs7zr/saddleflow/lic_pes/calibration \
  --output-dir /scratch/gqs7zr/saddleflow/lic_pes/E2
```

Run that training command for E0 and E1 as well. Initialization defaults to
`/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final`.

Resume an interrupted run with identical settings and output directory:

```bash
python examples/LiC/experiments/pes_and_recovery/train_pes_directions.py \
  --experiment E2 --calibration-dir /scratch/gqs7zr/saddleflow/lic_pes/calibration \
  --output-dir /scratch/gqs7zr/saddleflow/lic_pes/E2 \
  --resume-from /scratch/gqs7zr/saddleflow/lic_pes/E2/checkpoint_step_00001000
```

This restores the **raw** continuation state, not its EMA as a new initializer.
Log rows after the selected checkpoint are removed and replayed. This custom
research trainer is single-GPU; do not launch multiple ranks into one output.

## Calibration and audit

The deadband is explicitly the **maximum free-atom RMS UMA force on the 12
training saddle labels**, not p99 and not a statistic of the 179-label test union.
The teacher/mask are detached. Only states strictly above that envelope receive
the directional loss. The prediction's norm is never a masking criterion.
Smooth norm floors are 1e-4 A per FM time and 1e-6 eV/A in atom-vector RMS units.

The training-only audit samples 16 Gaussian bridge states per saddle (192
states), reports time-binned force/MMF cosines, lower tails, opposed fractions,
active fraction, force RMS, minimum curvature and eigengap, and saves raw values.
The force-to-FM RMS ratio has physical units; it is diagnostic, not a scale used
by the direction loss. Reflection preserves force magnitude.

Sixteen fixed calibration batches measure actual gradients over **all trainable
student parameters** for batch FM and the one-example E1/E2 priors. The common
weight is

`lambda = 0.1 * mean(||gFM||) / (0.5*(mean(||gF||)+mean(||gMMF||)))`.

The mean includes masked zero priors. Both resulting ratios are reported; they
need not individually equal 0.1. The weight is never retuned per experiment.
Zero calibration gradients raise an error rather than silently selecting lambda.

Separate generators control shuffling, Gaussian noise/time, and physics-example
selection. Every training log row stores a bridge-batch SHA256 plus selection;
the comparison refuses unmatched streams. Teacher RNG is isolated from student
RNG. Resume restores generator and optimizer states.

## Physics and inference

Generic teacher/loss functions live in `examples/LiC/experiments/pes_and_recovery/pes_direction.py`.
They use the existing `PESBackend`/`FreeDOFs` and `MinimumModeEstimator`: no bond,
hop, normal, Li index or target saddle enters the teacher. `FixAtoms` and fully
free systems are supported. Unsupported constraints are rejected. Small free
spaces use a dense FD Hessian; larger ones use Lanczos with finite-difference
HVPs. This is a numerical mode of UMA, not an exact DFT Hessian.

E0 has no separate UMA teacher. E1 uses approximately one force geometry per
update; E2 uses seven for this 3D LiC free space. Conservative UMA forces still
require internal position autograd, but their outputs are detached: no HVP
backward into the student, and no teacher parameter training.

The student **executes its trainable UMA backbone** at every inference step. It
does not call the frozen teacher, energy/force head or Hessian/minimum-mode
machinery. The accurate claim is **no online PES-oracle queries at inference**,
not no UMA-derived network computation.

## Matched evaluation and plots

```bash
python examples/LiC/experiments/pes_and_recovery/compare_pes_directions.py \
  --run-dir /scratch/gqs7zr/saddleflow/lic_pes \
  --out-dir /scratch/gqs7zr/saddleflow/lic_pes/comparison

python examples/LiC/experiments/pes_and_recovery/plot_pes_directions.py \
  /scratch/gqs7zr/saddleflow/lic_pes/comparison/saddle_search.npz
```

The comparison defaults to the **same 68 source coordinates** from the completed
`lic_saddle_search_threeway_21107961/saddle_search.npz`: false_exact, false_noise,
true_noise and shells. Only starts are reused; all neural predictions are new.
Training-parent noisy starts are diagnostic probes, not a held-out-parent test.
UMA force/index verification runs **after** autonomous 20-step Euler sampling.
The report prints catalog hits, false occupancy, index-1 rate, physical
certification, fmax and the two lowest eigenvalues, overall and per population.
Numerical archives/report are saved after each completed checkpoint, before PDF
generation. PDF pages compare E0/E1/E2 at increasing fine-tuning updates, draw
the carbon bond network, light-blue trajectories and outcome endpoints, and
omit starting dots. Use `--updates 2000` for final-only evaluation, or `--no-plot`
to render later without resampling.

Direction supervision does not preserve the original FM optimum or guarantee
physical fixed points. A null result alone does not establish that support
mismatch is the cause; the next separate experiment would supervise rollout
states. No E3 or magnitude-distillation ablation is implemented here.

## Michael: fixed-probe adoption diagnostic (no retraining)

Non-interactive launch for the existing run, Michael:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_pes_adoption.sh
```

Logs: `/scratch/gqs7zr/saddleflow/lic_pes_adoption_<JOBID>.out` and `.err`.
Or run directly on an allocated GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/experiments/pes_and_recovery/diagnose_pes_adoption.py \
  --run-dir /scratch/gqs7zr/saddleflow/lic_pes_directions_21242185
```

Results default to `<run-dir>/fixed_probe_adoption/`: `probes_teacher.npz`,
`adoption.npz`, `report.txt`, and `adoption.pdf`. The immutable teacher cache is
saved first, then numerical model results/report, then PDF. Subsequent runs
reuse teachers and make **zero additional UMA queries**. Use `--cache-only`
to prepare just the cache, or `--no-plot` to render later:

```bash
python examples/LiC/experiments/pes_and_recovery/plot_pes_adoption.py \
  /scratch/gqs7zr/saddleflow/lic_pes_directions_21242185/fixed_probe_adoption/adoption.npz
```

It reconstructs the original 192 training-distribution audit probes with the
original CPU generator (seed+4000), checks the training-file SHA256, and requires
exact agreement of parent indices and times. Any originally saved coordinate
hashes are verified. Recomputed force RMS, teacher/FM cosines, minimum eigenvalue,
eigengap and active mask must agree with the audit; numerical comparisons allow
5e-4 absolute/relative FD/GPU differences and fail before writing the cache.
**The original audit did not save coordinate hashes or positions.** That limit
is recorded honestly. Deterministic replay plus physics checks reconstructs it;
the new cache stores per-probe hashes and a payload checksum going forward.
These fixed probes are from training support, not exact optimizer-update samples.

The cache contains `(x_t,t,v_FM*,F,q,b_MMF)` for every probe, free mask and
provenance. It is write-once, with no force-overwrite option. Preparing it costs
about 1,344 UMA geometry evaluations once. No ODE rollout, optimizer, new loss,
weight calibration, E1 comparison or E3 is added.

Initializer, E0-final and E2-final are evaluated at identical coordinates and
times. EMA is primary; raw is diagnostic. Raw loading restores the **entire**
student including backbone and time-FiLM, not the older head-only raw loader.
Paired quantities are E0/E2 teacher cosines, delta cosine, RMS(E2-E0) velocity
change, E0/E2 FM errors and their difference. Velocity magnitudes and shifts
from initialization are also saved. Reports cover time bins, parents, and
teacher/FM disagreement bins, with descriptive Spearman correlation. Positive
delta cosine indicates adoption; positive FM-error change is a tradeoff.
Zero-prediction cosines are explicitly undefined. No automatic significance or
pass/fail threshold is asserted. PDF plots contrast disagreement/adoption,
velocity-change/adoption, and disagreement-bin FM tradeoffs. Student forwards
still execute the UMA backbone; cached-teacher reuse calls no PES oracle.
