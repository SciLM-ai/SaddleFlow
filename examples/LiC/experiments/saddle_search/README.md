# LiC physics saddle-search benchmark

The separate matched 100-step **learned translation** experiment is
documented in [../mmf_mobility/README.md](../mmf_mobility/README.md). It compares raw MMF, learned
global mobilities, and a state-dependent policy without changing this original
inference-only benchmark or the existing FM trainer.

This is an inference-only research benchmark, not another training objective.
It compares three searches on identical ASE geometries:

* **UMA + Sella:** an established index-one saddle optimizer.
* **Oracle MMF:** recompute the numerical lowest free-space Hessian eigenvector,
  then translate along `F - 2 * dot(q, F) * q`.
* **Two-flow:** keep a seeded random unit orientation; rotate it with
  `q <- normalize(q - eta_q * (Hq - dot(q, Hq) * q))`, then translate with the
  same reflected force. No stored saddle eigenmode is used.

`--ckpt` adds the ordinary unconditioned SaddleFlow sampler as a fourth row;
there is **no inference-time force guidance** in that row.

## Setup and first run

Use the SaddleFlow environment. Sella is an optional existing project dependency:

```bash
CC=gcc CXX=g++ pip install 'saddleflow[eval]'
```

Run a tiny smoke check before the full experiment. This tests execution, not
convergence: two starts and eight physics iterations are not a scientific result.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/rollout_eval/compare_saddle_search.py \
  --ckpt /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final \
  --out-dir /scratch/gqs7zr/saddleflow/saddle_search_smoke \
  --starts false --samples-per-site 0 --limit-starts 2 \
  --restarts 1 --max-steps 8 --max-force-calls 80 \
  --log-every-steps 1
```

First mechanistic experiment: all ten exact false bridges.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/rollout_eval/compare_saddle_search.py \
  --ckpt /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final \
  --out-dir /scratch/gqs7zr/saddleflow/saddle_search_false_bridges \
  --starts false --samples-per-site 0 --restarts 2
```

### Slurm batch submission

Use the batch launcher to run the mixed benchmark without keeping an
interactive session open:

```bash
sbatch examples/LiC/experiments/saddle_search/run_saddle_search.sh
```

It requests one GPU in the `gpu` partition, accepting any node with an `a100`,
`a40`, `b200`, `a6000`, or `rtxpro6000` feature; Slurm chooses the allocation.
Defaults are 4 CPUs, 64 GB host memory, and a 12-hour wall-time limit. It uses
the existing `saddleflow` Python interpreter and does not override Slurm's GPU
assignment. Add an `--account` option to `sbatch` if your allocation requires it.

The launcher uses the original TS-denoise checkpoint, `--starts mixed` and
`--restarts 2`. Results go to a fresh directory for each job:
`/scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_<JOBID>/`.
Stdout and stderr go to
`/scratch/gqs7zr/saddleflow/lic_saddle_search_<JOBID>.out` and `.err`.
The parent scratch directory must exist before submission, since Slurm opens
the logs before executing the script. This directory already holds the model
checkpoints in the current setup.

Watch progress with the job ID printed by `sbatch`:

```bash
tail -f /scratch/gqs7zr/saddleflow/lic_saddle_search_<JOBID>.out
```

Extra arguments after the script name are forwarded to the Python benchmark:

```bash
sbatch examples/LiC/experiments/saddle_search/run_saddle_search.sh \
  --limit-starts 2 --restarts 1 --max-steps 8 --max-force-calls 80 \
  --log-every-steps 1
```

Override the checkpoint or output directory through `LIC_SEARCH_CKPT` and
`LIC_SEARCH_OUT_DIR`; `LIC_SEARCH_PYTHON` and `SADDLEFLOW_REPO` override the
interpreter and repository paths. This launches a new experiment, not a resume
of an existing interactive run. Previously completed output is never deleted.

Broader comparison: ten exact false bridges, ten noisy false neighborhoods,
twelve noisy training saddles, and 36 shell starts (12 parents x 3 radii).
Defaults use XY shells to map the sheet; `--shell-dimensions xyz` uses a 3D sphere.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/rollout_eval/compare_saddle_search.py \
  --ckpt /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final \
  --out-dir /scratch/gqs7zr/saddleflow/saddle_search_mixed \
  --starts mixed --restarts 2
```

For a regular plane of starts use `--starts grid --grid 8`. For more directions
per shell use `--directions-per-radius 8`. A full 64x64 physical search is a very
large calculation; it is not comparable in cost to one learned-flow grid throw.
Each oracle translation costs roughly **seven geometry force evaluations** in
LiC (one center plus six finite-difference probes), versus one learned velocity
evaluation in SaddleFlow. Defaults permit 100 physics translations or 800 PES
queries per trial, whichever runs out first. Physics searches are serial across
starts; each FD Hessian's six probes are batched through UMA.

To reuse existing t=0 shell starts, add:

```bash
--starts-npz /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz \
--limit-starts 48
```

Source coordinates are read from `starts`, never from previous endpoints. An
archive's nonzero-time sources are excluded: physical search iteration is not FM
time. The checkpoint loader deliberately requires an unconditioned LiC model.

## Outputs

Each completed trial updates `saddle_search.npz` and `report.txt`; final tables
are also printed to stdout, so appear in a normal SLURM `.out` file. NPZ is saved
**before** rendering `saddle_search.pdf`. Output directories containing an
existing archive are not overwritten; partial runs are inspectable but this
first research runner does not implement resume.

Replot without repeating any inference:

```bash
python examples/LiC/rollout_eval/plot_saddle_search.py \
  /scratch/gqs7zr/saddleflow/saddle_search_false_bridges/saddle_search.npz
```

The PDF shows the C-C network, all 179 stored saddles (black stars, green training
and orange held-out rings), red false-bridge crosses, light-blue trajectories,
and endpoints colored by catalog outcome. Blue outline circles indicate a
separate force/index-one certificate. There are no start dots unless explicitly
requested with `--show-starts`. Shell pages are cumulative in increasing radius.
At most 300 paths/panel are drawn by default; **all** endpoint markers and all
statistics are retained. Curves are split at periodic cell crossings.

The pickle-free NPZ contains source populations, coordinates, radii, reference
labels, paths, orientations, force histories, lowest two eigenvalues where
computed, displacement RMS, force decomposition, actual query counts, solver
statuses, final metrics, budget snapshots and configuration. HVP two-flow does
not diagonalize at every step, so those trace eigenvalue slots are NaN; final
and budget-checkpoint eigenvalues are always evaluated independently.

## Scientific interpretation and budgets

The report deliberately separates these tests:

* **Catalog true hit:** distance below 0.10 A to any stored saddle.
* **False-region occupancy:** not a true hit and within 0.30 A of a known false
  bridge. Occupancy at a query limit is not proof of a fixed point.
* **Stationary:** maximum free-atom force at most `--fmax` (default 0.005 eV/A).
* **Index-one certified:** `lambda1 < -tol` and `lambda2 > tol`, with
  `--eigenvalue-tol 0.01` eV/A^2 by default.
* **Physical success:** stationary AND index-one certified.

These are UMA conditions, not independent DFT verification. A low-force minimum
does not pass. An endpoint outside the finite label catalog can still be a UMA
saddle. Full unconstrained systems can have zero/soft modes: the strict lambda2
test can refuse a certificate even if exactly one negative mode exists. Inspect
the eigenvalues before interpreting such a failure as a wrong saddle.

The dense 'oracle' Hessian is **central finite difference**, not an analytic
exact Hessian. Repeat with `--fd-eps 0.001` and `0.004` to check curvature
sensitivity, especially with float32 UMA. The default 0.002 A and fixed
mobilities are starting settings, not optimized physical constants. Translation
and orientation step sizes have units A^2/eV; the maximum translation is
per-free-atom RMS in A. There is no energy-descent line search, since the method
must climb in one direction. Small fixed steps can require many iterations;
step-size/rotation-count sensitivity is part of assessing the baseline.

Anytime tables use the last available translated state/accepted Sella state at
search query budgets 10,20,40,80. FD probes and orientation updates count;
cached reads do not. Energy+force at one geometry counts as one PES query,
even when six geometries execute in one GPU batch. Verification of each budget
checkpoint is extra cost and reported separately. Stops can occur before all
budgets, in which case the final state remains available.

All two-flow restarts are independent trials in the probability tables, with
total cost also reported. The runner never selects the 'best' restart by label
distance. SaddleFlow has zero **separate pretrained-PES search** queries but
uses K learned backbone/velocity evaluations; it is not computationally free.
Its final K-step state is shown for every PES-query budget, and its wall time
is batched amortized time per source, not single-source latency. Model loading
and independent verification time are excluded from search wall time. The
runner warms up scalar/batched UMA and the learned sampler before timing,
printing setup-only query counts separately.

## Generic implementation

Reusable code is in `saddleflow/classical_saddle_search/`. It reads ASE `FixAtoms`, preserves
fixed coordinates/metadata/cell/PBC, and works with any number of free atoms and
any ASE energy/force calculator. Unsupported constraint types fail explicitly.
No Li atom index or reference saddle catalog appears in the solver modules.
The existing `examples/MaterialsSaddles/sella_eval.py` uses the same corrected
constraint-preserving adapter; its `nfc` now counts PES calls, not optimizer
steps (`nsteps` is separate).

`MinimumModeEstimator(backend='auto')` forms a dense FD Hessian for free dimension
at most 12. Larger spaces use SciPy Lanczos with central FD HVPs. Force
`--mode-backend hvp` to test Lanczos in LiC; `--orientation-backend hvp` tests the
actual scalable orientation update without constructing a Hessian. Neither
backend needs second-order coordinate autograd.

CPU mathematical and archive/plot checks:

```bash
python -m unittest discover -s tests -p test_saddle_search.py -v
```

The real Sella integration check runs when the optional Sella dependency is
installed; otherwise it is explicitly skipped. Analytic-PES tests are numerical
implementation checks, not claims of LiC success with UMA.
