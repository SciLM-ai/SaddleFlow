# Michael: post-hoc LiC kappa audit

No training, steering, saddle optimization, or new model rollouts. Frozen UMA
evaluates existing paths; the diagnostic asks whether isopotential curvature
distinguishes eventual false endpoints BEFORE entering their geometric regions.

## Run

```bash
sbatch examples/LiC/experiments/kappa_audit/run_kappa_audit.sh
```

Defaults reuse BOTH existing archives:

1. `lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz`:
   all 2,892 shell trajectories, the main lead-time population.
2. `lic_saddle_search_threeway_21107961/saddle_search.npz`:
   68 trajectories EACH for SaddleFlow, oracle MMF, and Sella. Two-flow is
   excluded by default; use `--methods all` to include every stored method.

The current default dry run finds 3,096 trajectories and 68,380 unique Li
geometries INCLUDING the map: approximately 479,428 PES geometry evaluations.
These are geometries, not GPU forward batches or an estimated wall time.
One central energy/force evaluation plus six FD force geometries computes each
3x3 Hessian. Displaced geometries use the existing batched backend.

Results: `/scratch/gqs7zr/saddleflow/lic_kappa_audit_JOBID/`.
Logs: `/scratch/gqs7zr/saddleflow/lic_kappa_audit_JOBID.out` and `.err`.
The batch script requests any supported A100/A40/B200/A6000/RTXPRO6000 GPU,
64 GB CPU memory and up to 36 hours. It runs the numerical audit BEFORE the
separate plotting command. The time request is a limit, not a measured estimate.

To audit one archive interactively:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LiC/experiments/kappa_audit/kappa_audit.py \
  --archives /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz \
  --out-dir /scratch/gqs7zr/saddleflow/lic_kappa_shells \
  --force-floor 0.001
```

Add `--dry-run` to inspect selection/cost without loading UMA or creating
outputs. `--limit-per-method 8 --grid-size 9 --sensitivity-points 4` is an
optional execution check, not the full scientific evaluation.

## Calculation and safeguards

The underlying UMA structures are full ASE atom arrays (127x3 in current LiC),
with original cell/PBC/charge/spin and frozen-carbon constraints preserved.
Stored paths are only the one mobile Li coordinate. Physics operates on its
FULL 3D free subspace, NOT just the xy slice.

At nonzero force, construct orthonormal C perpendicular to the free force:

```text
C.T C = I; C.T F = 0
nu1 = min eig(C.T H_free C)
kappa = -nu1 / ||F_free||
```

C is 3x2, H_free is the symmetrized 3x3 central-FD Hessian, and the projected
matrix is 2x2. Do not take the minimum eigenvalue of the full projected 3x3
`P H P`: that would include an artificial zero normal mode.

- `--force-floor 0.001` is a NUMERICAL validity floor, not a learned-force or
  stationarity threshold. At/below it, BOTH nu1 and kappa are undefined (NaN).
  Force, energy, full Hessian and its eigenvalues remain available.
- Default warning is nu1<0; `--curvature-tol` can demand nu1<-tol. Strict
  negative-curvature fractions/first steps are saved separately.
- Positive kappa can occur on a successful saddle approach. It is a candidate
  predictor, NOT a failure label or a saddle certificate.
- Independent final labels: true-certified, true-geometric-uncertified,
  false bridge, other failure. True geometry <0.10 A to ANY stored saddle;
  false region <0.30 A, with true precedence. Certification requires free
  fmax<=0.005 eV/A, lambda1<-0.01 and lambda2>0.01 eV/A^2.
- Lead is first false-region entry minus first warning. Already-inside starts
  are excluded from advance-warning evaluation. Missing warnings count as
  misses; masked-only pre-entry paths are reported as unobservable.
- Duplicate physics trace records are removed. FM steps, even identical
  consecutive positions, are retained. Real FM times are saved; other solvers
  have local update counts and normalized progress, not a common time clock.
- A seeded subset of 64 geometries also uses FD steps 0.001 and 0.004 A,
  compared with the primary 0.002 A. Report sign flips, warning flips, nu1
  differences, and Hessian asymmetry. No thresholds are fit to these results.

## Cache, results, and plotting

`physics_cache/` stores atomic, checksum-verified chunks of 128 geometries.
Re-running the SAME command/output directory resumes existing physics. On
SIGINT, SIGTERM or SIGUSR1, the audit finishes its current F/H point, saves the
finished prefix of the current chunk and exports a partial NPZ/report. Resuming
fills the remaining chunk without recomputing that prefix. The batch script
requests a USR1 warning 120 seconds before its time limit; an unscheduled
shutdown is not guaranteed to provide that warning.

The batch script also uses `--save-every-chunks 8`, publishing a plot-ready
`kappa_audit.npz` and `report.txt` every 1,024 points. Pass
`--save-every-chunks 1` for every-chunk snapshots, or `0` for end-only export.
Snapshot replacements are atomic: an interrupted write leaves the previous
snapshot readable. SIGKILL/power loss cannot be caught: at most the current
unsaved chunk (normally up to 128 points) needs recomputation. Completed cache
chunks can still be exported on CPU, even if no final snapshot was written.

`audit_inputs.npz` stores the original audit plan, geometry and settings BEFORE
loading UMA. After interruption, export whatever was saved and render using CPU
only (no source archives or UMA model loading required):

```bash
python examples/LiC/experiments/kappa_audit/kappa_audit.py \
  --out-dir /scratch/gqs7zr/saddleflow/lic_kappa_audit_JOBID \
  --export-cached --plot
```

Or just plot the latest periodic snapshot using `plot_kappa_audit.py` below.
Partial PDFs/reports explicitly show point and trajectory coverage. ONLY fully
audited trajectories enter endpoint/lead statistics: unfinished trajectories
are excluded, not classified as failures or treated as shorter rollouts. All
completed physics points are still saved, including points of unfinished paths.
The missing map background is blank until those grid points are evaluated;
trajectory plots remain available without it. Partial samples follow processing
order and may not yet include every method, so do not interpret them as a
representative full comparison.

For a job started BEFORE this update (no `audit_inputs.npz`), `--export-cached`
can reconstruct the plan from the original archives. Supply the SAME original
selection, grid, FD, seed and cache-chunk arguments alongside that flag. Defaults
are sufficient if the original job used defaults. It still makes zero UMA calls.

Input archive hashes, fixed structure, potential name, FD settings,
point selection, and charge/spin metadata are recorded. Change those settings
only with a new output directory. Analysis force floor, curvature tolerance,
and final-classification tolerances can change using the same cached physics.

`kappa_audit.npz` saves paths, per-point full force vectors/Hessians, energy,
tangent bases, projected eigenvalues, validity, nu1/kappa histories, final
classifications, lead metrics and FD sensitivity. `report.txt` and stdout
contain outcome tables, valid coverage, warning fractions, lead denominators,
and false-positive rates on observable true-certified paths. These are
descriptive statistics: shell starts sharing parents are correlated.

The PDF includes actual full-3D trajectory histories and a 31x31 fixed-height
xy map around false bridge catalog ID 0 (configurable). Black contours show
relative UMA energy; dark/light blue show negative/positive kappa; grey is
undefined; white is the valid nu1=0 contour. Successful and false-ending paths
are shown separately, without start dots. z excursions are displayed because
an xy slice need not represent what a 3D trajectory experiences.

Replot without model, GPU, or repeated physics:

```bash
python examples/LiC/experiments/kappa_audit/plot_kappa_audit.py /scratch/gqs7zr/saddleflow/lic_kappa_audit_JOBID/kappa_audit.npz
```

The standalone audit optionally accepts `--plot`, but ALWAYS saves numerical
results/report first. Changing analysis thresholds requires rerunning the
audit from cache and then replotting, not just editing a plotting option.
Batch exports: `LIC_KAPPA_OUT`, `LIC_KAPPA_SHELLS`, `LIC_KAPPA_SEARCH`.

Reference: Xiao, Wu & Henkelman, *Basin constrained kappa-dimer method for
saddle point finding*, JCP 141, 164111 (2014), Eq. 6,
https://www.henkelmanlab.org/pubs/xiao14_164111.pdf. Its connected-saddle problem
is different from LiC's high-force neural false endpoints; discrimination here
must be demonstrated, not assumed.
