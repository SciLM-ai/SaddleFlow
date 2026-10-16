# Failure-mode analysis: false landing points of the unconditional model

How often does the released unconditional model (`SciLM/SaddleFlow`) land somewhere that is
not a first-order saddle, and does the UMA force at the landing point flag those cases? If one force
call separates them, false landing points can be filtered out cheaply after sampling.

## The experiment

1. **Starts.** Every reactant and product frame of the mp20bat test split is a minimum. Duplicates
   (every atom within 0.05 Å) are merged: 3,062 endpoints give 3,035 unique minima. A random 500
   (seed 0) are used by default. The model never sees a product or a saddle.
2. **Rollout.** `SaddleFlow.propose(minimum, n=1)`: the minimum plus the training noise (0.3 Å),
   then 5 passes of 4 Euler steps, the released defaults. One landing point per minimum.
3. **Label.** `sella_eval.py --check-index` runs Sella from each landing point with the paper's
   protocol (UMA-S-1.2, fmax 0.01 eV/Å, ≤ 300 steps, `delta0` 0.1). A landing point is **real** if
   Sella converges to an index-1 saddle within a maxd cutoff of it; otherwise it is **false**:
   `moved_far`, `minimum` (no negative mode), `higher_order` (two or more), `not_converged` or
   `error`. A low force cannot flag the stationary ones (`minimum`, `higher_order`); only the index
   check finds those.
4. **Filter test.** `fmax_start`, the UMA force at the landing point before Sella moves it, is compared
   between real and false points.

Why Sella and not the dataset: each minimum has one stored NEB saddle but usually several real ones,
so "not near the stored saddle" would mix false points with other real saddles. About 31% of mp20bat
labels are also not first order. And labelling by force would make the filter test circular.

## Run

```bash
sbatch examples/MaterialsSaddles/failure_mode_analysis/run_failure_mode_analysis.sh
sbatch --export=ALL,N_STARTS=0 examples/MaterialsSaddles/failure_mode_analysis/run_failure_mode_analysis.sh  # all 3,035
```

Overrides: `DATA` (another subset's split file), `TASK` (`omat`, `oc20`, `oc22`), `SEED`, `OUT`.
Output goes to `/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_<JOBID>/`; logs to
`/scratch/gqs7zr/saddleflow/failure_mode_analysis_<JOBID>.{out,err}`.

To re-analyse without rerunning anything (CPU, seconds):

```bash
python examples/MaterialsSaddles/failure_mode_analysis/analyze_failure_modes.py <OUT> --maxd-cutoffs 0.1,0.2,0.5
```

## Outputs

| file | contents |
|---|---|
| `selection.json` | every start's `ms_id`, role (R/P), source frame, stored saddle `ms_id`, plus the run settings |
| `minima.traj`, `proposals.traj`, `stored_saddles.traj` | aligned frame by frame (`info['tid']`) |
| `sella.json` | per landing point: `fmax_start`, converged, final fmax, force calls, maxd moved, eigenvalues, `nneg` |
| `report.txt` | false rate with a 95% interval, per-category force quantiles, filter AUC and a threshold table, share of real saddles that are the stored NEB one; repeated for each maxd cutoff |
| `failure_modes.pdf` | `fmax_start` vs Sella displacement coloured by category; ROC of `fmax_start` as a filter |

How to read the threshold table: for each force threshold, *false caught* is the share of false points
it would discard, *real discarded* the share of real saddles it would also lose, and *kept that are
real* the purity of what survives.
