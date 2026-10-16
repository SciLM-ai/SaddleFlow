# Basin-disjoint recovery FM

This is a mechanistic, first-fold experiment, not a strict leakage-free claim
about an unknown saddle catalog. It reuses the original shell archive and the
original TS-denoise EMA checkpoint. Nothing changes in shared training/inference.

Run the complete first fold:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh
```

Defaults: 2,000 updates EACH for R0/R1/R2, batch size 4, head/FiLM LR 1e-4,
backbone LR 1e-5, fresh AdamW and EMA, checkpoints at 0/250/500/1000/2000.
R0 uses original noise sources only. R1/R2 use exactly 3 original + 1 recovery
per batch, with matched original inputs across all three and identical full
bridge batches for R1/R2. R2 queries one detached MMF teacher per update, on
the recovery member. No endpoint force penalty or inference guidance is added.

## Small learnability control

Before interpreting generalization, deliberately fit six seen failures:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation learnability
```

This reuses ONLY fold 0's six training sources in basin 0 with parent saddle 0
from `lic_recovery_mmf_21289950/split.npz` (rows 148,152,193,202,227,234). It starts
from the original TS-denoise EMA, not an R1/R2 continuation. Every batch slot is
an exact recovery pair; there are no ordinary training samples, MMF teachers,
force penalties, new mining, or test examples in optimization.

Recovery times are 25% exactly zero, 50% uniform on [0,.25], and 25% uniform on
[0,1]. Thus 81.25% of supervision is below .25. No importance correction is
applied: emphasizing early states is the purpose of this diagnostic.

Defaults: 6,000 updates, batch size 4, head/FiLM LR 1e-3, backbone LR 1e-4,
100-step warmup then CONSTANT LR, existing trainable backbone blocks, fresh
AdamW and EMA (.999), gradient clip 1. The original basin-disjoint experiment
and its cosine schedule are unchanged. Extra CLI arguments after this ablation
are forwarded to `train_recovery_learnability.py`, for example:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation learnability \
  --learning-rate 1e-3 --uma-lr 1e-4 --updates 6000
```

At steps 0,250,500,1000,2000,4000,6000, save resumable checkpoints and evaluate
BOTH raw and EMA on identical fixed probes at t=0,.05,.125,.25,.5,.75, plus
actual autonomous 20-step rollouts. Report source/bridge velocity vector RMSE,
any-true success, false occupancy, nearest-true distance and first Euler-step
displacement. Fixed noisy starts around the 12 training and 167 test saddles
are regression diagnostics ONLY, never used to select a checkpoint. This is
GEOMETRIC evaluation; no post-hoc force or index certification is computed.
The student executes its usual UMA-derived backbone, but no separate PES oracle.

Results use the same `lic_recovery_mmf_JOBID` output/log naming. Checkpoints are
directly inside the new run directory. `learnability.npz` and `report.txt` are
saved at every diagnostic before the batch's separate PDF command. On CPU:

```bash
python examples/LiC/experiments/pes_and_recovery/plot_recovery_learnability.py \
  /scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/learnability.npz
```

The PDF shows raw/EMA fitting and success curves, then baseline/latest EMA
trajectories for recovery and clean populations, without start dots. Clean
regression is expected to be possible with recovery-only training; the goal
here is learnability, not balanced performance or basin generalization.

To resume, set `LIC_REC_OUT` to the SAME output directory, submit the same
ablation/options, and add `--resume-from DIR/checkpoint_step_00002000` (or an
available snapshot). Completed diagnostic rows after the resumed update are
discarded and recomputed; original source selection and optimizer settings are
checked. Change the selection using `--source-run`, `--basin`, `--parent`,
`--num-sources` only in a NEW run. All selected rows must be training rows of
the previous split and share the chosen parent; held-out rows are excluded.

Inspect selection on CPU without creating outputs or loading the student:

```bash
python examples/LiC/experiments/pes_and_recovery/train_recovery_learnability.py --run-dir /tmp/lic_control_check --dry-run
```

### Matched six-source R2: optional MMF direction

Create the MMF counterpart to successful plain-FM run 21302150:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation learnability --mmf
```

This starts from the SAME original TS-denoise EMA, not the completed R1 model.
It keeps exactly the same six basin-0/parent-0 source rows, 6,000 updates, batch
size 4, learning rates, constant-after-warmup schedule, and early-time draws.
No other basin, shell source, or false-bridge center enters training. The only
new term is a detached MMF smooth-cosine loss on one selected batch member:

`loss = mean(recovery FM over all four examples) + lambda * MMF_direction_loss`.

The teacher observes UMA at that member's ACTUAL prescribed recovery-bridge
state/time, not the algebraic endpoint. It reflects the free-coordinate force
along the numerical minimum-curvature mode. The same generic UMA/FD-Hessian
teacher and constraint-masked cosine already used by the larger R2 are reused.
Teacher vectors are detached: no force/Hessian gradient or teacher parameter
is included in the student optimizer/EMA. There is no endpoint force penalty
and no PES guidance during the ordinary 20-step diagnostics/inference.

The teacher RNG is independent of source selection and bridge-time sampling;
R1 and R2 retain identical source/time streams. At startup, calibrate a fixed
lambda on 16 batches of ONLY these selected training recovery pairs, targeting
an initial MMF/FM parameter-gradient norm ratio of 0.1. The teacher deadband
is the maximum UMA free force over the original 12 TRAIN saddle labels; no
held-out labels or outcomes are used. No time taper or student-velocity mask
is added. To predeclare a weight instead, pass `--lambda-pes VALUE` with
`--mmf`; calibration still reports its initial gradient ratio.

New files are `mmf_calibration.json` (immutable settings/weight/deadband) and
`mmf_probes.npz` (teachers on the existing 36 fixed recovery probes). The usual
`learnability.npz`, `report.txt`, PDF and checkpoints remain directly in the
new job directory. Reports/JSONL include FM/MMF/total losses, active teacher
status, force score, cosine, and geometry-query counts. Raw/EMA fixed-probe
adoption is included in the report and a PDF page, without repeated teacher
queries. Geometric success is still not force/index-1 certification.

The teacher costs about seven UMA geometries per update (central + six FD
probes), around 42,000 for 6,000 updates plus setup/calibration. Resume using
the SAME directory/options plus `--resume-from`; cached calibration/probes
are verified and reused. The plain `--ablation learnability` mode remains
FM-only and supports its existing checkpoints.

### Unseen-source and neighborhood evaluation

Compare the original initializer and the successful learnability EMA
checkpoint, without any more training:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation generalization \
  --ckpt /scratch/gqs7zr/saddleflow/lic_recovery_mmf_21302150/checkpoint_final
```

Rows 194 and 197 are the basin-0 source episodes withheld by the original fold.
BOTH were created from original TRAIN saddle 0 at radius 1 A; they are not
held-out saddle labels and are not starts centered on a false bridge. Their
original baseline ENDPOINTS landed near false bridge 0. The six exact sources
used in learnability training remain excluded from this withheld group.

Defaults evaluate 776 identical starts for each of the two models: six trained
exact sources, two withheld exact sources, and 32 fresh Gaussian perturbations
at each sigma .02,.05,.10 A around each of these eight source centers. Sigma is
per Cartesian component; it is added to the SHELL SOURCE, not the parent saddle
or false endpoint. Only Li moves. Seeded directions are shared across sigmas
and all starts across both models. No outcome-based filtering or label changes.

The report separates trained/withheld exact and jittered populations, and
counts baseline-failed starts rescued plus paired gains/losses. Rates are
descriptive: the perturbations share just eight centers. This tests within-basin
generalization, not transfer to unseen basins. Geometry only; no force/index
oracle is loaded. The student still runs its normal UMA-derived backbone.

The same `.out`/`.err` and run-directory naming applies. `generalization.npz`
and `report.txt` precede the PDF. Standalone options include `--sigmas`,
`--draws-per-source`, `--batch-size`, `--dry-run`, and `--no-plot`.
Replot on CPU without models or resampling:

```bash
python examples/LiC/experiments/pes_and_recovery/evaluate_recovery_learnability.py --plot-only \
  /scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/generalization.npz
```

### Cross-basin inference on exact/noisy false bridges

Keep the successful checkpoint UNCHANGED and test all ten catalog
false bridges, including the seen basin 0 and the nine other basins:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation generalization \
  --ckpt /scratch/gqs7zr/saddleflow/lic_recovery_mmf_21302150/checkpoint_final \
  --starts false-bridges --basins all
```

This runs the original initializer EMA and fine-tuned EMA on exactly the same
970 starts each: 10 exact bridges plus 32 Gaussian draws per bridge at each
sigma .02,.05,.10 A. Noise is centered on the FALSE BRIDGES themselves, not
the shell sources used in training. No parent saddle or target is supplied
to the student. Only Li moves; the sampler is the unchanged 20-step Euler
rollout. No new optimization, MMF teacher, energy/force/Hessian oracle, or
inference guidance is used. The inherited UMA-derived student backbone still
executes normally. Success means ANY catalog true saddle, not the parent.

Separate CLI tests, using new job directories:

```bash
# Transfer only: the nine basins with no explicit recovery supervision.
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation generalization \
  --ckpt /scratch/gqs7zr/saddleflow/lic_recovery_mmf_21302150/checkpoint_final \
  --starts false-bridges --basins unseen

# Quick exact-only check: ten starts per model, no noise clouds.
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation generalization \
  --ckpt /scratch/gqs7zr/saddleflow/lic_recovery_mmf_21302150/checkpoint_final \
  --starts false-bridges --basins all --sigmas none
```

`--basins 1,2,3` selects specific zero-based catalog bridge IDs. Their ordering
is checked against the training split. Subsetting preserves the same seeded
perturbations as `--basins all`. `--sigmas` and `--draws-per-source` control the
noise clouds; the exact bridge is always included once. Basin 0 is "seen"
because its failed shell sources were supervised, NOT because the exact bridge
was used for training. The unseen designation excludes explicit recovery
pairs, not necessarily every nearby state along a training bridge.

The `.out` table and `report.txt` show all/seen/unseen aggregates AND each
starting basin, by sigma: any-true %, false %, other %, median nearest-true
distance, baseline-failed rescues, and paired new/lost true hits. These are
geometric diagnostics, not force/index-1 certification. Jitter draws share
centers; ten basins are not hundreds of independent transfer problems.

Outputs use the existing `lic_recovery_mmf_JOBID` log/directory naming.
`false_bridges.npz` and `report.txt` are saved before `false_bridges.pdf`.
The PDF contains aggregate/per-basin outcomes, then paired trajectory maps
for each sigma and each basin; no start dots. Replot on CPU:

```bash
python examples/LiC/experiments/pes_and_recovery/evaluate_recovery_learnability.py --plot-only \
  /scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/false_bridges.npz
```

The original `--starts shell-sources` mode remains the default and reproduces
the earlier 776-start within-basin check. Standalone runs of the bridge mode
default to `<checkpoint-parent>/false_bridges/`, separate from that old output.

### Trajectory-time recovery horizon: matched R1 versus R2

Once the six-source MMF checkpoint has finished, compare it with the
successful plain-FM checkpoint using the actual saved failed trajectories:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation recovery-horizon \
  --r1-ckpt /scratch/gqs7zr/saddleflow/lic_recovery_mmf_21302150/checkpoint_final \
  --r2-ckpt /scratch/gqs7zr/saddleflow/lic_recovery_mmf_R2_JOBID/checkpoint_final
```

Replace `R2_JOBID` with the completed six-source `--mmf` job's ID. Do not use
the older larger-fold R2: the evaluator verifies matching initialization,
source rows, optimizer settings, checkpoint update, and actual training
source/time streams. R2 must have queried its directional teacher; R1 must be
FM-only. All model evaluations use EMA, and there are NO new training updates
or online PES calls. R0 in this diagnostic is the untouched original baseline
that generated the saved trajectories, not the older FM continuation.

The existing shell archive has 73 mined known-false episodes across basins
0--5. Only SIX of those were supervised in the small R1/R2 run. The default
`--episodes heldout` therefore uses the other 67: two in basin 0 and 65 in
the other observed basins. Membership in the OLD larger-fold training split
does not make a source supervised in the six-source run. Basins 6--9 have no
such episodes in this archive; the separate false-bridge test covers them.

For each episode, branch from its saved baseline states at steps
`0,5,10,15,18,19`. Keep the original time `t=k/20` and step `dt=1/20`; take
only `20-k` new steps. Every student receives the identical geometry/time.
The episode's parent identity is analysis metadata, NOT a conditioning input.
The untouched baseline should reproduce its saved original endpoint: a replay
error over .001 A prints a warning to inspect provenance/numerics.

Useful independent CLI tests:

```text
--episodes trained                 six supervised sources (fitting diagnostic)
--episodes heldout --basins 0       two unseen source episodes in the seen basin
--episodes heldout --basins unseen  65 episodes in other observed basins
--episodes all                     all 73, separating trained and holdout groups
--restart-steps 0,5,10,15,18,19     explicit original Euler-grid indices
--limit-episodes 12                 seeded small subset for a quick run
```

The stdout/`.out` table and `report.txt` include per-time/population/basin
true, false, other rates, median nearest-true/false distances, first new
displacement, and paired R2-versus-R1 new/lost true hits. Success is ANY known
true saddle, with the same .10/.30 A thresholds and true precedence. These are
geometric results, not force/index-1 certification. Late restarts have fewer
steps and less time to correct; the curve is a descriptive recovery horizon,
not proof of a fixed point or a monotonic basin-wide boundary.

`recovery_horizon.npz` preserves baseline paths, archive/parent/basin IDs,
restart indices, checkpoints, replay errors and all new suffix paths. Prefix
padding is NaN, not fabricated motion. Completed methods are saved before the
PDF, so a partially completed comparison can still be inspected. The PDF shows
probability/distance curves and paired trajectory maps: gray saved approach,
blue new student suffix, colored endpoints, no start dots. Replot on CPU:

```bash
python examples/LiC/experiments/pes_and_recovery/recovery_horizon.py --plot-only \
  /scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/recovery_horizon.npz
```

The new inference mode uses the usual `lic_recovery_mmf_JOBID` directory and
`.out`/`.err` paths. `--dry-run` verifies matching checkpoints/selection and
prints counts without loading students or writing new outputs.

## Data and split

- Mine every source with saved final free `force_max > 0.010 eV/A`. In LiC,
  fmax equals force RMS because there is exactly one free atom.
- Current archive: 452 high-force rows, including 73 known-false-region rows
  spread over catalog basin IDs 0..5. Do not exclude geometric true near-misses.
- First fold trains from known-false basin IDs **0,1,2**, withholds **3,4,5**.
- Keep 20% of episodes in each training basin for within-basin evaluation.
- High-force rows outside known false regions use a separate 20% episode
  holdout, stratified by training parent and radius. They are NOT assigned to
  their nearest false bridge as though they belonged to that basin. This
  generic episode split stays identical across complementary basin folds.
- Each archive row is a whole shell rollout episode. The trainer uses its
  exact t=0 START and originating training saddle, not its failed endpoint.
  No intermediate states are harvested. Fixed atoms/cell remain unchanged;
  the source-to-label displacement uses MIC before interpolation.
- `split.json` and write-once `split.npz` record all row IDs and checksums.
  The basin split is an **endpoint-group exclusion**, not a guarantee that
  original states or recovery bridges never pass near a held-out basin.
- Any stored true saddle counts as evaluation success; the parent is one valid
  recovery target, not a uniquely correct or nearest saddle assertion.

The calibration stage uses only training labels and TRAIN recovery bridges:
teacher deadband = maximum UMA free force RMS over the 12 training saddles;
one fixed lambda targets a 0.1 ratio of mean MMF/FM parameter-gradient norms.
`--lambda-pes` can instead supply a predeclared fixed weight. No holdout is used
for lambda or deadband selection. This does not imply a 0.1 ratio at every step.

## Outputs and evaluation

Results: `/scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/`

Logs: `/scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID.out` and `.err`.

The batch job runs preparation, calibration, R0/R1/R2 training, the fixed-probe
audit, and then final-checkpoint evaluation. It does not select a checkpoint
using test results. The untouched initializer is an extra reference.

`comparison/report.txt` (also stdout) reports training recoveries, unseen
sources in seen basins, withheld basins, generic high-force holdout, noisy
train/test true saddles, and exact/noisy false bridges. It includes per-basin
and basin-macro results, plus paired R2-vs-R1 gains/losses on holdouts.

Catalog hit: <0.10 A to ANY known true saddle. False occupancy: <0.30 A to a
known false bridge, with true hits taking precedence. Physical certification:
free fmax <=0.005 eV/A AND lambda1<-0.01 AND lambda2>0.01 eV/A^2. Stationarity
alone is not saddle certification. UMA verification is post-hoc, outside the
20-step autonomous Euler sampler. The student still runs its UMA-derived
neural backbone, but no energy/force/Hessian oracle during sampling.

`comparison/saddle_search.npz` saves source IDs, splits, paths, endpoints,
forces, eigenvalues, and outcomes BEFORE `saddle_search.pdf`. Plots show paths
and final markers, not a cluttered layer of start dots. Replot without sampling:

```bash
python examples/LiC/experiments/pes_and_recovery/plot_recovery_mmf.py /scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/comparison/saddle_search.npz
```

`recovery_teacher.npz` caches 4 fixed times for up to 48 TRAIN recovery sources,
including known-false and generic failures. `recovery_adoption/` compares R2
versus R1 using EMA (primary) and complete raw weights (diagnostic), saving
velocity changes, teacher cosines, FM errors, report, and PDF. Re-running this
audit needs no new UMA queries. The probes are fixed training-distribution
draws, not exact optimizer-update samples.

## Complementary fold, custom runs, and resume

```bash
sbatch --export=ALL,LIC_REC_FOLD=1 examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh
```

This swaps observed training/withheld basin IDs into a NEW job directory.
Do not reuse the first fold's split/calibration. One three-basin holdout is
small; report both folds before making a strong transfer claim.

Optional batch exports: `LIC_REC_OUT`, `LIC_REC_INIT`, `LIC_REC_ARCHIVE`,
`LIC_REC_UPDATES`, `LIC_REC_FRESH_ARCHIVE`. The last adds an independent existing
shell archive's sources (never its saved endpoints) to evaluation.

Standalone stages use `train_recovery_mmf.py --run-dir DIR --stage
prepare|calibrate|train`, with `--experiment R0|R1|R2` for training. Resume an
interrupted model using the SAME original arguments plus
`--resume-from DIR/R2/checkpoint_step_00001000`. No shell remine is required.

To evaluate all predeclared saved snapshots rather than only the final one:

```bash
python examples/LiC/experiments/pes_and_recovery/compare_recovery_mmf.py --run-dir DIR --updates 0,250,500,1000,2000
```

This is substantially more expensive because endpoint certification is done
for every checkpoint. The default batch evaluates the reference and three
final checkpoints only.

## Inference-only Sella endpoint restart

This experiment uses the **original TS-denoise EMA checkpoint**, not
R1/R2. There is no training and no mid-trajectory splice. Default: 68 identical
original starts (10 exact false bridges, 10 noisy false bridges, 12 noisy
training saddles, 36 shell starts). Every source first receives a 20-step flow
proposal. A separate frozen UMA force query flags endpoints with free fmax
above 0.010 eV/A. Catalog labels/parent IDs do not decide repair eligibility.

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation sella-restart
```

The matched branches are:

- `baseline`: original flow proposal only.
- `restart_none`: flagged endpoint, no displacement, fresh 20-step flow.
- `random_a1`, `force_a1`: random/downhill-force displacement of exactly the
  Sella step's norm, then fresh flow.
- `sella_a1`: one accepted Sella geometry update, then fresh flow from t=0.
- `full_sella`: full Sella from **every original source**, including sources
  the flow already gets right. This is the all-Sella pipeline alternative,
  not full Sella from only the failed model endpoints.

Only flagged endpoints get kicks/restarts; passing endpoints are unchanged.
Sella's initial mode estimation and force evaluations are counted in the
single-step cost. A failed/budget-limited Sella step returns its last accepted
geometry, not a displaced Hessian probe. Zero accepted displacement is visible
in `kick_norm`/`kick_status`; it is not silently replaced with a random kick.
`status=max_steps` is normal for the deliberately one-iteration kick, not
evidence that the geometry update was rejected.
No directional guard, cone sampling, repeated repair, or alpha selection.

Optional larger scale sweep, with identical norms for all controls:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation sella-restart \
  --alphas 0.25,0.5,1,2 --max-kick 0.2
```

Only alpha=1 **without clipping** is the actual accepted Sella displacement.
Other scales/capped displacements are explicit extrapolation ablations.
`--gate force-index` additionally checks free-Hessian index at every proposal;
its additional queries/time enter the hybrid cost. Final certification always
reports free fmax <=0.010, lambda1<-0.01, lambda2>0.01. This force tolerance is
configurable with `--fmax` and differs from the older 0.005 physics benchmark.

To use the existing shell archive's original sources, while freshly
measuring both the flow and all-Sella alternative:

```bash
sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation sella-restart \
  --starts-npz /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz \
  --limit-starts 128
```

Remove `--limit-starts` for all t=0 archived sources; full Sella then runs on
every one of them. The old archive's saved endpoints/forces are NOT reused for
failure selection or treated as a free timed proposal.

`report.txt` is printed in the `.out` log. It reports geometric and
physical success overall, for flagged/passed populations, and per baseline
false-endpoint basin. Timing includes:

- Proposal sampling, all-source failure detection, Sella kick, second sampling.
- Full-Sella search cost separately, from identical original sources.
- Actual PES geometry counts (not batch counts), and student-forward counts.
- Post-hoc final force/index verification separately AND included totals.

Shared measured proposal/gate/kick work is charged to each hypothetical
standalone branch; table sums are **not** the total runtime of running all
ablations together. Random/force controls are scientific norm-matched controls:
their timing includes the Sella norm-calibration cost. Full Sella reuses no
hybrid curvature state. CUDA synchronization precedes/follows timings. Model
loading/warmup and file/PDF I/O are excluded; batching is used for neural flow,
while independent Sella searches are serial. Compare throughput on this job,
not single-request latency or historical timings.

Output/log paths retain the existing batch naming:
`/scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/` and adjacent `.out`/`.err`.
`sella_restart.npz` and `report.txt` snapshots are saved before the PDF. This
runner is not resumable, but completed stages/trials can be plotted on CPU:

```bash
python examples/LiC/experiments/pes_and_recovery/sella_restart.py --plot-only \
  /scratch/gqs7zr/saddleflow/lic_recovery_mmf_JOBID/sella_restart.npz
```

Partial full-Sella comparisons have different completed N: do not compare
their raw total times until both rows cover the same original sources. Plots
show blue trajectories, orange endpoint kicks, and final markers, not a
starting-point dot cloud.
