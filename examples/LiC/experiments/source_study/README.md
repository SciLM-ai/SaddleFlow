# Michael: 12/48 saddles x initial-source memory

Four plain-FM runs: `n12_uncond`, `n12_source`, `n48_uncond`, `n48_source`.
No force/curvature/MMF inputs, losses, recovery pairs, or online PES guidance.
The student still runs its normal trainable UMA-derived neural backbone.

## Prepare and train

Michael: from the repository root in the saddleflow environment:

```bash
python examples/LiC/experiments/source_study/source_study.py prepare
sbatch examples/LiC/experiments/source_study/run_source_study.sh
```

Defaults write `/scratch/gqs7zr/saddleflow/lic_source_study_seed42/`.
The CPU preparation keeps the original 12 training triplets, selects 36 of the
167 other triplets without replacement with split seed 42, and reserves the
remaining **131** as a common holdout. It never selects using model outcomes.
The nested split's trajectory files, global catalog IDs, original input hashes,
and checksums are saved under `split/`. Repeat preparation reuses a verified
split; changing seed/inputs requires a new study directory. The old 167 labels
are NOT all held-out in the 48-saddle arms.

The Slurm array maps 0/1/2/3 to the four names above, each on one supported GPU.
`--array=0-3%2` limits concurrent jobs to two. Nothing submits automatically.
Training logs: `/scratch/gqs7zr/saddleflow/lic_source_study_ARRAYID_TASKID.out`
and `.err`. Checkpoints/configs go to `<study>/<arm>/`.

All four use **96000 optimizer updates**, batch 4, sigma .25 A, uniform time,
FP32, the same pretrained UMA initialization/seed, depth-1 head, last-two-block
time-FiLM, head/FiLM LR .001, backbone-block LR .01, warmup 100, EMA .999, and
the same cosine schedule. These are 16000 effective epochs for 12 triplets and
4000 for 48 (each triplet contributes two records). Budget matching is by
updates, NOT per-saddle exposure. Within each N, shuffling/noise/time streams
are matched through the same seed and identical model-construction RNG.
GPU arithmetic need not be bitwise identical across device types; this is a
first-seed study, not a multi-seed uncertainty estimate.

Every usual `train.py` argument can be forwarded, for example:

```bash
sbatch examples/LiC/experiments/source_study/run_source_study.sh --max-steps 6000 --log-every 100
```

Arm identity/output directory are set by the array; use `LIC_STUDY_ROOT` for a
separate cohort rather than overwriting runs. To use another split seed:

```bash
python examples/LiC/experiments/source_study/source_study.py prepare --split-seed 7 \
  --study-dir /scratch/gqs7zr/saddleflow/lic_source_study_seed7
sbatch --export=ALL,LIC_STUDY_ROOT=/scratch/gqs7zr/saddleflow/lic_source_study_seed7 \
  examples/LiC/experiments/source_study/run_source_study.sh
```

For a single arm/resume, invoke `train.py` with `--study-split DIR/split`,
`--train-saddles 12|48`, `--source-conditioning|--no-source-conditioning`,
the original output directory/settings and `--resume-from CHECKPOINT`.
The source-study CLI checks split/architecture identity and enough epochs for
the requested update budget. Checkpoints are large; 8 step saves plus final
per arm require substantial storage. Adjust `--save-every-steps` if needed.

## What the conditioning means

Only `MIC(x0-x_t)` is injected as an equivariant vector plus its scalar norm.
Here x0 is the ACTUAL noisy initial geometry, retained through the whole rollout.
It is not R/P, the parent saddle, a target eigenmode, or epsilon=x0-S.
Fixed carbons receive zero source displacement. Both arms have the same
32-channel projection/fusion branch; unconditioned arms always feed zeros.
Zero projection + identity geometry fusion preserves the common initial field,
while nonzero source-fusion columns allow the input projection to learn.
This first ablation injects memory into the velocity head, not into every
backbone block. A null result would not rule out richer source encoders.

Source memory supplies no extra information at t=0, and straight training
bridges admit the shortcut v*=(x_t-x0)/t for t>0. Consequently low interior FM
loss alone is NOT evidence of learning saddle physics. Evaluation includes
fixed bridge probes at t=0,.25,.5,.75 and autonomous endpoint/coverage metrics.
The general LiC loaders, exact/noisy false-bridge runner, random throws, and
shell sampler support source memory. A restart at t>0 must retain the ORIGINAL
x0; it cannot reset source memory to x_k. Standard study shells start at t=0.

## Shared inference

Michael: after the four runs finish:

```bash
sbatch examples/LiC/experiments/source_study/run_source_study_inference.sh
```

All four EMA models receive exactly the same exact/noisy false bridges,
Gaussian sources around the common12/extra36/heldout131, shells around the
common12 at radii .25,.43,.6,.8,1 A, and noisy starts around catalog minima.
All use 20-step Euler with a fresh clock at each initial source. The evaluation
seed is separate from the split/training seeds. No physics is used during
sampling; force/index verification is post-hoc and charged separately.

True hit <.10 A to ANY of the 179 stored saddles; false region <.30 A with true
precedence. Reports also give fixed-heldout hits, parent hits, unique saddles
reached, shell-radius curves, sampling/certification costs, final force, and
physical certification (fmax<=.005 eV/A,
lambda1<-.01, lambda2>.01). Parent recovery is not a nearest-saddle assertion.

Connected coverage uses stored R/S/P endpoints clustered with MIC tolerance
.05 A. For each starting minimum, count distinct adjacent catalog saddles
reached, then macro-average coverage over represented starting minima. Also
report catalog-connected hit rate. This is NOT independent PES connectivity
verification: no two-sided downhill relaxation is performed. Starts from minima
are an off-training-distribution stress test. Related perturbations are not
independent transfer units.

Logs: `/scratch/gqs7zr/saddleflow/lic_source_eval_JOBID.out` / `.err`.
Results: `<study>/comparison_checkpoint_final/source_study.npz`, `report.txt`,
and `source_study.pdf`. Numerical/report snapshots are saved after each batch;
incomplete methods have smaller N and must not be pooled as complete results.
The PDF shows outcomes and the same sampled trajectories for each model.

Michael: midpoint and quick inference options:

```bash
# Same optimizer checkpoint for ALL arms; never silently pick unequal "latest".
sbatch examples/LiC/experiments/source_study/run_source_study_inference.sh --checkpoint checkpoint_step_00048000
# Ten exact false bridges, skipping post-hoc physics for a fast geometry check.
sbatch examples/LiC/experiments/source_study/run_source_study_inference.sh --populations false_exact --geometry-only
# Evaluate one finished model while other arms are still training.
sbatch examples/LiC/experiments/source_study/run_source_study_inference.sh --models n12_source
```

Use separate `--out-dir` values for different evaluation selections. Other
options include `--samples-per-site`, `--false-noise-sigma`, `--radii`,
`--directions-per-radius`, `--coverage-draws`, `--coverage-noise-sigma`,
`--limit-starts`, `--batch-size`, `--K`, `--seed`, and `--no-plot`.
Repeated inference makes fresh predictions; it does not resume old models.

CPU-only replot, without oracle/model loading:

```bash
python examples/LiC/experiments/source_study/source_study.py --plot-only \
  /scratch/gqs7zr/saddleflow/lic_source_study_seed42/comparison_checkpoint_final/source_study.npz
```

## Coverage vs starts per minimum

Michael: does the plain model eventually reach every catalog-connected saddle
of a minimum if given enough noisy starts? Two inference-only runs of
`n12_uncond`, 128 starts per minimum, at the default noise and at twice it:

```bash
R=/scratch/gqs7zr/saddleflow/lic_source_study_seed42/coverage_curve
for s in 0.15 0.30; do
  sbatch --job-name=lic_cov_s$s examples/LiC/experiments/source_study/run_source_study_inference.sh \
    --models n12_uncond --populations connected \
    --coverage-draws 128 --coverage-noise-sigma $s --out-dir $R/sigma$s
done
python examples/LiC/experiments/source_study/coverage_curve.py $R/sigma0.15/source_study.npz \
  $R/sigma0.30/source_study.npz --draws 8,32,128 --out-dir $R
```

`coverage_curve.py` keeps the FIRST k starts of each minimum, so the 8/32/128
numbers are nested subsets of one run. It reports coverage next to a uniform
reference (each start hitting a random connected saddle at the observed hit
rate) and lists the connected saddles never reached. At k=8 on the main
comparison archive it reproduces the report: 73.94% coverage, 266/358, against
a uniform reference of 77.5%. Connectivity is still the stored catalog
adjacency, not downhill verification. About 3 h per run on one GPU.
