# LiC: lithium hopping on defective graphene

One Li atom on a frozen carbon sheet with two vacancies. The dataset has 179
first-order saddles for Li hops on this sheet, each within 0.22 Å of the
midpoint of a C–C bond. The model trains on 12 of them and is asked to find the
other 167.

If you are new to SaddleFlow, run [`examples/LiC_simpler`](../LiC_simpler) first:
the same physics on a pristine sheet, in about 20 minutes.

## The task

The model is **unconditioned**: it is never shown a reactant or a product. Each
training sample starts at a training saddle with Gaussian noise on the Li,
`x_0 = saddle + N(0, 0.25²)`, and the flow learns to carry it back to the saddle.
At inference any Li position on the sheet is a valid start, and the flow should
carry it to whichever saddle is nearest.

The test throws a Li at every point of a 64 × 64 grid covering the whole cell
and follows each one. A model that learned the physics sends almost every throw
onto a saddle, including the 167 it never saw.

## Run it

From the repository root:

```bash
python examples/LiC/train.py
python examples/LiC/random_throw.py --grid 64 \
    --ckpt examples/LiC/runs/tsdenoise_sigma0.25/checkpoint_final
python examples/LiC/plot_throws.py \
    examples/LiC/runs/tsdenoise_sigma0.25/random_throw/throw.npz
```

| step | cost |
|---|---|
| `train.py`: 16000 epochs, 96,000 steps, fp32 | about 4 h on one GH200 |
| `random_throw.py`: 4096 throws, 20 Euler steps each | about 8 min on one GH200 |
| `plot_throws.py` | a few seconds, CPU only |

Each checkpoint is about 4.6 GB, because it holds the weights, the EMA copy and
the optimizer state. The trainer saves one every 4000 epochs plus the final one,
about 23 GB in total. Pass `--save-every-epochs 0` to keep only the final one.

The last command writes `throws.pdf` next to `throw.npz`. Its first page shows
the sheet and the saddles alone. The second page overlays the trajectories and
prints the metrics below. A path that wraps through the cell edge is left out of
the drawing, because it would streak across the sheet, but it still counts in
the metrics.

## Expected result

The reference run was trained with exactly these defaults. It was scored with
EMA weights on the 64 × 64 grid, measuring in-plane distance to the nearest of
the 179 saddles:

| metric | value |
|---|---|
| median endpoint distance | 0.016 Å |
| p90 | 0.051 Å |
| p99 | 1.06 Å |
| endpoints within 0.10 Å of a saddle | 93.9 % |
| endpoints on a saddle-less midpoint | 4.9 % |
| endpoints nearest a different saddle than their start | 6.8 % |

Your numbers will differ slightly, because GPU training is not bit-for-bit
reproducible. `random_throw.py` also prints a summary in the terminal. Its
distances include the Li height, which the figure cannot show, but for a
trained model the two agree: 93.9 % reached on both counts for the reference run.

## Reading the figure

- **Grey:** the carbon sheet and its bonds. The two vacancies are the gaps.
- **Green rings:** the 12 training saddles, drawn at the 0.10 Å hit radius. An
  endpoint inside a ring has reached that saddle.
- **Orange rings:** the 167 test saddles, never seen in training.
- **Red crosses:** the 10 bond midpoints that carry no saddle. See below.
- **Blue lines:** one per throw, from the start to its endpoint, a black dot.
- **Right panel:** histograms of how far each start and each endpoint is from
  the nearest saddle. A working model moves the whole distribution below the
  dotted hit radius.

## The one failure you will see

About 5 % of throws end on a red cross. Those ten midpoints look exactly like
saddle sites, because the other 179 midpoints all carry a saddle within 0.22 Å.
But they are not stationary points. The force on the Li there is 0.06–0.28 eV/Å,
and a saddle optimiser started from one leaves the center. The geometry-only
baseline has learned false attraction regions there. This is not a fundamental
impossibility of learning the PES from geometry; it motivates testing richer
training support and explicit PES conditioning, described below.

## Why the model is not conditioned on R and P

The method in the main README starts from the midpoint of a reactant R and a
product P, and shows the model both endpoints. On this sheet that has nothing
to learn. R and P are neighbouring hollow sites, and the saddle is the bridge
between them. So the midpoint already sits on the saddle, a median 0.03 Å away
in-plane, and the flow would only adjust the Li height. The conditioned scheme is
used in [`examples/MP20Bat`](../MP20Bat), where the midpoint is far from the
saddle.

## Files

| file | what it is |
|---|---|
| `train.py` | trains the model; the defaults are the recipe above |
| `random_throw.py` | throws Li starts onto the sheet, integrates the flow, and saves every path |
| `plot_throws.py` | draws the figure and prints the metrics |
| `train_set.traj` | 12 `[R, S, P]` triplets used for training |
| `test_set.traj` | 167 held-out triplets, used only for scoring |

All 179 saddles were reconverged with Sella at fmax 0.005 eV/Å and verified as
first-order saddles. Li is atom 126; atoms 0–125 are the frozen carbons.

## Coverage and source-memory study

Michael: for the separate, oracle-free 12/48-saddle x source-conditioning study,
see [experiments/source_study/README.md](experiments/source_study/README.md). It provides a fixed nested split, four
matched training runs, and shared evaluation with a common 131-saddle holdout.

## Opt-in force and scalar-curvature inputs

Michael: these are architecture inputs, **not** a force/MMF loss or recovery
dataset. The defaults remain ordinary FM on the 12 training saddles, with
fresh sigma=0.25 A Li noise, uniform t, batch 4, 16000 epochs (96000 updates),
head depth 1, head/FiLM LR 1e-3, backbone LR 1e-2, warmup 100, EMA .999, FP32.
Both flags are off by default and can be used independently.

Michael: three matched baseline-recipe jobs, from pretrained UMA (no old FM
checkpoint initialization):

```bash
sbatch examples/LiC/experiments/force_conditioning/run_conditioning.sh
sbatch examples/LiC/experiments/force_conditioning/run_conditioning.sh --inject-force
sbatch examples/LiC/experiments/force_conditioning/run_conditioning.sh --inject-force --curvature-conditioning
```

Michael: `--inject-force` supplies detached frozen-UMA forces at the actual
current x_t to the velocity head (`--force-field-channels 32`). Carbon force
inputs are masked to zero. `--curvature-conditioning` supplies
`[lambda1, lambda2, lambda2-lambda1]` from the symmetrized Li-only 3x3 numerical
Hessian at the same x_t. Six central-FD probes perturb only Li, by
`--curvature-fd-eps 0.002` A; carbon positions/cell are untouched. Probes across
the batch are chunked with `--physics-batch-size 32`. No target eigenmode, false
bridge coordinates, or test labels enter the observations.

The scalar MLP (`--curvature-mlp-hidden 64`) uses fixed asinh scaling in units
of 1 eV/A^2, then applies invariant scalar FiLM to the head's equivariant
features. The last layer is zero-initialized. The force-fusion geometry block
starts as identity, with a zero force projection but nonzero force-fusion
columns, so gradients can activate the force path. Optional branches preserve
the common head's initialization/RNG. Construction is now seeded explicitly
before training; historical baseline weights are not reproduced bit-for-bit.
No extra velocity depth, force residual, mode predictor, or MMF loss is enabled.

Michael: the depth-1 baseline head has **41601 parameters**, including its time
MLP (128 channels). Curvature-only adds 16896 parameters; force-only adds
61696; both together have **120193**. Depth/width of the velocity trunk and
the trainable UMA backbone are unchanged. The curvature scale gate has a direct
nonlinear path into velocity even at depth 1. Force-only at depth 1 is more
restrictive: its explicit force-vector contribution is linear for fixed time,
and the injected l=0 force magnitude has no nonlinear gate into l=1 until a
deeper head is used. The nonlinear trainable backbone can still learn complex
geometry-dependent velocity features. Keep depth 1 for this first intervention;
if needed, repeat ALL arms with `--head-depth 3` rather than increasing only the
conditioned model's capacity. A null depth-1 result would not prove PES inputs
are useless in larger systems or richer heads.

Michael: observations are needed at **both training and sampling**. Costs per
state are force-only 1, curvature-only 6, both 7 UMA geometries. For batch 4,
the combined arm uses 28 per update, about **2.688 million** over the full
96000-update recipe. A 20-step rollout uses about 140 per source; batches reduce
GPU launch overhead, not the geometry count. There is no eigendecomposition
backpropagation or differentiation through the oracle. This is an expensive
oracle-conditioning proof of concept, not an oracle-free inference method.
Training logs include per-batch geometry counts and mean lambda1/lambda2/gap.

Michael: the batch launcher allows the usual A100/A40/B200/A6000/RTXPro6000 GPUs
and writes `/scratch/gqs7zr/saddleflow/lic_conditioning_JOBID.out`, `.err`, and
`lic_conditioning_JOBID/`. Its wall-time request is 24 hours; the full curvature
run has NOT been GPU-timed yet. Override the allocation as appropriate and use
the existing checkpoint/resume controls if needed, e.g. `--save-every-steps
12000`. Training defaults/checkpoint cadence otherwise match the baseline.
`--init-weights`/`--resume-from` require a matching conditioning architecture;
an unconditioned EMA cannot simply be loaded into a head with added parameters.

Michael: the checkpoint configuration reconstructs the exact inputs for the
existing inference commands:

```bash
python examples/LiC/rollout_eval/robustness_shells.py \
  --ckpt /scratch/gqs7zr/saddleflow/lic_conditioning_JOBID/checkpoint_final \
  --radii 0,0.25,0.43,0.6,0.8,1.0 --start-times 0 --bridge-noise-sigma 0
python examples/LiC/random_throw.py \
  --ckpt /scratch/gqs7zr/saddleflow/lic_conditioning_JOBID/checkpoint_final --grid 64
```

Michael: a short batch check of the ten exact false bridges using the force-only
run's 48000-step EMA checkpoint is available with:

```bash
sbatch examples/LiC/experiments/force_conditioning/run_conditioning_inference.sh
```

It writes `lic_conditioning_infer_JOBID.out` and `.err` under
`/scratch/gqs7zr/saddleflow/`, with numerical results and `report.txt` in
`lic_conditioning_21363449/false_exact_step48000/`. It uses architectural
conditioning but no extra force guidance, and reports geometry-only outcomes.
The launcher disables inherited CPU binding and preserves Slurm's GPU mask.
Pass `--ckpt` and `--out-dir` to evaluate a different checkpoint; remaining
Python CLI arguments are forwarded unchanged.

The shared LiC loader/velocity evaluator serves both scripts; input oracles are
queried at each current state, and saved archives record conditioning geometry
counts. `--no-uma` in the shell script skips ONLY post-hoc endpoint scoring;
it cannot disable architectural inputs that the model was trained to require.
`compare_saddle_search.py` also charges conditioning probes to its PES budgets.
The older recovery/PES-direction fine-tuning runners are separate fixed-architecture
experiments, not converted to this new conditioning recipe. The one-Sella-kick
restart experiment explicitly requires the original unconditioned model.
