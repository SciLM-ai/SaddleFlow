# LiC_simpler — the smallest end-to-end SaddleFlow example

One lithium atom hopping on a pristine graphene sheet. Small enough to train in
~20 minutes on a single GPU, and symmetric enough that you can *see* whether the
model learned the physics rather than just fitting a number.

**Start here if you are new to the codebase.**

## The system

`one_saddle.traj` holds a single `[R, S, P]` triplet: reactant, saddle, product
for one Li hop. 112 carbons (indices 0–111, frozen) + 1 Li (index 112, the only
mobile atom).

The point of this example is the symmetry. A Li adsorption site on graphene has
6-fold symmetry, so the hop has **six** equivalent saddles around it — but the
dataset contains only **one**. A correct model must recover all six from that
one, because the network is SO(3)-equivariant and the six are related by
rotation. That gives you a strong, visual pass/fail signal: the trajectories
should fan out into six petals (a "flower"), not spray outward uniformly.

## Run it

```bash
# (a) mode 1 — product-conditional. Given R and P, find the saddle between them.
python examples/LiC_simpler/train.py
python examples/LiC_simpler/viz_checkpoints.py --run-dir examples/LiC_simpler/runs/mode1

# (b) TS-denoise — unconditioned. Given any structure, flow to the nearest saddle.
python examples/LiC_simpler/train.py --ts-denoise-sigma 0.5 \
    --delta-endpoint-channels 0 --attn-layers 1 --ema-decay 0.99
python examples/LiC_simpler/viz_checkpoints.py \
    --run-dir examples/LiC_simpler/runs/tsdenoise_sigma0.5
```

Each is ~20 min on one GPU. Runs land in `runs/<objective>/`, so (a) and (b) do
not overwrite each other.

The visualiser reads the architecture from the run's own `config.json`, so you
never have to pass matching `--attn-layers` / `--head-depth` by hand. It writes
one figure per checkpoint plus a `flower_evolution.png` montage.

## The two objectives, and when to use which

|  | mode 1 (default) | TS-denoise |
|---|---|---|
| start `x_0` | the (R+P)/2 midpoint | saddle + Gaussian noise |
| sees R and P? | yes, at every step | **no** |
| answers | "the saddle between *these two* endpoints" | "the nearest saddle to *here*" |
| score against | the dataset saddle | the saddle a saddle-optimiser reaches |

Do not mix them up when evaluating: an unconditioned model was never told which
saddle you had in mind, so scoring it against a specific stored label punishes it
for working correctly.

## Reading the figures

Per panel: the red dot is the reactant Li, open stars are the six symmetry-
equivalent saddles, the black star is the one actually in the training data, and
grey lines mark the "atop" directions (30° off the saddles — where a broken model
tends to point).

Useful numbers to compute from the saved `li_paths_*.npz`:

- **hexatic order** `|mean(exp(6i·θ))|` over endpoint angles — 1.0 is six sharp
  petals, ~0.1 is angle-uniform.
- **on-orbit fraction** — endpoints landing within 0.35 Å of a true saddle.
- **balance** — entropy of the per-saddle counts. **Only meaningful when the
  on-orbit fraction is high**: if few endpoints land near any saddle, balance
  describes a handful of points and reads deceptively well.

## Other files

Three visualisers, three genuinely different questions — none is redundant:

| script | answers | works on |
|---|---|---|
| `viz_checkpoints.py` | *How did the field evolve over training?* One panel per checkpoint plus a montage, same perturbation seed throughout. | both objectives |
| `visualize.py` | *What does the velocity field look like everywhere?* Evaluates the field on a 2D grid around the Li at several flow times — including regions no trajectory visits. `--plot {trajectories,field,both}`. | both objectives |
| `visualize_mode1.py` | *Does conditioning actually steer it?* Synthesises the five missing partners by rotating (P−R) by k·60°, samples each separately, and checks you get six different saddles. | conditioned (mode 1) only |

Start with `viz_checkpoints.py`. Reach for the field map when trajectories look
wrong and you want to see *why* — it is what revealed that a broken model's field
points at the atop directions at every radius. Reach for `visualize_mode1.py` when
you want to test the conditioning mechanism itself rather than one trajectory set.

All three read the architecture from the run's `config.json`.

- `make_small_cell.py` — builds `small_n5_one_saddle.traj` / `small_n5_six_saddles.traj`,
  hexagonal cells that are **exactly** C6-symmetric about the Li site (the stock
  113-atom cell is rectangular and only symmetric to ~0.001 Å). Smaller and
  faster, and the sharpest test of symmetry behaviour.

## One trap worth knowing

Train in bf16 and geometry helpers can silently lose precision: `torch.autocast`
demotes matrix multiplies, and a Cartesian↔fractional coordinate round trip in
bf16 displaces atoms by up to ~0.09 Å — enough to destroy the six-fold structure
while the training loss actually looks *better*. This is fixed (the helpers in
`saddleflow/data/transforms.py` force fp32, and `tests/test_autocast_precision.py`
locks it), and this example defaults to fp32 anyway. It cost months to find, so
if you add coordinate arithmetic anywhere, keep it out of autocast. See
CLAUDE.md's latent-bug log.
