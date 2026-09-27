# LiC_simpler — the smallest end-to-end SaddleFlow example

One lithium atom hopping on a pristine graphene sheet. Small enough to train in
~20 minutes on a single GPU, and symmetric enough that you can *see* whether the
model learned the physics rather than just fitting a number.

**Start here if you are new to the codebase.** [`examples/LiC`](../LiC) is the
next step: the same physics on a defective sheet with 179 saddles.

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
python examples/LiC_simpler/train.py
python examples/LiC_simpler/viz_checkpoints.py \
    --run-dir examples/LiC_simpler/runs/tsdenoise_sigma0.5
```

The model is unconditioned. Every training sample starts at the one known saddle
with Gaussian noise on the Li, `x_0 = saddle + N(0, 0.5²)`, and the flow learns to
carry it back; it never sees the reactant or the product. The visualiser then
starts 48 trajectories from the reactant plus 0.15 Å of noise and follows each
one to wherever the learned field takes it.

The defaults unfreeze all four UMA blocks at `--uma-lr 1e-2` and apply
equivariant time-FiLM at every block. Both matter: with the backbone frozen this
example plateaus at hexatic order ~0.91 and never closes the last ~0.06 Å.
Capacity is not the missing ingredient: `--head-depth 3` on top of this is
slightly **worse** (hexatic 0.944), and dropping attention costs a little
(0.933).

Each checkpoint is about 4.6 GB with the backbone unfrozen, because it holds the
weights, the EMA copy and the optimizer state. The trainer saves one every 2000
epochs plus the final one, about 28 GB, and the visualiser draws one panel per
checkpoint. Pass `--save-every-epochs 0` to keep only the final one.

## Expected result

At 10000 epochs, fp32, with the defaults (48 perturbed starts, σ_inf 0.15,
K = 20, EMA weights), measured against the six symmetry-equivalent saddles,
for two runs with identical settings. The first four rows are the numbers the
visualiser prints in its panel titles:

| quantity | reference run | repeat run |
|---|---|---|
| hexatic order \|⟨e^{6iθ}⟩\| about the reactant site | **1.000** | **0.977** |
| endpoints within the 0.05 Å hit radius | **48 of 48** | **45 of 48** |
| median distance to the nearest true saddle | **0.002 Å** | **0.003 Å** |
| p90 distance | **0.004 Å** | **0.024 Å** |
| farthest endpoint | 0.025 Å | 0.21 Å |

In both runs every trajectory ends in one of the six saddle petals, so all six
saddles are recovered from the **one** that appears in the training data —
which is the whole point of the example. The runs differ only in how tightly the
last few trajectories converge: GPU training is not bit-for-bit reproducible,
and two runs with identical settings differed by this much. Ablations at the
same budget, scored at K = 50:

| variant | flags | hexatic | on-orbit | median dist |
|---|---|---|---|---|
| unfrozen + time-FiLM (the defaults) | | 1.000 | 100 % | 0.005 Å |
| + head depth 3 | `--head-depth 3` | 0.944 | 96 % | 0.005 Å |
| unfrozen, no attention | `--attn-layers 0` | 0.933 | 98 % | 0.005 Å |
| frozen backbone (head only) | `--no-unfreeze-uma-all --no-early-time-film` | 0.912 | 98 % | 0.056 Å |
| frozen + head depth 3 | `--no-unfreeze-uma-all --no-early-time-film --head-depth 3` | 0.763 | 90 % | 0.070 Å |

Each is ~20 min on one GPU. A run lands in `runs/` under a name built from its
flags, for example `runs/tsdenoise_sigma0.5_depth3`, so ablations do not
overwrite each other; pass that folder to the visualiser with `--run-dir`.

The visualiser reads the architecture from the run's own `config.json`, so you
never have to repeat the training flags. It writes one figure per checkpoint
plus a `flower_evolution.pdf` montage.

## Why the model is not conditioned on R and P

The method in the main README is product-conditional: it starts from the
midpoint of a reactant R and a product P and shows the model both endpoints. On
this sheet that leaves nothing to learn. R and P are neighbouring hollow sites
and the saddle is the bridge between them, so the midpoint already sits on the
saddle, 0.0006 Å away in-plane, and the flow would only lift the Li by 0.26 Å.
The conditioned scheme is used in [`examples/MP20Bat`](../MP20Bat), where the
midpoint is far from the saddle.

## Reading the figures

Panels are styled to match the defective-sheet example (`examples/LiC`) so the
two can be read side by side: dark grey carbon sheet with bonds, blue
trajectories from blue start markers to black endpoints, and the red dot is the
reactant Li. Saddles are drawn as **circles of the hit radius itself**, so a
trajectory has hit one exactly when its endpoint falls inside the circle:

- **green** — the saddle that is in the training data (1 of them)
- **orange** — the five symmetry-equivalent saddles, never trained on

Two deliberate differences from `examples/LiC`. The view is **zoomed to ±2.6 Å**
around the reactant rather than showing the whole sheet, because the cell is
pristine and every hexagon is equivalent. And the hit radius is **0.05 Å**
rather than the 0.10 Å used there: at 0.10 Å most endpoints of the
frozen-backbone variant (median 0.056 Å) would count as hits, hiding the
difference between the variants above. The radius is printed in every panel
title.

## Other files

- `viz_checkpoints.py` — the visualiser above. Every checkpoint gets the same
  perturbation draws, so the montage shows the flower forming over training.
- `make_small_cell.py` — builds `small_n5_one_saddle.traj` / `small_n5_six_saddles.traj`,
  hexagonal cells that are **exactly** C6-symmetric about the Li site (the stock
  113-atom cell is rectangular and only symmetric to ~0.001 Å). Smaller and
  faster, and the sharpest test of symmetry behaviour.
- `neb/` — the SaddleMill climbing-image NEB that regenerates the saddle in
  `one_saddle.traj` from its own reactant and product (7 images, `fmax` 0.01 eV/Å,
  serial on one GPU), together with the recorded run's outputs: it lands on the
  training saddle to 0.0000 Å with a 0.3084 eV barrier. `bash neb/run.sh fresh`
  reproduces it in ~20 s; see `neb/README.md`.

## One trap worth knowing

Train in bf16 and geometry helpers can silently lose precision: `torch.autocast`
demotes matrix multiplies, and a Cartesian↔fractional coordinate round trip in
bf16 displaces atoms by up to ~0.09 Å — enough to destroy the six-fold structure
while the training loss actually looks *better*. This is fixed (the helpers in
`saddleflow/data/transforms.py` force fp32, and `tests/test_autocast_precision.py`
locks it), and this example defaults to fp32 anyway. It cost months to find, so
if you add coordinate arithmetic anywhere, keep it out of autocast. See
CLAUDE.md's latent-bug log.
