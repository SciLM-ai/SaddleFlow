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
and a saddle optimiser started from one always leaves. A model that sees only
geometry cannot tell them apart from real saddles. Removing this tail needs
information beyond geometry, such as relabelling the model's own endpoints with
a saddle optimiser. That is beyond this example.

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
