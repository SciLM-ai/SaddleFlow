# Al_adatom: saddles around an Al adatom on Al(100)

An Al(100) slab (six layers of 64 atoms, the bottom two fixed) with one adatom on top: 385 atoms. `al_triplets.traj` holds
**353 reactions**, each a converged `[reactant, saddle, product]` triplet that starts at the same minimum. This example
fine-tunes SaddleFlow FM2 on them with FM2's own recipe and tests how many of them the model finds from that minimum,
before and after the fine-tune.

## The dataset

Built with UMA-S-1.2 (task `oc20`) from an archive of 490 dimer saddles around the adatom:

- every saddle reconverged with Sella to fmax 0.005 eV/Å and kept only if it has exactly one negative curvature;
- the minimum relaxed with FIRE to fmax 0.005 eV/Å;
- each saddle pushed both ways along its lowest-curvature mode and both sides relaxed with FIRE to fmax 0.005 eV/Å
  (verified minima, no negative curvature); a triplet is kept when one side returns to the reactant and the other is a
  different minimum, more than 0.2 Å from the saddle and at least 0.01 eV below it (the product);
- saddles that reconverge to the same point (within 0.1 Å) are kept once.

All 353 saddles are distinct (the closest two are 0.64 Å apart), connected to the reactant, and none is a periodic event
(a row of atoms bonded to its own image). Barriers are 0.19–1.62 eV; the shallowest six products lie 0.02–0.06 eV below
their saddle. Every frame carries `FixAtoms` on atoms 0–127, `info["task_name"] = "oc20"`, `info["triplet"]` and its UMA
energy in `info["energy"]` (eV).

## Run it

From the repository root, after accepting the terms of SciLM/SaddleFlow on Hugging Face and logging in:

```bash
python examples/Al_adatom/train.py --output-dir examples/Al_adatom/runs/fm2_ft
for m in movers ball; do
  python examples/Al_adatom/sample.py --model examples/Al_adatom/runs/fm2_ft/release --mode $m --n 10 \
      --out examples/Al_adatom/runs/fm2_ft/$m
  python examples/Al_adatom/sample.py --model SciLM/SaddleFlow --revision fm2 --mode $m --n 10 \
      --out examples/Al_adatom/runs/fm2/$m
done
python examples/Al_adatom/score.py examples/Al_adatom/runs/fm2/{movers,ball} examples/Al_adatom/runs/fm2_ft/{movers,ball}
```

`train.py` starts from the released FM2 (UMA-M backbone), trains 20 epochs on both directions of every triplet (R → S
and P → S) at batch 8 on one GPU, and writes `<output-dir>/release`, which `sample.py` loads exactly like the released
model. A smaller `--batch-size` lowers the memory; with several GPUs (`accelerate launch`), `--warmup-steps` counts
scheduler ticks, one per GPU per step. Each training checkpoint is about 22 GB (weights, EMA copy, optimizer state).

## Reading the numbers

- **movers**: for every saddle, starts from the minimum with noise on that saddle's own moving atoms (more than 0.5 Å,
  plus the most-moving one), flagged. **Recall** is the number of saddles that at least one of their own draws reaches.
- **ball**: starts with noise on the mobile atoms within 4 Å of a random adatom or top-layer atom, as a user who picks an
  atom of interest and a cutoff; **distinct saddles reached** counts how many of the 353 the draws find.
- An endpoint **hits** a saddle within 0.2 Å (largest per-atom distance, minimum image), **fell back** within 0.2 Å of the
  minimum; **elsewhere** is anything else, including saddles that are not in the dataset (symmetric copies at other sites
  are not listed). A saddle search from the endpoint tells those apart: `saddleflow.search.find_saddles` runs Sella on
  proposals.

## Files

| file | what it is |
|---|---|
| `train.py` | fine-tunes FM2 on the triplets with FM2's recipe and exports `<output-dir>/release` |
| `sample.py` | runs a model from the minimum (`movers`, `all` or `ball` atoms), five passes of four Euler steps |
| `score.py` | hits, fall-backs, distinct saddles and recall against the dataset |
| `al_triplets.traj` | the 353 `[R, S, P]` triplets |
