"""SaddleFlow quickstart: propose saddles for one structure, then converge them with Sella.

    python quickstart.py

Needs a GPU, `pip install "saddleflow[search]"`, and access to the gated Hugging Face repositories
SciLM/SaddleFlow (the model) and facebook/UMA (UMA-S-1.2, used to converge the proposals); accept their
licenses on the website and run `hf auth login` first.
"""
from pathlib import Path

from ase.io import read, write
from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate.pretrained_mlip import get_predict_unit

from saddleflow import SaddleFlow
from saddleflow.search import find_saddles, max_displacement

here = Path(__file__).resolve().parent
atoms = read(here / "reactant.traj")             # a Li-ion cathode cell from the MaterialsSaddles test split
sf = SaddleFlow.from_pretrained()                # SciLM/SaddleFlow, 5.6 GB on first use

# 1. Proposals only: 8 noise draws, each flowed toward a nearby saddle (a few seconds on a GPU).
proposals = sf.propose(atoms, n=8, seed=0, task_name="omat")
write(here / "proposals.traj", proposals)

# 2. Converge them with Sella on UMA-S-1.2 and keep the distinct first-order saddles (a few minutes).
calc = FAIRChemCalculator(get_predict_unit("uma-s-1p2", device="cuda"), task_name="omat")
saddles = find_saddles(atoms, calc, n=8, model=sf, seed=0, task_name="omat")
write(here / "saddles.traj", saddles)
start = atoms.copy()
start.calc = calc
e0 = start.get_potential_energy()
for i, s in enumerate(saddles):
    print(f"saddle {i}: {s.info['sella_steps']} Sella steps, {max_displacement(atoms, s):.2f} A from the input, "
          f"{s.info['energy'] - e0:.2f} eV above it")
