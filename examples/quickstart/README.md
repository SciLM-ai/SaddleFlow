# Quickstart

`quickstart.py` runs the released SaddleFlow model on one structure: it proposes 8 saddles from noisy copies of
the input, then converges them with Sella on UMA-S-1.2 and writes the distinct first-order saddles to
`saddles.traj`. See the top-level README for installation and the Hugging Face access it needs.

`reactant.traj` is a Li-ion cathode cell (F8Li5Mn8O32P8, 61 atoms) from the test split of
[MaterialsSaddles](https://huggingface.co/datasets/SciLM/MaterialsSaddles) (`mp20bat`, `ms_id` 102302658;
CC-BY 4.0). On one GH200, proposing takes a few seconds and the Sella refinement a few minutes.
