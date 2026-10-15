"""
Load a pretrained fairchem/UMA backbone for use inside a SaddleFlow model.

This is a thin wrapper around `fairchem.core.calculate.pretrained_mlip.get_predict_unit`
that strips the EMA wrapper and UMA's output heads, returning only the
`eSCNMDBackbone` we consume in `FlowMatchingLoss` and `sample_saddles`.
"""

import torch.nn as nn


def load_uma_backbone(
    name: str = "uma-s-1p2",
    device: str = "cuda",
    freeze: bool = True,
    eval_mode: bool = True,
    unfreeze_last_block: bool = False,
) -> nn.Module:
    """Return the `eSCNMDBackbone` module from a pretrained UMA checkpoint.

    Args:
        name: HuggingFace model tag. Defaults to the small UMA-S-1.2 variant
            (6.6M active / 290M total params).
        device: "cuda", "cpu", or a specific CUDA index.
        freeze: if True, sets `requires_grad=False` on every backbone parameter.
        eval_mode: if True, calls `.eval()` on the returned module. KEEP this
            True even with `unfreeze_last_block=True`: UMA-S-1.2's last block
            includes `composition_dropout=0.10` and `mole_dropout=0.05` which
            we explicitly suppress (CLAUDE.md latent-bug log §"UMA backbone
            dropout train/infer mismatch"). Eval mode keeps those off.
        unfreeze_last_block: if True, AFTER applying `freeze`, sets
            `requires_grad=True` on every parameter inside `backbone.blocks[-1]`
            (the final message-passing block). Used by Mode 1 v1 to let the
            backbone's last block adapt to the velocity-prediction objective.
            Keep `eval_mode=True` so dropouts inside that block stay off.

    A note on `blocks[-1]` size for UMA-S-1.2: 72.7M parameters (most of which
    are MoE experts; only a small subset routes per forward pass). Use a
    discriminative LR on those params (e.g. 1e-5) via a parameter-group split
    in the optimizer to avoid catastrophic overfitting on small datasets.
    """
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit

    predictor = get_predict_unit(name, device=device)
    # predictor.model is an AveragedModel(HydraModel); we want the backbone.
    backbone = predictor.model.module.backbone
    # get_predict_unit only sets the CURRENT_DEVICE env var — it does not move
    # module parameters. Force-move here so downstream code (evaluate.py without
    # accelerate, ad-hoc sampling) sees parameters on the intended device.
    backbone = backbone.to(device)
    if freeze:
        for p in backbone.parameters():
            p.requires_grad_(False)
    if unfreeze_last_block:
        for p in backbone.blocks[-1].parameters():
            p.requires_grad_(True)
    if eval_mode:
        backbone.eval()
    return backbone

def build_uma_backbone_custom(
    shape: str,
    dataset_list=None,
    device: str = "cuda",
):
    """Build a RANDOMLY-INITIALISED eSCN-MD MoE backbone with a custom shape.

    Motivation (2026-09-19, user-set): transition states are local, so a large
    receptive field may be unnecessary and extra depth only makes exact
    equivariance harder to maintain numerically.  A shallow/wide model with
    higher angular resolution tests that directly.  UMA-M is 10 layers, lmax 4,
    mmax 2, 128 channels, 32 experts; a MACE-like shape is 2 layers with larger
    lmax/mmax and more channels.

    NOTE this discards UMA's pretraining entirely -- there is no way to widen a
    pretrained tensor.  Whether that matters is itself untested in this project
    (no random-init control has ever been run), which is why a same-shape
    random-init UMA-M control should be trained alongside.

    `shape` is "layers=2,lmax=6,mmax=4,channels=256,experts=8"; omitted keys keep
    the UMA-M value.  Non-shape settings are copied from uma-m-1p1 so the only
    difference is the geometry of the network.

    Because the experts are random anyway, `dataset_list` can name the subset
    natively (e.g. ["oc22"]) -- no oc22->oc20 remap is needed, which removes the
    routing confound that applies to every pretrained run.
    """
    import torch.nn as nn  # noqa: F401
    from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

    # Defaults MATCH uma-m-1p1 so any override is a deliberate, visible deviation.
    # NOTE on `experts`: UMA routes the MoE by DATASET, so on a single subset the mixing
    # coefficients are constant and 32 experts collapse into one fixed linear combination --
    # extra parameters, zero extra expressivity per forward (they do cost Adam/EMA state).
    # Cutting them is therefore safe for single-subset runs and WRONG for multi-subset
    # training, where the routing is the whole point. Raise back to 32 before training on
    # more than one subset.
    cfg = {"layers": 10, "lmax": 4, "mmax": 2, "channels": 128, "experts": 32}
    for tok in [t for t in shape.split(",") if t.strip()]:
        k, v = tok.split("=", 1)
        k = k.strip()
        if k not in cfg:
            raise SystemExit(f"--backbone-shape: unknown key {k!r}; allowed {sorted(cfg)}")
        cfg[k] = int(v)

    ch = cfg["channels"]
    backbone = eSCNMDMoeBackbone(
        max_num_elements=100,
        sphere_channels=ch,
        lmax=cfg["lmax"],
        mmax=cfg["mmax"],
        num_layers=cfg["layers"],
        hidden_channels=ch,
        edge_channels=ch,
        num_experts=cfg["experts"],
        cutoff=6.0,
        max_neighbors=300,
        distance_function="gaussian",
        num_distance_basis=128,
        norm_type="rms_norm_sh",
        act_type="gate",
        ff_type="spectral",   # UMA-M uses spectral; "grid" materialises a 128-point
                              # spherical grid per edge and inflates activations ~5x -> OOM
        otf_graph=True,
        always_use_pbc=False,
        use_dataset_embedding=True,
        dataset_list=list(dataset_list) if dataset_list else ["oc22"],
        regress_forces=False,   # deliberate: we consume node_embedding only and discard UMA's
        direct_forces=False,    # output heads, so building the force/stress graph is wasted work
        chg_spin_emb_type="rand_emb",  # match uma-m-1p1 (inert on oc22, where charge=spin=0 always,
        cs_emb_grad=True,              # but a free variable to remove; matters on omol)
    ).to(device)
    for prm in backbone.parameters():
        prm.requires_grad_(True)
    n = sum(p.numel() for p in backbone.parameters())
    print(f"[backbone] CUSTOM random-init shape {cfg} -> {n/1e6:.1f}M params, "
          f"datasets={list(dataset_list) if dataset_list else ['oc22']}")
    return backbone
