"""Export a training checkpoint as a release directory loadable with `SaddleFlow.from_pretrained`.

    python export_pretrained.py --ckpt <run>/checkpoint_final --out <dir> [--live]

Writes <dir>/model.safetensors (the EMA weights by default, `--live` for the point estimate) and
<dir>/config.json (the UMA backbone configuration, the time-FiLM blocks, the head depth, the trained
tasks and the inference defaults). Only unconditional models (--delta-endpoint-channels 0) are supported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_full_testset_K10 import load_model  # noqa: E402

INFERENCE = {"sigma": 0.3, "passes": 5, "steps": 4}


def uma_backbone_config(name: str) -> dict:
    """The backbone section of a UMA checkpoint's model config, without the class path."""
    from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name

    ckpt = torch.load(pretrained_checkpoint_path_from_name(name), map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in dict(ckpt.model_config["backbone"]).items() if k != "model"}
    return json.loads(json.dumps(cfg, default=list))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="training checkpoint directory (config.json one level up)")
    p.add_argument("--out", required=True)
    p.add_argument("--live", action="store_true", help="export the live weights instead of the EMA shadow")
    a = p.parse_args()

    model, cfg = load_model(Path(a.ckpt), "cpu", use_ema=not a.live)
    ex = cfg["extras"]
    if int(ex.get("delta_endpoint_channels") or 0) != 0:
        raise SystemExit("only unconditional models (delta_endpoint_channels = 0) can be exported")
    if int(ex.get("attn_layers") or 0) != 0 or ex.get("inject_force") or ex.get("force_film"):
        raise SystemExit("global attention and force injection are not supported by SaddleFlow.from_pretrained")

    keep = ("backbone.", "velocity_head.")
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()
             if k.startswith(keep) and not k.startswith("backbone.film.")}   # `film` aliases films[-1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})

    task_map = {"omat": "omat", "oc20": "oc20", "oc22": "oc22"}
    for pair in filter(None, (ex.get("task_name_map") or "").split(",")):
        src, dst = pair.split("=")
        task_map[src.strip()] = dst.strip()
    config = {
        "model_type": "saddleflow-unconditional",
        "backbone_name": ex["backbone"],
        "backbone": uma_backbone_config(ex["backbone"]),
        "film_blocks": [int(b) for b in ex["early_time_film_blocks"].split(",")],
        "head_depth": int(ex["head_depth"]),
        "tasks": {"default": "omat", "map": task_map, "trained": sorted(set(task_map.values()))},
        "inference": {**INFERENCE, "sigma": float(ex.get("path_noise_sigma", INFERENCE["sigma"]))},
        "weights": "live" if a.live else "ema",
        "training_run": Path(a.ckpt).resolve().parent.name,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    n = sum(v.numel() for v in state.values())
    print(f"wrote {out}/model.safetensors ({len(state)} tensors, {n / 1e6:.1f} M parameters) and config.json")


if __name__ == "__main__":
    main()
