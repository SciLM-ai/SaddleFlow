#!/usr/bin/env python
"""
Combine per-config benchmark JSONs (written by train.py --bench-output) into:
  * a markdown summary (per-GPU throughput, scaling table, parallel efficiency,
    projected full-mp20bat-training wall-clock, and a Grace-Blackwell projection),
  * scaling.png (samples/s vs GPUs + parallel efficiency),
  * per_gpu_throughput.png (single-GPU throughput + memory headroom).

Robust to having only one machine's results present (run Perlmutter now; drop in
Vista JSONs later and re-run). Group key is the machine label parsed from each
filename ("<machine>_n<nodes>_g<gpus>_b<batch>.json"), with the GPU name read
from the JSON for display.

Usage:
    python analyze.py --results-dir DIR [--results-dir DIR2 ...] --out-dir OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- mp20bat production run (CLAUDE.md "Used in production") ------------------
# 34,742 saddles × 2 (R↔P doubling) = 69,484 records/epoch; 60 epochs.
PROD_RECORDS_PER_EPOCH = 69_484
PROD_EPOCHS = 60
PROD_SAMPLES = PROD_RECORDS_PER_EPOCH * PROD_EPOCHS

# --- Grace-Blackwell (Horizon) projection knobs ------------------------------
# Per-GPU GB200 vs GH200, from published dense specs. NOT measured — labelled as
# a projection everywhere it's used. Memory-bound kernels track HBM bandwidth;
# compute-bound kernels track dense bf16 tensor throughput. We report the band.
GB200_OVER_GH200 = {
    "bf16_dense_tflops": 2.25,   # B200 ~2.25 PFLOPS dense bf16 vs H100/GH200 ~0.99
    "hbm_bandwidth": 2.0,        # ~8 TB/s (B200 HBM3e) vs ~4 TB/s (GH200 HBM3e)
    "fp4_vs_bf16_bonus": 2.0,    # 2nd-gen Transformer Engine FP4 path, IF kernels adopt it
}


def _machine_from_name(p: Path) -> str:
    stem = p.stem
    return stem.split("_n")[0] if "_n" in stem else stem


def load_results(dirs: list[str]) -> dict[str, list[dict]]:
    by_machine: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for d in dirs:
        for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
            rp = os.path.realpath(fp)
            if rp in seen:
                continue
            seen.add(rp)
            try:
                rec = json.loads(Path(fp).read_text())
            except Exception:
                continue
            if "samples_per_sec" not in rec:
                continue  # not a bench JSON (e.g. a report file)
            rec["_machine"] = _machine_from_name(Path(fp))
            rec["_file"] = fp
            by_machine[rec["_machine"]].append(rec)
    return by_machine


def gpu_label(recs: list[dict]) -> str:
    names = {r.get("gpu_name") for r in recs if r.get("gpu_name")}
    return sorted(names)[0] if names else "?"


def scaling_points(recs: list[dict]) -> list[dict]:
    """Configs forming the scaling curve: fixed (modal) per-GPU batch, vary GPUs."""
    if not recs:
        return []
    batches = [r["batch_size_per_gpu"] for r in recs]
    modal_b = max(set(batches), key=batches.count)
    pts = [r for r in recs if r["batch_size_per_gpu"] == modal_b]
    # dedupe by world_size (keep fastest if a config ran twice)
    best: dict[int, dict] = {}
    for r in pts:
        w = r["world_size"]
        if w not in best or r["samples_per_sec"] > best[w]["samples_per_sec"]:
            best[w] = r
    return sorted(best.values(), key=lambda r: r["world_size"])


def batch_points(recs: list[dict]) -> list[dict]:
    """Single-GPU configs across batch sizes (memory-headroom probe)."""
    pts = [r for r in recs if r["world_size"] == 1]
    best: dict[int, dict] = {}
    for r in pts:
        b = r["batch_size_per_gpu"]
        if b not in best or r["samples_per_sec"] > best[b]["samples_per_sec"]:
            best[b] = r
    return sorted(best.values(), key=lambda r: r["batch_size_per_gpu"])


def fmt(x, nd=1):
    return "—" if x is None else f"{x:,.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", action="append", required=True,
                    help="Directory of bench JSONs (repeatable).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_machine = load_results(args.results_dir)
    if not by_machine:
        raise SystemExit(f"No bench JSONs found under {args.results_dir}")

    machines = sorted(by_machine)
    md: list[str] = ["# SaddleFlow MP20Bat training throughput & scaling\n"]
    md.append(f"_Model: production mp20bat config (UMA-S-1.2 full unfreeze, 4-block "
              f"time-FiLM, bf16). Each point times the real training step for a fixed "
              f"window after warmup._\n")

    # ---- Per-GPU headline table --------------------------------------------
    # Best single-GPU operating point across ALL benched batch sizes. This is the
    # fair "what one GPU can do" number — a GPU that keeps speeding up with batch
    # (GH200) should be credited at its best batch, not pinned to the small one.
    md.append("## Per-GPU throughput (best single-GPU operating point, across batch sizes)\n")
    md.append("| machine | GPU | best batch | samples/s | ms/step | peak mem (GiB) |")
    md.append("|---|---|--:|--:|--:|--:|")
    per_gpu_best: dict[str, dict] = {}
    for m in machines:
        singles = [r for r in by_machine[m] if r["world_size"] == 1]
        one = (max(singles, key=lambda r: r["samples_per_sec"]) if singles
               else min(by_machine[m], key=lambda r: r["world_size"]))
        per_gpu_best[m] = one
        md.append(f"| {m} | {one.get('gpu_name','?')} | {one['batch_size_per_gpu']} | "
                  f"{fmt(one['samples_per_sec'])} | {fmt(one.get('mean_ms_per_step'))} | "
                  f"{fmt(one.get('peak_mem_alloc_gib'),2)} |")
    md.append("")
    # speedup line if exactly two machines
    if len(machines) == 2:
        a, b = machines
        ra, rb = per_gpu_best[a]["samples_per_sec_per_gpu"], per_gpu_best[b]["samples_per_sec_per_gpu"]
        hi, lo = (a, b) if ra >= rb else (b, a)
        md.append(f"**Per-GPU speedup (best operating point):** {hi} is "
                  f"**{max(ra,rb)/max(1e-9,min(ra,rb)):.2f}×** faster per GPU than {lo} "
                  f"(at batch {per_gpu_best[hi]['batch_size_per_gpu']} vs "
                  f"{per_gpu_best[lo]['batch_size_per_gpu']}).\n")

    # ---- Scaling table ------------------------------------------------------
    md.append("## Strong/weak scaling (fixed per-GPU batch)\n")
    for m in machines:
        sp = scaling_points(by_machine[m])
        if not sp:
            continue
        base = sp[0]
        base_per_gpu = base["samples_per_sec_per_gpu"]
        md.append(f"### {m} — {gpu_label(by_machine[m])}\n")
        md.append("| GPUs | nodes | global batch | samples/s | samples/s/GPU | "
                  "parallel eff. | ms/step | peak mem (GiB) |")
        md.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
        for r in sp:
            eff = r["samples_per_sec_per_gpu"] / base_per_gpu if base_per_gpu else 0
            md.append(f"| {r['world_size']} | {r.get('num_nodes','—')} | {r['global_batch']} | "
                      f"{fmt(r['samples_per_sec'])} | {fmt(r['samples_per_sec_per_gpu'])} | "
                      f"{eff*100:.0f}% | {fmt(r.get('mean_ms_per_step'))} | "
                      f"{fmt(r.get('peak_mem_alloc_gib'),2)} |")
        md.append("")

    # ---- Memory-headroom (batch sweep), if present --------------------------
    for m in machines:
        bp = batch_points(by_machine[m])
        if len(bp) > 1:
            md.append(f"## Memory headroom — {m} (1 GPU, batch sweep)\n")
            md.append("| batch | samples/s | samples/s/GPU | peak mem (GiB) |")
            md.append("|--:|--:|--:|--:|")
            for r in bp:
                md.append(f"| {r['batch_size_per_gpu']} | {fmt(r['samples_per_sec'])} | "
                          f"{fmt(r['samples_per_sec_per_gpu'])} | "
                          f"{fmt(r.get('peak_mem_alloc_gib'),2)} |")
            md.append("")

    # ---- Projected full mp20bat production wall-clock -----------------------
    md.append("## Projected full mp20bat training wall-clock\n")
    # Apples-to-apples: project at the largest GPU count benched on ALL machines,
    # so a machine isn't credited just for having run more GPUs.
    ws_sets = [set(r["world_size"] for r in scaling_points(by_machine[m])) for m in machines]
    common = set.intersection(*ws_sets) if ws_sets else set()
    proj_n = max(common) if common else None
    prod: dict[str, float] = {}
    if proj_n is not None:
        md.append(f"_Production run = {PROD_EPOCHS} epochs × {PROD_RECORDS_PER_EPOCH:,} "
                  f"records = {PROD_SAMPLES:,} sample-presentations. Projected at "
                  f"**{proj_n} GPU** (the largest count benched on every machine) for an "
                  f"apples-to-apples comparison; wall-clock = samples ÷ samples/s._\n")
        md.append(f"| machine | config | samples/s | projected wall-clock |")
        md.append("|---|---|--:|--:|")
        for m in machines:
            r = next((x for x in scaling_points(by_machine[m]) if x["world_size"] == proj_n), None)
            if r is None:
                continue
            hrs = PROD_SAMPLES / max(1e-9, r["samples_per_sec"]) / 3600.0
            prod[m] = hrs
            md.append(f"| {m} | {proj_n} GPU (global batch {r['global_batch']}) | "
                      f"{fmt(r['samples_per_sec'])} | {hrs:.2f} h |")
        md.append("")
        if len(prod) == 2:
            a, b = list(prod)
            fast, slow = (a, b) if prod[a] <= prod[b] else (b, a)
            md.append(f"**{fast}** finishes the full run **{prod[slow]/prod[fast]:.2f}×** "
                      f"faster than **{slow}** at {proj_n} GPU.\n")

    # ---- Grace-Blackwell (Horizon) projection ------------------------------
    md.append("## Grace-Blackwell (Horizon) projection — extrapolated, not measured\n")
    md.append("_Per-GPU GB200-vs-GH200 ratios from published dense specs. Real "
              "kernels fall between the memory-bandwidth bound (≈2.0×) and the "
              "dense-bf16 compute bound (≈2.25×); FP4 via 2nd-gen Transformer "
              "Engine could add up to ≈2× more **if** the eSCN/UMA kernels adopt "
              "it (they run bf16 today, so treat FP4 as upside, not baseline)._\n")
    hopper = None
    for cand in ("vista", "gh200"):
        for m in machines:
            if cand in m.lower() or "gh200" in gpu_label(by_machine[m]).lower():
                hopper = m
                break
        if hopper:
            break
    if hopper:
        h = per_gpu_best[hopper]["samples_per_sec_per_gpu"]
        lo = h * GB200_OVER_GH200["hbm_bandwidth"]
        hi = h * GB200_OVER_GH200["bf16_dense_tflops"]
        fp4 = hi * GB200_OVER_GH200["fp4_vs_bf16_bonus"]
        md.append(f"Measured Hopper (GH200) per-GPU throughput: **{fmt(h)} samples/s**.\n")
        md.append(f"- Projected GB200 per-GPU (bf16): **{fmt(lo)}–{fmt(hi)} samples/s** "
                  f"({GB200_OVER_GH200['hbm_bandwidth']:.2f}–{GB200_OVER_GH200['bf16_dense_tflops']:.2f}×).")
        md.append(f"- With FP4 kernels (upside): up to **{fmt(fp4)} samples/s** per GPU.")
        md.append(f"- Plus 2nd-gen NVLink (NVL72 all-to-all domain) → higher parallel "
                  f"efficiency than Vista's per-node NDR IB at the same GPU count.\n")
    else:
        md.append("_No Hopper/Vista result present yet — add Vista JSONs and re-run "
                  "to populate this section._\n")

    md.append("\n---\n_Generated by examples/scaling_bench/analyze.py_\n")
    (out / "summary.md").write_text("\n".join(md))
    print(f"[analyze] wrote {out/'summary.md'}")

    # ---- Plots --------------------------------------------------------------
    _plot_scaling(by_machine, out)
    _plot_per_gpu(by_machine, per_gpu_best, out)
    print(f"[analyze] wrote {out/'scaling.png'} and {out/'per_gpu_throughput.png'}")


def _plot_scaling(by_machine, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.cm.tab10.colors
    for i, m in enumerate(sorted(by_machine)):
        sp = scaling_points(by_machine[m])
        if not sp:
            continue
        g = np.array([r["world_size"] for r in sp], float)
        s = np.array([r["samples_per_sec"] for r in sp], float)
        base_pg = sp[0]["samples_per_sec_per_gpu"]
        eff = np.array([r["samples_per_sec_per_gpu"] / base_pg for r in sp]) * 100
        c = colors[i % len(colors)]
        lab = f"{m} ({gpu_label(by_machine[m])})"
        ax1.plot(g, s, "o-", color=c, label=lab)
        ideal = s[0] * (g / g[0])
        ax1.plot(g, ideal, "--", color=c, alpha=0.4)
        ax2.plot(g, eff, "o-", color=c, label=lab)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log", base=2)
    ax1.set_xlabel("GPUs"); ax1.set_ylabel("samples / s")
    ax1.set_title("Throughput vs GPUs (dashed = ideal linear)")
    ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=8)
    ax2.axhline(100, color="grey", ls=":", alpha=0.6)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("GPUs"); ax2.set_ylabel("parallel efficiency (%)")
    ax2.set_title("Parallel efficiency (vs single GPU)")
    ax2.set_ylim(0, 115); ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=150)
    plt.close(fig)


def _plot_per_gpu(by_machine, per_gpu_best, out):
    machines = sorted(per_gpu_best)
    if not machines:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(machines) + 2.5, 4.5))
    vals = [per_gpu_best[m]["samples_per_sec_per_gpu"] for m in machines]
    labs = [f"{m}\n{per_gpu_best[m].get('gpu_name','?')}" for m in machines]
    bars = ax.bar(labs, vals, color=plt.cm.tab10.colors[:len(machines)])
    for b, m in zip(bars, machines):
        mem = per_gpu_best[m].get("peak_mem_alloc_gib")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():,.0f}\n{mem:.1f} GiB" if mem else f"{b.get_height():,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("samples / s per GPU")
    ax.set_title("Single-GPU training throughput (peak mem annotated)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "per_gpu_throughput.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
