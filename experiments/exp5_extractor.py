"""Exp5: feature-extractor axes in front of the projection.

Extractors: identity (raw), frozen random MLP, CE-trained MLP, metric-learning
variants (SupCon / proxy-anchor / angular-margin), k-means weight clustering,
and co-training of the extractor with the HDC prototype objective (fixed
random projection, straight-through estimator).

Projections: Gaussian RP, tuned SignRFF, learned projection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.bench import BenchConfig, ExtractorCache, Spec, get_task, run_spec

RESULTS = Path(__file__).resolve().parents[1] / "results"
TASKS = ["linear", "fine", "heavy_tail", "shift"]

EXTRACTORS = {
    "raw": ("raw", {}),
    "randmlp": ("randmlp", {}),
    "ce": ("ce", {}),
    "supcon": ("supcon", {}),
    "proxy": ("proxy", {}),
    "arc": ("arc", {}),
    "cluster4": ("clustered", dict(levels=4)),
    "cluster8": ("clustered", dict(levels=8)),
    "ce_wide": ("ce", dict(hidden=(256,))),
    "ce_deep": ("ce", dict(hidden=(128, 128))),
    "cotrain": ("cotrain", dict(out_dim=128, epochs=60)),
    "cotrain_learnP": ("cotrain", dict(out_dim=128, epochs=60, learn_proj=True)),
}

PROJECTIONS = {
    "gauss": dict(family="linear", kwargs=dict(rp="gaussian")),
    "signrff": dict(family="rff", kwargs=dict(bw_select="val")),
    "learned": dict(family="learned", kwargs=dict(epochs=60)),
}


def build_specs() -> list[Spec]:
    specs = []
    for ename, (kind, ekw) in EXTRACTORS.items():
        for pname, p in PROJECTIONS.items():
            name = f"{ename}+{pname}"
            specs.append(Spec(name=name, family=p["family"], kwargs=dict(p["kwargs"]),
                              extractor=kind, ext_kwargs=dict(ekw)))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()

    specs = build_specs()
    cfg = BenchConfig(dim=args.dim, epochs=30)
    cache = ExtractorCache()
    rows = []
    t0 = time.time()
    for task_name in args.tasks:
        for seed in range(args.seeds):
            task = get_task(task_name, seed)
            for spec in specs:
                rows.append(run_spec(spec, task, seed, cfg, cache))
            print(f"[{task_name} s{seed}] {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / "exp5_extractor.csv"
    df.to_csv(csv, index=False)

    agg = df.groupby(["spec", "task"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(5.2 * len(args.tasks), 6),
                             squeeze=False)
    for j, task in enumerate(args.tasks):
        sub = agg[agg.task == task].copy()
        order = sub.sort_values("id_acc", ascending=True)
        axes[0][j].barh(order.spec, order.id_acc, color="#4878cf")
        axes[0][j].set_xlim(0, 1.02)
        axes[0][j].set_title(f"{task}: ID accuracy")
        axes[0][j].grid(alpha=0.3, axis="x")
        axes[0][j].tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exp5_extractor.png", dpi=150)
    plt.close(fig)

    print("\nID accuracy by extractor+projection:")
    print(df.pivot_table(index="spec", columns="task", values="id_acc").round(3).to_string())
    print(f"\nsaved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
