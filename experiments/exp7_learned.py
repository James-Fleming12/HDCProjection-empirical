"""Exp7: learned / data-informed / input-modulated projections vs random.

At D=1024 (where random RP loses most accuracy) compare:
  * PIONEER-style STE-trained projections (epochs / init / lr)
  * best-of-K data-informed random projection selection
  * input-modulated projection (random gating)
  * co-training of a feature extractor with the HDC objective (fixed vs
    learned projection)
against a Gaussian RP and tuned SignRFF baseline, on all vector tasks.
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
TASKS = ["linear", "fine", "heavy_tail", "xor", "shift"]


def build_specs() -> list[Spec]:
    S = [
        Spec(name="gaussian", family="linear", kwargs=dict(rp="gaussian")),
        Spec(name="signrff_tuned", family="rff", kwargs=dict(bw_select="val")),
    ]
    for tag, kw in {
        "learned_e10": dict(epochs=10),
        "learned_e40": dict(epochs=40),
        "learned_e120": dict(epochs=120),
        "learned_init_rad": dict(epochs=60, init="rademacher"),
        "learned_lr3e-3": dict(epochs=60, lr=3e-3),
        "learned_lr3e-2": dict(epochs=60, lr=3e-2),
    }.items():
        S.append(Spec(name=tag, family="learned", kwargs=kw))
    for K in (1, 4, 16, 64, 256):
        S.append(Spec(name=f"datainformed_k{K}", family="datainformed", kwargs=dict(K=K)))
    for g in (0.25, 0.5, 1.0):
        S.append(Spec(name=f"imp_g{g}", family="imp", kwargs=dict(gain=g)))
    S.append(Spec(name="cotrain_fixedP", family="linear", kwargs=dict(rp="gaussian"),
                  extractor="cotrain", ext_kwargs=dict(out_dim=128, epochs=60)))
    S.append(Spec(name="cotrain_learnP", family="linear", kwargs=dict(rp="gaussian"),
                  extractor="cotrain",
                  ext_kwargs=dict(out_dim=128, epochs=60, learn_proj=True)))
    return S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=1024)
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
    csv = RESULTS / "exp7_learned.csv"
    df.to_csv(csv, index=False)

    agg = df.groupby(["spec", "task"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(4.2 * len(args.tasks), 6),
                             squeeze=False)
    for j, task in enumerate(args.tasks):
        sub = agg[agg.task == task].sort_values("id_acc")
        axes[0][j].barh(sub.spec, sub.id_acc, color="#9a6fb0")
        axes[0][j].set_xlim(0, 1.02)
        axes[0][j].set_title(f"{task}")
        axes[0][j].grid(alpha=0.3, axis="x")
        axes[0][j].tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exp7_learned.png", dpi=150)
    plt.close(fig)

    print("\nID accuracy:")
    print(df.pivot_table(index="spec", columns="task", values="id_acc").round(3).to_string())
    print(f"\nsaved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
