"""Exp2: sweep hyperspace dimension D for representative methods.

Key question: do projection families behave differently in the *compression*
regime (D < F, where structured/sketching projections were designed to work)
and the *expansion* regime (D > F, the usual HDC operating point)?
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.specs import sweep_specs
from modules.bench import BenchConfig, ExtractorCache, get_task, run_spec

RESULTS = Path(__file__).resolve().parents[1] / "results"
DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", nargs="+", default=["linear", "fine", "shells", "xor", "shift"])
    ap.add_argument("--dims", type=int, nargs="+", default=DIMS)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    specs = sweep_specs()
    cache = ExtractorCache()
    rows = []
    t0 = time.time()
    for task_name in args.tasks:
        for seed in range(args.seeds):
            task = get_task(task_name, seed)
            for D in args.dims:
                cfg = BenchConfig(dim=D, epochs=args.epochs, robustness=False)
                for spec in specs:
                    row = run_spec(spec, task, seed, cfg, cache)
                    rows.append(row)
            print(f"[{task_name} s{seed}] done in {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / "exp2_dimension_sweep.csv"
    df.to_csv(csv, index=False)

    # curves: 2 x len(tasks) grid
    tasks = list(dict.fromkeys(args.tasks))
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.6 * len(tasks), 7.5), squeeze=False)
    agg = df.groupby(["spec", "task", "dim"]).mean(numeric_only=True).reset_index()
    for j, task in enumerate(tasks):
        sub = agg[agg.task == task]
        for spec_name, s in sub.groupby("spec"):
            axes[0][j].plot(s.dim, s.id_acc, marker="o", ms=3, label=spec_name)
            col = "acc_shift_12" if "acc_shift_12" in s else None
            if col is not None:
                axes[1][j].plot(s.dim, s[col], marker="o", ms=3, label=spec_name)
        axes[0][j].axvline(64, ls=":", color="gray", lw=1)
        axes[0][j].set_xscale("log", base=2)
        axes[0][j].set_title(f"{task}: ID")
        axes[1][j].axvline(64, ls=":", color="gray", lw=1)
        axes[1][j].set_xscale("log", base=2)
        axes[1][j].set_title(f"{task}: " + ("shift-12 OOD" if task == "shift" else "ID (dup)"))
        if task != "shift":
            axes[1][j].plot(sub.dim, sub.id_acc, alpha=0)
    axes[0][0].set_ylabel("accuracy")
    axes[1][0].set_ylabel("accuracy")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.set_xlabel("D")
    axes[0][0].legend(fontsize=6, ncol=1)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exp2_dimension.png", dpi=150)
    plt.close(fig)
    print(f"saved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
