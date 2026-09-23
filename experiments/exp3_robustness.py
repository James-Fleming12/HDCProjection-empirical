"""Exp3: robustness curves.

Three corruption models, all applied at inference to a cleanly trained model:

* feature noise    x + sigma * N(0, I)  (input-space robustness)
* feature dropout  zero each input feature independently with prob p
* model bit flips  each stored prototype coordinate flips sign with prob p

Reported per method x task, plus the interpolated corruption level at which
accuracy falls below 90% of its clean value (a half-life-like summary).
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
from modules.bench import BenchConfig, ExtractorCache, Spec, get_task, run_spec

RESULTS = Path(__file__).resolve().parents[1] / "results"
NOISE = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
DROPS = (0.1, 0.3, 0.5)
FLIPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)


def robustness_specs() -> list[Spec]:
    specs = sweep_specs()
    specs += [
        Spec(name="cauchy_l1", family="linear", kwargs=dict(rp="cauchy"), out="real",
             sim="l1"),
        Spec(name="gaussian_l1", family="linear", kwargs=dict(rp="gaussian"), out="real",
             sim="l1"),
        Spec(name="nystrom_gauss", family="nystrom", kwargs=dict(kernel="gaussian")),
    ]
    return specs


def crossing(levels: list[float], accs: list[float], clean: float, frac: float = 0.9) -> float:
    """Interpolated level where accuracy first drops below frac * clean.

    The clean point is prepended at level 0 so curves that are already below
    target at the first measured corruption level interpolate correctly.
    """
    xs = [0.0] + list(levels)
    ys = [clean] + list(accs)
    target = frac * clean
    for i in range(1, len(xs)):
        if ys[i] < target <= ys[i - 1]:
            if ys[i - 1] <= ys[i]:
                return float(xs[i - 1])
            w = (target - ys[i]) / (ys[i - 1] - ys[i])
            return float(xs[i - 1] + w * (xs[i] - xs[i - 1]))
    return float(xs[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", nargs="+", default=["linear", "fine", "heavy_tail", "shift"])
    ap.add_argument("--dim", type=int, default=4096)
    args = ap.parse_args()

    specs = robustness_specs()
    cfg = BenchConfig(dim=args.dim, epochs=30, noise_sigmas=NOISE, drop_ps=DROPS, flip_ps=FLIPS)
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
    csv = RESULTS / "exp3_robustness.csv"
    df.to_csv(csv, index=False)

    # summary: corruption level at which accuracy hits 90% of clean
    summary = []
    for (spec, task), sub in df.groupby(["spec", "task"]):
        row = {"spec": spec, "task": task,
               "clean": sub.id_acc.mean()}
        for kind, levels in (("noise", NOISE), ("drop", DROPS), ("flip", FLIPS)):
            cols = [f"acc_{kind}_{l:g}" for l in levels]
            row[f"{kind}_p90"] = float(np.mean([
                crossing(list(levels), [r[c] for c in cols], r["id_acc"])
                for _, r in sub.iterrows()]))
        summary.append(row)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(RESULTS / "exp3_robustness_summary.csv", index=False)

    fig, axes = plt.subplots(len(args.tasks), 3, figsize=(15, 3.2 * len(args.tasks)),
                             squeeze=False)
    agg = df.groupby(["spec", "task"]).mean(numeric_only=True).reset_index()
    for i, task in enumerate(args.tasks):
        sub = agg[agg.task == task]
        for ax, kind, levels, xlabel in (
                (axes[i][0], "noise", NOISE, "feature noise sigma"),
                (axes[i][1], "drop", DROPS, "feature dropout p"),
                (axes[i][2], "flip", FLIPS, "prototype bit-flip p")):
            cols = [f"acc_{kind}_{l:g}" for l in levels]
            for _, r in sub.iterrows():
                ax.plot(levels, [r[c] for c in cols], marker="o", ms=3, label=r["spec"])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("accuracy")
            ax.set_title(f"{task}: {kind}")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exp3_robustness.png", dpi=150)
    plt.close(fig)
    print(sdf.round(3).to_string(index=False))
    print(f"saved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
