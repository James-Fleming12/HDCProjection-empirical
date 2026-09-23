"""Exp9: projection x codec x dimension interaction.

The interaction tests the hypothesis that the projection family and the
hypervector algebra / output codec are not separable design choices:
a codec that is nearly free under one projection can be destructive under
another (and vice versa).

Heatmaps: ID accuracy (and OOD accuracy on the shift task) for
projection in {Gaussian RP, SignRFF, FPE 1-bit, learned} x
codec in {fp32, int4, ternary, bipolar} x D in {256, 1024, 4096}.
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
DIMS = [256, 1024, 4096]
CODECS = ["", "int4", "ternary", "bipolar"]
TASKS = ["fine", "shift", "novel"]

PROJECTIONS = {
    "gauss": Spec(name="gauss", family="linear", kwargs=dict(rp="gaussian")),
    "signrff": Spec(name="signrff", family="rff", out="real", kwargs=dict(bw_scale=1.0)),
    "fpe_b1": Spec(name="fpe_b1", family="fpe", kwargs=dict(bw_scale=0.25, phase_bits=1)),
    "learned": Spec(name="learned", family="learned", kwargs=dict(epochs=60)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()

    cache = ExtractorCache()
    rows = []
    t0 = time.time()
    for task_name in args.tasks:
        for seed in range(args.seeds):
            task = get_task(task_name, seed)
            for D in DIMS:
                cfg = BenchConfig(dim=D, epochs=30, robustness=False)
                for pname, p in PROJECTIONS.items():
                    for codec in CODECS:
                        name = f"{pname}|{codec or 'fp32'}"
                        spec = Spec(name=name, family=p.family, out=p.out,
                                    kwargs=dict(p.kwargs), codec=codec or None)
                        row = run_spec(spec, task, seed, cfg, cache)
                        row["spec"] = name
                        row["proj"] = pname
                        row["codec"] = codec or "fp32"
                        rows.append(row)
            print(f"[{task_name} s{seed}] {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / "exp9_interactions.csv"
    df.to_csv(csv, index=False)

    agg = df.groupby(["proj", "codec", "dim", "task"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(len(args.tasks), len(DIMS),
                             figsize=(4.6 * len(DIMS), 3.4 * len(args.tasks)), squeeze=False)
    for i, task in enumerate(args.tasks):
        for j, D in enumerate(DIMS):
            sub = agg[(agg.task == task) & (agg.dim == D)]
            mat = sub.pivot(index="proj", columns="codec", values="id_acc").reindex(
                index=list(PROJECTIONS), columns=[c or "fp32" for c in CODECS])
            ax = axes[i][j]
            im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(mat.shape[1]), mat.columns, fontsize=7)
            ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    ax.text(c, r, f"{mat.values[r, c]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if mat.values[r, c] < 0.6 else "black")
            ax.set_title(f"{task} D={D}", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.savefig(RESULTS / "fig_exp9_interactions.png", dpi=150)
    plt.close(fig)

    print("\nID accuracy, D=4096:")
    print(df[df.dim == 4096].pivot_table(index="proj", columns=["codec", "task"],
                                         values="id_acc").round(3).to_string())
    print(f"\nsaved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
