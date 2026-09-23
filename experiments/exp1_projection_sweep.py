"""Exp1: sweep projection/encoder families on all synthetic tasks.

The core comparison: every method runs through the same pipeline
(raw features -> projection -> OnlineHD prototype classifier, D=4096) and is
scored on in-distribution accuracy, domain-shift OOD accuracy, novel-class
discovery, feature-noise robustness, model bit-flip robustness, geometry
fidelity, and resource cost.
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

from experiments.specs import TASKS_VECTOR, core_specs
from modules.bench import BenchConfig, ExtractorCache, get_task, run_spec

RESULTS = Path(__file__).resolve().parents[1] / "results"


def ood_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("acc_shift")]


def plot_bars(df: pd.DataFrame, out_png: Path) -> None:
    tasks = sorted(df.task.unique())
    order = df.groupby("spec").id_acc.mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), squeeze=False)
    for ax, task in zip(axes.ravel(), tasks):
        sub = df[df.task == task]
        m = sub.groupby("spec").id_acc.mean().reindex(order)
        s = sub.groupby("spec").id_acc.std().reindex(order)
        ax.barh(np.arange(len(m)), m.values, xerr=s.values, color="#4878cf", alpha=0.9)
        ax.set_yticks(np.arange(len(m)), m.index, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{task} (ID accuracy)", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ood_novel(df: pd.DataFrame, out_png: Path) -> None:
    has_shift = (df.task == "shift").any()
    has_novel = (df.task == "novel").any()
    if not (has_shift or has_novel):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (present, title, ylabel, ycol) in zip(axes, [
            (has_shift, "(a) domain shift", "avg domain-shift OOD accuracy", "ood"),
            (has_novel, "(b) novel-class discovery", "novel-class NMI", "nmi")]):
        if not present:
            ax.axis("off")
            continue
        task = "shift" if ycol == "ood" else "novel"
        sub = df[df.task == task].copy()
        if ycol == "ood":
            sub["ood"] = sub[ood_cols(sub)].mean(axis=1)
        m = sub.groupby("spec")[["id_acc", ycol]].mean()
        ax.scatter(m.id_acc, m[ycol], s=28)
        for name, r in m.iterrows():
            ax.annotate(name, (r.id_acc, r[ycol]), fontsize=6.5, alpha=0.85)
        ax.set_xlabel("ID accuracy")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_robustness(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    m = df.groupby("spec")[["id_acc", "acc_noise_0.5", "acc_flip_0.05"]].mean()
    for ax, col, xlabel in ((axes[0], "acc_noise_0.5", "accuracy @ feature noise sigma=0.5"),
                            (axes[1], "acc_flip_0.05", "accuracy @ 5% prototype bit flips")):
        ax.scatter(m.id_acc, m[col], s=28)
        for name, r in m.iterrows():
            ax.annotate(name, (r.id_acc, r[col]), fontsize=6.5, alpha=0.85)
        lim = [0, 1.02]
        ax.plot(lim, lim, ls=":", color="gray", lw=1)
        ax.set_xlabel("ID accuracy (clean)")
        ax.set_ylabel(xlabel)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", nargs="+", default=TASKS_VECTOR)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset of spec names (default: all)")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    specs = core_specs()
    if args.methods:
        specs = [s for s in specs if s.name in args.methods]
    cfg = BenchConfig(dim=args.dim, epochs=args.epochs)
    cache = ExtractorCache(cfg.device)
    RESULTS.mkdir(exist_ok=True)

    rows = []
    t0 = time.time()
    for task_name in args.tasks:
        for seed in range(args.seeds):
            task = get_task(task_name, seed)
            for spec in specs:
                row = run_spec(spec, task, seed, cfg, cache)
                rows.append(row)
                extra = ""
                ood = [row[c] for c in row if c.startswith("acc_shift")]
                if ood:
                    extra = f" ood={np.mean(ood):.3f}"
                if "nmi" in row:
                    extra += f" nmi={row['nmi']:.3f} auroc={row['auroc']:.3f}"
                print(f"[{task_name} s{seed}] {spec.name:18s} id={row['id_acc']:.3f} "
                      f"noise={row.get('acc_noise_0.5', float('nan')):.3f} "
                      f"flip={row.get('acc_flip_0.05', float('nan')):.3f}{extra}",
                      flush=True)
        print(f"--- {task_name}: {time.time() - t0:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / f"exp1_projection_sweep{args.tag}.csv"
    df.to_csv(csv, index=False)
    piv = df.pivot_table(index="spec", columns="task", values="id_acc", aggfunc="mean")
    print("\nID accuracy (mean over seeds):")
    print(piv.round(3).to_string())
    if args.seeds >= 1 and len(specs) > 3:
        plot_bars(df, RESULTS / f"fig_exp1_bars{args.tag}.png")
        plot_ood_novel(df, RESULTS / f"fig_exp1_ood_novel{args.tag}.png")
        if "acc_noise_0.5" in df.columns:
            plot_robustness(df, RESULTS / f"fig_exp1_robustness{args.tag}.png")
    print(f"\nsaved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
