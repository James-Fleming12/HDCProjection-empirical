"""Exp8: geometry / theory validation for projection matrices.

Sections:
  (a) Johnson-Lindenstrauss distortion of pairwise L2 distances vs D
  (b) L1 vs L2 rank preservation (Cauchy should preserve L1, Gaussian L2)
  (c) kernel-approximation error of RFF / SignRFF / FPE / Nystrom vs D
  (d) bandwidth effects: kernel error and ID/OOD accuracy
  (e) cross-links with exp1: geometry fidelity / margins vs accuracy and OOD
  (f) row incoherence (max off-diagonal |cos|) of each construction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.bench import BenchConfig, Spec, get_task, run_spec
from modules.data import make_task
from modules.encoders import FPEEncoder, NystromEncoder, RFFEncoder
from modules.metrics import accuracy, jl_distortion, kernel_mse
from modules.rp import RP_NAMES, make_rp, rp_stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
DIMS = [64, 128, 256, 512, 1024, 2048, 4096]


def section_a(out_dir: Path) -> pd.DataFrame:
    task = make_task("linear", seed=0)
    x = task.x_train[:2500]
    rows = []
    for rp in RP_NAMES:
        for D in DIMS:
            P = make_rp(rp, D, 64, seed=0)
            st = jl_distortion(P, x, n_pairs=2000, seed=0)
            rows.append(dict(rp=rp, D=D, **st,
                             ratio=st["jl_std_rel"] / st["jl_pred_std"]))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp8_jl.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for rp, sub in df.groupby("rp"):
        ax.plot(sub.D, sub.jl_std_rel, marker="o", ms=3, label=rp)
    ref = df[df.rp == "gaussian"]
    ax.plot(ref.D, np.sqrt(2 / ref.D), "k--", lw=1.5, label="sqrt(2/D) prediction")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel("std of relative L2 distance error")
    ax.set_title("(a) JL distortion vs dimension")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp8_jl.png", dpi=150)
    plt.close(fig)
    return df


def section_b(out_dir: Path) -> pd.DataFrame:
    task = make_task("linear", seed=0)
    x = task.x_train[:2500]
    g = torch.Generator().manual_seed(0)
    n = x.shape[0]
    i = torch.randint(0, n, (3000,), generator=g)
    j = torch.randint(0, n, (3000,), generator=g)
    diff = x[i] - x[j]
    l1 = diff.abs().sum(dim=1)
    l2 = diff.norm(dim=1)
    rows = []
    for rp in ("gaussian", "cauchy", "laplace", "student_t", "rademacher"):
        for D in (512, 4096):
            P = make_rp(rp, D, 64, seed=0)
            y = x @ P.T
            dy = y[i] - y[j]
            est_l1 = dy.abs().median(dim=1).values  # Cauchy scale estimator
            est_l2 = dy.norm(dim=1)
            rows.append(dict(
                rp=rp, D=D,
                rho_l1_l1=spearmanr(est_l1.numpy(), l1.numpy()).statistic,
                rho_l2_l2=spearmanr(est_l2.numpy(), l2.numpy()).statistic,
                rho_l1_l2=spearmanr(est_l1.numpy(), l2.numpy()).statistic,
                rho_l2_l1=spearmanr(est_l2.numpy(), l1.numpy()).statistic,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp8_l1_l2.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, D in zip(axes, (512, 4096)):
        sub = df[df.D == D].set_index("rp")
        sub[["rho_l2_l2", "rho_l1_l1"]].plot.bar(ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"D={D}: rank preservation (L2 vs L1)")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp8_l1_l2.png", dpi=150)
    plt.close(fig)
    return df


def section_c(out_dir: Path, dims=DIMS) -> pd.DataFrame:
    task = make_task("linear", seed=0)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(task.x_train.shape[0], generator=g)[:1500]
    x = task.x_train[idx]

    rows = []
    for D in dims:
        encs = {
            "rff_real": (RFFEncoder(D, 64, bw_scale=1.0, out="real", seed=0), False),
            "signrff": (RFFEncoder(D, 64, bw_scale=1.0, out="bipolar", seed=0), True),
            "fpe_phasor": (FPEEncoder(D, 64, bw_scale=1.0, seed=0), True),
            "nystrom_gauss": (NystromEncoder(D, 64, bw_scale=1.0, kernel="gaussian", seed=0), True),
        }
        for name, (enc, norm) in encs.items():
            enc.fit(task.x_train)
            sigma2 = enc.sigma2 if hasattr(enc, "sigma2") else enc.scale
            kern = lambda a, b, s=sigma2: torch.exp(-((a - b) ** 2).sum(dim=1) / (2 * s))
            mse = kernel_mse(enc.encode, x, kern, n_pairs=1500, seed=0, normalize=norm)
            rows.append(dict(method=name, D=D, kernel_mse=mse))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp8_kernel.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, sub in df.groupby("method"):
        ax.plot(sub.D, sub.kernel_mse, marker="o", ms=3, label=name)
    ref = df[df.method == "rff_real"]
    ax.plot(ref.D, ref.kernel_mse.iloc[0] * ref.D.iloc[0] / ref.D, "k--", lw=1,
            label="1/D reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel("MSE vs Gaussian kernel")
    ax.set_title("(c) kernel approximation vs dimension")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp8_kernel.png", dpi=150)
    plt.close(fig)
    return df


def section_d(out_dir: Path, seeds: int = 3) -> pd.DataFrame:
    """Bandwidth effects on kernel error and ID/OOD accuracy (shift task)."""
    bws = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    rows = []
    for seed in range(seeds):
        task = get_task("shift", seed)
        cfg = BenchConfig(dim=2048, epochs=30, robustness=False)
        for bw in bws:
            for out in ("real", "bipolar"):
                spec = Spec(name=f"rff_{out}_b{bw}", family="rff", out=out,
                            kwargs=dict(bw_scale=bw))
                row = run_spec(spec, task, seed, cfg)
                row["bw"] = bw
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp8_bandwidth.csv", index=False)
    agg = df.groupby(["spec", "bw"]).mean(numeric_only=True).reset_index()
    agg["ood"] = agg[["acc_shift_6", "acc_shift_12", "acc_shift_18"]].mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, ycol, title in ((axes[0], "id_acc", "ID accuracy"),
                            (axes[1], "ood", "avg domain-shift OOD")):
        for spec_name, sub in agg.groupby("spec"):
            ax.plot(sub.bw, sub[ycol], marker="o", ms=3, label=spec_name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bandwidth scale (x median distance)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp8_bandwidth.png", dpi=150)
    plt.close(fig)
    return df


def section_e(out_dir: Path) -> pd.DataFrame | None:
    csv = out_dir / "exp1_projection_sweep.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    rows = []
    for metric, target in (("margin_id", "id_acc"), ("margin_id", "acc_noise_0.5"),
                           ("margin_ood", "acc_shift_12"),
                           ("geom_fid", "id_acc")):
        idx = df[df.task.isin(["shift", "linear", "fine"])].dropna(subset=[metric, target])
        rho, p = spearmanr(idx[metric], idx[target])
        rows.append(dict(metric=metric, target=target, spearman=float(rho), p=float(p)))
    # per-task geometry fidelity vs OOD
    sub = df[df.task == "shift"].copy()
    sub["ood"] = sub[["acc_shift_6", "acc_shift_12", "acc_shift_18"]].mean(axis=1)
    rho, p = spearmanr(sub.geom_fid, sub.ood)
    rows.append(dict(metric="geom_fid", target="avg_ood_shift", spearman=float(rho), p=float(p)))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "exp8_crosslinks.csv", index=False)
    return out


def section_f(out_dir: Path) -> pd.DataFrame:
    rows = []
    for rp in RP_NAMES:
        P = make_rp(rp, 1024, 64, seed=0)
        Pn = P / P.norm(dim=1, keepdim=True).clamp_min(1e-12)
        G = (Pn @ Pn.T).abs()
        mask = ~torch.eye(G.shape[0], dtype=torch.bool)
        rows.append(dict(rp=rp, max_offdiag_cos=float(G[mask].max().item()),
                         mean_offdiag_cos=float(G[mask].mean().item()),
                         **{k: v for k, v in rp_stats(P).items()}))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp8_incoherence.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=["a", "b", "c", "d", "e", "f"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if "a" in args.sections:
        a = section_a(RESULTS)
        print("(a) JL: mean |ratio| per family:")
        print(a.groupby("rp").ratio.mean().round(2).sort_values().to_string())
    if "b" in args.sections:
        print("\n(b) L1/L2 rank preservation:")
        print(section_b(RESULTS).round(3).to_string(index=False))
    if "c" in args.sections:
        print("\n(c) kernel MSE vs D:")
        print(section_c(RESULTS).pivot(index="method", columns="D", values="kernel_mse")
              .round(4).to_string())
    if "d" in args.sections:
        d = section_d(RESULTS, seeds=args.seeds)
        d["ood"] = d[["acc_shift_6", "acc_shift_12", "acc_shift_18"]].mean(axis=1)
        print("\n(d) RFF bandwidth: ID/OOD")
        print(d.groupby(["spec", "bw"])[["id_acc", "ood"]].mean().round(3).to_string())
    if "e" in args.sections:
        e = section_e(RESULTS)
        if e is not None:
            print("\n(e) cross-links (Spearman):")
            print(e.round(3).to_string(index=False))
    if "f" in args.sections:
        print("\n(f) incoherence:")
        print(section_f(RESULTS)[["rp", "max_offdiag_cos", "density", "rank"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
