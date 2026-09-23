"""Build compact markdown summary tables from the experiment CSVs.

Run after the experiments: `python experiments/make_summaries.py`.
Writes results/summary_*.md and prints them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TASKS_V = ["linear", "fine", "xor", "shells", "heavy_tail", "shift", "novel"]


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def read(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def save(name: str, text: str) -> None:
    (RESULTS / name).write_text(text)
    print(f"--- {name} ---")
    print(text)
    print()


def shift_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("acc_shift")]


def exp1() -> None:
    df = read("exp1_projection_sweep.csv")
    if df is None:
        return
    if "vec_bits" not in df.columns:
        df["vec_bits"] = np.nan
    idp = df.pivot_table(index="spec", columns="task", values="id_acc")
    cols = [c for c in TASKS_V if c in idp.columns]
    idp = idp[cols]
    idp["mean"] = idp.mean(axis=1)
    idp = idp.sort_values("mean", ascending=False)

    rob = df[df.task.isin(["linear", "fine", "heavy_tail", "shift"])]
    r = rob.groupby("spec")[["acc_noise_0.5", "acc_drop_0.2", "acc_flip_0.05",
                             "margin_id", "geom_fid"]].mean()
    s = df[df.task == "shift"].groupby("spec")[["acc_shift_6", "acc_shift_12",
                                                "acc_shift_18", "margin_ood"]].mean()
    s["avg_ood"] = s[["acc_shift_6", "acc_shift_12", "acc_shift_18"]].mean(axis=1)
    n = df[df.task == "novel"].groupby("spec")[["nmi", "ari", "auroc"]].mean()
    res = df[df.task == "linear"].groupby("spec")[["mem_kb", "ops_per_query",
                                                   "rp_density", "rp_rank"]].mean()

    out = ["# Exp1 summaries\n", "## ID accuracy\n", md(idp),
           "\n## Robustness / margins (linear+fine+heavy_tail+shift)\n", md(r),
           "\n## Domain shift\n", md(s),
           "\n## Novel-class discovery\n", md(n),
           "\n## Resources (linear task)\n", md(res, "{:.2f}")]
    save("summary_exp1.md", "\n".join(out))


def exp2() -> None:
    df = read("exp2_dimension_sweep.csv")
    if df is None:
        return
    p = df.pivot_table(index="spec", columns=["task", "dim"], values="id_acc")
    save("summary_exp2.md", "# Exp2: ID accuracy by (task, D)\n\n" + md(p, "{:.3f}"))


def exp3() -> None:
    df = read("exp3_robustness_summary.csv")
    if df is None:
        return
    p = df.pivot_table(index="spec", columns="task",
                       values=["noise_p90", "drop_p90", "flip_p90"])
    save("summary_exp3.md", "# Exp3: 90%-of-clean corruption levels\n\n" + md(p, "{:.3f}"))


def exp4() -> None:
    df = read("exp4_quantization.csv")
    if df is None:
        return
    codec = df[~df.spec.str.contains("_p[0-9]|_fp32|proto")]
    ptq = df[df.spec.str.contains("_p[0-9]|_fp32|ternary|_sign")]
    qat = df[df.spec.str.contains("proto")]
    out = ["# Exp4 summaries\n",
           "\n## Output codec: ID accuracy\n",
           md(codec.pivot_table(index="spec", columns="task", values="id_acc")),
           "\n## Output codec: avg-shift OOD\n",
           md(codec[codec.task == "shift"].assign(
               ood=codec[codec.task == "shift"][shift_cols(codec)].mean(axis=1)
           ).pivot_table(index="spec", values="ood")),
           "\n## Projection-matrix PTQ: ID accuracy\n",
           md(ptq.pivot_table(index="spec", columns="task", values="id_acc")),
           "\n## Prototype-bits QAT vs PTQ: ID accuracy\n",
           md(qat.pivot_table(index="spec", columns="task", values="id_acc"))]
    save("summary_exp4.md", "\n".join(out))


def exp5() -> None:
    df = read("exp5_extractor.csv")
    if df is None:
        return
    p = df.pivot_table(index="spec", columns="task", values="id_acc")
    ood = df[df.task == "shift"].assign(
        ood=df[df.task == "shift"][shift_cols(df)].mean(axis=1)
    ).pivot_table(index="spec", values="ood")
    p["shift_ood"] = ood["ood"]
    save("summary_exp5.md", "# Exp5: feature extractors\n\n" + md(p))


def exp6() -> None:
    df = read("exp6_processing.csv")
    if df is None:
        return
    p = df.pivot_table(index="spec", columns="task", values="id_acc")
    save("summary_exp6.md", "# Exp6: processing / classifier variants\n\n" + md(p))


def exp6b() -> None:
    df = read("exp6b_sequence.csv")
    if df is None:
        return
    p = df.pivot_table(index=["mode", "bind"], columns="pos", values="id_acc")
    save("summary_exp6b.md", "# Exp6b: symbolic binding / position encoding\n\n" + md(p))


def exp7() -> None:
    df = read("exp7_learned.csv")
    if df is None:
        return
    p = df.pivot_table(index="spec", columns="task", values="id_acc")
    save("summary_exp7.md", "# Exp7: learned / data-informed projections\n\n" + md(p))


def exp8() -> None:
    parts = []
    for name, title in [("exp8_jl.csv", "JL distortion (std of relative error)"),
                        ("exp8_l1_l2.csv", "L1/L2 rank preservation"),
                        ("exp8_kernel.csv", "kernel-approximation MSE"),
                        ("exp8_crosslinks.csv", "cross-links (Spearman)"),
                        ("exp8_incoherence.csv", "row incoherence")]:
        df = read(name)
        if df is None:
            continue
        if name == "exp8_kernel.csv":
            df = df.pivot(index="method", columns="D", values="kernel_mse").reset_index()
        if name == "exp8_jl.csv":
            df = df.pivot_table(index="rp", columns="D", values="jl_std_rel").reset_index()
        parts.append(f"\n## {title}\n\n" + md(df))
    save("summary_exp8.md", "# Exp8 summaries\n" + "\n".join(parts))


def exp9() -> None:
    df = read("exp9_interactions.csv")
    if df is None:
        return
    parts = []
    for task in df.task.unique():
        p = df[df.task == task].pivot_table(index="proj", columns=["dim", "codec"],
                                            values="id_acc")
        parts.append(f"\n## {task}: ID accuracy\n\n" + md(p))
    save("summary_exp9.md", "# Exp9: projection x codec x dimension\n" + "\n".join(parts))


def main() -> None:
    for fn in (exp1, exp2, exp3, exp4, exp5, exp6, exp6b, exp7, exp8, exp9):
        fn()


if __name__ == "__main__":
    main()
