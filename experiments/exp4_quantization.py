"""Exp4: quantization of hypervectors, prototypes and the projection matrix.

Axes:
  (a) output codec       real / int8 / int4 / int2 / bipolar / binary01 /
                         ternary / block-sparse / thermometer / VQ
  (b) projection matrix  fp32 / int8 / int4 / int2 (MSE vs max-scaled) /
                         ternary / sign, applied to Gaussian and SRHT RPs
  (c) quantization-aware training (QuantHD-style) for low-bit prototypes
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.bench import BenchConfig, ExtractorCache, Spec, get_task, run_spec

RESULTS = Path(__file__).resolve().parents[1] / "results"
TASKS = ["fine", "heavy_tail", "shift", "novel"]
CODECS = ["real", "int8", "int4", "int2", "bipolar", "binary01", "ternary",
          "block", "thermo", "vq"]

BASE_ENCODERS = {
    "gauss": Spec(name="", family="linear", kwargs=dict(rp="gaussian")),
    "signrff": Spec(name="", family="rff", out="real", kwargs=dict(bw_scale=0.25)),
    "fpe_b3": Spec(name="", family="fpe", kwargs=dict(bw_scale=0.25, phase_bits=3)),
}


def codec_specs() -> list[Spec]:
    out = []
    for base, spec in BASE_ENCODERS.items():
        for codec in CODECS:
            out.append(Spec(name=f"{base}_{codec}", family=spec.family, out=spec.out,
                            kwargs=dict(spec.kwargs), codec=codec))
    return out


def proj_ptq_specs() -> list[Spec]:
    out = []
    for rp in ("gaussian", "srht"):
        out.append(Spec(name=f"{rp}_fp32", family="linear", kwargs=dict(rp=rp)))
    for rp in ("gaussian", "srht"):
        for bits in (2, 4, 8):
            for scheme in ("mse", "max"):
                out.append(Spec(name=f"{rp}_p{bits}_{scheme}", family="linear",
                                kwargs=dict(rp=rp, proj_bits=bits, proj_scheme=scheme)))
        out.append(Spec(name=f"{rp}_ternary", family="linear",
                        kwargs=dict(rp=rp, proj_bits=2, proj_scheme="ternary")))
        out.append(Spec(name=f"{rp}_sign", family="linear",
                        kwargs=dict(rp=rp, proj_bits=1, proj_scheme="sign")))
    return out


def qat_variants():
    """(label, spec, cfg overrides) for the quantization-aware training study."""
    out = []
    for bits in (2, 3, 4):
        for qat in (False, True):
            label = f"proto{bits}_{'qat' if qat else 'ptq'}"
            out.append((label, Spec(name=label, family="linear", kwargs=dict(rp="gaussian")),
                        dict(proto_bits=bits, quant_train=qat)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--part", nargs="+", default=["codec", "ptq", "qat"])
    args = ap.parse_args()

    specs = []
    if "codec" in args.part:
        specs += codec_specs()
    if "ptq" in args.part:
        specs += proj_ptq_specs()
    variants = qat_variants() if "qat" in args.part else []

    cache = ExtractorCache()
    rows = []
    t0 = time.time()
    for task_name in args.tasks:
        for seed in range(args.seeds):
            task = get_task(task_name, seed)
            cfg = BenchConfig(dim=args.dim, epochs=30)
            for spec in specs:
                rows.append(run_spec(spec, task, seed, cfg, cache))
            for label, spec, over in variants:
                c = replace(cfg, **over)
                row = run_spec(spec, task, seed, c, cache)
                row["spec"] = label
                row["quant_train"] = int(over.get("quant_train", False))
                rows.append(row)
            print(f"[{task_name} s{seed}] {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / "exp4_quantization.csv"
    df.to_csv(csv, index=False)

    agg = df.groupby(["spec", "task"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, len(args.tasks), figsize=(4.2 * len(args.tasks), 8),
                             squeeze=False)
    for j, task in enumerate(args.tasks):
        sub = agg[agg.task == task]
        for part, ax, col in (("codec", axes[0][j], "id_acc"),
                              ("ptq", axes[1][j], "id_acc")):
            if part == "codec":
                s = sub[sub.spec.str.startswith(("gauss_", "signrff_", "fpe_"))]
            else:
                s = sub[sub.spec.str.contains("_p") | sub.spec.str.contains("_fp32")
                        | sub.spec.str.contains("ternary") | sub.spec.str.contains("sign")]
            ax.barh(s.spec, s[col], color="#4878cf")
            ax.set_xlim(0, 1.02)
            ax.set_title(f"{task}: {part}")
            ax.grid(alpha=0.3, axis="x")
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exp4_quantization.png", dpi=150)
    plt.close(fig)

    print("\nOutput codec, ID accuracy (mean over seeds):")
    codec = df[df.spec.str.startswith(("gauss_", "signrff_", "fpe_"))]
    print(codec.pivot_table(index="spec", columns="task", values="id_acc").round(3).to_string())
    print("\nProjection PTQ, ID accuracy:")
    ptq = df[df.spec.str.contains("_p") | df.spec.str.contains("_fp32")]
    print(ptq.pivot_table(index="spec", columns="task", values="id_acc").round(3).to_string())
    print(f"\nsaved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
