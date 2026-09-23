"""Exp6: processing and classifier variants.

Axes:
  similarity      cosine / dot / L1 / RBF / Hamming (on {0,1} codes)
  bundling        single-pass accumulation vs OnlineHD retraining, margin-
                  and confidence-weighted bundling, multi-prototype classes
  prototype bits  fp32 / 8 / 4 / 2 / ternary / sign (with and without QAT)
  preprocessing   none / L2 / standardize / PCA-whiten, for Gaussian RP and
                  tuned SignRFF
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


def gaussian_spec(**kw) -> Spec:
    out = dict(rp="gaussian")
    out.update(kw)
    return Spec(name="", family="linear", kwargs=out)


def variants() -> list[tuple[str, Spec, dict]]:
    """(label, spec, cfg overrides)."""
    V = []
    V.append(("base_cos", gaussian_spec(), {}))
    V.append(("sim_dot", gaussian_spec(), dict(sim="dot")))
    V.append(("sim_l1", gaussian_spec(out="real"), dict(sim="l1")))
    V.append(("sim_rbf", gaussian_spec(out="real"), dict(sim="rbf")))
    V.append(("bin_hamming", gaussian_spec(out="binary01"), dict(sim="hamming")))
    V.append(("bin_cosine", gaussian_spec(out="binary01"), dict(sim="cosine")))
    V.append(("weight_margin", gaussian_spec(), dict(weight="margin")))
    V.append(("weight_conf", gaussian_spec(), dict(weight="confidence")))
    V.append(("multi2", gaussian_spec(), dict(multi=2)))
    V.append(("multi4", gaussian_spec(), dict(multi=4)))
    V.append(("multi8", gaussian_spec(), dict(multi=8)))
    V.append(("single_pass", gaussian_spec(), dict(epochs=0)))
    V.append(("proto8", gaussian_spec(), dict(proto_bits=8)))
    V.append(("proto4", gaussian_spec(), dict(proto_bits=4)))
    V.append(("proto2", gaussian_spec(), dict(proto_bits=2)))
    V.append(("proto_ternary", gaussian_spec(), dict(proto_bits=2, proto_scheme="ternary")))
    V.append(("proto_sign", gaussian_spec(), dict(proto_bits=1, proto_scheme="sign")))
    for pp in ("l2", "std", "whiten"):
        V.append((f"gauss_preproc_{pp}",
                  Spec(name="", family="linear", kwargs=dict(rp="gaussian"), preproc=pp), {}))
        V.append((f"signrff_preproc_{pp}",
                  Spec(name="", family="rff", kwargs=dict(bw_select="val"), preproc=pp), {}))
    V.append(("signrff_base",
              Spec(name="", family="rff", kwargs=dict(bw_select="val")), {}))
    return V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()

    cfg = BenchConfig(dim=args.dim, epochs=30)
    cache = ExtractorCache()
    rows = []
    t0 = time.time()
    for task_name in args.tasks:
        for seed in range(args.seeds):
            task = get_task(task_name, seed)
            for label, spec, over in variants():
                spec.name = label
                c = replace(cfg, **over)
                row = run_spec(spec, task, seed, c, cache)
                row["spec"] = label
                rows.append(row)
            print(f"[{task_name} s{seed}] {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / "exp6_processing.csv"
    df.to_csv(csv, index=False)

    agg = df.groupby(["spec", "task"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(5.2 * len(args.tasks), 7),
                             squeeze=False)
    for j, task in enumerate(args.tasks):
        sub = agg[agg.task == task].sort_values("id_acc")
        axes[0][j].barh(sub.spec, sub.id_acc, color="#5b9a5b")
        axes[0][j].set_xlim(0, 1.02)
        axes[0][j].set_title(f"{task}: ID accuracy")
        axes[0][j].grid(alpha=0.3, axis="x")
        axes[0][j].tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exp6_processing.png", dpi=150)
    plt.close(fig)

    print("\nID accuracy by processing variant:")
    print(df.pivot_table(index="spec", columns="task", values="id_acc").round(3).to_string())
    print(f"\nsaved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
