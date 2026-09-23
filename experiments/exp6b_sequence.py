"""Exp6b: symbolic-sequence processing (binding / positional encoding).

The symbolic task's class is the presence of an ordered bigram, so:

* ``bag``            (order-invariant bundling) cannot separate the classes;
* ``bigram``         (bind symbol t with a shifted symbol t+1, then a
                     position code) constructs the conjunction explicitly;
* ``circ`` binding   uses HRR-style circular convolution on real HVs;
* ``xor`` binding    uses binary codes with XOR binding and majority bundling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.data import make_task
from modules.encoders import SequenceEncoder
from modules.hd import PrototypeClassifier
from modules.metrics import accuracy

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODES = ["bag", "bag_pos", "bigram", "bigram_nopos"]
BINDS = ["mul", "xor", "circ"]
POS = ["perm", "random"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for seed in range(args.seeds):
        task = make_task("symbolic", seed=seed)
        for mode in MODES:
            for bind in BINDS:
                for pos in POS:
                    enc = SequenceEncoder(args.dim, 8, 12, mode=mode, bind=bind,
                                          pos_mode=pos, out="bipolar", seed=seed).fit()
                    clf = PrototypeClassifier(enc, task.num_classes,
                                              epochs=args.epochs, seed=seed).fit(
                        task.x_train, task.y_train)
                    acc = accuracy(clf.predict(task.x_test), task.y_test)
                    val = accuracy(clf.predict(task.x_val), task.y_val)
                    rows.append(dict(mode=mode, bind=bind, pos=pos, seed=seed,
                                     val_acc=val, id_acc=acc,
                                     mem_kb=enc.memory_bits() / 8 / 1024,
                                     ops=enc.ops_per_query()))
                    print(f"[s{seed}] {mode:13s} bind={bind:4s} pos={pos:6s} "
                          f"id={acc:.3f}", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS / "exp6b_sequence.csv"
    df.to_csv(csv, index=False)

    piv = df.pivot_table(index=["mode", "bind"], columns="pos", values="id_acc")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    piv.plot.barh(ax=ax)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("ID accuracy")
    ax.set_title("symbolic bigram task: binding / position encoding")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exp6b_sequence.png", dpi=150)
    with open(RESULTS / "exp6b_sequence_pivot.csv", "w") as f:
        piv.to_csv(f)
    print(piv.round(3).to_string())
    print(f"saved {csv} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
