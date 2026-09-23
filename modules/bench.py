"""Common evaluation harness: Spec -> result row.

Every experiment runs (spec, task, seed) through the same pipeline::

    [feature extractor] -> [projection encoder] -> [prototype classifier]
        -> ID / OOD / novel-class / robustness / geometry / resource metrics

so differences between rows are attributable to the spec.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import torch

from .data import Task, make_task
from .encoders import CodecWrapper, FixedProjectionEncoder, make_encoder, resolve_device
from .extractors import Extractor, co_train, make_extractor
from .hd import PrototypeClassifier, add_feature_noise, drop_features, sign_flip
from .metrics import (accuracy, clustering_metrics, geometry_fidelity, margins,
                      ood_detection_auroc)
from .rp import make_rp, rp_stats

__all__ = ["Spec", "BenchConfig", "run_spec", "ExtractorCache", "get_task"]


@dataclass
class Spec:
    """One projection/processing configuration to evaluate."""

    name: str
    family: str = "linear"
    out: str = "bipolar"
    preproc: str = "none"
    extractor: str = "raw"
    kwargs: dict = field(default_factory=dict)
    ext_kwargs: dict = field(default_factory=dict)
    sim: str | None = None
    codec: str | None = None

    def meta(self) -> dict:
        k = dict(self.kwargs)
        rp = k.pop("rp", None)
        return {
            "spec": self.name,
            "family": self.family,
            "rp": rp if rp is not None else self.family,
            "out": self.out,
            "preproc": self.preproc,
            "extractor": self.extractor,
            "params": json.dumps(k, sort_keys=True) if k else "",
            "spec_sim": self.sim or "",
            "codec": self.codec or "",
        }


@dataclass
class BenchConfig:
    """Shared evaluation settings."""

    dim: int = 4096
    epochs: int = 30
    sim: str = "cosine"
    weight: str = "uniform"
    multi: int = 1
    proto_bits: int | None = None
    proto_scheme: str = "mse"
    quant_train: bool = False
    noise_sigmas: tuple = (0.5,)
    drop_ps: tuple = (0.2,)
    flip_ps: tuple = (0.05,)
    geom_points: int = 400
    robustness: bool = True
    device: str | None = None


class ExtractorCache:
    """Cache fitted extractors across specs (they depend only on task/seed)."""

    def __init__(self, device: str | None = None) -> None:
        self.cache: dict = {}
        self.device = device

    def get(self, spec: Spec, task: Task, seed: int, dim: int):
        key = (spec.extractor, json.dumps(spec.ext_kwargs, sort_keys=True),
               spec.kwargs.get("rp", "gaussian") if spec.extractor == "cotrain" else "",
               task.name, seed, dim if spec.extractor == "cotrain" else 0)
        if key in self.cache:
            return self.cache[key]
        F = task.x_train.shape[1]
        if spec.extractor == "cotrain":
            ekw = dict(spec.ext_kwargs)
            out_dim = ekw.pop("out_dim", 128)
            learn_proj = ekw.pop("learn_proj", False)
            co_keys = ("epochs", "lr", "weight_decay", "batch_size")
            co_kw = {k: ekw.pop(k) for k in co_keys if k in ekw}
            ext = make_extractor("randmlp", F, out_dim=out_dim, seed=seed,
                                 device=self.device, **ekw)
            P = make_rp(spec.kwargs.get("rp", "gaussian"), dim, out_dim, seed=seed)
            ext, P = co_train(ext, P, task.x_train, task.y_train, seed=seed,
                              learn_proj=learn_proj, device=self.device, **co_kw)
            val = (ext, P)
        else:
            ext = make_extractor(spec.extractor, F, seed=seed, device=self.device,
                                 **spec.ext_kwargs)
            if spec.extractor != "raw":
                ext.fit(task.x_train, task.y_train)
            val = (ext, None)
        self.cache[key] = val
        return val


def get_task(name: str, seed: int, **overrides) -> Task:
    return make_task(name, seed=seed, **overrides)


@torch.no_grad()
def _score_flipped(clf: PrototypeClassifier, h: torch.Tensor,
                   p: float, seed: int) -> torch.Tensor:
    W = sign_flip(clf._prototypes(), p, seed=seed)
    return clf._sim(h.float(), W)


def run_spec(spec: Spec, task: Task, seed: int, cfg: BenchConfig,
             cache: ExtractorCache | None = None) -> dict:
    """Evaluate one spec on one task/seed; returns a flat result row."""
    cache = cache or ExtractorCache(cfg.device)
    t0 = time.time()

    ext, P_fixed = cache.get(spec, task, seed, cfg.dim)
    xtr = ext.transform(task.x_train)
    xva = ext.transform(task.x_val)
    xte = ext.transform(task.x_test)
    targets = [(n, ext.transform(x), y) for n, x, y in task.targets]
    xnov = ext.transform(task.x_novel) if task.x_novel is not None else None

    F = xtr.shape[1]
    C = task.num_classes
    if spec.extractor == "cotrain":
        encoder = FixedProjectionEncoder(P_fixed, out=spec.out, device=cfg.device)
    else:
        encoder = make_encoder(spec.family, cfg.dim, F, seed=seed, device=cfg.device,
                               **spec.kwargs)
        if spec.codec:
            encoder = CodecWrapper(encoder, spec.codec)
    encoder.fit(xtr, task.y_train, xva, task.y_val)
    fit_enc = time.time() - t0

    clf = PrototypeClassifier(
        encoder, C, sim=spec.sim or cfg.sim, epochs=cfg.epochs, weight=cfg.weight,
        multi=cfg.multi, proto_bits=cfg.proto_bits, proto_scheme=cfg.proto_scheme,
        quant_train=cfg.quant_train, seed=seed,
    )
    t1 = time.time()
    clf.fit(xtr, task.y_train)
    fit_clf = time.time() - t1

    row = {
        **spec.meta(),
        "task": task.name,
        "seed": seed,
        "dim": cfg.dim,
        "F": F,
        "C": C,
        "epochs": cfg.epochs,
        "sim": cfg.sim,
        "weight": cfg.weight,
        "multi": cfg.multi,
        "proto_bits": cfg.proto_bits if cfg.proto_bits else 32,
        "quant_train": int(cfg.quant_train),
        "val_acc": accuracy(clf.predict(xva), task.y_val),
        "id_acc": accuracy(clf.predict(xte), task.y_test),
        "margin_id": margins(clf.scores(xte), task.y_test)["margin"],
        "geom_fid": geometry_fidelity(encoder, xte, n_points=cfg.geom_points, seed=seed),
        "fit_s": round(fit_enc + fit_clf, 3),
    }

    for tname, xt, yt in targets:
        row[f"acc_{tname}"] = accuracy(clf.predict(xt), yt)
        if tname == targets[0][0]:
            row["margin_ood"] = margins(clf.scores(xt), yt)["margin"]

    if xnov is not None:
        hn = encoder.encode(xnov)
        cl = clustering_metrics(hn, task.num_novel_classes, task.y_novel.numpy(), seed=seed)
        row["nmi"] = cl["nmi"]
        row["ari"] = cl["ari"]
        s_id = clf.scores(xte).max(dim=1).values.cpu().numpy()
        s_no = clf.scores(xnov).max(dim=1).values.cpu().numpy()
        row["auroc"] = ood_detection_auroc(s_id, s_no)

    if cfg.robustness:
        for s in cfg.noise_sigmas:
            row[f"acc_noise_{s:g}"] = accuracy(
                clf.predict(add_feature_noise(xte, s, seed=seed + 999)), task.y_test)
        for p in cfg.drop_ps:
            row[f"acc_drop_{p:g}"] = accuracy(
                clf.predict(drop_features(xte, p, seed=seed + 999)), task.y_test)
        hex_ = encoder.encode(xte)
        if cfg.multi == 1:
            for p in cfg.flip_ps:
                row[f"acc_flip_{p:g}"] = accuracy(
                    _score_flipped(clf, hex_, p, seed + 999).argmax(dim=1), task.y_test)

    proj_bits = int(encoder.memory_bits())
    proto_bits = cfg.proto_bits if cfg.proto_bits else 32
    total_bits = proj_bits + C * encoder.dim_out * proto_bits
    row["mem_kb"] = round(total_bits / 8.0 / 1024.0, 2)
    row["vec_bits"] = round(encoder.storage_bits_per_coord(), 3)
    row["ops_per_query"] = int(encoder.ops_per_query()
                               + C * getattr(encoder, "dim_out", cfg.dim))
    row["ext_bits"] = int(ext.memory_bits())
    row["ext_ops"] = int(ext.ops_per_query())
    if spec.family == "linear" and hasattr(encoder, "P") and encoder.P is not None:
        st = rp_stats(encoder.P)
        row["rp_density"] = round(st["density"], 4)
        row["rp_rank"] = st["rank"]
    return row
