"""Synthetic workloads for probing projection methods in HDC pipelines.

Tasks (all F=64-dimensional vectors unless noted):

* ``linear``     : class-conditional Gaussians with a shared anisotropic
                   covariance (power-law eigenspectrum). A linear encoder is
                   sufficient; this is the "RP-friendly" regime.
* ``fine``       : same family with a small inter-class spread (high class
                   similarity / fine-grained regime).
* ``xor``        : class depends on the parity of latent sign bits; a linear
                   projection can only reach chance, a nonlinear kernel
                   approximation can do better.
* ``shells``     : classes are radial quantiles of a latent Gaussian; class
                   means are identical, so linear methods are at chance.
* ``heavy_tail`` : Gaussian classes with Student-t within-class noise plus
                   random per-feature outliers (tests L1/outlier robustness).
* ``shift``      : domain shift (covariate shift + noise inflation) with
                   graded severity targets.
* ``novel``      : held-out classes from the same generative family
                   (unsupervised discovery + OOD detection).
* ``symbolic``   : variable-length-free symbolic sequences (alphabet of 8),
                   class = presence of an ordered bigram (order-sensitive;
                   used to compare binding / positional encodings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["Task", "make_task", "TASK_NAMES"]

TASK_NAMES = ["linear", "fine", "xor", "shells", "heavy_tail", "shift", "novel", "symbolic"]


@dataclass
class Task:
    name: str
    num_classes: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    targets: list = field(default_factory=list)  # (name, X, y) extra splits
    x_novel: torch.Tensor | None = None
    y_novel: torch.Tensor | None = None
    num_novel_classes: int = 0
    shift_dir: np.ndarray | None = None
    modality: str = "vector"  # "vector" | "sequence"


def _t(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)


def _orthonormal(F: int, L: int, rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((F, L)))
    return q[:, :L]


def _shared_cov(F: int, decay: float, rng):
    eig = np.power(np.arange(1, F + 1), -decay)
    eig = eig / eig.mean()
    q, _ = np.linalg.qr(rng.standard_normal((F, F)))
    return (q * eig) @ q.T, eig


def _student_noise(rng, n: int, F: int, nu: float) -> np.ndarray:
    z = rng.standard_t(nu, size=(n, F))
    if nu > 2:
        z = z / np.sqrt(nu / (nu - 2.0))
    return z


def _balanced_pool(sample_fn, C: int, n: int, max_iter: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` samples per class from a labeled sampler."""
    xs, ys = [], []
    counts = np.zeros(C, dtype=np.int64)
    for _ in range(max_iter):
        if (counts >= n).all():
            break
        x, y = sample_fn(max(2 * n, 256))
        for c in range(C):
            need = n - counts[c]
            if need <= 0:
                continue
            idx = np.flatnonzero(y == c)[:need]
            if len(idx):
                xs.append(x[idx])
                ys.append(np.full(len(idx), c, dtype=np.int64))
                counts[c] += len(idx)
    out_x = np.concatenate(xs, axis=0)
    out_y = np.concatenate(ys, axis=0)
    return out_x, out_y


def _gaussian_sampler(rng, means, chol, noise: str, nu: float,
                      outlier_frac: float, outlier_scale: float):
    k, F = means.shape

    def sample(n_total: int):
        z = rng.standard_normal((n_total, F))
        if noise == "t":
            z = _student_noise(rng, n_total, F, nu)
        x = z @ chol.T
        # class assignment: cycle through classes so the sampler is balanced
        y = np.tile(np.arange(k, dtype=np.int64), n_total // k + 1)[:n_total]
        x = x + means[y]
        if outlier_frac > 0:
            mask = rng.random(x.shape) < outlier_frac
            x[mask] *= outlier_scale
        return x, y

    return sample


def _assemble(xtr, ytr, xva, yva, xte, yte, *, name, C, targets=(), **kw) -> Task:
    return Task(
        name=name,
        num_classes=C,
        x_train=_t(xtr), y_train=_t(ytr, torch.long),
        x_val=_t(xva), y_val=_t(yva, torch.long),
        x_test=_t(xte), y_test=_t(yte, torch.long),
        targets=[(n, _t(x), _t(y, torch.long)) for n, x, y in targets],
        **kw,
    )


def _gaussian_splits(seed, *, C, F, spread, decay, noise, nu, outlier_frac,
                     outlier_scale, n_train, n_val, n_test, target_specs=(),
                     shift_dir=None):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, spread, size=(C, F))
    cov, _ = _shared_cov(F, decay, rng)
    chol = np.linalg.cholesky(cov)
    samp = _gaussian_sampler(rng, means, chol, noise, nu, outlier_frac, outlier_scale)
    xtr, ytr = _balanced_pool(samp, C, n_train)
    xva, yva = _balanced_pool(samp, C, n_val)
    xte, yte = _balanced_pool(samp, C, n_test)
    targets = []
    for tname, shift_norm, tnoise in target_specs:
        if shift_dir is None:
            sdir = rng.standard_normal(F)
            sdir /= np.linalg.norm(sdir)
        else:
            sdir = shift_dir
        z = rng.standard_normal((C * n_test, F)) * tnoise
        y = np.tile(np.arange(C, dtype=np.int64), n_test)
        xs = z @ chol.T + means[y] + sdir * shift_norm
        targets.append((tname, xs, y))
    return xtr, ytr, xva, yva, xte, yte, targets


def make_task(name: str, seed: int = 0, **overrides) -> Task:
    """Build a synthetic task by name (extra kwargs override defaults)."""
    if name == "linear":
        kw = dict(C=30, F=64, spread=0.7, decay=0.4, noise="gauss", nu=3.0,
                  outlier_frac=0.0, outlier_scale=1.0, n_train=150, n_val=40,
                  n_test=50, target_specs=())
    elif name == "fine":
        kw = dict(C=30, F=64, spread=0.35, decay=0.4, noise="gauss", nu=3.0,
                  outlier_frac=0.0, outlier_scale=1.0, n_train=150, n_val=40,
                  n_test=50, target_specs=())
    elif name == "heavy_tail":
        kw = dict(C=30, F=64, spread=0.7, decay=0.4, noise="t", nu=3.0,
                  outlier_frac=0.05, outlier_scale=6.0, n_train=150, n_val=40,
                  n_test=50, target_specs=())
    else:
        kw = {}

    if name in ("linear", "fine", "heavy_tail"):
        kw.update(overrides)
        xtr, ytr, xva, yva, xte, yte, targets = _gaussian_splits(seed, **kw)
        return _assemble(xtr, ytr, xva, yva, xte, yte,
                         name=name, C=kw["C"], targets=targets)

    if name == "shift":
        kw = dict(C=30, F=64, spread=0.7, decay=0.4, noise="gauss", nu=3.0,
                  outlier_frac=0.0, outlier_scale=1.0, n_train=150, n_val=40,
                  n_test=50,
                  target_specs=(("shift_6", 6.0, 1.15), ("shift_12", 12.0, 1.30),
                                ("shift_18", 18.0, 1.45)))
        kw.update(overrides)
        rng = np.random.default_rng(seed + 12345)
        shift_dir = rng.standard_normal(kw["F"])
        shift_dir /= np.linalg.norm(shift_dir)
        xtr, ytr, xva, yva, xte, yte, targets = _gaussian_splits(
            seed, shift_dir=shift_dir, **kw)
        return _assemble(xtr, ytr, xva, yva, xte, yte,
                         name=name, C=kw["C"], targets=targets, shift_dir=shift_dir)

    if name == "novel":
        kw = dict(C_total=24, C_known=14, F=64, spread=0.55, decay=0.4,
                  n_train=150, n_val=40, n_test=50, n_novel=60)
        kw.update(overrides)
        rng = np.random.default_rng(seed)
        means = rng.normal(0.0, kw["spread"], size=(kw["C_total"], kw["F"]))
        cov, _ = _shared_cov(kw["F"], kw["decay"], rng)
        chol = np.linalg.cholesky(cov)

        def samp(sel: np.ndarray):
            def sample(n_total: int):
                z = rng.standard_normal((n_total, kw["F"])) @ chol.T
                y = np.tile(np.arange(len(sel), dtype=np.int64), n_total // len(sel) + 1)[:n_total]
                return z + means[sel][y], y
            return sample

        known = np.arange(kw["C_known"])
        novel = np.arange(kw["C_known"], kw["C_total"])
        xtr, ytr = _balanced_pool(samp(known), kw["C_known"], kw["n_train"])
        xva, yva = _balanced_pool(samp(known), kw["C_known"], kw["n_val"])
        xte, yte = _balanced_pool(samp(known), kw["C_known"], kw["n_test"])
        xnov, ynov = _balanced_pool(samp(novel), len(novel), kw["n_novel"])
        return _assemble(xtr, ytr, xva, yva, xte, yte, name=name,
                         C=kw["C_known"], x_novel=_t(xnov), y_novel=_t(ynov, torch.long),
                         num_novel_classes=len(novel))

    if name == "xor":
        kw = dict(C=4, F=64, L=8, signal=1.5, noise=0.5,
                  n_train=200, n_val=50, n_test=60)
        kw.update(overrides)
        rng = np.random.default_rng(seed)
        A = _orthonormal(kw["F"], kw["L"], rng)

        def sample(n_total: int):
            z = rng.standard_normal((n_total, kw["L"]))
            b = z[:, :4] > 0
            y = (b[:, 0] ^ b[:, 1]).astype(np.int64) + 2 * (b[:, 2] ^ b[:, 3]).astype(np.int64)
            x = (z @ A.T) * kw["signal"] + rng.standard_normal((n_total, kw["F"])) * kw["noise"]
            return x, y

        xtr, ytr = _balanced_pool(sample, kw["C"], kw["n_train"])
        xva, yva = _balanced_pool(sample, kw["C"], kw["n_val"])
        xte, yte = _balanced_pool(sample, kw["C"], kw["n_test"])
        return _assemble(xtr, ytr, xva, yva, xte, yte, name=name, C=kw["C"])

    if name == "shells":
        kw = dict(C=6, F=64, L=8, signal=1.5, noise=0.5,
                  n_train=200, n_val=50, n_test=60)
        kw.update(overrides)
        rng = np.random.default_rng(seed)
        A = _orthonormal(kw["F"], kw["L"], rng)
        from scipy.stats import chi2

        edges = chi2.ppf(np.arange(1, kw["C"]) / kw["C"], df=kw["L"])

        def sample(n_total: int):
            z = rng.standard_normal((n_total, kw["L"]))
            r = np.linalg.norm(z, axis=1)
            y = np.digitize(r, edges).astype(np.int64)
            x = (z @ A.T) * kw["signal"] + rng.standard_normal((n_total, kw["F"])) * kw["noise"]
            return x, y

        xtr, ytr = _balanced_pool(sample, kw["C"], kw["n_train"])
        xva, yva = _balanced_pool(sample, kw["C"], kw["n_val"])
        xte, yte = _balanced_pool(sample, kw["C"], kw["n_test"])
        return _assemble(xtr, ytr, xva, yva, xte, yte, name=name, C=kw["C"])

    if name == "symbolic":
        kw = dict(V=8, T=12, C=6, n_train=200, n_val=50, n_test=60)
        kw.update(overrides)
        rng = np.random.default_rng(seed)
        V, T, C = kw["V"], kw["T"], kw["C"]
        pairs = [(k, (k + 3) % V) for k in range(C)]

        def sample(n_per_class: int):
            seqs = np.empty((C * n_per_class, T), dtype=np.int64)
            labels = np.repeat(np.arange(C, dtype=np.int64), n_per_class)
            for k, (a, b) in enumerate(pairs):
                for i in range(n_per_class):
                    s = rng.integers(0, V, size=T)
                    t = rng.integers(0, T - 1)
                    s[t], s[t + 1] = a, b
                    seqs[k * n_per_class + i] = s
            return seqs, labels

        str_, ytr = sample(kw["n_train"])
        sva, yva = sample(kw["n_val"])
        ste, yte = sample(kw["n_test"])
        return Task(
            name=name, num_classes=C,
            x_train=_t(str_, torch.long), y_train=_t(ytr, torch.long),
            x_val=_t(sva, torch.long), y_val=_t(yva, torch.long),
            x_test=_t(ste, torch.long), y_test=_t(yte, torch.long),
            modality="sequence",
        )

    raise ValueError(f"unknown task {name!r}; options: {TASK_NAMES}")
