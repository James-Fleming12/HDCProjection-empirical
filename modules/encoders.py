"""Projection encoders for HDC pipelines.

    LinearRPEncoder       sign/quantized linear random projection (any RP
                          construction in modules/rp.py)
    RFFEncoder            random Fourier features, cos(omega x + b), optional
                          sign (SignRFF) / binary output
    FPEEncoder            fractional power encoding (phasor / group-VSA style,
                          optional phase quantization)
    NystromEncoder        Nystrom kernel-approximation features (Gaussian or
                          Laplacian kernel)
    LearnedProjectionEncoder  PIONEER/LeHDC-style projection trained with a
                          straight-through estimator on a prototype-cosine
                          surrogate objective
    DataInformedEncoder   best-of-K random projections selected on a held-out
                          split (cheap data-dependent selection)
    IMPEncoder            input-modulated projection (per-sample feature
                          gating before a fixed random projection)
    SequenceEncoder       symbolic sequences: symbol HVs + binding +
                          positional encoding + bundling

Every encoder exposes ``fit(x, y=None, x_val=None, y_val=None)``,
``encode(x) -> (n, dim_out)``, ``memory_bits()`` and ``ops_per_query()``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import quant
from .rp import make_rp, rp_nnz, rp_storage_bits

__all__ = [
    "resolve_device",
    "OutputCode",
    "LinearRPEncoder",
    "RFFEncoder",
    "FPEEncoder",
    "NystromEncoder",
    "LearnedProjectionEncoder",
    "DataInformedEncoder",
    "IMPEncoder",
    "SequenceEncoder",
    "FixedProjectionEncoder",
    "CodecWrapper",
    "make_encoder",
    "ENCODER_FAMILIES",
]

ENCODER_FAMILIES = ["linear", "rff", "fpe", "nystrom", "learned",
                    "datainformed", "imp", "seq"]


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    env = os.environ.get("HDCPROJ_DEVICE")
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------


class Preproc:
    """Feature preprocessing fitted on the training split."""

    def __init__(self, kind: str = "none") -> None:
        assert kind in ("none", "center", "std", "l2", "whiten")
        self.kind = kind
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.W: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "Preproc":
        x = x.float()
        if self.kind in ("center", "std", "whiten"):
            self.mean = x.mean(dim=0)
        if self.kind == "std":
            self.std = x.std(dim=0).clamp_min(1e-8)
        if self.kind == "whiten":
            xc = x - self.mean
            cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
            evals, evecs = torch.linalg.eigh(cov)
            evals = evals.clamp_min(1e-10)
            evals = evals / evals.mean()
            self.W = evecs * (1.0 / evals.sqrt())  # (F, F)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.kind == "center":
            return x - self.mean.to(x.device)
        if self.kind == "std":
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        if self.kind == "whiten":
            return (x - self.mean.to(x.device)) @ self.W.to(x.device)
        if self.kind == "l2":
            return F.normalize(x, dim=1)
        return x

    def memory_bits(self, F: int) -> int:
        if self.kind in ("center", "std"):
            return 2 * F * 32 if self.kind == "std" else F * 32
        if self.kind == "whiten":
            return (2 * F + F * F) * 32
        return 0

    def ops_per_query(self, F: int) -> int:
        if self.kind == "whiten":
            return F * F
        if self.kind in ("center", "std"):
            return 2 * F
        return 0


# --------------------------------------------------------------------------
# output coding
# --------------------------------------------------------------------------


class OutputCode:
    """Maps pre-activations (n, D) to the stored hypervector code.

    ``kind``:
      real      - identity (32-bit floats)
      bipolar   - sign in {-1, +1}
      binary01  - {0, 1}
      int       - MSE-optimal symmetric quantization at ``bits``
      ternary   - MSE-optimal {-a, 0, +a}
      block     - keep the top-1 magnitude per block of ``block`` dims
      thermo    - thermometer code with ``levels`` bits per coordinate
      vq        - k-means codebook (fit on encoded training vectors)
    """

    def __init__(self, kind: str = "bipolar", *, bits: int = 8, levels: int = 4,
                 block: int = 8, seed: int = 0) -> None:
        if kind.startswith("int") and kind[3:].isdigit():
            bits, kind = int(kind[3:]), "int"
        self.kind = kind
        self.bits = bits
        self.levels = levels
        self.block = block
        self.scale = None
        self.tau = None
        self.a = None
        self.lo = None
        self.hi = None
        self.vq: quant.VectorQuantizer | None = None
        self._seed = seed

    @property
    def dim_multiplier(self) -> int:
        return self.levels if self.kind == "thermo" else 1

    def fit(self, pre: torch.Tensor) -> "OutputCode":
        if self.kind == "int":
            self.scale = quant.mse_scale(pre, self.bits)
        elif self.kind == "ternary":
            self.tau, self.a = quant.ternary_fit(pre)
        elif self.kind == "thermo":
            self.lo, self.hi = quant.thermometer_fit(pre)
        return self

    def apply(self, pre: torch.Tensor) -> torch.Tensor:
        if self.kind == "real":
            return pre
        if self.kind == "bipolar":
            return quant.sign(pre)
        if self.kind == "binary01":
            return quant.binary01(pre)
        if self.kind == "int":
            return quant.apply_scale(pre, self.scale, self.bits)
        if self.kind == "ternary":
            return quant.ternary_apply(pre, self.tau, self.a)
        if self.kind == "block":
            return quant.sparse_block(pre, self.block, keep=1, mode="sign")
        if self.kind == "thermo":
            return quant.thermometer_apply(pre, self.levels, self.lo, self.hi)
        if self.kind == "vq":
            return self.vq.encode(pre)
        raise ValueError(self.kind)

    def storage_bits_per_coord(self) -> float:
        return {
            "real": 32.0,
            "bipolar": 1.0,
            "binary01": 1.0,
            "int": float(self.bits),
            "ternary": 2.0,
            "block": (1.0 + math.log2(self.block)) / self.block,
            "thermo": float(self.levels),
            "vq": 0.0,
        }[self.kind]


# --------------------------------------------------------------------------
# scalar-projection encoders
# --------------------------------------------------------------------------


class LinearRPEncoder:
    """Linear random projection followed by an output code."""

    family = "linear"
    kind = "linear"

    def __init__(self, dim: int, num_features: int, *, rp: str = "gaussian",
                 rp_params: dict | None = None, seed: int = 0, preproc: str = "none",
                 out: str = "bipolar", out_bits: int = 8, thermo_levels: int = 4,
                 block: int = 8, proj_bits: int | None = None,
                 proj_scheme: str = "mse", device: str | None = None) -> None:
        self.dim = dim
        self.num_features = num_features
        self.rp = rp
        self.rp_params = dict(rp_params or {})
        self.seed = seed
        self.preproc = Preproc(preproc)
        self.out_code = OutputCode(out, bits=out_bits, levels=thermo_levels, block=block, seed=seed)
        self.out = out
        self.proj_bits = proj_bits
        self.proj_scheme = proj_scheme
        self.device = resolve_device(device)
        self.P: torch.Tensor | None = None
        self.dim_out = dim * self.out_code.dim_multiplier

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None,
            x_val: torch.Tensor | None = None, y_val: torch.Tensor | None = None) -> "LinearRPEncoder":
        self.preproc.fit(x)
        P = make_rp(self.rp, self.dim, self.num_features, seed=self.seed, **self.rp_params)
        if self.proj_bits is not None:
            if self.proj_scheme == "mse":
                P = quant.mse_ptq(P, self.proj_bits)
            elif self.proj_scheme == "max":
                P = quant.uniform_max(P, self.proj_bits)
            elif self.proj_scheme == "ternary":
                P = quant.ternary_quantize(P)
            elif self.proj_scheme == "sign":
                P = quant.sign(P)
            else:
                raise ValueError(self.proj_scheme)
        self.P = P.to(self.device)
        pre = self._pre(x)
        self.out_code.fit(pre)
        if self.out == "vq":
            self.out_code.vq = quant.VectorQuantizer(256, seed=self.seed).fit(pre)
        return self

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.preproc.apply(x.float().to(self.device))
        return xp @ self.P.T

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            outs.append(self.out_code.apply(self._pre(x[i:i + batch_size])))
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        if self.P is None:
            proj = self.dim * self.num_features * 32
        elif self.proj_bits is not None:
            bits_p = 32 if self.proj_scheme == "mse" else (2 if self.proj_scheme == "ternary" else 1)
            if self.proj_scheme == "max":
                bits_p = self.proj_bits
            proj = int(rp_nnz(self.P) * bits_p)
        else:
            proj = rp_storage_bits(self.rp, self.P)
        extra = self.preproc.memory_bits(self.num_features)
        if self.out == "vq":
            extra += self.out_code.vq.memory_bits()
        return int(proj) + extra

    def ops_per_query(self) -> int:
        nnz = rp_nnz(self.P) if self.P is not None else self.dim * self.num_features
        return int(nnz + self.dim) + self.preproc.ops_per_query(self.num_features)

    def storage_bits_per_coord(self) -> float:
        return self.out_code.storage_bits_per_coord()


def _select_bandwidth(enc, grid: tuple, x: torch.Tensor, y: torch.Tensor,
                      x_val: torch.Tensor, y_val: torch.Tensor,
                      subsample: int = 2000, seed: int = 0) -> float:
    """Pick the bandwidth with the best single-pass prototype accuracy on a
    validation split (no retraining), for kernel-approximation encoders."""
    g = torch.Generator().manual_seed(seed)
    if x.shape[0] > subsample:
        idx = torch.randperm(x.shape[0], generator=g)[:subsample]
        x, y = x[idx], y[idx]
    y = y.long()
    best_bw, best_acc = grid[0], -1.0
    for bw in grid:
        enc.set_bandwidth(bw)
        h = enc.encode(x).float()
        C = int(y.max().item()) + 1
        W = torch.zeros(C, h.shape[1], device=h.device)
        W.index_add_(0, y.to(h.device), h)
        hval = enc.encode(x_val).float()
        scores = F.normalize(hval, dim=1) @ F.normalize(W, dim=1).T
        acc = float((scores.argmax(dim=1) == y_val.to(h.device)).float().mean().item())
        if acc > best_acc:
            best_acc, best_bw = acc, bw
    enc.set_bandwidth(best_bw)
    return best_bw


class RFFEncoder:
    """Random Fourier features for the Gaussian kernel.

    phi(x) = sqrt(2/D) * cos(omega^T x + b), omega ~ N(0, I/sigma^2),
    b ~ U(0, 2 pi); the bandwidth sigma^2 is set from the data by
    ``sigma^2 = bw_scale * median_pairwise_sqdist``. ``out='bipolar'`` gives
    SignRFF; ``out='binary01'`` gives binary RFF HDC.
    """

    family = "rff"
    kind = "rff"

    def __init__(self, dim: int, num_features: int, *, seed: int = 0,
                 bw_scale: float = 0.25, use_bias: bool = True, out: str = "real",
                 preproc: str = "none", bw_select: str | None = None,
                 bw_grid: tuple = (0.05, 0.1, 0.25, 0.5, 1.0),
                 device: str | None = None) -> None:
        self.dim = dim
        self.num_features = num_features
        self.seed = seed
        self.bw_scale = bw_scale
        self.use_bias = use_bias
        self.out = out
        self.bw_select = bw_select
        self.bw_grid = tuple(bw_grid)
        self.preproc = Preproc(preproc)
        self.device = resolve_device(device)
        self.omega: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self._med2: float | None = None
        self.selected_bw: float | None = None
        self.sigma2: float = 1.0
        self.dim_out = dim

    def _fit_bandwidth(self, x: torch.Tensor) -> float:
        g = torch.Generator().manual_seed(self.seed)
        n = min(2000, x.shape[0])
        idx = torch.randperm(x.shape[0], generator=g)[:n]
        xs = F.normalize(x[idx].float(), dim=1) * math.sqrt(self.num_features)
        d = torch.cdist(xs, xs)
        return float(d.median().item() ** 2)

    def _set_omega(self, sigma2: float) -> None:
        self.sigma2 = float(sigma2)
        g = torch.Generator().manual_seed(self.seed + 7)
        self.omega = (torch.randn(self.dim, self.num_features, generator=g)
                      / math.sqrt(max(sigma2, 1e-12))).to(self.device)
        if self.use_bias:
            self.bias = (torch.rand(self.dim, generator=g) * 2 * math.pi).to(self.device)

    def set_bandwidth(self, bw_scale: float) -> None:
        self.bw_scale = bw_scale
        self._set_omega(self.bw_scale * self._med2)

    def fit(self, x, y=None, x_val=None, y_val=None) -> "RFFEncoder":
        self.preproc.fit(x)
        xp = self.preproc.apply(x.float())
        self._med2 = self._fit_bandwidth(xp)
        self._set_omega(self.bw_scale * self._med2)
        if self.bw_select == "val" and x_val is not None and y_val is not None:
            self.selected_bw = _select_bandwidth(
                self, self.bw_grid, xp, y, self.preproc.apply(x_val.float()), y_val,
                seed=self.seed)
        return self

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.preproc.apply(x.float().to(self.device))
        z = xp @ self.omega.T
        if self.use_bias:
            z = z + self.bias
        return math.sqrt(2.0 / self.dim) * torch.cos(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            z = self._pre(x[i:i + batch_size])
            if self.out == "bipolar":
                z = quant.sign(z)
            elif self.out == "binary01":
                z = quant.binary01(z)
            outs.append(z)
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        bits = self.dim * self.num_features * 32 + (self.dim * 32 if self.use_bias else 0)
        return bits + self.preproc.memory_bits(self.num_features)

    def ops_per_query(self) -> int:
        return int(self.dim * self.num_features + self.dim) + self.preproc.ops_per_query(self.num_features)

    def storage_bits_per_coord(self) -> float:
        return 32.0 if self.out == "real" else 1.0


class FPEEncoder:
    """Fractional power encoding with diagonal (Gaussian) frequencies.

    A real value vector x is mapped to a phasor hypervector
    z_k(x) = exp(i * omega_k * x) with omega_k ~ N(0, 1/sigma^2), so that
    Re <z(x), z(y)> / D = E[cos(omega (x - y))] -> Gaussian kernel. The
    complex pair (cos, sin) is stored as a real 2D-dimensional vector, which
    makes phasor binding/bundling/similarity identical to real-vector
    operations. ``phase_bits`` quantizes the phase (group-VSA style); with
    ``phase_bits=1`` the code reduces to sign(cos) (a D-dimensional bipolar
    SignRFF variant without the random bias).
    """

    family = "fpe"
    kind = "fpe"

    def __init__(self, dim: int, num_features: int, *, seed: int = 0,
                 bw_scale: float = 0.25, phase_bits: int | None = None,
                 preproc: str = "none", bw_select: str | None = None,
                 bw_grid: tuple = (0.05, 0.1, 0.25, 0.5, 1.0),
                 device: str | None = None) -> None:
        self.dim = dim
        self.num_features = num_features
        self.seed = seed
        self.bw_scale = bw_scale
        self.phase_bits = phase_bits
        self.preproc = Preproc(preproc)
        self.bw_select = bw_select
        self.bw_grid = tuple(bw_grid)
        self.device = resolve_device(device)
        self.omega: torch.Tensor | None = None
        self._med2: float | None = None
        self.selected_bw: float | None = None
        self.sigma2: float = 1.0
        self.dim_out = dim if phase_bits == 1 else 2 * dim

    def _set_omega(self, sigma2: float) -> None:
        self.sigma2 = float(sigma2)
        g = torch.Generator().manual_seed(self.seed + 11)
        self.omega = (torch.randn(self.dim, self.num_features, generator=g)
                      / math.sqrt(max(sigma2, 1e-12))).to(self.device)

    def set_bandwidth(self, bw_scale: float) -> None:
        self.bw_scale = bw_scale
        self._set_omega(self.bw_scale * self._med2)

    def fit(self, x, y=None, x_val=None, y_val=None) -> "FPEEncoder":
        self.preproc.fit(x)
        xp = self.preproc.apply(x.float())
        g = torch.Generator().manual_seed(self.seed)
        n = min(2000, xp.shape[0])
        idx = torch.randperm(xp.shape[0], generator=g)[:n]
        xs = F.normalize(xp[idx], dim=1) * math.sqrt(self.num_features)
        self._med2 = float(torch.cdist(xs, xs).median().item() ** 2)
        self._set_omega(self.bw_scale * self._med2)
        if self.bw_select == "val" and x_val is not None and y_val is not None:
            self.selected_bw = _select_bandwidth(
                self, self.bw_grid, xp, y, self.preproc.apply(x_val.float()), y_val,
                seed=self.seed)
        return self

    def _phase(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.preproc.apply(x.float().to(self.device))
        phase = xp @ self.omega.T
        if self.phase_bits is not None:
            step = 2 * math.pi / (2 ** self.phase_bits)
            phase = torch.round(phase / step) * step
        return phase

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            ph = self._phase(x[i:i + batch_size])
            if self.phase_bits == 1:
                outs.append(quant.sign(torch.cos(ph)))
            else:
                outs.append(torch.cat((torch.cos(ph), torch.sin(ph)), dim=1) / math.sqrt(2.0))
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        bits_per = self.phase_bits if self.phase_bits is not None else 32
        return self.dim * self.num_features * 32 + self.dim * bits_per \
            + self.preproc.memory_bits(self.num_features)

    def ops_per_query(self) -> int:
        mult = 1 if self.phase_bits == 1 else 2
        return int(self.dim * self.num_features + mult * self.dim) \
            + self.preproc.ops_per_query(self.num_features)

    def storage_bits_per_coord(self) -> float:
        if self.phase_bits == 1:
            return 1.0
        if self.phase_bits is not None:
            return float(self.phase_bits) / 2.0
        return 32.0


class NystromEncoder:
    """Nystrom kernel-approximation features (arbitrary PSD kernel).

    Uses a Cholesky factorization of the landmark kernel ``K = L L^T`` and
    the whitening map ``L^{-T}`` (equivalently ``U Lambda^{-1/2}`` up to an
    orthogonal rotation), giving the standard Nystrom approximation
    ``K_xz K^{-1} K_xz^T``. The factorization runs on CPU: cuSOLVER's syevd on
    a 4096x4096 landmark matrix is a multi-second GPU kernel that can starve
    a shared display GPU (see README "GPU stability" note).
    """

    family = "nystrom"
    kind = "nystrom"

    def __init__(self, dim: int, num_features: int, *, seed: int = 0,
                 kernel: str = "gaussian", bw_scale: float = 0.25,
                 n_landmarks: int | None = None, out: str = "bipolar",
                 preproc: str = "none", factor_device: str = "cpu",
                 device: str | None = None) -> None:
        self.dim = dim
        self.num_features = num_features
        self.seed = seed
        self.kernel = kernel
        self.bw_scale = bw_scale
        self.n_landmarks = n_landmarks or dim
        self.out = out
        self.preproc = Preproc(preproc)
        self.factor_device = factor_device
        self.device = resolve_device(device)
        self.landmarks: torch.Tensor | None = None
        self.W: torch.Tensor | None = None
        self.scale = 1.0
        self.dim_out = dim

    def _kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.kernel == "gaussian":
            d2 = torch.cdist(a, b) ** 2
            return torch.exp(-d2 / (2 * self.scale))
        if self.kernel == "laplacian":
            d1 = torch.cdist(a, b, p=1)
            return torch.exp(-d1 / self.scale)
        raise ValueError(self.kernel)

    def fit(self, x, y=None, x_val=None, y_val=None) -> "NystromEncoder":
        self.preproc.fit(x)
        xp = self.preproc.apply(x.float())
        g = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(xp.shape[0], generator=g)[: min(self.n_landmarks, xp.shape[0])]
        self.landmarks = xp[idx].clone()
        # bandwidth from a subsample
        n = min(1000, xp.shape[0])
        sub = xp[torch.randperm(xp.shape[0], generator=g)[:n]]
        if self.kernel == "gaussian":
            self.scale = max(float(torch.cdist(sub, sub).median().item() ** 2) * self.bw_scale, 1e-12)
        else:
            self.scale = max(float(torch.cdist(sub, sub, p=1).median().item()) * self.bw_scale, 1e-12)
        K = self._kernel(self.landmarks, self.landmarks).float()
        L = K.shape[0]
        eye = torch.eye(L, device=K.device)
        K = 0.5 * (K + K.T) + 1e-6 * eye  # symmetrize + jitter for PSD safety
        # whitening W = L_chol^{-T} so that Phi Phi^T = K_xz K^{-1} K_xz^T
        Kc = K.to(self.factor_device)
        Lc = torch.linalg.cholesky(Kc)
        Wc = torch.linalg.solve_triangular(
            Lc, torch.eye(L, device=Kc.device, dtype=Kc.dtype), upper=False).T
        self.W = Wc.to(self.device)
        self.landmarks = self.landmarks.to(self.device)
        self.dim_out = min(self.dim, L)
        self.W = self.W[:, : self.dim_out].contiguous()
        return self

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.preproc.apply(x.float().to(self.device))
        Kxz = self._kernel(xp, self.landmarks)
        return Kxz @ self.W

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            z = self._pre(x[i:i + batch_size])
            if self.out == "bipolar":
                z = quant.sign(z)
            elif self.out == "binary01":
                z = quant.binary01(z)
            outs.append(z)
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        # inference state: landmarks (L x F) + whitening map (L x dim_out);
        # the landmark kernel itself is only needed during fit.
        L = self.landmarks.shape[0] if self.landmarks is not None else self.n_landmarks
        r = self.dim_out
        return (L * self.num_features * 32 + L * r * 32) \
            + self.preproc.memory_bits(self.num_features)

    def ops_per_query(self) -> int:
        return int(self.n_landmarks * self.num_features + self.n_landmarks * self.dim_out) \
            + self.preproc.ops_per_query(self.num_features)

    def storage_bits_per_coord(self) -> float:
        return 32.0 if self.out == "real" else 1.0


# --------------------------------------------------------------------------
# learned / data-informed encoders
# --------------------------------------------------------------------------


class LearnedProjectionEncoder:
    """Projection trained end-to-end with a prototype-cosine surrogate.

    The forward pass binarizes ``P x`` with a straight-through estimator and
    scores against learnable unit-norm class prototypes; after training the
    learned proxies are discarded and the HDC classifier re-accumulates
    prototypes from the binarized features (PIONEER/LeHDC-style).
    """

    family = "learned"
    kind = "learned"

    def __init__(self, dim: int, num_features: int, *, seed: int = 0,
                 out: str = "bipolar", epochs: int = 40, lr: float = 1e-2,
                 temp: float = 0.05, weight_decay: float = 0.0,
                 batch_size: int = 512, preproc: str = "std", init: str = "gaussian",
                 device: str | None = None) -> None:
        self.dim = dim
        self.num_features = num_features
        self.seed = seed
        self.out = out
        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.preproc = Preproc(preproc)
        self.init = init
        self.device = resolve_device(device)
        self.P: torch.Tensor | None = None
        self.dim_out = dim

    def fit(self, x: torch.Tensor, y: torch.Tensor, x_val=None, y_val=None) -> "LearnedProjectionEncoder":
        self.preproc.fit(x)
        xp = self.preproc.apply(x.float())
        C = int(y.max().item()) + 1
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        if self.init == "gaussian":
            P0 = torch.randn(self.dim, self.num_features, generator=g) / math.sqrt(self.num_features)
        else:
            P0 = make_rp(self.init, self.dim, self.num_features, seed=self.seed)
        P = torch.nn.Parameter(P0.to(self.device))
        xp = xp.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            h = quant.sign(xp @ P.T)
            W = torch.zeros(C, self.dim, device=self.device)
            W.index_add_(0, y, h)
            W = F.normalize(W, dim=1)
        W = torch.nn.Parameter(W)
        opt = torch.optim.Adam([P, W], lr=self.lr, weight_decay=self.weight_decay)
        n = xp.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = xp[idx], y[idx]
                pre = xb @ P.T
                soft = torch.clamp(pre, -1.0, 1.0)
                hard = quant.sign(pre)
                h = hard + soft - soft.detach()  # straight-through
                logits = F.normalize(h, dim=1) @ F.normalize(W, dim=1).T * 20.0
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.P = P.detach().to(self.device)
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            pre = self.preproc.apply(x[i:i + batch_size].float().to(self.device)) @ self.P.T
            outs.append(quant.sign(pre) if self.out == "bipolar" else pre)
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        return self.dim * self.num_features * 32 + self.preproc.memory_bits(self.num_features)

    def ops_per_query(self) -> int:
        return int(self.dim * self.num_features + self.dim) \
            + self.preproc.ops_per_query(self.num_features)

    def storage_bits_per_coord(self) -> float:
        return 1.0 if self.out == "bipolar" else 32.0


class FixedProjectionEncoder:
    """Wrap an externally supplied projection matrix (e.g. from co-training)."""

    family = "fixed"
    kind = "fixed"

    def __init__(self, P: torch.Tensor, *, out: str = "bipolar",
                 device: str | None = None) -> None:
        self.P = P.to(resolve_device(device)).float()
        self.dim, self.num_features = self.P.shape
        self.out = out
        self.device = self.P.device
        self.dim_out = self.dim
        self.out_code = OutputCode(out)

    def fit(self, x=None, y=None, x_val=None, y_val=None) -> "FixedProjectionEncoder":
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            pre = x[i:i + batch_size].float().to(self.device) @ self.P.T
            outs.append(self.out_code.apply(pre))
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        return int(self.dim * self.num_features * 32)

    def ops_per_query(self) -> int:
        return int(self.dim * self.num_features + self.dim)

    def storage_bits_per_coord(self) -> float:
        return self.out_code.storage_bits_per_coord()


class CodecWrapper:
    """Apply an output codec (int/ternary/binary/block/thermo/VQ) to any encoder.

    Used to cross the output-quantization axis with encoders that do not
    expose their own output code (RFF, FPE, Nystrom, learned, ...).
    """

    family = "codec"
    kind = "codec"

    def __init__(self, base, codec: str, *, out_bits: int = 8, thermo_levels: int = 4,
                 block: int = 8, seed: int = 0) -> None:
        self.base = base
        self.codec = codec
        self.code = OutputCode(codec, bits=out_bits, levels=thermo_levels, block=block,
                               seed=seed)
        self.device = base.device
        self.dim_out = base.dim_out * self.code.dim_multiplier

    def fit(self, x, y=None, x_val=None, y_val=None) -> "CodecWrapper":
        self.base.fit(x, y, x_val, y_val)
        with torch.no_grad():
            h = self.base.encode(x)
            if h.shape[0] > 20000:
                h = h[:20000]
            self.code.fit(h)
            if self.codec == "vq":
                q = quant.VectorQuantizer(256, seed=self.code._seed)
                self.code.vq = q.fit(h)
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.code.apply(self.base.encode(x))

    def memory_bits(self) -> int:
        extra = 0
        if self.codec == "vq":
            extra = self.code.vq.memory_bits()
        return int(self.base.memory_bits()) + extra

    def ops_per_query(self) -> int:
        return int(self.base.ops_per_query()) + int(getattr(self.base, "dim_out", 0))

    def storage_bits_per_coord(self) -> float:
        return self.code.storage_bits_per_coord()


class DataInformedEncoder:
    """Select the best of K random projections on a held-out split.

    A cheap data-dependent construction: no gradients, only a single-pass
    prototype evaluation per candidate.
    """

    family = "datainformed"
    kind = "datainformed"

    def __init__(self, dim: int, num_features: int, *, rp: str = "gaussian",
                 rp_params: dict | None = None, K: int = 16, seed: int = 0,
                 out: str = "bipolar", preproc: str = "none",
                 device: str | None = None) -> None:
        self.dim = dim
        self.num_features = num_features
        self.rp = rp
        self.rp_params = dict(rp_params or {})
        self.K = K
        self.seed = seed
        self.out = out
        self.preproc = Preproc(preproc)
        self.device = resolve_device(device)
        self.P: torch.Tensor | None = None
        self.candidate_scores: list[float] = []
        self.dim_out = dim

    @torch.no_grad()
    def _val_acc(self, P: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
        h = quant.sign(x @ P.T)
        C = int(y.max().item()) + 1
        W = torch.zeros(C, self.dim, device=h.device)
        W.index_add_(0, y, h)
        scores = F.normalize(h, dim=1) @ F.normalize(W, dim=1).T
        return float((scores.argmax(dim=1) == y).float().mean().item())

    def fit(self, x, y=None, x_val=None, y_val=None) -> "DataInformedEncoder":
        assert y is not None
        self.preproc.fit(x)
        xp = self.preproc.apply(x.float())
        if x_val is not None and y_val is not None and x_val.shape[0] > 0:
            xtr, ytr, xva, yva = xp, y, self.preproc.apply(x_val.float()), y_val
        else:
            g = torch.Generator().manual_seed(self.seed)
            n = xp.shape[0]
            idx = torch.randperm(n, generator=g)
            n_val = max(1, int(0.2 * n))
            va, tr = idx[:n_val], idx[n_val:]
            xtr, ytr, xva, yva = xp[tr], y[tr], xp[va], y[va]
        xtr, ytr = xtr.to(self.device), ytr.to(self.device)
        xva, yva = xva.to(self.device), yva.to(self.device)
        best_acc, best_P = -1.0, None
        for k in range(self.K):
            P = make_rp(self.rp, self.dim, self.num_features,
                        seed=self.seed + 7919 * k, **self.rp_params).to(self.device)
            acc = self._val_acc(P, xtr, ytr) + self._val_acc(P, xva, yva)
            self.candidate_scores.append(acc)
            if acc > best_acc:
                best_acc, best_P = acc, P
        self.P = best_P
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            pre = self.preproc.apply(x[i:i + batch_size].float().to(self.device)) @ self.P.T
            outs.append(quant.sign(pre) if self.out == "bipolar" else pre)
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        return self.dim * self.num_features * 32 + self.preproc.memory_bits(self.num_features)

    def ops_per_query(self) -> int:
        return int(self.dim * self.num_features + self.dim) \
            + self.preproc.ops_per_query(self.num_features)

    def storage_bits_per_coord(self) -> float:
        return 1.0 if self.out == "bipolar" else 32.0


class IMPEncoder:
    """Input-modulated projection: x -> sign((x * g(x)) @ P^T).

    ``g(x) = 1 + gain * tanh(x V^T / sqrt(F))`` is a fixed random per-feature
    gain that adapts the projection to each input without learning.
    """

    family = "imp"
    kind = "imp"

    def __init__(self, dim: int, num_features: int, *, gain: float = 0.5,
                 rp: str = "gaussian", seed: int = 0, out: str = "bipolar",
                 preproc: str = "none", device: str | None = None) -> None:
        self.dim = dim
        self.num_features = num_features
        self.gain = gain
        self.rp = rp
        self.seed = seed
        self.out = out
        self.preproc = Preproc(preproc)
        self.device = resolve_device(device)
        self.P: torch.Tensor | None = None
        self.V: torch.Tensor | None = None
        self.dim_out = dim

    def fit(self, x, y=None, x_val=None, y_val=None) -> "IMPEncoder":
        self.preproc.fit(x)
        self.P = make_rp(self.rp, self.dim, self.num_features, seed=self.seed).to(self.device)
        g = torch.Generator().manual_seed(self.seed + 3)
        self.V = (torch.randn(self.num_features, self.num_features, generator=g)
                  / math.sqrt(self.num_features)).to(self.device)
        return self

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.preproc.apply(x.float().to(self.device))
        gate = 1.0 + self.gain * torch.tanh((xp @ self.V.T) / math.sqrt(self.num_features))
        return (xp * gate) @ self.P.T

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], batch_size):
            pre = self._pre(x[i:i + batch_size])
            outs.append(quant.sign(pre) if self.out == "bipolar" else pre)
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        return (self.dim * self.num_features + self.num_features ** 2) * 32 \
            + self.preproc.memory_bits(self.num_features)

    def ops_per_query(self) -> int:
        return int(self.num_features ** 2 + self.dim * self.num_features + self.dim) \
            + self.preproc.ops_per_query(self.num_features)

    def storage_bits_per_coord(self) -> float:
        return 1.0 if self.out == "bipolar" else 32.0


# --------------------------------------------------------------------------
# symbolic sequence encoder
# --------------------------------------------------------------------------


class SequenceEncoder:
    """Symbolic sequence encoder: symbol HVs + binding + positional coding.

    ``mode``:
      bag          - bundle symbol HVs (order invariant)
      bag_pos      - bundle bind(sym[x_t], pos_t)
      bigram       - bundle bind(bind(sym[x_t], roll(sym[x_{t+1}], 1)), pos_t)
      bigram_nopos - bundle bind(sym[x_t], roll(sym[x_{t+1}], 1))
    ``bind``: "mul" (bipolar elementwise), "xor" (binary {0,1} XOR),
              "circ" (HRR circular convolution on real HVs).
    ``pos_mode``: "perm" (cyclic roll of one base vector) or "random".
    """

    family = "seq"
    kind = "seq"

    def __init__(self, dim: int, vocab: int, length: int, *, mode: str = "bag_pos",
                 bind: str = "mul", pos_mode: str = "perm", out: str = "bipolar",
                 seed: int = 0, device: str | None = None) -> None:
        self.dim = dim
        self.vocab = vocab
        self.length = length
        self.mode = mode
        self.bind = bind
        self.pos_mode = pos_mode
        self.out = out
        self.seed = seed
        self.device = resolve_device(device)
        self.sym: torch.Tensor | None = None
        self.pos: torch.Tensor | None = None
        self.dim_out = dim

    def fit(self, x=None, y=None, x_val=None, y_val=None) -> "SequenceEncoder":
        g = torch.Generator().manual_seed(self.seed)
        if self.bind == "circ":
            self.sym = torch.randn(self.vocab, self.dim, generator=g) / math.sqrt(self.dim)
        else:
            self.sym = torch.where(torch.rand(self.vocab, self.dim, generator=g) < 0.5, -1.0, 1.0)
        if self.pos_mode == "random":
            if self.bind == "circ":
                self.pos = torch.randn(self.length, self.dim, generator=g) / math.sqrt(self.dim)
            else:
                self.pos = torch.where(torch.rand(self.length, self.dim, generator=g) < 0.5, -1.0, 1.0)
        else:
            if self.bind == "circ":
                self.pos = torch.randn(self.length, self.dim, generator=g) / math.sqrt(self.dim)
            else:
                self.pos = torch.where(torch.rand(self.length, self.dim, generator=g) < 0.5, -1.0, 1.0)
        self.sym = self.sym.to(self.device)
        self.pos = self.pos.to(self.device)
        return self

    def _bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.bind == "circ":
            fa = torch.fft.rfft(a, dim=-1)
            fb = torch.fft.rfft(b, dim=-1)
            return torch.fft.irfft(fa * fb, n=self.dim, dim=-1)
        return a * b

    def _roll(self, a: torch.Tensor, k: int = 1) -> torch.Tensor:
        return torch.roll(a, shifts=k, dims=-1)

    def _pos_code(self, t: int) -> torch.Tensor:
        if self.pos_mode == "perm":
            return self._roll(self.pos[0], t)
        return self.pos[t]

    @torch.no_grad()
    def encode(self, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        x = x.long().to(self.device)
        outs = []
        for i in range(0, x.shape[0], batch_size):
            seq = x[i:i + batch_size]                  # (n, T)
            s = self.sym[seq]                          # (n, T, D)
            if self.mode == "bag":
                h = s.sum(dim=1)
            elif self.mode == "bag_pos":
                p = torch.stack([self._pos_code(t) for t in range(seq.shape[1])])  # (T, D)
                h = (s * p.unsqueeze(0)).sum(dim=1)
            elif self.mode in ("bigram", "bigram_nopos"):
                a, b = s[:, :-1], s[:, 1:]
                term = self._bind(a, self._roll(b, 1))
                if self.mode == "bigram":
                    p = torch.stack([self._pos_code(t) for t in range(seq.shape[1] - 1)])
                    term = term * p.unsqueeze(0)
                h = term.sum(dim=1)
            else:
                raise ValueError(self.mode)
            if self.out == "bipolar":
                h = quant.sign(h)
            elif self.out == "binary01":
                h = quant.binary01(h)
            outs.append(h)
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        return (self.vocab * self.dim + (self.length if self.pos_mode == "random" else 1) * self.dim) * 32

    def ops_per_query(self) -> int:
        T = self.length
        if self.mode == "bag":
            return T * self.dim
        if self.mode == "bag_pos":
            return 2 * T * self.dim
        return 4 * T * self.dim

    def storage_bits_per_coord(self) -> float:
        return 1.0 if self.out == "bipolar" else 32.0


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------


def make_encoder(family: str, dim: int, num_features: int, *, seed: int = 0,
                 device: str | None = None, **kwargs):
    """Instantiate an (unfitted) encoder by family name."""
    if family == "linear":
        return LinearRPEncoder(dim, num_features, seed=seed, device=device, **kwargs)
    if family == "rff":
        return RFFEncoder(dim, num_features, seed=seed, device=device, **kwargs)
    if family == "fpe":
        return FPEEncoder(dim, num_features, seed=seed, device=device, **kwargs)
    if family == "nystrom":
        return NystromEncoder(dim, num_features, seed=seed, device=device, **kwargs)
    if family == "learned":
        return LearnedProjectionEncoder(dim, num_features, seed=seed, device=device, **kwargs)
    if family == "datainformed":
        return DataInformedEncoder(dim, num_features, seed=seed, device=device, **kwargs)
    if family == "imp":
        return IMPEncoder(dim, num_features, seed=seed, device=device, **kwargs)
    if family == "seq":
        return SequenceEncoder(dim, seed=seed, device=device, **kwargs)
    raise ValueError(f"unknown encoder family {family!r}; options: {ENCODER_FAMILIES}")
