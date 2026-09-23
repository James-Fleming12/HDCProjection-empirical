"""Quantization and coding schemes for projected hypervectors.

Two families:

* Scalar quantizers that dequantize back to floats, so the downstream
  prototype classifier is unchanged:

    - ``mse_ptq``      : MSE-optimal symmetric scale search (DPQ-HD Alg. 1)
    - ``uniform_max``  : MicroHD-style max-magnitude symmetric quantization
    - ``sign``/``binary01`` : 1-bit codes
    - ``ternary``      : MSE-optimal {-a, 0, +a} threshold search
                         (QuantHD-style storage)
    - ``sparse_block`` : block-sparse top-k codes (SparseHD-style)
    - ``thermometer``  : TQHD-style thermometer codes (expands the dimension)
    - ``VectorQuantizer`` : k-means codebook quantization (HDVQ-VAE-style)
"""

from __future__ import annotations

import torch

__all__ = [
    "sign",
    "binary01",
    "mse_scale",
    "apply_scale",
    "mse_ptq",
    "uniform_max",
    "ternary_fit",
    "ternary_apply",
    "ternary_quantize",
    "sparse_block",
    "thermometer_fit",
    "thermometer_apply",
    "VectorQuantizer",
]

# DPQ-HD Algorithm 1 candidate fractions of the max-based scale.
_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@torch.no_grad()
def sign(t: torch.Tensor) -> torch.Tensor:
    """Bipolar sign code {-1, +1} (zeros map to +1)."""
    return torch.where(t >= 0, 1.0, -1.0)


@torch.no_grad()
def binary01(t: torch.Tensor) -> torch.Tensor:
    """Unipolar {0, 1} code (Hamming-friendly)."""
    return (t >= 0).to(t.dtype)


@torch.no_grad()
def mse_scale(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Return the MSE-optimal symmetric quantization scale for ``t``."""
    q_max = 2 ** (bits - 1) - 1
    s0 = t.abs().max().clamp_min(1e-12) / q_max
    best_s, best_err = s0, float("inf")
    for frac in _FRACS:
        s = frac * s0
        tq = torch.round(t / s).clamp(-(q_max + 1), q_max) * s
        err = ((tq - t) ** 2).mean().item()
        if err < best_err:
            best_err, best_s = err, s
    return best_s


@torch.no_grad()
def apply_scale(t: torch.Tensor, scale: torch.Tensor | float, bits: int) -> torch.Tensor:
    """Symmetric uniform quantization at a fixed scale, dequantized to float."""
    q_max = 2 ** (bits - 1) - 1
    return torch.round(t / scale).clamp(-(q_max + 1), q_max) * scale


@torch.no_grad()
def mse_ptq(t: torch.Tensor, bits: int) -> torch.Tensor:
    """MSE-optimal symmetric PTQ (DPQ-HD Algorithm 1), dequantized."""
    if bits >= 31:
        return t
    return apply_scale(t, mse_scale(t, bits), bits)


@torch.no_grad()
def uniform_max(t: torch.Tensor, bits: int) -> torch.Tensor:
    """MicroHD-style symmetric quantization over the observed max magnitude."""
    if bits >= 31:
        return t
    q_max = 2 ** (bits - 1) - 1
    s = t.abs().max().clamp_min(1e-12) / q_max
    return apply_scale(t, s, bits)


@torch.no_grad()
def ternary_fit(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE-optimal ternary codebook: (threshold tau, kept magnitude a).

    The dequantized code is ``sign(t) * a`` where ``|t| > tau`` and 0 else.
    ``tau`` is searched over fractions of the mean |t| and ``a`` is the mean
    magnitude of the kept entries (MSE-optimal for a fixed support).
    """
    mean_abs = t.abs().mean().clamp_min(1e-12)
    best = (float("inf"), mean_abs, torch.zeros((), device=t.device, dtype=t.dtype))
    for frac in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
        tau = frac * mean_abs
        kept = t.abs() > tau
        if kept.sum() == 0:
            continue
        a = t[kept].abs().mean()
        tq = torch.where(kept, torch.sign(t) * a, torch.zeros_like(t))
        err = ((tq - t) ** 2).mean().item()
        if err < best[0]:
            best = (err, tau, a)
    return best[1], best[2]


@torch.no_grad()
def ternary_apply(t: torch.Tensor, tau: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    kept = t.abs() > tau
    return torch.where(kept, torch.sign(t) * a, torch.zeros_like(t))


@torch.no_grad()
def ternary_quantize(t: torch.Tensor) -> torch.Tensor:
    tau, a = ternary_fit(t)
    return ternary_apply(t, tau, a)


@torch.no_grad()
def sparse_block(t: torch.Tensor, block: int = 8, keep: int = 1, mode: str = "sign") -> torch.Tensor:
    """Keep the top-``keep`` magnitudes per ``block`` (SparseHD-style codes).

    ``mode="sign"`` stores +/-1 for kept entries (1 bit + position);
    ``mode="magnitude"`` stores the value.
    """
    n, d = t.shape
    assert d % block == 0, "dimension must be divisible by block size"
    tb = t.view(n, d // block, block)
    _, idx = tb.abs().topk(keep, dim=-1)
    out = torch.zeros_like(tb)
    if mode == "sign":
        out.scatter_(-1, idx, torch.sign(torch.gather(tb, -1, idx)))
    else:
        out.scatter_(-1, idx, torch.gather(tb, -1, idx))
    return out.view(n, d)


@torch.no_grad()
def thermometer_fit(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor (lo, hi) range for thermometer encoding (TQHD)."""
    return t.min(), t.max()


@torch.no_grad()
def thermometer_apply(
    t: torch.Tensor, levels: int, lo: torch.Tensor, hi: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:
    """Thermometer code: expands (n, d) -> (n, d * levels) binary vectors.

    Coordinate ``x`` is normalized to [0, 1] with the fitted range and coded
    as ``levels`` bits; bit k is 1 iff ``x > k / (levels + 1)`` (levels + 1
    distinguishable values). Rows are processed in chunks to bound the size of
    the (chunk, d, levels) intermediate.
    """
    thr = torch.arange(1, levels + 1, device=t.device, dtype=t.dtype) / (levels + 1)
    outs = []
    for i in range(0, t.shape[0], chunk):
        tb = t[i:i + chunk]
        v = ((tb - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        outs.append((v.unsqueeze(-1) > thr).to(t.dtype).reshape(v.shape[0], -1))
    return torch.cat(outs, dim=0)


class VectorQuantizer:
    """k-means codebook quantization of hypervectors (HDVQ-VAE-style)."""

    def __init__(self, num_codes: int = 256, iters: int = 10, seed: int = 0,
                 max_fit: int = 8192) -> None:
        self.num_codes = num_codes
        self.iters = iters
        self.seed = seed
        self.max_fit = max_fit
        self.codebook: torch.Tensor | None = None

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "VectorQuantizer":
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        x = x.float()
        n = x.shape[0]
        if n > self.max_fit:  # fit the codebook on a subsample
            idx = torch.randperm(n, generator=g)[: self.max_fit].to(x.device)
            x = x[idx]
            n = x.shape[0]
        num_codes = min(self.num_codes, n)
        idx = torch.randperm(n, generator=g)[:num_codes].to(x.device)
        c = x[idx].clone()
        for _ in range(self.iters):
            assign = torch.cdist(x, c).argmin(dim=1)
            counts = torch.bincount(assign, minlength=num_codes)
            sums = torch.zeros_like(c)
            sums.index_add_(0, assign, x)
            alive = counts > 0
            c = torch.where(alive.view(-1, 1), sums / counts.clamp_min(1).view(-1, 1), c)
        self.codebook = c
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert self.codebook is not None, "call fit first"
        d = torch.cdist(x.float(), self.codebook.to(x.device))
        return self.codebook.to(x.device)[d.argmin(dim=1)]

    def memory_bits(self, bits: int = 32) -> int:
        k, d = self.codebook.shape
        return k * d * bits
