"""Random projection (RP) matrix constructions for HDC encoders.

Scaling convention
------------------
All dense constructions are scaled so that for ``x ~ N(0, I_F)`` the
projected vector ``y = P x`` has ``E[y_i^2] ~= 1`` (unit output variance per
coordinate). This makes sign()/quantization thresholds comparable across
constructions and gives the ops model a common unit. Cauchy entries have no
finite variance; their nominal 1/sqrt(F) scaling only fixes the median
magnitude (the distance preserved by Cauchy projections is the L1 distance,
see exp9).

Structured constructions report their nonzero count for the ops model.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["RP_NAMES", "make_rp", "fwht", "rp_nnz", "rp_stats", "rp_description",
           "rp_storage_bits"]

RP_NAMES = [
    "gaussian",
    "rademacher",
    "uniform",
    "laplace",
    "student_t",
    "cauchy",
    "achlioptas",
    "ternary",
    "very_sparse",
    "count_sketch",
    "srht",
    "hadamard_det",
    "orthogonal",
    "orth_blend",
    "sobol",
    "binary01",
]


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Fast Walsh-Hadamard transform along the last dimension.

    The last dimension must be a power of two. The unscaled transform is
    orthogonal up to a factor sqrt(n).
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"fwht needs a power-of-two length, got {n}"
    orig = x.shape
    x = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.view(x.shape[0], n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.cat((a + b, a - b), dim=2)
        h *= 2
    return x.reshape(orig)


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1)).bit_length()


# --------------------------------------------------------------------------
# generators: f(D, F, rng, **params) -> np.ndarray (D, F)
# --------------------------------------------------------------------------


def _gaussian(D, F, rng, **_):
    return rng.standard_normal((D, F)) / math.sqrt(F)


def _rademacher(D, F, rng, **_):
    return np.where(rng.random((D, F)) < 0.5, -1.0, 1.0) / math.sqrt(F)


def _uniform(D, F, rng, **_):
    return rng.uniform(-1.0, 1.0, (D, F)) * math.sqrt(3.0 / F)


def _laplace(D, F, rng, **_):
    return rng.laplace(0.0, 1.0, (D, F)) / math.sqrt(2.0 * F)


def _student_t(D, F, rng, nu=3.0, **_):
    z = rng.standard_t(nu, (D, F))
    if nu > 2:
        z = z / math.sqrt(nu / (nu - 2.0))
    return z / math.sqrt(F)


def _cauchy(D, F, rng, **_):
    return rng.standard_cauchy((D, F)) / math.sqrt(F)


def _ternary(D, F, rng, density=2.0 / 3.0, **_):
    """Symmetric sparse {-1, 0, +1} projection (Achlioptas family).

    ``density`` is the probability of a nonzero entry; per-entry variance is
    kept at 1/F, so the nonzero value is ``1/sqrt(density * F)``.
    """
    u = rng.random((D, F))
    P = np.zeros((D, F), dtype=np.float64)
    v = 1.0 / math.sqrt(density * F)
    P[u < density / 2] = v
    P[u > 1 - density / 2] = -v
    return P


def _achlioptas(D, F, rng, **_):
    """Achlioptas' sparse RP: {-1, 0, +1} at 1/6, 2/3, 1/6."""
    return _ternary(D, F, rng, density=1.0 / 3.0)


def _very_sparse(D, F, rng, s=None, **_):
    """Very sparse RP (Li, Hastie & Church, 2006).

    Entries are nonzero with probability 1/s. Default ``s = sqrt(F)``, the
    paper's recommendation; ``s = F`` gives ~1 nonzero per row.
    """
    if s is None:
        s = math.sqrt(F)
    return _ternary(D, F, rng, density=1.0 / s)


def _count_sketch(D, F, rng, **_):
    """Count-Sketch / feature-hashing projection: one nonzero per column."""
    P = np.zeros((D, F), dtype=np.float64)
    rows = rng.integers(0, D, size=F)
    signs = rng.choice(np.array([-1.0, 1.0]), size=F)
    P[rows, np.arange(F)] = signs * math.sqrt(D / F)
    return P


def _srht(D, F, rng, **_):
    """Subsampled Randomized Hadamard Transform: P = S H D / sqrt(F)."""
    Fp = _next_pow2(F)
    rows = rng.integers(0, Fp, size=D)
    S = np.zeros((D, Fp), dtype=np.float64)
    S[np.arange(D), rows] = 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=Fp)
    Hs = fwht(torch.from_numpy(S).float()).numpy()
    return (Hs[:, :F] * signs[:F]) / math.sqrt(F)


def _hadamard_det(D, F, rng, **_):
    """Deterministic subsampled Hadamard: first D rows, no random signs."""
    Fp = _next_pow2(F)
    S = np.zeros((D, Fp), dtype=np.float64)
    S[np.arange(D), np.arange(D) % Fp] = 1.0
    Hs = fwht(torch.from_numpy(S).float()).numpy()
    return Hs[:, :F] / math.sqrt(F)


def _orthogonal(D, F, rng, **_):
    """Haar-distributed (semi-)orthogonal projection with unit output variance.

    * ``D <= F``: rows are orthonormal (P = Q^T, Q from QR of an F x D
      Gaussian), so P maps R^F isometrically onto a random D-dim subspace.
    * ``D > F``: columns are orthonormal and rows are scaled by sqrt(D/F)
      (the standard semi-orthogonal RP, rank F).
    """
    if D <= F:
        G = rng.standard_normal((F, D))
        Q, _ = np.linalg.qr(G)
        return Q.T
    G = rng.standard_normal((D, F))
    Q, _ = np.linalg.qr(G)
    return Q * math.sqrt(D / F)


def _orth_blend(D, F, rng, alpha=0.5, **_):
    """Partially orthogonalized RP: sqrt(a) P_orth + sqrt(1-a) P_gauss."""
    a = float(alpha)
    return math.sqrt(a) * _orthogonal(D, F, rng) + math.sqrt(1.0 - a) * _gaussian(D, F, rng)


def _sobol(D, F, rng, seed=0, **_):
    """Low-discrepancy (scrambled Sobol) projection, inverse-normal entries.

    Each *row* of P is an F-dimensional scrambled Sobol point pushed through
    the normal inverse CDF, so the matrix is a deterministic function of the
    seed (reproducible; no run-to-run PRNG variation). A balanced power-of-two
    number of points is drawn and truncated to D rows.
    """
    import warnings

    import scipy.stats
    from scipy.stats import qmc

    sob = qmc.Sobol(d=F, scramble=True, seed=int(seed))
    n = 1 << max(1, math.ceil(math.log2(max(D, 2))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = sob.random_base2(int(math.log2(n)))
    u = np.clip(u[:D], 1e-12, 1.0 - 1e-12)
    z = scipy.stats.norm.ppf(u)
    return z / math.sqrt(F)


def _binary01(D, F, rng, **_):
    """Unipolar binary RP with entries in {0, sqrt(2/F)} (nonzero mean)."""
    return (rng.random((D, F)) < 0.5).astype(np.float64) * math.sqrt(2.0 / F)


_GENERATORS = {
    "gaussian": _gaussian,
    "rademacher": _rademacher,
    "uniform": _uniform,
    "laplace": _laplace,
    "student_t": _student_t,
    "cauchy": _cauchy,
    "achlioptas": _achlioptas,
    "ternary": _ternary,
    "very_sparse": _very_sparse,
    "count_sketch": _count_sketch,
    "srht": _srht,
    "hadamard_det": _hadamard_det,
    "orthogonal": _orthogonal,
    "orth_blend": _orth_blend,
    "sobol": _sobol,
    "binary01": _binary01,
}

_PARAMS = {
    "student_t": ("nu",),
    "ternary": ("density",),
    "very_sparse": ("s",),
    "orth_blend": ("alpha",),
}


def rp_description(name: str, **params) -> str:
    """Short human-readable descriptor used in result tables."""
    if name in _PARAMS and params:
        spec = ",".join(f"{k}={params[k]}" for k in _PARAMS[name] if k in params)
        return f"{name}({spec})"
    if name == "very_sparse":
        return "very_sparse(s=sqrt(F))"
    if name == "student_t":
        return "student_t(nu=3)"
    if name == "ternary":
        return "ternary(d=2/3)"
    if name == "orth_blend":
        return "orth_blend(a=0.5)"
    return name


def make_rp(name: str, D: int, F: int, *, seed: int = 0, **params) -> torch.Tensor:
    """Build a (D, F) projection matrix of the requested construction."""
    if name not in _GENERATORS:
        raise ValueError(f"unknown RP construction {name!r}; options: {RP_NAMES}")
    # Offset the RNG stream per construction so that changing the RP type
    # changes the matrix, while each type stays reproducible from `seed`.
    rng = np.random.default_rng(seed + 1000 * RP_NAMES.index(name))
    if name == "sobol":
        params.setdefault("seed", seed)
    P = _GENERATORS[name](D, F, rng, **params)
    return torch.as_tensor(P, dtype=torch.float32)


@torch.no_grad()
def rp_nnz(P: torch.Tensor) -> int:
    return int((P != 0).sum().item())


@torch.no_grad()
def rp_storage_bits(name: str, P: torch.Tensor) -> int:
    """Bits needed to *store* a projection matrix in its natural encoding.

    Dense float constructions: 32 bits/entry. Sign matrices (Rademacher,
    SRHT, subsampled Hadamard, unipolar binary): 1 bit/entry. Sparse ternary
    families: 2 bits per nonzero. Count-Sketch: 1 sign bit + log2(D) index
    bits per nonzero.
    """
    D, _F = P.shape
    nnz = rp_nnz(P)
    if name in ("rademacher", "srht", "hadamard_det", "binary01"):
        return int(nnz)
    if name in ("achlioptas", "ternary", "very_sparse"):
        return int(2 * nnz)
    if name == "count_sketch":
        return int(nnz * (1 + max(1, (D - 1).bit_length())))
    return int(P.numel() * 32)


@torch.no_grad()
def rp_stats(P: torch.Tensor) -> dict:
    """density / rank / mean-abs statistics of a projection matrix."""
    nnz = rp_nnz(P)
    sv = torch.linalg.svdvals(P.float())
    return {
        "nnz": nnz,
        "density": nnz / P.numel(),
        "rank": int((sv > 1e-6 * sv[0]).sum().item()),
        "mean_abs": float(P.abs().mean().item()),
    }
