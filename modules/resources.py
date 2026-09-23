"""Memory / compute accounting for projection-HDC pipelines.

Model (bits):
    projection parameters   nnz_P * bits_P
    class prototypes        C * dim * bits_W
    encoder aux state       (e.g. VQ codebook, thermometer range)

Compute proxy (elementwise ops per query):
    projection              nnz-equivalent multiply-adds (dense: dim*F)
    output nonlinearity     dim (sign, cos, ...)
    classification          C * dim similarity multiply-adds
Sparse and structured projections report their true nonzero count.
"""

from __future__ import annotations

from .quant import VectorQuantizer

__all__ = ["bits_to_kb", "model_memory_kb", "dense_proj_ops", "classifier_ops"]


def bits_to_kb(bits: float) -> float:
    return float(bits) / 8.0 / 1024.0


def model_memory_kb(
    proj_bits: int,
    num_classes: int,
    dim: int,
    *,
    proto_bits: int = 32,
    extra_bits: int = 0,
) -> float:
    return bits_to_kb(proj_bits + num_classes * dim * proto_bits + extra_bits)


def dense_proj_ops(nnz: int, dim: int) -> int:
    return int(nnz) + int(dim)


def classifier_ops(num_classes: int, dim: int) -> int:
    return int(num_classes) * int(dim)
