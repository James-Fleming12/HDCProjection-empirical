"""Shared Spec definitions for the experiments."""

from __future__ import annotations

from modules.bench import Spec

# --------------------------------------------------------------------------
# core projection/encoder library (exp1, exp2, exp3)
# --------------------------------------------------------------------------

RP_METHODS = [
    "gaussian", "rademacher", "uniform", "laplace", "student_t", "cauchy",
    "achlioptas", "ternary", "very_sparse", "count_sketch", "srht",
    "hadamard_det", "orthogonal", "orth_blend", "sobol", "binary01",
]


def linear_specs() -> list[Spec]:
    return [Spec(name=rp, family="linear", kwargs=dict(rp=rp), out="bipolar") for rp in RP_METHODS]


def nonlinear_specs() -> list[Spec]:
    return [
        Spec(name="rff_b0.25", family="rff", out="bipolar", kwargs=dict(bw_scale=0.25)),
        Spec(name="rff_b1.0", family="rff", out="bipolar", kwargs=dict(bw_scale=1.0)),
        Spec(name="rff_real_b1.0", family="rff", out="real", kwargs=dict(bw_scale=1.0)),
        Spec(name="signrff_tuned", family="rff", out="bipolar",
             kwargs=dict(bw_select="val", bw_grid=(0.05, 0.1, 0.25, 0.5, 1.0))),
        Spec(name="fpe_phasor", family="fpe", kwargs=dict(bw_scale=0.25)),
        Spec(name="fpe_b1", family="fpe", kwargs=dict(bw_scale=0.25, phase_bits=1)),
        Spec(name="fpe_b3", family="fpe", kwargs=dict(bw_scale=0.25, phase_bits=3)),
        Spec(name="nystrom_gauss", family="nystrom", kwargs=dict(kernel="gaussian")),
        Spec(name="nystrom_lap", family="nystrom", kwargs=dict(kernel="laplacian")),
        Spec(name="learned", family="learned", kwargs=dict(epochs=60)),
        Spec(name="datainformed_k16", family="datainformed", kwargs=dict(K=16)),
        Spec(name="imp_a0.5", family="imp", kwargs=dict(gain=0.5)),
    ]


def core_specs() -> list[Spec]:
    return linear_specs() + nonlinear_specs() + [
        Spec(name="linear_real_fp32", family="linear", kwargs=dict(rp="gaussian"), out="real"),
    ]


# shortlist used by the sweeps that vary another axis (dimension, robustness, ...)
SWEEP_METHODS = ["gaussian", "rademacher", "very_sparse", "count_sketch", "srht",
                 "orthogonal", "sobol", "signrff_tuned", "fpe_b1", "learned",
                 "datainformed_k16"]


def sweep_specs() -> list[Spec]:
    core = {s.name: s for s in core_specs()}
    return [core[n] for n in SWEEP_METHODS]


TASKS_VECTOR = ["linear", "fine", "xor", "shells", "heavy_tail", "shift", "novel"]
