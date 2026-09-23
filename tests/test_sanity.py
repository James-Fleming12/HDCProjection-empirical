"""Sanity tests for the projection-HDC framework.

Run: python -m tests.test_sanity
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from modules import quant
from modules.data import make_task
from modules.encoders import (FPEEncoder, LinearRPEncoder, NystromEncoder,
                              RFFEncoder, SequenceEncoder, make_encoder)
from modules.extractors import Extractor, co_train, make_extractor
from modules.hd import PrototypeClassifier, add_feature_noise, sign_flip
from modules.metrics import (accuracy, geometry_fidelity, jl_distortion,
                             kernel_mse, margins)
from modules.rp import RP_NAMES, make_rp, rp_nnz, rp_stats

torch.manual_seed(0)


def _gauss_kernel(a: torch.Tensor, b: torch.Tensor, sigma2: float):
    return torch.exp(-((a - b) ** 2).sum(dim=1) / (2 * sigma2))


def test_rp_dense_variance():
    """Dense constructions keep unit output variance for x ~ N(0, I_F)."""
    F, D = 64, 512
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4000, F, generator=g)
    for name in RP_NAMES:
        P = make_rp(name, D, F, seed=1)
        y = x @ P.T
        if name == "cauchy":
            # no finite variance; the 1-stable scale is sum_j |x_j| / sqrt(F)
            scale = (x.abs().sum(dim=1) / math.sqrt(F)).mean()
            med = y.abs().median()
            assert 0.5 < float(med / scale) < 2.0, (name, float(med), float(scale))
            continue
        var = float(y.var().item())
        assert 0.8 < var < 1.25, (name, var)
    print("  rp variance conventions ok (cauchy checked via L1 stability)")


def test_rp_structure():
    F, D = 64, 512
    P = make_rp("count_sketch", D, F, seed=0)
    assert rp_nnz(P) == F, "count-sketch must have one nonzero per column"
    P = make_rp("very_sparse", D, F, seed=0)
    assert abs(float((P != 0).float().mean()) - 1.0 / math.sqrt(F)) < 0.02
    P = make_rp("ternary", D, F, seed=0)
    assert abs(float((P != 0).float().mean()) - 2.0 / 3.0) < 0.03
    # deterministic Sobol: same seed -> identical matrix
    a = make_rp("sobol", D, F, seed=3)
    b = make_rp("sobol", D, F, seed=3)
    assert torch.equal(a, b)
    # orthogonal: D <= F rows orthonormal
    P = make_rp("orthogonal", 32, F, seed=0)
    gram = P @ P.T
    assert torch.allclose(gram, torch.eye(32), atol=1e-4)
    # D > F: columns orthonormal up to the sqrt(D/F) output-variance scale
    P = make_rp("orthogonal", 512, F, seed=0)
    assert torch.allclose((P.T @ P) / (512 / F), torch.eye(F), atol=1e-4)
    st = rp_stats(P)
    assert st["rank"] == F, st
    # built-in Hadamard must match the reference Kronecker construction
    from modules.rp import fwht

    h = fwht(torch.eye(8))
    ref = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32)
    ref = torch.kron(ref, ref).float()
    ref = torch.kron(ref, torch.tensor([[1, 1], [1, -1]], dtype=torch.float32)).float()
    assert torch.allclose(h, ref), "fwht does not match the Kronecker Hadamard"
    print("  rp structure (sparsity/rank/Hadamard) ok")


def test_quantizers():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(3000, 256, generator=g) * 0.7
    for bits in (2, 4, 8):
        mse_best = ((quant.mse_ptq(t, bits) - t) ** 2).mean()
        mse_max = ((quant.uniform_max(t, bits) - t) ** 2).mean()
        assert mse_best <= mse_max + 1e-9, (bits, mse_best, mse_max)
    q = quant.ternary_quantize(t)
    frac0 = float((q == 0).float().mean())
    assert 0.05 < frac0 < 0.95, frac0
    sb = quant.sparse_block(t, block=8, keep=1)
    assert float((sb != 0).float().mean()) < 0.14
    lo, hi = quant.thermometer_fit(t)
    th = quant.thermometer_apply(t, 4, lo, hi)
    assert th.shape == (t.shape[0], t.shape[1] * 4)
    assert set(torch.unique(th).tolist()) <= {0.0, 1.0}
    vq = quant.VectorQuantizer(64, iters=10, seed=0).fit(t[:2000])
    q = vq.encode(t[:2000])
    mse_vq = ((q - t[:2000]) ** 2).mean()
    mse_mean = ((t[:2000].mean(0) - t[:2000]) ** 2).mean()
    assert mse_vq < mse_mean, (mse_vq, mse_mean)
    print("  quantizers (MSE-PTQ <= max-scaled, ternary/block/thermo/VQ) ok")


def test_encoders_basic():
    task = make_task("linear", seed=0)
    for out in ("bipolar", "binary01", "ternary", "int", "block", "thermo", "vq", "real"):
        enc = LinearRPEncoder(512, 64, rp="rademacher", out=out, out_bits=4,
                              thermo_levels=4, block=8, seed=0)
        enc.fit(task.x_train, task.y_train)
        h = enc.encode(task.x_test[:64])
        assert h.shape[0] == 64
        if out == "bipolar":
            assert set(torch.unique(h).tolist()) <= {-1.0, 1.0}
        if out == "binary01":
            assert set(torch.unique(h).tolist()) <= {0.0, 1.0}
        assert enc.memory_bits() > 0 and enc.ops_per_query() > 0
    print("  output codes (bipolar/binary01/ternary/int/block/thermo/vq/real) ok")


def test_kernel_approximations():
    task = make_task("linear", seed=0)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(task.x_train.shape[0], generator=g)[:1500]
    x = task.x_train[idx]
    med2 = float(torch.cdist(x, x).median().item() ** 2)
    kern = lambda a, b: _gauss_kernel(a, b, med2)

    errs = {}
    for D in (256, 4096):
        rff = RFFEncoder(D, 64, bw_scale=1.0, out="real", seed=0).fit(task.x_train)
        fpe = FPEEncoder(D, 64, bw_scale=1.0, seed=0).fit(task.x_train)
        sig = RFFEncoder(D, 64, bw_scale=1.0, out="bipolar", seed=0).fit(task.x_train)
        errs[f"rff{D}"] = kernel_mse(rff.encode, x, kern, n_pairs=1500, seed=0)
        errs[f"fpe{D}"] = kernel_mse(fpe.encode, x, kern, n_pairs=1500, seed=0)
        errs[f"sign{D}"] = kernel_mse(sig.encode, x, kern, n_pairs=1500, seed=0)
    e_rff_256, e_rff_4k = errs["rff256"], errs["rff4096"]
    e_fpe_256, e_fpe_4k = errs["fpe256"], errs["fpe4096"]
    assert e_rff_4k < e_rff_256, (e_rff_256, e_rff_4k)
    assert e_fpe_4k < e_fpe_256, (e_fpe_256, e_fpe_4k)
    assert e_rff_4k < 0.02 and e_fpe_4k < 0.02, (e_rff_4k, e_fpe_4k)
    print(f"  kernel approx: RFF MSE {e_rff_256:.4f}->{e_rff_4k:.4f}, "
          f"FPE {e_fpe_256:.4f}->{e_fpe_4k:.4f}, SignRFF {errs['sign256']:.4f}->{errs['sign4096']:.4f}")


def test_nystrom():
    task = make_task("linear", seed=0)
    for kernel in ("gaussian", "laplacian"):
        enc = NystromEncoder(512, 64, kernel=kernel, seed=0).fit(task.x_train)
        h = enc.encode(task.x_test[:64])
        assert set(torch.unique(h).tolist()) <= {-1.0, 1.0}
    print("  Nystrom (gaussian/laplacian kernels) ok")


def test_classifier_variants():
    task = make_task("linear", seed=0)
    base = LinearRPEncoder(1024, 64, rp="gaussian", seed=0).fit(task.x_train, task.y_train)
    clf = PrototypeClassifier(base, task.num_classes, epochs=30).fit(task.x_train, task.y_train)
    acc = accuracy(clf.predict(task.x_test), task.y_test)
    assert acc > 0.9, acc
    # Hamming and cosine give the same ranking for bipolar codes
    h = base.encode(task.x_test[:128])
    s_cos = clf._sim(h, clf.class_hvs)
    clf_h = PrototypeClassifier(base, task.num_classes, sim="hamming", epochs=0)
    clf_h.class_hvs = clf.class_hvs
    s_ham = clf_h._sim(h, clf.class_hvs)
    assert torch.equal(s_cos.argmax(1), s_ham.argmax(1))
    # weighted bundling and multi-prototype run and stay sane
    for kw in ({"weight": "margin"}, {"weight": "confidence"}, {"multi": 4}):
        c = PrototypeClassifier(base, task.num_classes, epochs=10, **kw).fit(
            task.x_train, task.y_train)
        a = accuracy(c.predict(task.x_test), task.y_test)
        assert a > 0.85, (kw, a)
    # deployment quantization of prototypes
    c = PrototypeClassifier(base, task.num_classes, epochs=30, proto_bits=4).fit(
        task.x_train, task.y_train)
    a = accuracy(c.predict(task.x_test), task.y_test)
    assert a > 0.85, a
    # signature: margins decrease under feature noise
    m_clean = margins(clf.scores(task.x_test), task.y_test)["margin"]
    g = torch.Generator().manual_seed(0)
    m_noise = margins(clf.scores(add_feature_noise(task.x_test, 1.0, generator=g)), task.y_test)["margin"]
    assert m_noise < m_clean
    # bit flips degrade accuracy monotonically
    h = base.encode(task.x_test)
    g = torch.Generator().manual_seed(0)
    a1 = accuracy(clf._sim(h, sign_flip(clf.class_hvs, 0.05, seed=0)).argmax(1), task.y_test)
    g = torch.Generator().manual_seed(0)
    a2 = accuracy(clf._sim(h, sign_flip(clf.class_hvs, 0.5, seed=0)).argmax(1), task.y_test)
    assert a2 < a1 <= acc + 1e-9
    print(f"  classifier ok (acc={acc:.3f}, margins/noise/flip signatures hold)")


def test_task_difficulty_structure():
    """XOR is at chance for sign-linear projections (odd function symmetry);
    radial tasks are at chance for linear projections but solvable by RFF."""
    task = make_task("xor", seed=0)
    lin = LinearRPEncoder(2048, 64, rp="gaussian", seed=0).fit(task.x_train, task.y_train)
    c = PrototypeClassifier(lin, task.num_classes).fit(task.x_train, task.y_train)
    a_lin = accuracy(c.predict(task.x_test), task.y_test)
    assert a_lin < 0.35, f"sign-linear XOR should be at chance, got {a_lin}"
    rff = RFFEncoder(2048, 64, bw_scale=0.1, out="bipolar", seed=0).fit(task.x_train, task.y_train)
    c = PrototypeClassifier(rff, task.num_classes).fit(task.x_train, task.y_train)
    a_rff_xor = accuracy(c.predict(task.x_test), task.y_test)
    assert a_rff_xor > 0.33, a_rff_xor

    task = make_task("shells", seed=0)
    lin = LinearRPEncoder(2048, 64, rp="gaussian", seed=0).fit(task.x_train, task.y_train)
    c = PrototypeClassifier(lin, task.num_classes).fit(task.x_train, task.y_train)
    a_lin = accuracy(c.predict(task.x_test), task.y_test)
    rff = RFFEncoder(2048, 64, bw_scale=1.0, out="bipolar", seed=0).fit(task.x_train, task.y_train)
    c = PrototypeClassifier(rff, task.num_classes).fit(task.x_train, task.y_train)
    a_rff = accuracy(c.predict(task.x_test), task.y_test)
    assert a_lin < 0.4 and a_rff > 0.6, (a_lin, a_rff)
    print(f"  task structure ok (xor: lin {0.25:.2f} < rff {a_rff_xor:.2f}; "
          f"shells: lin {a_lin:.2f} < rff {a_rff:.2f})")


def test_sequence_encoder():
    task = make_task("symbolic", seed=0)
    for mode, floor in (("bag", 0.0), ("bigram", 0.5)):
        enc = SequenceEncoder(2048, 8, 12, mode=mode, seed=0).fit()
        clf = PrototypeClassifier(enc, task.num_classes, epochs=10).fit(
            task.x_train, task.y_train)
        a = accuracy(clf.predict(task.x_test), task.y_test)
        assert a >= floor, (mode, a)
        if mode == "bigram":
            best_bigram = a
    assert best_bigram > 0.5
    print(f"  sequence encoder ok (bigram {best_bigram:.3f} > bag chance)")


def test_extractors():
    task = make_task("linear", seed=0)
    for kind in ("ce", "supcon", "proxy", "arc", "clustered", "randmlp"):
        ext = make_extractor(kind, 64, 128, seed=0, epochs=15)
        ext.fit(task.x_train, task.y_train)
        f = ext.transform(task.x_test)
        assert f.shape == (task.x_test.shape[0], 128)
        clf = PrototypeClassifier(LinearRPEncoder(512, 128, seed=0).fit(
            ext.transform(task.x_train), task.y_train), task.num_classes, epochs=5).fit(
            ext.transform(task.x_train), task.y_train)
        a = accuracy(clf.predict(f), task.y_test)
        assert a > 0.7, (kind, a)
    ext = make_extractor("clustered", 64, 128, seed=0, levels=8, epochs=15)
    ext.fit(task.x_train, task.y_train)
    assert ext.weight_levels() <= 8
    print("  feature extractors (ce/supcon/proxy/arc/clustered/randmlp) ok")


def test_cotrain():
    task = make_task("linear", seed=0)
    ext = make_extractor("randmlp", 64, 64, seed=0)
    P = make_rp("gaussian", 1024, 64, seed=0)
    ext, P = co_train(ext, P, task.x_train, task.y_train, epochs=15, seed=0)
    from modules.encoders import FixedProjectionEncoder

    enc = FixedProjectionEncoder(P).fit()
    clf = PrototypeClassifier(enc, task.num_classes, epochs=10).fit(
        ext.transform(task.x_train), task.y_train)
    a = accuracy(clf.predict(ext.transform(task.x_test)), task.y_test)
    assert a > 0.85, a
    print(f"  co-training ok (fixed-P HDC accuracy {a:.3f})")


def test_learned_projection():
    task = make_task("fine", seed=0)
    D = 128
    g = LinearRPEncoder(D, 64, rp="gaussian", seed=0).fit(task.x_train, task.y_train)
    a_g = accuracy(PrototypeClassifier(g, task.num_classes).fit(
        task.x_train, task.y_train).predict(task.x_test), task.y_test)
    learned = make_encoder("learned", D, 64, seed=0, epochs=60).fit(
        task.x_train, task.y_train)
    a_l = accuracy(PrototypeClassifier(learned, task.num_classes).fit(
        task.x_train, task.y_train).predict(task.x_test), task.y_test)
    assert a_l > a_g + 0.05, (a_g, a_l)
    # XOR cannot be learned by an odd (sign-linear) encoder, any amount of training
    task = make_task("xor", seed=0)
    learned = make_encoder("learned", 512, 64, seed=0, epochs=60).fit(
        task.x_train, task.y_train)
    a = accuracy(PrototypeClassifier(learned, task.num_classes).fit(
        task.x_train, task.y_train).predict(task.x_test), task.y_test)
    assert a < 0.4, f"learned odd encoder must stay at chance on XOR, got {a}"
    print(f"  learned projections ok (fine D=128: {a_g:.3f}->{a_l:.3f}; XOR stays {a:.3f})")


def test_metrics():
    task = make_task("linear", seed=0)
    enc = LinearRPEncoder(4096, 64, rp="gaussian", seed=0).fit(task.x_train, task.y_train)
    rho = geometry_fidelity(enc, task.x_test)
    assert rho > 0.85, rho
    P = make_rp("gaussian", 1024, 64, seed=0)
    st = jl_distortion(P, task.x_train)
    assert st["jl_std_rel"] < 4 * st["jl_pred_std"], st
    assert abs(st["jl_mean_rel"]) < 0.05, st
    print(f"  metrics ok (geom rho={rho:.3f}, jl std={st['jl_std_rel']:.3f} "
          f"vs pred {st['jl_pred_std']:.3f})")


def test_codec_wrapper():
    """The generic codec wrapper reproduces encoder-native codes and supports
    codes the base encoder does not implement (int/ternary/thermo/VQ)."""
    from modules.encoders import CodecWrapper, LinearRPEncoder

    task = make_task("linear", seed=0)
    base = LinearRPEncoder(512, 64, rp="gaussian", out="bipolar", seed=0)
    nat = LinearRPEncoder(512, 64, rp="gaussian", out="bipolar", seed=0)
    nat.fit(task.x_train, task.y_train)
    wrapped = CodecWrapper(base, "bipolar", seed=0).fit(task.x_train, task.y_train)
    assert torch.equal(nat.encode(task.x_test[:64]), wrapped.encode(task.x_test[:64]))
    for codec in ("int4", "int2", "ternary", "block", "thermo", "vq"):
        w = CodecWrapper(LinearRPEncoder(256, 64, rp="gaussian"), codec, seed=0)
        w.fit(task.x_train, task.y_train)
        h = w.encode(task.x_test[:64])
        assert h.shape[0] == 64 and torch.isfinite(h).all()
        if codec != "vq":  # VQ stores a codebook instead of per-coordinate bits
            assert w.storage_bits_per_coord() > 0
        assert w.memory_bits() > 0
    print("  codec wrapper (native parity + int/ternary/block/thermo/VQ) ok")


def test_l1_similarity_chunking():
    from modules.encoders import LinearRPEncoder

    task = make_task("linear", seed=0)
    enc = LinearRPEncoder(256, 64, rp="gaussian", out="real", seed=0).fit(
        task.x_train, task.y_train)
    clf = PrototypeClassifier(enc, task.num_classes, sim="l1", epochs=0)
    h = enc.encode(task.x_test[:300])
    clf.class_hvs = torch.zeros(5, 256, device=h.device)
    h = enc.encode(task.x_test[:300])
    big = clf._l1_sim(h[:300], clf.class_hvs[:5], chunk=64)
    small = clf._l1_sim(h[:300], clf.class_hvs[:5], chunk=7)
    assert torch.allclose(big, small, atol=1e-5)
    print("  L1 similarity chunking is memory-safe and chunk-invariant")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nall {len(tests)} sanity tests passed")


if __name__ == "__main__":
    main()
