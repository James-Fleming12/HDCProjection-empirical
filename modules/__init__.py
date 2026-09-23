"""Empirical study of random projection / kernel-approximation matrices in
HDC pipelines (feature extractor -> projection -> prototype classification).

Modules:
    rp.py         projection-matrix constructions (Gaussian, Rademacher,
                  Cauchy, Student-t, sparse, count-sketch, SRHT, orthogonal,
                  Sobol, ...)
    quant.py      hypervector / model coding schemes (MSE-PTQ, max-scaled,
                  sign, ternary, thermometer, sparse-block, VQ)
    data.py       synthetic workloads (Gaussian, fine-grained, XOR, shells,
                  heavy-tail, domain shift, novel class, symbolic sequences)
    encoders.py   projection encoders (linear RP, RFF, SignRFF, FPE, Nystrom,
                  learned, data-informed, input-modulated, sequences)
    extractors.py feature extractors (raw, random MLP, CE/SupCon/proxy/arc
                  trained MLPs, weight clustering, co-training)
    hd.py         OnlineHD-style prototype classifier + variants
    metrics.py    accuracy, OOD, clustering, margins, geometry fidelity, JL
    resources.py  memory / ops accounting
    bench.py      common evaluation harness (Spec -> CSV row)
"""
