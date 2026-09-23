"""Generalization, robustness and geometry metrics for HDC pipelines."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

__all__ = [
    "accuracy",
    "hungarian_accuracy",
    "clustering_metrics",
    "ood_detection_auroc",
    "geometry_fidelity",
    "margins",
    "jl_distortion",
    "kernel_mse",
    "cosine_sim",
]


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred.view(-1) == y.view(-1).to(pred.device)).float().mean().item()


def hungarian_accuracy(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Best-match cluster<->class assignment accuracy."""
    c_true = int(true_labels.max()) + 1
    c_pred = int(cluster_labels.max()) + 1
    conf = np.zeros((c_pred, c_true), dtype=np.int64)
    for p, t in zip(cluster_labels, true_labels):
        conf[p, t] += 1
    row, col = linear_sum_assignment(-conf)
    return float(conf[row, col].sum() / len(true_labels))


def clustering_metrics(embeddings: torch.Tensor, k: int, labels: np.ndarray, seed: int = 0) -> dict:
    emb = embeddings.detach().cpu().numpy().astype(np.float64)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(emb)
    return {
        "nmi": float(normalized_mutual_info_score(labels, km.labels_)),
        "ari": float(adjusted_rand_score(labels, km.labels_)),
        "hungarian_acc": float(hungarian_accuracy(km.labels_, labels)),
    }


def ood_detection_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC separating ID from OOD with a higher-is-more-ID score."""
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))


@torch.no_grad()
def geometry_fidelity(encoder, x: torch.Tensor, *, n_points: int = 400, seed: int = 0) -> float:
    """Spearman rho of pairwise cosine similarities, input vs encoded space."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[: min(n_points, x.shape[0])]
    xs = x[idx].float()
    xin = torch.nn.functional.normalize(xs, dim=1)
    tri = torch.triu_indices(len(xs), len(xs), offset=1)
    sim_in = (xin @ xin.T)[tri.unbind()]
    h = encoder.encode(xs).float()
    hin = torch.nn.functional.normalize(h, dim=1)
    sim_out = (hin @ hin.T)[tri.unbind()]
    rho, _ = spearmanr(sim_in.cpu().numpy(), sim_out.cpu().numpy())
    return float(rho)


def margins(scores: torch.Tensor, y: torch.Tensor) -> dict:
    """Mean (correct-class - best-wrong-class) similarity.

    ``scores`` are (n, C) similarities; the margin is reported raw and in
    units of the 1/sqrt(d) cosine-noise floor of unit-norm random vectors.
    """
    y = y.to(scores.device)
    correct = scores.gather(1, y.view(-1, 1)).squeeze(1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, y.view(-1, 1), False)
    best_wrong = scores.masked_fill(~mask, float("-inf")).max(dim=1).values
    m = (correct - best_wrong)
    return {
        "margin": float(m.mean().item()),
        "margin_std": float(m.std().item()),
    }


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.float(), dim=1)
    b = torch.nn.functional.normalize(b.float(), dim=1)
    return a @ b.T


@torch.no_grad()
def jl_distortion(P: torch.Tensor, x: torch.Tensor, *, n_pairs: int = 2000, seed: int = 0) -> dict:
    """Relative error of projected pairwise distances (Johnson-Lindenstrauss).

    The projection is rescaled by sqrt(F/D) inside the metric so that it
    preserves input norms in expectation (our RP convention is unit output
    variance, i.e. ``E||P d||^2 = (D/F) ||d||^2``). Returns the mean relative
    error and the empirical std against the JL prediction ``sqrt(2/D)``.
    """
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    P = P.to(x.device)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    diff = x[i].float() - x[j].float()
    d_in = (diff ** 2).sum(dim=1)
    F, D = P.shape[1], P.shape[0]
    d_out = ((diff @ P.T) ** 2).sum(dim=1) * (F / D)
    rel = (d_out - d_in) / d_in.clamp_min(1e-12)
    return {
        "jl_mean_rel": float(rel.mean().item()),
        "jl_std_rel": float(rel.std().item()),
        "jl_pred_std": float(np.sqrt(2.0 / D)),
    }


@torch.no_grad()
def kernel_mse(encode, x: torch.Tensor, kernel_fn, *, n_pairs: int = 2000,
               seed: int = 0, normalize: bool = True) -> float:
    """MSE between encoded-space similarities and a target kernel function.

    ``normalize=True`` uses cosine similarity (appropriate for signed/binary
    codes); ``normalize=False`` uses the raw inner product, which is the
    unbiased kernel estimate for random Fourier features and phasor codes.
    """
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    h = encode(x).float()
    dev = h.device
    if normalize:
        h = torch.nn.functional.normalize(h, dim=1)
    sim = (h[i] * h[j]).sum(dim=1)
    k = kernel_fn(x[i].float().to(dev), x[j].float().to(dev))
    return float(((sim - k) ** 2).mean().item())
