"""Feature extractors sitting in front of the projection stage.

Kinds:

    raw        identity (no extractor; projections see the raw features)
    randmlp    frozen random 2-layer ReLU MLP (random features)
    ce         MLP trained with cross-entropy
    supcon     MLP trained with cross-entropy + supervised contrastive loss
    proxy      MLP trained with cross-entropy + proxy-anchor metric loss
    arc        MLP trained with an angular-margin (ArcFace-style) loss
    clustered  CE-trained MLP with k-means weight clustering (no fine-tune)
    cotrain    MLP co-trained end-to-end with a fixed random projection and
               the HDC prototype objective (straight-through estimator)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import resolve_device

__all__ = ["MLP", "Extractor", "make_extractor", "co_train", "sign_ste",
           "EXTRACTOR_KINDS"]

EXTRACTOR_KINDS = ["raw", "randmlp", "ce", "supcon", "proxy", "arc", "clustered"]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple = (128,), out_dim: int = 128) -> None:
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers[:-1])  # no activation on the output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sign_ste(t: torch.Tensor) -> torch.Tensor:
    """Sign with a straight-through estimator (hard forward, soft gradient)."""
    hard = torch.where(t >= 0, 1.0, -1.0)
    soft = torch.tanh(t)
    return hard + soft - soft.detach()


def _supcon(f: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    z = F.normalize(f, dim=1)
    sim = z @ z.T / tau
    n = f.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=f.device)
    pos = (y.view(-1, 1) == y.view(1, -1)) & ~eye
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum(dim=1).div(pos.sum(dim=1).clamp_min(1)).mean()


def _cluster_1d(t: torch.Tensor, levels: int, iters: int = 25) -> torch.Tensor:
    """1-D k-means weight clustering (sorted centroids)."""
    flat = t.detach().reshape(-1)
    if flat.numel() <= levels or levels <= 1:
        return t
    centers = torch.linspace(float(flat.min()), float(flat.max()), levels, device=flat.device)
    for _ in range(iters):
        idx = torch.bucketize(flat, (centers[:-1] + centers[1:]) / 2)
        for k in range(levels):
            sel = flat[idx == k]
            if sel.numel():
                centers[k] = sel.mean()
    idx = torch.bucketize(flat, (centers[:-1] + centers[1:]) / 2)
    return centers[idx].reshape(t.shape)


class Extractor:
    """Feature extractor wrapper with fit/transform and resource accounting."""

    def __init__(self, kind: str = "raw", in_dim: int = 64, out_dim: int = 128, *,
                 seed: int = 0, hidden: tuple = (128,), epochs: int = 30,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256,
                 levels: int = 8, lam: float = 0.5, margin: float = 0.2,
                 device: str | None = None) -> None:
        self.kind = kind
        self.in_dim = in_dim
        self.out_dim = out_dim if kind != "raw" else in_dim
        self.seed = seed
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.levels = levels
        self.lam = lam
        self.margin = margin
        self.device = resolve_device(device)
        self.model: nn.Module | None = None
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    # ------------------------------------------------------------------

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(self.device)
        if self.kind == "raw" or self.mean is None:
            return x
        return (x - self.mean.to(self.device)) / self.std.to(self.device)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "Extractor":
        if self.kind == "raw":
            return self
        torch.manual_seed(self.seed)
        x = x.float()
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0).clamp_min(1e-8)
        xp = self._standardize(x)
        y = y.long().to(self.device)
        self.model = MLP(self.in_dim, self.hidden, self.out_dim).to(self.device)
        if self.kind == "randmlp":
            return self
        self._train(xp, y)
        if self.kind == "clustered":
            with torch.no_grad():
                for p in self.model.parameters():
                    p.copy_(_cluster_1d(p, self.levels))
        return self

    def _train(self, xp: torch.Tensor, y: torch.Tensor) -> None:
        C = int(y.max().item()) + 1
        head = nn.Linear(self.out_dim, C).to(self.device)
        params = list(self.model.parameters()) + list(head.parameters())
        proxies = None
        if self.kind in ("proxy", "arc"):
            proxies = nn.Parameter(
                F.normalize(torch.randn(C, self.out_dim, device=self.device), dim=1))
            params.append(proxies)
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        n = xp.shape[0]
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = xp[idx], y[idx]
                f = self.model(xb)
                if self.kind in ("ce", "clustered"):
                    loss = F.cross_entropy(head(f), yb)
                elif self.kind == "supcon":
                    loss = F.cross_entropy(head(f), yb) + self.lam * _supcon(f, yb)
                elif self.kind == "proxy":
                    loss = F.cross_entropy(head(f), yb) + self.lam * self._proxy_loss(f, yb, proxies)
                elif self.kind == "arc":
                    loss = self._arc_loss(f, yb, proxies)
                else:
                    raise ValueError(self.kind)
                opt.zero_grad()
                loss.backward()
                opt.step()

    @staticmethod
    def _proxy_loss(f: torch.Tensor, y: torch.Tensor, proxies: torch.Tensor,
                    margin: float = 0.1, tau: float = 0.05) -> torch.Tensor:
        z = F.normalize(f, dim=1)
        p = F.normalize(proxies, dim=1)
        sim = z @ p.T
        pos = sim.gather(1, y.view(-1, 1))
        neg = sim.masked_fill(
            F.one_hot(y, sim.shape[1]).bool(), float("-inf")).max(dim=1, keepdim=True).values
        return F.softplus((neg - pos + margin) / tau).mean()

    def _arc_loss(self, f: torch.Tensor, y: torch.Tensor, proxies: torch.Tensor) -> torch.Tensor:
        z = F.normalize(f, dim=1)
        p = F.normalize(proxies, dim=1)
        cos = z @ p.T
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta.gather(1, y.view(-1, 1)).squeeze(1) + self.margin)
        logits = cos.scatter(1, y.view(-1, 1), target.view(-1, 1))
        return F.cross_entropy(logits * 16.0, y)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def transform(self, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        if self.kind == "raw":
            return x.float()
        outs = []
        for i in range(0, x.shape[0], batch_size):
            xb = self._standardize(x[i:i + batch_size])
            outs.append(self.model(xb))
        return torch.cat(outs, dim=0)

    def memory_bits(self) -> int:
        if self.kind == "raw":
            return 0
        elems = sum(p.numel() for p in self.model.parameters())
        bits_per = math.log2(self.levels) if self.kind == "clustered" else 32.0
        return int(elems * bits_per) + self.in_dim * 2 * 32

    def ops_per_query(self) -> int:
        if self.kind == "raw":
            return 0
        dims = [self.in_dim, *self.hidden, self.out_dim]
        return sum(a * b for a, b in zip(dims[:-1], dims[1:])) + self.in_dim * 2

    def weight_levels(self) -> int:
        """Largest number of distinct values in any weight tensor."""
        if self.kind != "clustered" or self.model is None:
            return 0
        return max(int(torch.unique(p.detach()).numel()) for p in self.model.parameters())


def make_extractor(kind: str, in_dim: int, out_dim: int = 128, *, seed: int = 0,
                   device: str | None = None, **kwargs) -> Extractor:
    return Extractor(kind, in_dim, out_dim, seed=seed, device=device, **kwargs)


# --------------------------------------------------------------------------
# co-training (DNN + HDC objective, gradients through the projection)
# --------------------------------------------------------------------------


def co_train(extractor: Extractor, P: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *,
             epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-4,
             batch_size: int = 256, learn_proj: bool = False, seed: int = 0,
             device: str | None = None) -> tuple[Extractor, torch.Tensor]:
    """Train ``extractor`` (and optionally the projection) with the HDC
    prototype-cosine objective on sign(P f(x)), using a straight-through
    estimator. Returns ``(extractor, P)``.
    """
    dev = resolve_device(device)
    torch.manual_seed(seed)
    x = x.float()
    extractor.mean = x.mean(dim=0)
    extractor.std = x.std(dim=0).clamp_min(1e-8)
    xp = ((x - extractor.mean) / extractor.std).to(dev)
    y = y.long().to(dev)
    extractor.model = MLP(extractor.in_dim, extractor.hidden, extractor.out_dim).to(dev)
    C = int(y.max().item()) + 1
    D = P.shape[0]
    P = torch.nn.Parameter(P.to(dev).float(), requires_grad=learn_proj)
    # initialize prototypes from a first pass so the cosine head starts useful
    with torch.no_grad():
        h0 = torch.where(extractor.model(xp) @ P.T >= 0, 1.0, -1.0)
        W0 = torch.zeros(C, D, device=dev)
        W0.index_add_(0, y, h0)
    W = torch.nn.Parameter(W0)
    params = list(extractor.model.parameters()) + ([P] if learn_proj else []) + [W]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = xp.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            f = extractor.model(xp[idx])
            h = sign_ste(f @ P.T)
            hn = F.normalize(h, dim=1)
            wn = F.normalize(W, dim=1)
            loss = F.cross_entropy(hn @ wn.T * 20.0, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return extractor, P.detach()
