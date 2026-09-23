"""OnlineHD-style prototype classifier with processing variants.

Training = single-pass bundling of encoded training samples followed by
error-driven retraining passes (OnlineHD, lr=1). Variants:

* similarity: cosine / dot / hamming / L1 / RBF
* bundling  : uniform, margin- or confidence-weighted, multi-prototype
* storage   : prototype quantization (MSE int, ternary, sign) with optional
              quantization-aware retraining (QuantHD-style)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import quant

__all__ = ["PrototypeClassifier", "sign_flip", "add_feature_noise", "drop_features"]


class PrototypeClassifier:
    def __init__(self, encoder, num_classes: int, *, sim: str = "cosine",
                 epochs: int = 30, lr: float = 1.0, weight: str = "uniform",
                 multi: int = 1, proto_bits: int | None = None,
                 proto_scheme: str = "mse", quant_train: bool = False,
                 rbf_gamma: float | None = None, batch_size: int = 4096,
                 seed: int = 0, device: str | None = None) -> None:
        self.encoder = encoder
        self.num_classes = num_classes
        self.sim = sim
        self.epochs = epochs
        self.lr = lr
        self.weight = weight
        self.multi = multi
        self.proto_bits = proto_bits
        self.proto_scheme = proto_scheme
        self.quant_train = quant_train
        self.rbf_gamma = rbf_gamma
        self.batch_size = batch_size
        self.seed = seed
        self.device = encoder.device if hasattr(encoder, "device") else torch.device("cpu")
        self.dim = getattr(encoder, "dim_out", getattr(encoder, "dim", None))
        self.class_hvs: torch.Tensor | None = None
        self.proto_class: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # encoding helpers
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder.encode(x)
        return h.float()

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def _accumulate(self, h: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None):
        W = torch.zeros(self.num_classes, h.shape[1], device=h.device, dtype=torch.float32)
        if w is None:
            W.index_add_(0, y, h)
        else:
            W.index_add_(0, y, h * w.view(-1, 1))
        return W

    def _margin_weights(self, h: torch.Tensor, y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        sims = self._sim(h, W)
        correct = sims.gather(1, y.view(-1, 1)).squeeze(1)
        mask = torch.ones_like(sims, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), False)
        best_wrong = sims.masked_fill(~mask, float("-inf")).max(dim=1).values
        m = correct - best_wrong
        T = m.std().clamp_min(1e-6)
        w = torch.exp(m / T)
        return (w / w.mean().clamp_min(1e-6)).clamp(0.05, 20.0)

    def _confidence_weights(self, h: torch.Tensor, y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        sims = self._sim(h, W)
        p = F.softmax(sims * 10.0, dim=1).gather(1, y.view(-1, 1)).squeeze(1)
        return (p / p.mean().clamp_min(1e-6)).clamp(0.05, 20.0)

    def fit(self, x: torch.Tensor, y: torch.Tensor, x_val=None, y_val=None) -> "PrototypeClassifier":
        h = self._encode(x)
        y = y.long().to(h.device)
        if self.weight == "uniform" or self.multi > 1:
            self.class_hvs = self._accumulate(h, y)
        else:
            W0 = self._accumulate(h, y)
            if self.weight == "margin":
                w = self._margin_weights(h, y, W0)
            elif self.weight == "confidence":
                w = self._confidence_weights(h, y, W0)
            else:
                raise ValueError(self.weight)
            self.class_hvs = self._accumulate(h, y, w)

        if self.multi > 1:
            self._fit_multi(h, y)
        elif self.epochs > 0:
            self._retrain(h, y)
        if self.quant_train and self.proto_bits is not None:
            self.class_hvs = self._quantize(self.class_hvs)
        if self.sim == "rbf" and self.rbf_gamma is None:
            hs = F.normalize(h[: min(2000, h.shape[0])].float(), dim=1)
            d2 = torch.cdist(hs, hs) ** 2
            self.rbf_gamma = float(1.0 / d2.median().clamp_min(1e-12))
        return self

    def _retrain(self, h: torch.Tensor, y: torch.Tensor) -> None:
        g = torch.Generator().manual_seed(self.seed)
        n = h.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                hb, yb = h[idx], y[idx]
                cn = F.normalize(self.class_hvs, dim=1)
                pred = (hb @ cn.T).argmax(dim=1)
                wrong = pred != yb
                if wrong.any():
                    upd = self.lr * hb[wrong]
                    self.class_hvs.index_add_(0, yb[wrong], upd)
                    self.class_hvs.index_add_(0, pred[wrong], -upd)
            if self.quant_train and self.proto_bits is not None:
                self.class_hvs = self._quantize(self.class_hvs)

    def _fit_multi(self, h: torch.Tensor, y: torch.Tensor) -> None:
        """Cosine k-means per class; prototypes are cluster means."""
        g = torch.Generator().manual_seed(self.seed)
        dim = h.shape[1]
        protos, cls = [], []
        for c in range(self.num_classes):
            Xc = h[y == c]
            m = Xc.shape[0]
            k = min(self.multi, m)
            idx = torch.randperm(m, generator=g)[:k]
            centers = Xc[idx].clone()
            for _ in range(12):
                sims = F.normalize(Xc, dim=1) @ F.normalize(centers, dim=1).T
                assign = sims.argmax(dim=1)
                for j in range(k):
                    sel = Xc[assign == j]
                    if sel.shape[0]:
                        centers[j] = sel.mean(dim=0)
            protos.append(centers)
            cls.append(torch.full((k,), c, dtype=torch.long))
        self.class_hvs = torch.cat(protos, dim=0)
        self.proto_class = torch.cat(cls).to(h.device)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def _quantize(self, W: torch.Tensor) -> torch.Tensor:
        if self.proto_bits is None or self.proto_bits >= 31:
            return W
        if self.proto_scheme == "mse":
            return quant.mse_ptq(W, self.proto_bits)
        if self.proto_scheme == "ternary":
            return quant.ternary_quantize(W)
        if self.proto_scheme == "sign":
            return quant.sign(W)
        raise ValueError(self.proto_scheme)

    @torch.no_grad()
    def _prototypes(self) -> torch.Tensor:
        W = self.class_hvs
        if self.proto_bits is not None and self.proto_scheme == "mse":
            W = quant.mse_ptq(W, self.proto_bits)
        elif self.proto_bits is not None and self.proto_scheme == "ternary":
            W = quant.ternary_quantize(W)
        elif self.proto_bits is not None and self.proto_scheme == "sign":
            W = quant.sign(W)
        return W

    @torch.no_grad()
    def _sim(self, h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        if self.sim in ("cosine", "hamming"):
            kind = getattr(getattr(self.encoder, "out_code", None), "kind", None)
            if self.sim == "hamming" and kind == "binary01":
                hm = h.mean(dim=1, keepdim=True)
                wm = W.mean(dim=1, keepdim=True).T
                return 1.0 - (hm + wm - 2.0 * (h @ W.T) / h.shape[1])
            return F.normalize(h, dim=1) @ F.normalize(W, dim=1).T
        if self.sim == "dot":
            return h @ W.T
        if self.sim == "l1":
            return self._l1_sim(h, W)
        if self.sim == "rbf":
            hn = F.normalize(h, dim=1)
            wn = F.normalize(W, dim=1)
            gamma = self.rbf_gamma or 1.0
            return torch.exp(-gamma * torch.cdist(hn, wn) ** 2)
        raise ValueError(self.sim)

    @torch.no_grad()
    def _l1_sim(self, h: torch.Tensor, W: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        """Negative mean L1 distance between L2-normalized codes/prototypes.

        Normalization first is essential: accumulated prototypes have a much
        larger scale than single encodings, and raw L1 distance would rank by
        prototype norm rather than by geometry (accuracy collapses to chance).
        """
        hn = F.normalize(h.float(), dim=1)
        wn = F.normalize(W.float(), dim=1)
        out = torch.empty(h.shape[0], W.shape[0], device=h.device, dtype=torch.float32)
        for i in range(0, h.shape[0], chunk):
            out[i:i + chunk] = -torch.cdist(hn[i:i + chunk], wn, p=1) / h.shape[1]
        return out

    @torch.no_grad()
    def scores(self, x: torch.Tensor, *, batch_size: int | None = None) -> torch.Tensor:
        batch_size = batch_size or self.batch_size
        W = self._prototypes()
        outs = []
        for i in range(0, x.shape[0], batch_size):
            h = self._encode(x[i:i + batch_size])
            s = self._sim(h, W)
            if self.proto_class is not None:
                C, P = self.num_classes, self.multi
                s = s.view(s.shape[0], -1)[:, : P * C].view(s.shape[0], C, P).amax(dim=2)
            outs.append(s)
        return torch.cat(outs, dim=0)

    def predict(self, x: torch.Tensor, **kw) -> torch.Tensor:
        return self.scores(x, **kw).argmax(dim=1)


# --------------------------------------------------------------------------
# corruption helpers (robustness experiments)
# --------------------------------------------------------------------------


def _device_generator(device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def sign_flip(W: torch.Tensor, p: float, *, seed: int = 0,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Multiply each entry by +/-1 with flip probability p (model noise)."""
    if p <= 0:
        return W
    if generator is None:
        generator = _device_generator(W.device, seed)
    mask = torch.rand(W.shape, generator=generator, device=W.device) < p
    return W * torch.where(mask, -1.0, 1.0)


@torch.no_grad()
def add_feature_noise(x: torch.Tensor, sigma: float, *, seed: int = 0,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    if sigma <= 0:
        return x
    if generator is None:
        generator = _device_generator(x.device, seed)
    return x + sigma * torch.randn(x.shape, generator=generator, device=x.device)


@torch.no_grad()
def drop_features(x: torch.Tensor, p: float, *, seed: int = 0,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    if p <= 0:
        return x
    if generator is None:
        generator = _device_generator(x.device, seed)
    mask = (torch.rand(x.shape, generator=generator, device=x.device) >= p).float()
    return x * mask
