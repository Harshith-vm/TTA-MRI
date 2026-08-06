# tta/methods/lame.py
import torch
import torch.nn.functional as F
from tta.base_tta import BaseTTA


class LAME(BaseTTA):
    """
    Laplacian Adjusted Maximum-likelihood Estimation.
    Boudiaf et al., CVPR 2022. Parameter-free, gradient-free: refines the
    output probabilities so that nearby features (kNN affinity in feature
    space) receive similar labels. No weight updates -> cannot collapse.

    Mechanism: minimise  -sum z_i·log p_i  -  sum_ij w_ij z_i·z_j  over the
    assignment Z, solved by a few fixed-point iterations.
    """
    def __init__(self, model, lr=0.0, device='cuda', knn=5, n_iter=10):
        super().__init__(model, lr, device)
        self.model.eval()
        self.knn = knn
        self.n_iter = n_iter

    @torch.no_grad()
    def _features(self, x):
        if hasattr(self.model, "get_features"):
            return self.model.get_features(x)
        # generic: penultimate via forward hook-free fallback = logits proxy
        return self.model(x)

    @torch.no_grad()
    def adapt(self, x):
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        feats = F.normalize(self._features(x).flatten(1), dim=1)

        # kNN affinity matrix (cosine), symmetric, self excluded
        sim = feats @ feats.t()
        n = sim.size(0)
        sim.fill_diagonal_(-1)
        k = min(self.knn, n - 1)
        if k < 1:
            return logits.detach()
        topv, topi = sim.topk(k, dim=1)
        W = torch.zeros_like(sim)
        W.scatter_(1, topi, topv.clamp(min=0))
        W = 0.5 * (W + W.t())

        # Fixed-point: Z <- softmax(log p + W Z)
        Z = probs.clone()
        for _ in range(self.n_iter):
            Z = F.softmax(torch.log(probs + 1e-8) + W @ Z, dim=1)
        return torch.log(Z + 1e-8)  # log-probs usable as logits
