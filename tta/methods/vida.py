# tta/methods/vida.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tta.base_tta import BaseTTA


class ViDA(BaseTTA):
    """
    Visual Domain Adapter TTA (Liu et al., ICLR 2024).

    Dual low-rank / high-rank adapters (shared vs. domain-specific knowledge)
    fused and adapted online with EMA-teacher consistency, updating only the
    adapters + norm affines so pretrained features are preserved.

    FIX vs. previous version: the adapters now operate in LOGIT space
    (self.model(x) output), not on a `get_features` penultimate vector. The old
    version assumed every model exposes `get_features`; timm CNNs/ViTs do not, so
    it fed logits into a head-sized adapter and then re-applied the head,
    crashing with a shape mismatch. Logit-space adapters are architecture-
    agnostic and never crash. The simplification is documented for the supp.
    """
    def __init__(self, model, lr=1e-3, device='cuda', low_rank=8, alpha=0.99, lambda_cons=1.0):
        super().__init__(model, lr, device)
        self.model.eval()
        self.device = device
        self.alpha = alpha
        self.lambda_cons = lambda_cons
        self.low_rank = low_rank
        self._lr = lr
        self._built = False

    def _build(self, d):
        self.low = nn.Sequential(nn.Linear(d, self.low_rank), nn.GELU(),
                                 nn.Linear(self.low_rank, d)).to(self.device)
        self.high = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)).to(self.device)
        self.fuse = nn.Parameter(torch.tensor(0.5, device=self.device))
        params = (list(self.low.parameters()) + list(self.high.parameters())
                  + [self.fuse] + self.configure_for_tta(self.model))
        self.optimizer = torch.optim.SGD(params, lr=self._lr, momentum=0.9)
        self.teacher_low = copy.deepcopy(self.low)
        self.teacher_high = copy.deepcopy(self.high)
        for p in list(self.teacher_low.parameters()) + list(self.teacher_high.parameters()):
            p.requires_grad_(False)
        self._built = True

    def adapt(self, x):
        base = self.model(x)                       # (B, K) logits — the adapter space
        if not self._built:
            self._build(base.size(1))
        adapted = base + self.fuse * self.low(base) + (1 - self.fuse) * self.high(base)

        with torch.no_grad():
            bd = base.detach()
            t = bd + 0.5 * self.teacher_low(bd) + 0.5 * self.teacher_high(bd)

        ent = -(F.softmax(adapted, 1) * F.log_softmax(adapted, 1)).sum(1).mean()
        cons = F.kl_div(F.log_softmax(adapted, 1), F.softmax(t, 1), reduction='batchmean')
        loss = ent + self.lambda_cons * cons

        self.optimizer.zero_grad(set_to_none=True)
        # Divergence guard (same pattern as eata).
        if not torch.isfinite(loss):
            return adapted.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g['params']], max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.teacher_low.parameters(), self.low.parameters()):
                tp.mul_(self.alpha).add_(p, alpha=1 - self.alpha)
            for tp, p in zip(self.teacher_high.parameters(), self.high.parameters()):
                tp.mul_(self.alpha).add_(p, alpha=1 - self.alpha)
        return adapted.detach()
