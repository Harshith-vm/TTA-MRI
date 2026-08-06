# tta/methods/rem.py
import torch
import torch.nn.functional as F
from tta.base_tta import BaseTTA


class REM(BaseTTA):
    """
    REM — Ranked Entropy Minimization for Continual TTA
    (Han, Na & Hwang, ICML 2025). arXiv:2505.16441.

    Entropy minimisation collapses because it pushes every sample to maximal
    confidence indiscriminately. REM instead structures prediction *difficulty*
    via a progressive masking strategy: harder (more-masked) views should not be
    more confident than easier views. It minimises entropy while preserving the
    rank order of entropy across difficulty levels, which stabilises long
    continual streams.

    Pragmatic realisation: L input-masking levels with increasing mask ratio.
    Loss = mean entropy over views + ranking hinge enforcing
           H(level_0) <= H(level_1) <= ... <= H(level_{L-1}).
    The patch-masking simplification (vs. the paper's token masking) is noted in
    the supplementary.
    """
    def __init__(self, model, lr=1e-3, device='cuda', levels=(0.0, 0.25, 0.5),
                 margin=0.0, lambda_rank=1.0, patch=16):
        super().__init__(model, lr, device)
        params = self.configure_for_tta(model)
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        self.levels = levels
        self.margin = margin
        self.lambda_rank = lambda_rank
        self.patch = patch

    def _mask(self, x, ratio):
        if ratio <= 0:
            return x
        B, C, H, W = x.shape
        ph, pw = H // self.patch, W // self.patch
        m = (torch.rand(B, 1, ph, pw, device=x.device) > ratio).float()
        m = F.interpolate(m, size=(H, W), mode='nearest')
        return x * m

    @staticmethod
    def _entropy(logits):
        p = F.softmax(logits, 1)
        return -(p * torch.log(p + 1e-8)).sum(1).mean()

    def adapt(self, x):
        # Backward per level so at most ONE forward's autograd graph is resident.
        # Holding all L graphs simultaneously OOMs the A6000 on resnet50 at
        # bs=256 (peak ~45 GB observed for L=3). The rank hinge uses the
        # previous level's entropy detached — gradient flows only through the
        # current level, which is the standard chain-of-gradient stop-grad
        # simplification used in memory-efficient ranking losses.
        self.optimizer.zero_grad(set_to_none=True)
        L = len(self.levels)
        prev_ent = None
        clean_logits = None
        any_finite = False
        for i, r in enumerate(self.levels):
            logits = self.model(self._mask(x, r))
            ent = self._entropy(logits)
            loss = ent / L
            if prev_ent is not None:
                loss = loss + self.lambda_rank * F.relu(prev_ent - ent + self.margin)
            # Divergence guard (same pattern as eata): a non-finite loss drives the
            # adapted params to NaN, and every later batch then yields NaN logits.
            if torch.isfinite(loss):
                loss.backward()
                any_finite = True
            if i == 0:
                clean_logits = logits.detach()
            prev_ent = ent.detach()
        if any_finite:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g['params']], max_norm=1.0)
            self.optimizer.step()
        return clean_logits
