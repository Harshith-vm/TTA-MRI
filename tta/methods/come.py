# tta/methods/come.py
import torch
import torch.nn.functional as F
from tta.base_tta import BaseTTA


class COME(BaseTTA):
    """
    COME — Conservatively Minimizing Entropy (Zhang et al., ICLR 2025).
    arXiv:2410.10894.

    Standard entropy minimisation (Tent) drives the model to overconfidence and
    eventual collapse. COME instead places a Dirichlet prior over predictions
    (subjective-logic / evidential view) and minimises the entropy of the
    resulting *opinion*, which carries an explicit uncertainty mass `u`. This
    naturally regularises the model toward conservative confidence on unreliable
    samples — a perfect probe for our calibration hypothesis.

    Opinion from logits:  evidence e = softplus(logits);  alpha = e + 1
                          S = sum(alpha);  belief b_k = e_k / S;  u = K / S
    Conservative entropy: H = -Σ b_k log b_k  -  u log u    (minimised)
    """
    def __init__(self, model, lr=1e-3, device='cuda'):
        super().__init__(model, lr, device)
        params = self.configure_for_tta(model)
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    def adapt(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = self._opinion_entropy(logits)
        # Divergence guard (same pattern as eata).
        if not torch.isfinite(loss):
            return logits.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g['params']], max_norm=1.0)
        self.optimizer.step()
        return logits.detach()

    @staticmethod
    def _opinion_entropy(logits):
        eps = 1e-8
        K = logits.size(1)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(1, keepdim=True)
        belief = evidence / S                     # (B,K)
        u = K / S.squeeze(1)                       # (B,) uncertainty mass
        h_belief = -(belief * torch.log(belief + eps)).sum(1)
        h_unc = -(u * torch.log(u + eps))
        return (h_belief + h_unc).mean()
