# tta/methods/tent.py
import torch
import torch.nn as nn
from tta.base_tta import BaseTTA

class Tent(BaseTTA):
    """
    Entropy minimization on BN/LN affine parameters.
    Wang et al. NeurIPS 2021. https://github.com/DequanWang/tent
    """
    def __init__(self, model, lr=1e-3, device='cuda'):
        super().__init__(model, lr, device)
        params = self.configure_for_tta(model)
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    def adapt(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = self._entropy_loss(logits)
        loss.backward()
        self.optimizer.step()
        return logits.detach()

    @staticmethod
    def _entropy_loss(logits):
        p = torch.softmax(logits, dim=1)
        return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
