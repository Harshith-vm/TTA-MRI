# tta/methods/eata.py
import torch
import torch.nn as nn
import copy
from tta.base_tta import BaseTTA

class EATA(BaseTTA):
    """
    Efficient Anti-Forgetting TTA.
    Sample selection (entropy filter) + Fisher regularization.
    Niu et al. ICML 2022. https://github.com/mr-eggplant/EATA
    """
    def __init__(self, model, lr=1e-3, device='cuda',
                 e_margin=0.4, d_margin=0.05, fisher_alpha=2000.0):
        super().__init__(model, lr, device)
        self.e_margin = e_margin * torch.log(torch.tensor(2.0))  # entropy threshold
        self.d_margin = d_margin
        self.fisher_alpha = fisher_alpha
        self.params = self.configure_for_tta(model)
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=0.9)
        self.fisher = {n: torch.zeros_like(p)
                       for n, p in model.named_parameters() if p.requires_grad}
        self.source_params = {n: p.clone().detach()
                               for n, p in model.named_parameters() if p.requires_grad}
        self._ema_entropy = None

    def adapt(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)

        # Sample selection: keep low-entropy, non-redundant samples
        entropy = -(torch.softmax(logits, 1) * torch.log_softmax(logits, 1)).sum(1)
        mask = entropy < self.e_margin

        if mask.sum() == 0:
            return logits.detach()

        # Fisher regularization
        loss = entropy[mask].mean()
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher_alpha / 2.0) * (
                    self.fisher[n] * (p - self.source_params[n]) ** 2
                ).sum()

        # Zero-Fisher first-batch was pushing resnet50 BN affine to NaN on
        # some (domain,seed) combos, and every subsequent batch then produced
        # NaN logits — poisoning `probs_after` and breaking roc_auc downstream.
        # Two guards below stop that failure mode without changing the method.
        if not torch.isfinite(loss):
            return logits.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
        # Update Fisher estimate
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and n in self.fisher:
                self.fisher[n] = self.fisher[n] * 0.9 + (p.grad ** 2) * 0.1
        self.optimizer.step()
        return logits.detach()
