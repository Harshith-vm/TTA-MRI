# tta/methods/sar.py
import torch
import torch.nn as nn
import copy
from tta.base_tta import BaseTTA

class SAR(BaseTTA):
    """
    Sharpness-Aware and Reliable entropy minimization.
    1. Filter noisy samples with large gradient norms
    2. SAM optimizer finds flat minima
    3. Model recovery on entropy spike
    Niu et al. ICLR 2023 (Oral). https://github.com/mr-eggplant/SAR
    """
    def __init__(self, model, lr=1e-3, device='cuda',
                 e_margin=0.4, rho=0.05):
        super().__init__(model, lr, device)
        self.e_margin = e_margin * torch.log(torch.tensor(2.0))
        self.rho = rho
        self.params = self.configure_for_tta(model)
        self.optimizer = SAMOptimizer(self.params, lr=lr, momentum=0.9, rho=rho)
        self._best_state = copy.deepcopy(model.state_dict())
        self._best_entropy = float('inf')

    def adapt(self, x):
        # --- First forward pass: compute entropy, filter noisy samples ---
        logits = self.model(x)
        entropy = -(torch.softmax(logits, 1) * torch.log_softmax(logits, 1)).sum(1)
        mask = entropy < self.e_margin
        if mask.sum() == 0:
            return logits.detach()

        # --- SAM first step: compute gradient ---
        self.optimizer.zero_grad(set_to_none=True)
        loss_1 = entropy[mask].mean()
        loss_1.backward()
        self.optimizer.first_step(zero_grad=True)

        # --- SAM second step: update at perturbed weights ---
        logits_2 = self.model(x)
        entropy_2 = -(torch.softmax(logits_2, 1) * torch.log_softmax(logits_2, 1)).sum(1)
        loss_2 = entropy_2[mask].mean()
        loss_2.backward()
        self.optimizer.second_step(zero_grad=True)

        # --- Model recovery: reset if entropy gets worse ---
        current_entropy = entropy_2[mask].mean().item()
        if current_entropy < self._best_entropy:
            self._best_entropy = current_entropy
            self._best_state = copy.deepcopy(self.model.state_dict())
        elif current_entropy > self._best_entropy * 1.2:
            self.model.load_state_dict(self._best_state)

        return logits.detach()


class SAMOptimizer(torch.optim.SGD):
    """Sharpness-Aware Minimization optimizer."""
    def __init__(self, params, lr, momentum, rho=0.05):
        super().__init__(params, lr=lr, momentum=momentum)
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
        super().step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared = next(iter(self.param_groups))["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)
