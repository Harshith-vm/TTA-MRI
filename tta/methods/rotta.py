# tta/methods/rotta.py
import torch
import torch.nn as nn
import copy
from collections import deque
import random
from tta.base_tta import BaseTTA

class RoTTA(BaseTTA):
    """
    Robust continual TTA with memory bank + timeliness-aware reweighting.
    Yuan et al. CVPR 2023. https://github.com/BIT-DA/RoTTA
    """
    def __init__(self, model, lr=1e-3, device='cuda',
                 memory_size=64, nu=0.001, alpha=0.05, lambda_t=1.0):
        super().__init__(model, lr, device)
        self.memory = deque(maxlen=memory_size)
        self.nu = nu          # EMA momentum
        self.alpha = alpha    # timeliness decay
        self.lambda_t = lambda_t
        self.params = self.configure_for_tta(model)
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.step = 0

    def adapt(self, x):
        self.step += 1

        # Update memory bank with current batch samples
        with torch.no_grad():
            probs = torch.softmax(self.ema_model(x), dim=1)
            confidence, pseudo_labels = probs.max(1)
        for i in range(x.size(0)):
            if confidence[i] > 0.5:  # only store reasonably confident samples
                self.memory.append((
                    x[i].detach().cpu().clone(), pseudo_labels[i].item(),
                    confidence[i].item(), self.step
                ))

        if len(self.memory) < 16:
            with torch.no_grad():
                return self.model(x)

        # Sample from memory for adaptation
        batch = random.sample(list(self.memory), min(32, len(self.memory)))
        imgs = torch.stack([b[0] for b in batch]).to(self.device)
        labels = torch.tensor([b[1] for b in batch]).to(self.device)
        ages = torch.tensor([self.step - b[3] for b in batch], dtype=torch.float32)
        weights = torch.exp(-self.alpha * ages).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(imgs)
        loss = (weights * nn.CrossEntropyLoss(reduction='none')(logits, labels)).mean()
        loss.backward()
        self.optimizer.step()

        # EMA update
        for ep, mp in zip(self.ema_model.parameters(), self.model.parameters()):
            ep.data = (1 - self.nu) * ep.data + self.nu * mp.data

        with torch.no_grad():
            return self.model(x)
