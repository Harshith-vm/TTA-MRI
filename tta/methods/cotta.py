# tta/methods/cotta.py
import torch
import torch.nn as nn
import copy
import torchvision.transforms as T
from tta.base_tta import BaseTTA

class CoTTA(BaseTTA):
    """
    Continual TTA with augmentation averaging + stochastic restoration.
    Wang et al. CVPR 2022. https://github.com/qinenergy/cotta
    Note: Uses 8 augmentations (reduced from 32 for VRAM safety on RTX 8000).
    """
    def __init__(self, model, lr=1e-3, device='cuda',
                 n_aug=8, restore_prob=0.01, ema_factor=0.999):
        super().__init__(model, lr, device)
        self.n_aug = n_aug
        self.restore_prob = restore_prob
        self.ema_factor = ema_factor
        self.params = self.configure_for_tta(model)
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        # EMA teacher
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
        ])

    def adapt(self, x):
        # Teacher augmentation averaging for pseudo-labels
        with torch.no_grad():
            aug_preds = torch.stack([
                torch.softmax(self.teacher(self.augment(x)), dim=1)
                for _ in range(self.n_aug)
            ]).mean(0)

        # Student prediction
        self.optimizer.zero_grad(set_to_none=True)
        student_logits = self.model(x)
        # Cross-entropy with soft teacher pseudo-labels
        loss = -(aug_preds.detach() * torch.log_softmax(student_logits, dim=1)).sum(1).mean()
        loss.backward()
        self.optimizer.step()

        # Update EMA teacher
        for tp, sp in zip(self.teacher.parameters(), self.model.parameters()):
            tp.data = self.ema_factor * tp.data + (1 - self.ema_factor) * sp.data

        # Stochastic restoration
        for p, p0 in zip(self.model.parameters(),
                          self.teacher.parameters()):
            if torch.rand(1).item() < self.restore_prob:
                p.data = self.source_state_param(p.data, p0.data)

        return student_logits.detach()

    def source_state_param(self, current, src):
        return src  # restore to teacher (which is EMA of source)
