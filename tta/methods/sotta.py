# tta/methods/sotta.py
import torch
import torch.nn.functional as F
from tta.base_tta import BaseTTA
from tta.methods.sar import SAMOptimizer


class SoTTA(BaseTTA):
    """
    Self-stabilised online TTA (Gong et al., NeurIPS 2023).
    Robust to noisy / temporally-correlated streams via:

      1. High-confidence uniform-class sampling memory (HUS) — only confident,
         class-balanced samples enter a fixed-capacity memory bank.
      2. Entropy-sharpness minimisation (SAM) over the memory, so single noisy
         batches cannot derail adaptation.
    """
    def __init__(self, model, lr=1e-3, device='cuda', mem_size=64,
                 conf_thresh=0.99, rho=0.05, num_classes=2):
        super().__init__(model, lr, device)
        self.params = self.configure_for_tta(model)
        self.optimizer = SAMOptimizer(self.params, lr=lr, momentum=0.9, rho=rho)
        self.mem_size = mem_size
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes
        self.mem_x, self.mem_y = [], []

    @torch.no_grad()
    def _update_memory(self, x, probs):
        conf, pred = probs.max(1)
        for i in range(x.size(0)):
            if conf[i].item() < self.conf_thresh:
                continue
            cls = pred[i].item()
            cls_count = sum(1 for c in self.mem_y if c == cls)
            if len(self.mem_x) < self.mem_size:
                self.mem_x.append(x[i].detach().clone()); self.mem_y.append(cls)
            elif cls_count <= self.mem_size // self.num_classes:
                # evict from the most populous class (uniform sampling)
                from collections import Counter
                maj = Counter(self.mem_y).most_common(1)[0][0]
                j = self.mem_y.index(maj)
                self.mem_x[j] = x[i].detach().clone(); self.mem_y[j] = cls

    def adapt(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, 1)
        self._update_memory(x, probs)

        if len(self.mem_x) >= max(4, self.num_classes * 2):
            mem = torch.stack(self.mem_x)
            self.optimizer.zero_grad(set_to_none=True)
            # Divergence guard (same pattern as eata). With SAM, second_step must
            # still run once first_step has perturbed the weights -- otherwise the
            # model is left sitting at the perturbed point. Skipping only the
            # backward leaves grads at zero, so second_step just restores.
            params = [p for g in self.optimizer.param_groups for p in g['params']]
            l1 = self._ent(self.model(mem))
            if torch.isfinite(l1):
                l1.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                self.optimizer.first_step(zero_grad=True)
                l2 = self._ent(self.model(mem))
                if torch.isfinite(l2):
                    l2.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                self.optimizer.second_step(zero_grad=True)

        return self.model(x).detach()

    @staticmethod
    def _ent(logits):
        p = F.softmax(logits, 1)
        return -(p * torch.log(p + 1e-8)).sum(1).mean()
