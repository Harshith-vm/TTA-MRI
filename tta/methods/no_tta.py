# tta/methods/no_tta.py
import torch
from tta.base_tta import BaseTTA

class NoTTA(BaseTTA):
    def adapt(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
