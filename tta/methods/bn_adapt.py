# tta/methods/bn_adapt.py
import torch
import torch.nn as nn
from tta.base_tta import BaseTTA

class BNAdapt(BaseTTA):
    """
    Updates BN running statistics with test batch stats.
    No gradient computation. No parameter updates.
    Schneider et al. NeurIPS 2020.
    """
    def adapt(self, x):
        # Enable BN train mode (updates running stats) for forward pass
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()
                m.momentum = None  # cumulative moving average
        with torch.no_grad():
            return self.model(x)
