# tta/methods/sicl.py
import torch
import torch.nn.functional as F
from tta.base_tta import BaseTTA


class SICL(BaseTTA):
    """
    SICL — Style Invariance as a Correctness Likelihood (2025).
    arXiv:2512.07390.

    A plug-and-play, BACKPROPAGATION-FREE calibration module for TTA, motivated
    by high-stakes domains (healthcare). It estimates predictive correctness by
    measuring prediction consistency across style-altered variants of the input
    (here: H&E-appropriate brightness/contrast/gamma jitter, which leaves tissue
    morphology intact). Confident-but-style-unstable predictions are down-
    weighted via temperature scaling, recalibrating uncertainty without any
    weight update — so it cannot collapse and is ideal as a calibration probe.
    """
    def __init__(self, model, lr=0.0, device='cuda', n_views=4, t_max=3.0):
        super().__init__(model, lr, device)
        self.model.eval()
        self.n_views = n_views
        self.t_max = t_max

    @staticmethod
    def _style(x):
        # forward-only photometric jitter; morphology preserved
        b = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.4
        c = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.4
        mean = x.mean(dim=(2, 3), keepdim=True)
        return ((x - mean) * c + mean) * b

    @torch.no_grad()
    def adapt(self, x):
        base_logits = self.model(x)
        base_p = F.softmax(base_logits, 1)
        pred = base_p.argmax(1)

        agree = torch.zeros(x.size(0), device=x.device)
        for _ in range(self.n_views):
            vp = F.softmax(self.model(self._style(x)), 1)
            agree += (vp.argmax(1) == pred).float()
        agree /= self.n_views                       # style-invariance ∈ [0,1]

        # low invariance -> high temperature -> softer (less overconfident) probs
        temp = 1.0 + (1.0 - agree) * (self.t_max - 1.0)   # (B,)
        calibrated = base_logits / temp.unsqueeze(1)
        return calibrated.detach()
