# tta/methods/roid.py
import copy
import torch
import torch.nn.functional as F
from tta.base_tta import BaseTTA


class ROID(BaseTTA):
    """
    Robust / Universal TTA (Marsden et al., WACV 2024).
    Certainty+diversity-weighted entropy minimisation, consistency to a
    weight-ensembled EMA teacher, and running prior correction.

    FIX vs. previous version: the teacher is a SEPARATE deep-copied model that is
    EMA-updated, instead of copying EMA weights in-place into the live model.
    The old in-place `p.copy_()` mutated parameters that were part of the active
    autograd graph -> "variable needed for gradient has been modified" on
    backward. A separate teacher removes that entirely.
    """
    def __init__(self, model, lr=1e-3, device='cuda', alpha=0.99, lambda_cons=1.0):
        super().__init__(model, lr, device)
        self.params = self.configure_for_tta(model)
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=0.9)
        self.lambda_cons = lambda_cons
        self.alpha = alpha
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.running_prior = None

    @torch.no_grad()
    def _update_teacher(self):
        for tp, p in zip(self.teacher.parameters(), self.model.parameters()):
            tp.mul_(self.alpha).add_(p.detach(), alpha=1 - self.alpha)

    def adapt(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = F.softmax(logits, 1)
        K = probs.size(1)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(1)
        w_cert = torch.exp(-entropy.detach())
        logK = torch.log(torch.tensor(float(K), device=x.device))
        w_div = (1.0 - entropy.detach() / logK).clamp(min=0)
        weights = (w_cert * w_div).detach()
        loss = (weights * entropy).mean()

        with torch.no_grad():
            tlog = self.teacher(x)
        loss = loss + self.lambda_cons * F.kl_div(
            F.log_softmax(logits, 1), F.softmax(tlog, 1), reduction='batchmean')

        # Divergence guard (same pattern as eata): a non-finite loss pushes the
        # adapted params to NaN, and since run_evaluation reads pre-TTA logits
        # from this same model object, BOTH prob arrays are NaN from that batch
        # on — which is what broke roc_auc downstream.
        if not torch.isfinite(loss):
            return torch.log(probs.detach() + 1e-8)

        marg = probs.mean(0).detach()
        if torch.isfinite(marg).all():
            self.running_prior = (marg if self.running_prior is None
                                  else 0.9 * self.running_prior + 0.1 * marg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g['params']], max_norm=1.0)
        self.optimizer.step()
        self._update_teacher()

        corrected = probs.detach() / (self.running_prior + 1e-8)
        corrected = corrected / corrected.sum(1, keepdim=True)
        return torch.log(corrected + 1e-8)
