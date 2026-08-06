# tta/methods/rmemsafe.py
import torch
import torch.nn.functional as F
import copy
from tta.base_tta import BaseTTA


class RMemSafe(BaseTTA):
    """
    RMemSafe — Reliability-Gated Source Anchoring for Continual TTA (2026).
    arXiv:2605.14063.

    Continual TTA on a non-stationary stream, anchored to a FROZEN source model.
    A reliability gate derived from the frozen source's normalised predictive
    entropy controls how much the source is trusted: when the source posterior
    is sharp (low entropy -> reliable), the objective anchors the adapting model
    to the source; as the source posterior approaches uniformity (gate closes),
    the objective falls back to a source-agnostic term — entropy minimisation
    plus a marginal-calibration regulariser that keeps the predicted class
    marginal close to uniform, preventing collapse.
    """
    def __init__(self, model, lr=1e-3, device='cuda', lambda_cal=1.0):
        super().__init__(model, lr, device)
        self.frozen = copy.deepcopy(model).eval()
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        params = self.configure_for_tta(model)
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        self.lambda_cal = lambda_cal

    def adapt(self, x):
        with torch.no_grad():
            src_logits = self.frozen(x)
            src_p = F.softmax(src_logits, 1)
            K = src_p.size(1)
            src_ent = -(src_p * torch.log(src_p + 1e-8)).sum(1)
            gate = 1.0 - (src_ent / torch.log(torch.tensor(float(K), device=x.device)))
            gate = gate.clamp(0, 1)                  # 1 = source reliable

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        logp = F.log_softmax(logits, 1)
        p = logp.exp()

        # anchored term: KL(current || source), weighted by reliability gate
        anchor = (gate * F.kl_div(logp, src_p, reduction='none').sum(1)).mean()
        # source-agnostic fallback: entropy min + marginal calibration
        ent = -(p * logp).sum(1)
        marg = p.mean(0)
        cal = (marg * (marg.add(1e-8).log() + torch.log(torch.tensor(float(K),
                device=x.device)))).sum()          # KL(marg || uniform)
        fallback = ((1.0 - gate) * ent).mean() + self.lambda_cal * cal

        # Divergence guard (same pattern as eata): skip non-finite updates and
        # clip, so the adapted params can never become NaN and poison every
        # subsequent batch's logits.
        loss = anchor + fallback
        if not torch.isfinite(loss):
            return logits.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g['params']], max_norm=1.0)
        self.optimizer.step()
        return logits.detach()
