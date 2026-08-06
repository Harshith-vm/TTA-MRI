# tta/methods/t3a.py
import torch
import torch.nn.functional as F
from tta.base_tta import BaseTTA


class T3A(BaseTTA):
    """
    Test-Time Template Adjustment — parameter-free, no backprop.
    Iwasawa & Matsuo, NeurIPS 2021.

    FIX vs. the previous version: prototypes and features are L2-NORMALISED and
    matched by cosine similarity. The old code did `logits = feats @ W.T` and
    then EMA-updated `W` (weight-space, O(0.01)) with raw feature means
    (feature-space, O(1-10)); the scale mismatch made prototypes drift to
    garbage within a few batches, collapsing AUC to ~0.5. Cosine matching keeps
    everything on the unit sphere, so the classifier stays well-conditioned.
    """
    def __init__(self, model, device='cuda', lr=0.0, top_k_frac=0.1, num_classes=2, temp=10.0):
        super().__init__(model, lr=lr, device=device)
        self.top_k_frac = top_k_frac
        self.num_classes = num_classes
        self.temp = temp
        self.model.eval()
        proto = self._init_prototypes()
        self.prototypes = F.normalize(proto, dim=1)      # unit-norm class templates

    def _init_prototypes(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == self.num_classes:
                return m.weight.data.clone().float()      # (num_classes, feat_dim)
        raise ValueError("Cannot find classification head for T3A prototype init")

    def _features(self, x):
        """Penultimate representation, shape (B, feat_dim) matching the head weight.

        Must never return logits: prototypes are (num_classes, feat_dim), so a
        (B, num_classes) tensor makes `feats @ prototypes.T` fail with
        "mat1 and mat2 shapes cannot be multiplied (Bx2 and 768x2)".
        """
        if hasattr(self.model, 'get_features'):          # FoundationModelWrapper
            return self.model.get_features(x).flatten(1).float()

        # timm exposes the pre-logits representation directly. swin_t and vit_b16
        # contain no AdaptiveAvgPool2d/Flatten, so the hook path below found
        # nothing and the old code silently fell back to self.model(x) -- logits.
        if hasattr(self.model, 'forward_features') and hasattr(self.model, 'forward_head'):
            f = self.model.forward_features(x)
            return self.model.forward_head(f, pre_logits=True).flatten(1).float()

        feats = []
        h = None
        for layer in reversed(list(self.model.modules())):
            if isinstance(layer, (torch.nn.AdaptiveAvgPool2d, torch.nn.Flatten)):
                h = layer.register_forward_hook(lambda m, i, o: feats.append(o))
                break
        self.model(x)
        if h:
            h.remove()
        if feats:
            return feats[0].flatten(1).float()
        raise RuntimeError(
            f"T3A could not extract penultimate features from "
            f"{type(self.model).__name__}; refusing to fall back to logits.")

    @torch.no_grad()
    def adapt(self, x):
        self.model.eval()
        feats = F.normalize(self._features(x), dim=1)          # (B, d) unit-norm
        logits = (feats @ self.prototypes.T) * self.temp        # cosine logits

        probs = torch.softmax(logits, dim=1)
        conf, pseudo = probs.max(1)
        k = max(1, int(x.size(0) * self.top_k_frac))
        idx = conf.topk(k).indices
        for c in range(self.num_classes):
            mask = pseudo[idx] == c
            if mask.sum() > 0:
                new_proto = F.normalize(feats[idx][mask].mean(0), dim=0)
                self.prototypes[c] = F.normalize(
                    0.9 * self.prototypes[c] + 0.1 * new_proto, dim=0)
        return logits
