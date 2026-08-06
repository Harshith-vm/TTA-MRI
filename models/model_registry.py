# models/model_registry.py
"""
Model lineup — ZERO gated access required:

CNN baselines (torchvision / timm, Apache 2.0):
  - resnet50       : 25M params, BN-based, canonical pathology baseline
  - efficientnet_b3: 12M params, BN + SE blocks, better scaling

Transformer baselines (timm, Apache 2.0):
  - swin_t         : 28M params, LayerNorm, hierarchical attention
  - vit_b16        : 86M params, pure ViT, LayerNorm only

Pathology foundation models (all instant download):
  - phikon_v2      : 307M params ViT-L, Owkin, CC-BY-NC 4.0, owkin/phikon-v2 on HF
                     Trained on public slides only. No gating, no registration.
  - h_optimus_0    : 1.1B params ViT-G, Bioptimus, Apache 2.0, bioptimus/H-optimus-0 on HF
                     Fully open source. Most permissive license.
  - prov_gigapath  : 1.14B params ViT-g/14, Microsoft/Nature 2024, CC-BY-NC-SA 4.0
                     Requires free HuggingFace account + single ToS click (instant).
                     github.com/prov-gigapath/prov-gigapath

REMOVED (require manual approval / days of waiting):
  - UNI2   (Mahmood Lab — manual review gate)
  - CONCH  (Mahmood Lab — manual review gate)
"""
import torch
import torch.nn as nn
from timm import create_model
from transformers import AutoModel
import os

CKPT_ROOT = os.environ.get("CKPT_ROOT", "./checkpoints")

def get_model(name: str, num_classes: int = 2, pretrained: bool = True):
    """
    Returns (model, feature_dim, norm_layer_type).
    norm_layer_type: 'bn' | 'ln' — determines which TTA adaptation strategy applies.
    """
    if name == "resnet50":
        m = create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
        return m, 2048, 'bn'

    elif name == "efficientnet_b3":
        m = create_model('efficientnet_b3', pretrained=pretrained, num_classes=num_classes)
        return m, 1536, 'bn'

    elif name == "swin_t":
        m = create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
        return m, 768, 'ln'

    elif name == "vit_b16":
        m = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        return m, 768, 'ln'

    elif name == "phikon_v2":
        # owkin/phikon-v2 — ViT-L/16, no registration needed
        backbone = AutoModel.from_pretrained("owkin/phikon-v2")
        feat_dim = backbone.config.hidden_size  # 1024
        model = FoundationModelWrapper(backbone, feat_dim, num_classes)
        return model, feat_dim, 'ln'

    elif name == "h_optimus_0":
        # bioptimus/H-optimus-0 — ViT-G, Apache 2.0
        backbone = AutoModel.from_pretrained(
            "bioptimus/H-optimus-0",
            trust_remote_code=True
        )
        feat_dim = backbone.config.hidden_size  # 1536
        model = FoundationModelWrapper(backbone, feat_dim, num_classes)
        return model, feat_dim, 'ln'

    elif name == "prov_gigapath":
        # prov-gigapath — ViT-g/14, requires HF account + ToS click
        backbone = create_model("hf-hub:prov-gigapath/prov-gigapath", pretrained=True)
        feat_dim = 1536
        model = FoundationModelWrapper(backbone, feat_dim, num_classes)
        return model, feat_dim, 'ln'

    else:
        raise ValueError(f"Unknown model: {name}")


class FoundationModelWrapper(nn.Module):
    """
    Wraps a HuggingFace or timm foundation model backbone with a linear classification head.
    Backbone is FROZEN by default — only the linear head is trained initially.
    For TTA: only LayerNorm affine params (gamma, beta) in the LAST 2 transformer blocks
    are unfrozen. This prevents catastrophic forgetting of pathology features.
    """
    def __init__(self, backbone, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, num_classes)
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_last_n_blocks(self, n: int = 2):
        """Unfreeze LN params in last N transformer blocks for partial fine-tuning."""
        all_blocks = [m for m in self.backbone.modules()
                      if isinstance(m, nn.LayerNorm)]
        for ln in all_blocks[-n*4:]:  # ~4 LN per block
            for p in ln.parameters():
                p.requires_grad_(True)

    def forward(self, x):
        out = self.backbone(x)
        # Handle both HuggingFace BaseModelOutput and plain tensors
        if hasattr(out, 'last_hidden_state'):
            feat = out.last_hidden_state[:, 0]  # CLS token
        elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out  # plain tensor (timm models)
        return self.head(feat)

    def get_features(self, x):
        """Return CLS token features without classification head."""
        with torch.no_grad():
            out = self.backbone(x)
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]
        return out
