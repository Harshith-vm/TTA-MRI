# xai/xai_runner.py
"""
Unified saliency generation across multiple XAI families so we can show that
the ESI trend generalises beyond a single attribution method (a Q1 requirement).

Families:
  - CAM-based:   GradCAM, GradCAM++, ScoreCAM          (grad-cam library)
  - Gradient:    Integrated Gradients                  (captum)
  - Transformer: Attention rollout / relevance         (native hooks)

Per-architecture target layers are mapped EXPLICITLY (no brittle name heuristics)
so foundation ViTs produce correct maps.
"""
import torch
import numpy as np


def _last_block_norm(backbone):
    """Last transformer block's PRE-attention norm (the ViT analogue of
    vit_b16's blocks[-1].norm1).

    Picking the post-attention norm (norm2) or the final backbone layernorm
    yields an all-zero CAM: both sit after the last token-mixing op, and the
    head reads only the CLS token (last_hidden_state[:, 0]), so patch tokens
    can no longer influence the output and their gradients are exactly zero.
    The old `lns[-2]` heuristic landed on exactly that layer for phikon_v2.
    """
    import torch.nn as nn
    blocks = None
    enc = getattr(backbone, "encoder", None)
    if enc is not None and hasattr(enc, "layer"):
        blocks = enc.layer                      # HuggingFace ViT
    elif hasattr(backbone, "blocks"):
        blocks = backbone.blocks                # timm ViT
    if blocks is not None and len(blocks) > 0:
        last = blocks[-1]
        for attr in ("norm1", "layernorm_before"):
            m = getattr(last, attr, None)
            if isinstance(m, nn.LayerNorm):
                return m
        inner = [m for m in last.modules() if isinstance(m, nn.LayerNorm)]
        if inner:
            return inner[0]
    lns = [m for m in backbone.modules() if isinstance(m, nn.LayerNorm)]
    if not lns:
        raise ValueError("No LayerNorm found in transformer backbone")
    return lns[-2] if len(lns) >= 2 else lns[-1]


def get_target_layer(model, model_name):
    """Explicit CAM target layer per architecture."""
    if model_name == "resnet50":
        return [model.layer4[-1]]
    if model_name == "efficientnet_b3":
        return [model.conv_head]
    if model_name == "swin_t":
        return [model.layers[-1].blocks[-1].norm2]
    if model_name == "vit_b16":
        return [model.blocks[-1].norm1]
    if model_name in ("phikon_v2", "h_optimus_0", "prov_gigapath"):
        bb = getattr(model, "backbone", model)
        return [_last_block_norm(bb)]
    raise ValueError(f"No target layer defined for {model_name}")


# grid sizes for reshape (224/patch). ViT-B/16 -> 14, GigaPath /14 -> 16.
GRID = {"vit_b16": 14, "phikon_v2": 14, "swin_t": 7,
        "h_optimus_0": 14, "prov_gigapath": 16}


def vit_reshape_transform_factory(model_name):
    g = GRID.get(model_name, 14)

    def _t(tensor, height=g, width=g):
        n = tensor.shape[1]
        if n == height * width + 1:
            tensor = tensor[:, 1:, :]
        elif n != height * width:
            side = int(round(n ** 0.5)); height = width = side
            tensor = tensor[:, -height * width:, :]
        r = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        return r.permute(0, 3, 1, 2)
    return _t


def _is_transformer(model_name):
    return model_name in ("vit_b16", "swin_t", "phikon_v2",
                          "h_optimus_0", "prov_gigapath")


def generate_saliency(model, imgs, model_name, device, method="gradcam", targets=None):
    """
    imgs: (B,3,H,W) tensor on `device`. Returns list[np.ndarray | None] length B.
    method in {gradcam, gradcampp, scorecam, integrated_gradients,
               transformer_attribution}.
    """
    try:
        if method in ("gradcam", "gradcampp", "scorecam"):
            return _cam_saliency(model, imgs, model_name, method, targets)
        if method == "integrated_gradients":
            return _ig_saliency(model, imgs, device, targets)
        if method == "transformer_attribution":
            return _attn_saliency(model, imgs, model_name, device)
        raise ValueError(f"Unknown XAI method {method}")
    except Exception as e:
        print(f"    [xai:{method}] failed on {model_name}: {e}", flush=True)
        return [None] * imgs.size(0)


def _cam_saliency(model, imgs, model_name, method, targets):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cam_cls = {"gradcam": GradCAM, "gradcampp": GradCAMPlusPlus,
               "scorecam": ScoreCAM}[method]
    rt = vit_reshape_transform_factory(model_name) if _is_transformer(model_name) else None
    layers = get_target_layer(model, model_name)
    cam_targets = ([ClassifierOutputTarget(int(t)) for t in targets]
                   if targets is not None else None)
    # FoundationModelWrapper freezes its whole backbone, so for TTA methods that
    # never unfreeze it (no_tta, bn_adapt, lame, sicl, t3a) the target activation
    # carries no grad_fn and grad-cam hands us None -> "'NoneType' has no
    # attribute 'shape'". Enabling grad on the target layer's own affine params is
    # enough to put its output in the graph; nothing is optimised here, and the
    # previous flags are restored so TTA state is untouched. No-op for timm models.
    saved = [(p, p.requires_grad) for layer in layers for p in layer.parameters()]
    for p, _ in saved:
        p.requires_grad_(True)
    try:
        with cam_cls(model=model, target_layers=layers, reshape_transform=rt) as cam:
            maps = cam(input_tensor=imgs, targets=cam_targets)
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)
    return [maps[i] for i in range(imgs.size(0))]


def _ig_saliency(model, imgs, device, targets):
    from captum.attr import IntegratedGradients
    model.zero_grad(set_to_none=True)
    ig = IntegratedGradients(model)
    if targets is None:
        with torch.no_grad():
            targets = model(imgs).argmax(1)
    targets = torch.as_tensor(targets, device=device)
    baseline = torch.zeros_like(imgs)
    attr = ig.attribute(imgs, baselines=baseline, target=targets, n_steps=32,
                        internal_batch_size=imgs.size(0))
    attr = attr.abs().sum(1)
    return [attr[i].detach().cpu().numpy() for i in range(imgs.size(0))]


@torch.no_grad()
def _attn_saliency(model, imgs, model_name, device):
    """Attention rollout (Abnar & Zuidema 2020) for ViT-style models."""
    attentions, hooks = [], []
    try:
        for m in model.modules():
            if 'attention' in type(m).__name__.lower() or hasattr(m, 'attn_drop'):
                def hook(mod, inp, out):
                    a = out[0] if isinstance(out, tuple) else out
                    if torch.is_tensor(a) and a.dim() == 4:
                        attentions.append(a.detach())
                hooks.append(m.register_forward_hook(hook))
        model(imgs)
        if not attentions:
            return [None] * imgs.size(0)

        B = imgs.size(0)
        out_maps = []
        seq = attentions[0].shape[-1]
        eye = torch.eye(seq, device=device)
        for b in range(B):
            rollout = eye.clone()
            for attn in attentions:
                a = attn[b].mean(0)
                a = 0.5 * a + 0.5 * eye
                a = a / a.sum(-1, keepdim=True)
                rollout = a @ rollout
            mask = rollout[0, 1:]
            side = int(mask.shape[0] ** 0.5)
            out_maps.append(mask[:side * side].reshape(side, side).cpu().numpy())
        return out_maps
    finally:
        # Always remove hooks and drop cached activations — a leftover hook or
        # a retained (B, heads, seq, seq) tensor on GPU costs hundreds of MB
        # per invocation and would compound across the 3000-combo sweep.
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        attentions.clear()
