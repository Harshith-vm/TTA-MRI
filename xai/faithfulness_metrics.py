# xai/faithfulness_metrics.py
"""
Faithfulness metrics: do saliency maps actually reflect what the model uses?

Self-consistency (stability_metrics.py) measures whether an explanation drifts
under TTA. Faithfulness measures whether the explanation is *correct* at all.
A Q1 reviewer expects both. Implemented:

  - Insertion / Deletion AUC  (Petsiuk et al., BMVC 2018)
  - Pointing Game accuracy     (Zhang et al., IJCV 2018) — uses Camelyon17 masks
  - Infidelity                 (Yeh et al., NeurIPS 2019)

All operate on a single image tensor and its saliency map.
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _prob_of_class(model, x, target, device):
    logits = model(x.to(device))
    return torch.softmax(logits, dim=1)[0, target].item()


@torch.no_grad()
def insertion_deletion_auc(model, image, saliency, target, device,
                           step_frac=0.05, mode="both", baseline="blur"):
    """
    image:    (3,H,W) normalised tensor
    saliency: (H,W) numpy array (will be resized to image H,W)
    Returns dict with insertion_auc, deletion_auc (higher insertion / lower
    deletion = more faithful).
    """
    C, H, W = image.shape
    sal = torch.from_numpy(np.asarray(saliency, dtype=np.float32))
    sal = F.interpolate(sal[None, None], size=(H, W), mode="bilinear",
                        align_corners=False)[0, 0]
    order = torch.argsort(sal.flatten(), descending=True)  # most salient first
    n_pix = H * W
    step = max(1, int(step_frac * n_pix))

    if baseline == "blur":
        k = 11
        blur = F.avg_pool2d(image[None], k, stride=1, padding=k // 2)[0]
    else:
        blur = torch.zeros_like(image)

    out = {}

    if mode in ("insertion", "both"):
        cur = blur.clone()
        scores = []
        for i in range(0, n_pix, step):
            idx = order[i:i + step]
            ys, xs = idx // W, idx % W
            cur[:, ys, xs] = image[:, ys, xs]
            scores.append(_prob_of_class(model, cur[None], target, device))
        out["insertion_auc"] = float(np.trapz(scores, dx=1.0 / len(scores)))

    if mode in ("deletion", "both"):
        cur = image.clone()
        scores = []
        for i in range(0, n_pix, step):
            idx = order[i:i + step]
            ys, xs = idx // W, idx % W
            cur[:, ys, xs] = blur[:, ys, xs]
            scores.append(_prob_of_class(model, cur[None], target, device))
        out["deletion_auc"] = float(np.trapz(scores, dx=1.0 / len(scores)))

    return out


def pointing_game(saliency, gt_mask, tolerance=15):
    """
    saliency: (H,W) array. gt_mask: (H,W) binary ground-truth region (tumour).
    Hit if the saliency argmax falls within `tolerance` px of a positive pixel.
    Returns 1.0 (hit) or 0.0 (miss); average over a dataset for accuracy.
    """
    sal = np.asarray(saliency, dtype=np.float32)
    gt = np.asarray(gt_mask) > 0
    if gt.sum() == 0:
        return None  # no annotated region on this patch; skip
    H, W = sal.shape
    if gt.shape != (H, W):
        from skimage.transform import resize
        gt = resize(gt.astype(float), (H, W), order=0, preserve_range=True) > 0.5
    py, px = np.unravel_index(np.argmax(sal), sal.shape)
    ys, xs = np.where(gt)
    d = np.sqrt((ys - py) ** 2 + (xs - px) ** 2).min()
    return 1.0 if d <= tolerance else 0.0


def infidelity(model, image, saliency, target, device, n_pert=20, noise_std=0.1):
    """
    Yeh et al. (2019) infidelity: expected squared difference between
    (perturbation·saliency) and the actual change in model output. Lower=better.
    """
    C, H, W = image.shape
    sal = torch.from_numpy(np.asarray(saliency, dtype=np.float32))
    sal = F.interpolate(sal[None, None], size=(H, W), mode="bilinear",
                        align_corners=False)[0, 0].to(device)
    sal3 = sal[None].expand(C, H, W).to(device)
    base = _prob_of_class(model, image[None], target, device)
    errs = []
    for _ in range(n_pert):
        pert = torch.randn(C, H, W, device=device) * noise_std
        x_pert = (image.to(device) - pert)
        pred_drop = (pert * sal3).sum().item()
        actual_drop = base - _prob_of_class(model, x_pert[None], target, device)
        errs.append((pred_drop - actual_drop) ** 2)
    return float(np.mean(errs))
