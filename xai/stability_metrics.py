# xai/stability_metrics.py
"""
Explanation-stability metrics (self-consistency of a saliency map before vs.
after TTA) plus the ESI composite index.

FIX (vs. original): ESI weights now form a proper convex combination of a
"similarity" term and a penalised "divergence" term, so ESI is bounded in
[0,1] by construction and every component carries a non-zero, reported weight.
The previous version's weights summed to 0.60 and silently dropped
entropy_delta — both fixed here.
"""
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from skimage.metrics import structural_similarity
from skimage.transform import resize


def normalize_map(smap):
    s_min, s_max = smap.min(), smap.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(smap)
    return (smap - s_min) / (s_max - s_min)


def resize_map(smap, target_size=(224, 224)):
    return resize(smap, target_size, anti_aliasing=True, preserve_range=True)


def compute_all_metrics(S_before, S_after, target_size=(224, 224)):
    """S_before/S_after: raw saliency maps (any size). Returns metric dict."""
    S_b = normalize_map(resize_map(S_before, target_size))
    S_a = normalize_map(resize_map(S_after,  target_size))

    eps = 1e-8
    b_flat, a_flat = S_b.flatten(), S_a.flatten()
    metrics = {}

    # 1. Cosine similarity
    denom = np.linalg.norm(b_flat) * np.linalg.norm(a_flat) + eps
    metrics['cosine_sim'] = float(np.dot(b_flat, a_flat) / denom)

    # 2. SSIM
    metrics['ssim'] = float(structural_similarity(S_b, S_a, data_range=1.0))

    # 3. Spearman rank correlation
    r, p = spearmanr(b_flat, a_flat)
    metrics['spearman_r'] = float(r) if not np.isnan(r) else 0.0
    metrics['spearman_p'] = float(p) if not np.isnan(p) else 1.0

    # 4. Earth Mover's Distance (Wasserstein-1)
    metrics['emd'] = float(wasserstein_distance(b_flat, a_flat))

    # 5. Symmetric KL divergence (distributional)
    p_dist = b_flat + eps; p_dist /= p_dist.sum()
    q_dist = a_flat + eps; q_dist /= q_dist.sum()
    kl_pq = np.sum(p_dist * np.log(p_dist / q_dist))
    kl_qp = np.sum(q_dist * np.log(q_dist / p_dist))
    metrics['kl_div'] = float(0.5 * (kl_pq + kl_qp))

    # 6. Saliency entropy change
    def saliency_entropy(s):
        sn = s.flatten() + eps; sn /= sn.sum()
        return -np.sum(sn * np.log(sn))
    metrics['entropy_delta'] = float(abs(saliency_entropy(S_a) - saliency_entropy(S_b)))

    # 7. Top-20% ROI IoU (clinical: do high-attention regions agree?)
    roi_b = (S_b >= np.percentile(S_b, 80)).astype(float)
    roi_a = (S_a >= np.percentile(S_a, 80)).astype(float)
    inter = (roi_b * roi_a).sum()
    union = ((roi_b + roi_a) > 0).sum()
    metrics['roi_iou'] = float(inter / (union + eps))

    return metrics


# Normalisation constants for the divergence terms. Empirically the ~95th
# percentile of the no-TTA baseline distribution; report in supplementary.
KL_MAX = 5.0
EMD_MAX = 0.5
ENT_MAX = 2.0


def compute_esi(metrics_dict, blend=0.70, kl_max=KL_MAX, emd_max=EMD_MAX, ent_max=ENT_MAX):
    """
    Explanation Stability Index (ESI) in [0,1]; higher = more stable.

        SIM = 0.30·SSIM + 0.25·cos + 0.25·spearman' + 0.20·roi_iou   (sums to 1)
        PEN = 0.40·kl_n + 0.40·emd_n + 0.20·ent_n                    (sums to 1)
        ESI = clip( blend·SIM + (1-blend)·(1 - PEN), 0, 1 )

    spearman' rescales Spearman r from [-1,1] to [0,1]. SIM and PEN are each in
    [0,1], so ESI is a proper convex combination bounded in [0,1]. Every metric
    (including entropy_delta, previously unused) contributes. `blend` is swept
    in the ablation figure.
    """
    ssim = np.clip(metrics_dict.get('ssim', 0.0), 0.0, 1.0)
    cos  = np.clip(metrics_dict.get('cosine_sim', 0.0), 0.0, 1.0)
    spr  = np.clip(0.5 * (metrics_dict.get('spearman_r', 0.0) + 1.0), 0.0, 1.0)
    roi  = np.clip(metrics_dict.get('roi_iou', 0.0), 0.0, 1.0)
    sim = 0.30 * ssim + 0.25 * cos + 0.25 * spr + 0.20 * roi

    kl_n  = min(metrics_dict.get('kl_div', 0.0), kl_max) / kl_max
    emd_n = min(metrics_dict.get('emd', 0.0), emd_max) / emd_max
    ent_n = min(metrics_dict.get('entropy_delta', 0.0), ent_max) / ent_max
    pen = 0.40 * kl_n + 0.40 * emd_n + 0.20 * ent_n

    esi = blend * sim + (1.0 - blend) * (1.0 - pen)
    return float(np.clip(esi, 0.0, 1.0))
