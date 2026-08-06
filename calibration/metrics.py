# calibration/metrics.py
"""
Calibration metrics for binary and multi-class outputs.

Binary  : pass probs_pos = P(y=1), labels in {0,1}.
Multi   : pass probs = (N,K) array, labels in {0..K-1} via compute_calibration_multiclass.
Both return {ece, adaptive_ece, brier, nll}.
"""
import numpy as np


def _ece_from_conf(conf, correct, n_bins=15):
    """Top-label ECE given per-sample confidence and correctness."""
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    # adaptive (equal-mass) ECE
    order = np.argsort(conf)
    aece = 0.0
    bs = max(1, len(conf) // n_bins)
    for i in range(0, len(conf), bs):
        idx = order[i:i + bs]
        if len(idx) == 0:
            continue
        aece += (len(idx) / len(conf)) * abs(correct[idx].mean() - conf[idx].mean())
    return float(ece), float(aece)


def compute_calibration_metrics(probs_pos, labels, n_bins=15):
    """Binary calibration. probs_pos = P(y=1)."""
    probs = np.asarray(probs_pos, dtype=float)
    labels = np.asarray(labels, dtype=float)
    conf = np.maximum(probs, 1.0 - probs)          # top-label confidence
    pred = (probs >= 0.5).astype(float)
    correct = (pred == labels).astype(float)
    ece, aece = _ece_from_conf(conf, correct, n_bins)
    brier = float(np.mean((probs - labels) ** 2))
    eps = 1e-8
    nll = float(-np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)))
    return {"ece": ece, "adaptive_ece": aece, "brier": brier, "nll": nll}


def compute_calibration_multiclass(probs, labels, n_bins=15):
    """probs: (N,K) softmax. labels: (N,) ints."""
    P = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    N, K = P.shape
    conf = P.max(1)
    pred = P.argmax(1)
    correct = (pred == y).astype(float)
    ece, aece = _ece_from_conf(conf, correct, n_bins)
    onehot = np.eye(K)[y]
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    eps = 1e-8
    nll = float(-np.mean(np.log(P[np.arange(N), y] + eps)))
    return {"ece": ece, "adaptive_ece": aece, "brier": brier, "nll": nll}
