"""Metrics: ranking (ROC-AUC, PR-AUC), probability quality (Brier, log loss,
calibration error) and an operational one - precision at a fixed alert budget."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

SEED = 42


def top_k_mask(p: np.ndarray, frac: float, seed: int = SEED) -> np.ndarray:
    """Flag the `frac` highest-risk cases. Ties (e.g. every case of one type gets the
    same baseline score) are broken at random so no model benefits from row order."""
    k = int(np.ceil(frac * len(p)))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(p)), -p))
    mask = np.zeros(len(p), dtype=bool)
    mask[order[:k]] = True
    return mask


def precision_at_budget(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    m = top_k_mask(p, frac)
    tp = int(y[m].sum())
    return {"budget": frac, "n_flagged": int(m.sum()), "precision": tp / m.sum(),
            "recall": tp / max(int(y.sum()), 1), "lift": (tp / m.sum()) / y.mean()}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def all_metrics(y: np.ndarray, p: np.ndarray, budgets=(0.05, 0.10, 0.20)) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "late_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece_10bins": expected_calibration_error(y, p),
    }
    for b in budgets:
        r = precision_at_budget(y, p, b)
        tag = f"{int(round(b * 100))}pct"
        out[f"precision_at_{tag}"] = float(r["precision"])
        out[f"recall_at_{tag}"] = float(r["recall"])
    return out


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos
