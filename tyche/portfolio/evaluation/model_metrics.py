"""Predictive-model evaluation metrics.

Point-forecast quality (MAE, RMSE, directional accuracy), cross-sectional ranking
skill (Pearson IC, Spearman rank IC), distributional fit (Gaussian NLL), and
calibration (predictive-interval coverage). All operate on saved predictions vs the
realized forward returns.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

from tyche.portfolio.model.predict import Predictions


def _flat(pred: Predictions) -> tuple[np.ndarray, np.ndarray]:
    return pred.mu.reshape(-1), pred.target.reshape(-1)


def mae(pred: Predictions) -> float:
    m, r = _flat(pred)
    return float(np.mean(np.abs(m - r)))


def rmse(pred: Predictions) -> float:
    m, r = _flat(pred)
    return float(np.sqrt(np.mean((m - r) ** 2)))


def directional_accuracy(pred: Predictions) -> float:
    m, r = _flat(pred)
    return float(np.mean(np.sign(m) == np.sign(r)))


def information_coefficient(pred: Predictions) -> float:
    """Mean per-date cross-sectional Pearson IC of predicted vs realized returns."""
    ics = [
        pearsonr(pred.mu[i], pred.target[i])[0]
        for i in range(len(pred.mu))
        if np.std(pred.mu[i]) > 0
    ]
    return float(np.nanmean(ics)) if ics else float("nan")


def rank_ic(pred: Predictions) -> float:
    ics = [
        spearmanr(pred.mu[i], pred.target[i]).correlation for i in range(len(pred.mu))
    ]
    return float(np.nanmean(ics)) if ics else float("nan")


def negative_log_likelihood(pred: Predictions) -> float:
    """Mean multivariate-Gaussian NLL over rebalance dates."""
    n = pred.mu.shape[1]
    total = 0.0
    for i in range(len(pred.mu)):
        diff = pred.target[i] - pred.mu[i]
        cov = pred.cov[i] + 1e-6 * np.eye(n)
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, diff)
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        total += 0.5 * (z @ z + logdet + n * np.log(2 * np.pi))
    return float(total / len(pred.mu))


def interval_coverage(pred: Predictions, level: float = 0.9) -> float:
    """Fraction of realized returns inside the per-asset central predictive interval."""
    z = norm.ppf(0.5 + level / 2.0)
    var = np.stack([np.diag(c) for c in pred.cov])
    lo = pred.mu - z * np.sqrt(var)
    hi = pred.mu + z * np.sqrt(var)
    return float(np.mean((pred.target >= lo) & (pred.target <= hi)))


def evaluate_model(pred: Predictions) -> dict[str, float]:
    return {
        "MAE": mae(pred),
        "RMSE": rmse(pred),
        "directional_accuracy": directional_accuracy(pred),
        "IC": information_coefficient(pred),
        "rank_IC": rank_ic(pred),
        "NLL": negative_log_likelihood(pred),
        "coverage_90": interval_coverage(pred, 0.9),
    }
