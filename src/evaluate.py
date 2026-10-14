"""Evaluation helpers: screening-oriented metrics, CV, calibration."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

SCORING = {
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def cv_report(model, X, y, n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified k-fold CV; returns mean and std for each metric.

    `model` should be a Pipeline so imputation/scaling happen inside each fold
    (otherwise information leaks from the validation fold).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = cross_validate(clone(model), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    out = {}
    for name in SCORING:
        vals = res[f"test_{name}"]
        out[name] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
    return out


def compare_models(models: dict, X, y, **kwargs) -> pd.DataFrame:
    """Run cv_report for several named models and return a tidy table."""
    rows = {name: cv_report(m, X, y, **kwargs) for name, m in models.items()}
    return pd.DataFrame(rows).T


def holdout_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Threshold-dependent and threshold-free metrics on a held-out set."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def calibration_points(y_true, y_prob, n_bins: int = 10):
    """Return (mean predicted prob, observed fraction positive) per bin."""
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return mean_pred, frac_pos
