"""Calibration analysis and plotting for IPL betting models."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def plot_calibration_curves(models_dict, feature_sets, market_configs):
    """Plot calibration curves for all trained models.

    Args:
        models_dict: {market_name: calibrated_model}
        feature_sets: dict of feature DataFrames
        market_configs: MARKETS dict from train.py
    """
    NOTEBOOKS_DIR.mkdir(exist_ok=True)

    n_models = len(models_dict)
    if n_models == 0:
        print("No models to plot calibration curves for.")
        return

    fig, axes = plt.subplots(
        (n_models + 2) // 3, 3,
        figsize=(15, 5 * ((n_models + 2) // 3)),
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, (market_name, model) in enumerate(models_dict.items()):
        ax = axes[idx]
        config = market_configs.get(market_name, {})
        data_key = config.get("data_key")
        target_col = config.get("target")
        feature_cols = config.get("features", [])

        if data_key not in feature_sets or feature_sets[data_key] is None:
            ax.set_title(f"{market_name}\n(no data)")
            continue

        df = feature_sets[data_key]
        if target_col not in df.columns:
            ax.set_title(f"{market_name}\n(no target)")
            continue

        # Get available features
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            continue

        mask = df[target_col].notna()
        subset = df[mask]
        X = subset[available].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.fillna(X.median()).fillna(0)
        y = subset[target_col].values

        try:
            raw_pred = model.predict_proba(X)[:, 1]
            if hasattr(model, "_platt_calibrator") and model._platt_calibrator is not None:
                y_pred = model._platt_calibrator.predict_proba(raw_pred.reshape(-1, 1))[:, 1]
            elif hasattr(model, "_iso_calibrator") and model._iso_calibrator is not None:
                y_pred = model._iso_calibrator.predict(raw_pred)
            else:
                y_pred = raw_pred
            y_pred = np.clip(y_pred, 0.02, 0.98)
            fraction_pos, mean_predicted = calibration_curve(y, y_pred, n_bins=10, strategy="uniform")

            ax.plot(mean_predicted, fraction_pos, "s-", label="Model", color="#2196F3")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")

            brier = brier_score_loss(y, y_pred)
            ax.set_title(f"{market_name}\nBrier: {brier:.4f}")
            ax.legend(loc="lower right", fontsize=8)
        except Exception as e:
            ax.set_title(f"{market_name}\n(error: {str(e)[:30]})")

    # Hide unused axes
    for idx in range(len(models_dict), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = NOTEBOOKS_DIR / "calibration_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration curves saved to {path}")


def plot_feature_importance(models_dict, market_configs, feature_sets):
    """Plot feature importance for each model."""
    NOTEBOOKS_DIR.mkdir(exist_ok=True)

    for market_name, model in models_dict.items():
        config = market_configs.get(market_name, {})
        feature_cols = config.get("features", [])
        data_key = config.get("data_key")

        if data_key not in feature_sets:
            continue

        df = feature_sets[data_key]
        available = [c for c in feature_cols if c in df.columns]

        # Get feature importances from model
        try:
            importances = model.feature_importances_
        except AttributeError:
            continue

        if len(importances) != len(available):
            continue

        fig, ax = plt.subplots(figsize=(10, max(4, len(available) * 0.3)))
        sorted_idx = np.argsort(importances)
        ax.barh(
            [available[i] for i in sorted_idx],
            importances[sorted_idx],
            color="#4CAF50",
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{market_name} - Feature Importance")
        plt.tight_layout()

        path = NOTEBOOKS_DIR / f"importance_{market_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Feature importance plots saved to {NOTEBOOKS_DIR}")
