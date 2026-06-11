"""Train XGBoost + LightGBM ensemble models for IPL betting markets.

Each market (player runs thresholds, wickets, team totals, match winner)
gets an ensemble of XGBoost and LightGBM, with calibrated probability outputs.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent


# ── Feature columns for each market ──────────────────────────────────────

BATTING_FEATURE_COLS = [
    # Rolling form
    "runs_roll_5", "runs_roll_10", "runs_ewm",
    "strike_rate_roll_5", "strike_rate_roll_10", "strike_rate_ewm",
    "balls_faced_roll_5", "balls_faced_roll_10", "balls_faced_ewm",
    # Career
    "career_runs_mean", "career_sr_mean", "career_innings",
    "last_5_avg", "runs_std_10",
    # Context
    "batting_position",
    "is_second_innings",
    # Venue
    "venue_bat_avg", "venue_bat_sr", "venue_bat_innings",
    # Opponent bowling strength
    "opp_bowl_econ", "opp_bowl_wr",
    # Matchup history
    "avg_matchup_sr", "avg_matchup_dismissal_rate", "total_matchup_balls",
    # Powerplay
    "pp_runs_roll_5", "pp_sr_roll_5",
    # Conditions
    "venue_recent_avg_score", "venue_overall_avg_score",
    # Toss
    "toss_elected_bat", "toss_elected_field",
    # Tournament phase
    "match_number", "is_playoff",
]

BOWLING_FEATURE_COLS = [
    # Rolling form
    "wickets_roll_5", "wickets_roll_10", "wickets_ewm",
    "economy_roll_5", "economy_roll_10", "economy_ewm",
    "runs_conceded_roll_5", "runs_conceded_roll_10", "runs_conceded_ewm",
    # Career
    "career_wickets_mean", "career_economy_mean", "career_bowl_innings",
    # Context
    "is_second_innings",
    # Venue
    "venue_bowl_econ", "venue_bowl_wickets_avg", "venue_bowl_innings",
    # Opponent batting strength
    "opp_bat_avg", "opp_bat_sr",
    # Phase-specific
    "death_econ_roll_5", "pp_bowl_econ_roll_5",
    # Conditions
    "venue_recent_avg_score", "venue_overall_avg_score",
    # Toss
    "toss_elected_bat", "toss_elected_field",
    # Tournament phase
    "match_number", "is_playoff",
]

TEAM_FEATURE_COLS = [
    "team_total_roll_3", "team_total_roll_5", "team_total_ewm",
    "team_avg_total",
    # Venue scoring history
    "venue_avg_first_inn", "venue_avg_first_inn_last5", "venue_matches_played",
    # Conditions
    "venue_recent_avg_score", "venue_overall_avg_score",
    # Toss
    "toss_elected_bat", "toss_elected_field",
    # Tournament phase
    "match_number", "is_playoff",
]

MATCH_FEATURE_COLS = [
    # NOTE: first_inn_score is NOT used — it's only known AFTER innings 1 ends,
    # which makes it unusable for pre-match predictions (data leakage at inference).
    # ELO
    "team1_elo", "team2_elo", "elo_diff",
    # Team form
    "team1_win_rate_5", "team1_win_rate_10", "team1_career_win_rate",
    "team2_win_rate_5", "team2_win_rate_10", "team2_career_win_rate",
    # Head-to-head at venue
    "h2h_venue_matches", "h2h_venue_win_rate",
    # Home advantage
    "team1_is_home", "team2_is_home",
    # Toss
    "toss_elected_bat", "toss_elected_field",
    # Tournament phase
    "match_number", "is_playoff",
]


# ── Market definitions ───────────────────────────────────────────────────

MARKETS = {}

# Player runs markets — each threshold gets its own career/recent hit rate feature
for threshold in [10, 15, 20, 25, 30, 40, 50, 70, 100]:
    MARKETS[f"player_runs_{threshold}"] = {
        "features": BATTING_FEATURE_COLS + [
            f"career_rate_{threshold}_runs",
            f"recent_rate_{threshold}_runs",
        ],
        "target": f"target_{threshold}_runs",
        "data_key": "batting",
        "description": f"Player scores {threshold}+ runs",
    }

# Player wickets markets
for threshold in [1, 2, 3, 4, 5]:
    MARKETS[f"player_wickets_{threshold}"] = {
        "features": BOWLING_FEATURE_COLS + [
            f"career_rate_{threshold}_wickets",
            f"recent_rate_{threshold}_wickets",
        ],
        "target": f"target_{threshold}_wickets",
        "data_key": "bowling",
        "description": f"Player takes {threshold}+ wickets",
    }

# Team total markets
MARKETS["first_innings_over_160"] = {
    "features": TEAM_FEATURE_COLS,
    "target": "target_over_160",
    "data_key": "first_innings",
    "description": "First innings total over 160",
}

# Match winner
MARKETS["match_winner"] = {
    "features": MATCH_FEATURE_COLS,
    "target": "team1_won",
    "data_key": "match_results",
    "description": "Team batting first wins the match",
}


def _get_season(df):
    """Extract season/year from the data for train/test splitting."""
    if "season" in df.columns:
        return df["season"]
    if "start_date" in df.columns:
        return pd.to_datetime(df["start_date"], errors="coerce").dt.year
    return None


def _prepare_data(df, feature_cols, target_col):
    """Prepare X, y arrays, dropping rows with missing target or all-NaN features."""
    available_features = [c for c in feature_cols if c in df.columns]
    if not available_features:
        raise ValueError(f"No feature columns found. Available: {list(df.columns)}")

    mask = df[target_col].notna()
    subset = df[mask].copy()

    X = subset[available_features].copy()
    y = subset[target_col].values

    # Convert to numeric but keep NaN — XGBoost and LightGBM handle NaN natively
    # and learn optimal split directions for missing values. Filling with
    # median/0 destroys this signal and creates train/predict mismatches.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, available_features, subset


class EnsembleModel:
    """Wrapper that averages XGBoost and LightGBM predictions."""

    def __init__(self, xgb_model, lgb_model, xgb_weight=0.5):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.xgb_weight = xgb_weight
        self.lgb_weight = 1.0 - xgb_weight
        self._platt_calibrator = None
        self._iso_calibrator = None

    def predict_proba(self, X):
        """Average predicted probabilities from both models."""
        xgb_proba = self.xgb_model.predict_proba(X)
        lgb_proba = self.lgb_model.predict_proba(X)
        return self.xgb_weight * xgb_proba + self.lgb_weight * lgb_proba

    def get_booster(self):
        """Delegate to XGBoost for feature names."""
        return self.xgb_model.get_booster()

    @property
    def feature_importances_(self):
        xgb_imp = self.xgb_model.feature_importances_
        lgb_imp = self.lgb_model.feature_importances_
        xgb_norm = xgb_imp / (xgb_imp.sum() or 1)
        lgb_norm = lgb_imp / (lgb_imp.sum() or 1)
        return (xgb_norm + lgb_norm) / 2


def train_market_model(df, market_name, market_config, train_seasons, val_seasons):
    """Train and calibrate a single market model.

    Returns: (calibrated_model, metrics_dict, feature_importances)
    """
    feature_cols = market_config["features"]
    target_col = market_config["target"]

    if target_col not in df.columns:
        print(f"  Skipping {market_name}: target column '{target_col}' not found")
        return None, None, None

    X, y, used_features, subset = _prepare_data(df, feature_cols, target_col)

    if len(X) < 50:
        print(f"  Skipping {market_name}: only {len(X)} samples")
        return None, None, None

    season = _get_season(subset)
    if season is not None:
        season_vals = pd.to_numeric(season, errors="coerce")
        train_mask = season_vals.isin(train_seasons) | season_vals.le(max(train_seasons))
        val_mask = season_vals.isin(val_seasons)

        # Fallback if season split yields too few samples
        if val_mask.sum() < 10:
            split_idx = int(len(X) * 0.8)
            train_mask = pd.Series([True] * split_idx + [False] * (len(X) - split_idx), index=X.index)
            val_mask = ~train_mask
    else:
        split_idx = int(len(X) * 0.8)
        train_mask = pd.Series([True] * split_idx + [False] * (len(X) - split_idx), index=X.index)
        val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask.values]
    X_val, y_val = X[val_mask], y[val_mask.values]

    # Extract sample weights if available
    sample_weights_train = None
    if "sample_weight" in subset.columns:
        w = subset["sample_weight"].values
        sample_weights_train = w[train_mask.values]

    if len(X_val) < 5:
        print(f"  Skipping {market_name}: validation set too small ({len(X_val)})")
        return None, None, None

    print(f"  {market_name}: train={len(X_train)}, val={len(X_val)}, "
          f"pos_rate={y_train.mean():.3f}")

    # Handle class imbalance
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 0.01)

    # ── Train XGBoost ──
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights_train,
        verbose=False,
    )

    # ── Train LightGBM ──
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights_train,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Ensemble: create wrapper that averages both models ──
    ensemble = EnsembleModel(xgb_model, lgb_model)

    # ── Calibration via Platt scaling on cross-validated OOF predictions ──
    from sklearn.model_selection import StratifiedKFold

    n_cal_folds = 5
    oof_raw = np.zeros(len(X_train))
    oof_y = y_train.copy()
    skf = StratifiedKFold(n_splits=n_cal_folds, shuffle=True, random_state=42)

    xgb_iters = xgb_model.best_iteration + 1 if hasattr(xgb_model, "best_iteration") and xgb_model.best_iteration else 200
    lgb_iters = lgb_model.best_iteration_ if hasattr(lgb_model, "best_iteration_") and lgb_model.best_iteration_ else 200

    for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
        X_ft = X_train.iloc[fold_train_idx]
        y_ft = y_train[fold_train_idx]
        X_fv = X_train.iloc[fold_val_idx]
        w_ft = sample_weights_train[fold_train_idx] if sample_weights_train is not None else None

        fold_xgb = xgb.XGBClassifier(
            n_estimators=xgb_iters, max_depth=6, learning_rate=0.03,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        fold_xgb.fit(X_ft, y_ft, sample_weight=w_ft, verbose=False)

        fold_lgb = lgb.LGBMClassifier(
            n_estimators=lgb_iters, max_depth=6, learning_rate=0.03,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42, verbosity=-1,
        )
        fold_lgb.fit(X_ft, y_ft, sample_weight=w_ft)

        # Average the two models' predictions for OOF calibration
        oof_raw[fold_val_idx] = (
            fold_xgb.predict_proba(X_fv)[:, 1] +
            fold_lgb.predict_proba(X_fv)[:, 1]
        ) / 2.0

    # Fit Platt scaling on ensemble OOF predictions
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(oof_raw.reshape(-1, 1), oof_y)
    ensemble._platt_calibrator = platt

    # ── Evaluate on val set (raw ensemble, no Platt) ──
    raw_proba_val = ensemble.predict_proba(X_val)[:, 1]
    y_pred_proba = np.clip(raw_proba_val, 0.02, 0.98)

    has_both_classes = len(np.unique(y_val)) > 1
    metrics = {
        "brier_score": brier_score_loss(y_val, y_pred_proba) if has_both_classes else None,
        "log_loss": log_loss(y_val, y_pred_proba, labels=[0, 1]) if has_both_classes else None,
        "roc_auc": roc_auc_score(y_val, y_pred_proba) if has_both_classes else None,
        "val_size": len(y_val),
        "val_pos_rate": y_val.mean(),
        "pred_mean": y_pred_proba.mean(),
        "train_size": len(y_train),
        "features_used": used_features,
    }

    # Feature importance (average of both models)
    xgb_imp = xgb_model.feature_importances_
    lgb_imp = lgb_model.feature_importances_ / (lgb_model.feature_importances_.sum() or 1)
    xgb_imp_norm = xgb_imp / (xgb_imp.sum() or 1)
    avg_imp = (xgb_imp_norm + lgb_imp) / 2
    importance = dict(zip(used_features, avg_imp))

    brier_str = f"{metrics['brier_score']:.4f}" if metrics['brier_score'] is not None else "N/A"
    auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
    print(f"    Brier: {brier_str}, AUC: {auc_str}")

    return ensemble, metrics, importance


def train_all_models(feature_sets, train_seasons=None, val_seasons=None):
    """Train models for all defined markets.

    Args:
        feature_sets: dict from build_all_features()
        train_seasons: list of years for training (default: 2008-2025)
        val_seasons: list of years for validation (default: 2026)

    Returns: dict of {market_name: (model, metrics)}
    """
    if train_seasons is None:
        train_seasons = list(range(2008, 2026))
    if val_seasons is None:
        val_seasons = [2026]

    print("\n=== Training Models ===")
    results = {}
    all_metrics = {}

    for market_name, config in MARKETS.items():
        data_key = config["data_key"]
        if data_key not in feature_sets or feature_sets[data_key] is None:
            print(f"  Skipping {market_name}: no {data_key} data")
            continue

        df = feature_sets[data_key]
        if df.empty:
            print(f"  Skipping {market_name}: empty data")
            continue

        model, metrics, importance = train_market_model(
            df, market_name, config, train_seasons, val_seasons
        )

        if model is not None:
            # Save ensemble + calibrator
            model_path = MODELS_DIR / f"{market_name}.joblib"
            calibrator = getattr(model, "_platt_calibrator", getattr(model, "_iso_calibrator", None))
            joblib.dump({
                "model": model,
                "calibrator": calibrator,
                "calibrator_type": "platt",
                "is_ensemble": isinstance(model, EnsembleModel),
            }, model_path)
            print(f"    Saved: {model_path}")

            results[market_name] = (model, metrics)
            all_metrics[market_name] = metrics

    # Train batting regression model
    print("\n  --- Batting Regression Model ---")
    reg_model, reg_metrics = train_batting_regression(
        feature_sets, train_seasons, val_seasons
    )
    if reg_model is not None:
        results["batting_regression"] = (reg_model, reg_metrics)
        all_metrics["batting_regression"] = {
            k: v for k, v in reg_metrics.items() if k != "threshold_metrics"
        }
        # Add threshold metrics separately
        if "threshold_metrics" in reg_metrics:
            for t, m in reg_metrics["threshold_metrics"].items():
                all_metrics[f"batting_reg_runs_{t}"] = m

    # Save metrics summary
    metrics_path = MODELS_DIR / "metrics.json"
    # Convert non-serializable items
    serializable_metrics = {}
    for k, v in all_metrics.items():
        serializable_metrics[k] = {
            mk: (float(mv) if isinstance(mv, (np.floating, float)) else mv)
            for mk, mv in v.items()
        }
    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    print("=== Training complete ===\n")
    return results


def load_models():
    """Load all trained models from disk."""
    models = {}
    for market_name in MARKETS:
        path = MODELS_DIR / f"{market_name}.joblib"
        if path.exists():
            saved = joblib.load(path)
            if isinstance(saved, dict):
                model = saved["model"]
                calibrator = saved.get("calibrator")
                cal_type = saved.get("calibrator_type", "isotonic")
                if cal_type == "platt":
                    model._platt_calibrator = calibrator
                elif calibrator is not None:
                    model._iso_calibrator = calibrator
                models[market_name] = model
            else:
                models[market_name] = saved
    return models


def predict_single(model, features_dict, feature_cols):
    """Make a prediction for a single player/match.

    Args:
        model: trained calibrated model
        features_dict: dict of feature_name -> value
        feature_cols: list of feature column names the model expects

    Returns: predicted probability
    """
    X = pd.DataFrame([features_dict])
    available = [c for c in feature_cols if c in X.columns]
    missing = [c for c in feature_cols if c not in X.columns]
    for col in missing:
        X[col] = float("nan")  # NaN for missing — XGBoost/LightGBM handle natively

    X = X[feature_cols]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    raw_prob = model.predict_proba(X)[0, 1]
    return _calibrate_prob(model, raw_prob)


def _calibrate_prob(model, raw_prob):
    """Apply clipping to a raw probability.

    After fixing data leakage, the Platt calibrator was overcorrecting
    (compressing all probabilities toward 0). The raw ensemble output
    (XGBoost + LightGBM average) is already well-behaved, so we just clip
    extremes to prevent 0.0/1.0 outputs.
    """
    PROB_FLOOR = 0.02
    PROB_CEIL = 0.98
    return float(np.clip(raw_prob, PROB_FLOOR, PROB_CEIL))


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION-BASED BATTING MODEL
# Instead of separate binary classifiers per threshold, train one model that
# predicts the player's actual runs scored. Then derive P(runs >= threshold)
# from the predicted distribution using historical residuals.
# ══════════════════════════════════════════════════════════════════════════════


class BattingRegressionModel:
    """Predicts runs scored, then derives threshold probabilities from residuals."""

    def __init__(self, xgb_reg, lgb_reg, residuals):
        """
        Args:
            xgb_reg: trained XGBoost regressor
            lgb_reg: trained LightGBM regressor
            residuals: array of (actual - predicted) from training data,
                       used to build empirical error distribution
        """
        self.xgb_reg = xgb_reg
        self.lgb_reg = lgb_reg
        self.residuals = np.sort(residuals)
        self._feature_names = None

    def predict_runs(self, X):
        """Predict expected runs (average of XGB + LGB)."""
        xgb_pred = self.xgb_reg.predict(X)
        lgb_pred = self.lgb_reg.predict(X)
        return (xgb_pred + lgb_pred) / 2.0

    def predict_threshold_prob(self, X, threshold):
        """Predict P(runs >= threshold) using predicted mean + empirical residuals.

        For each prediction, we compute: what fraction of historical residuals
        would push the prediction above the threshold?
        P(runs >= T) = P(predicted + residual >= T) = P(residual >= T - predicted)
        """
        predicted = self.predict_runs(X)
        probs = np.zeros(len(predicted))
        for i, pred in enumerate(predicted):
            needed = threshold - pred  # residual needed to exceed threshold
            # What fraction of historical residuals exceed this?
            probs[i] = np.mean(self.residuals >= needed)
        return np.clip(probs, 0.02, 0.98)

    def get_booster(self):
        return self.xgb_reg.get_booster()

    @property
    def feature_importances_(self):
        xgb_imp = self.xgb_reg.feature_importances_
        lgb_imp = self.lgb_reg.feature_importances_
        xgb_norm = xgb_imp / (xgb_imp.sum() or 1)
        lgb_norm = lgb_imp / (lgb_imp.sum() or 1)
        return (xgb_norm + lgb_norm) / 2


def train_batting_regression(feature_sets, train_seasons=None, val_seasons=None):
    """Train a single regression model for batting runs prediction.

    Returns (BattingRegressionModel, metrics_dict)
    """
    if train_seasons is None:
        train_seasons = list(range(2008, 2026))
    if val_seasons is None:
        val_seasons = [2026]

    df = feature_sets.get("batting")
    if df is None or df.empty:
        return None, None

    feature_cols = BATTING_FEATURE_COLS
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return None, None

    # Target: actual runs scored
    mask = df["runs"].notna()
    subset = df[mask].copy()
    X = subset[available].copy()
    y = subset["runs"].values.astype(float)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Train/val split by season
    season = _get_season(subset)
    if season is not None:
        season_vals = pd.to_numeric(season, errors="coerce")
        train_mask = season_vals.isin(train_seasons) | season_vals.le(max(train_seasons))
        val_mask = season_vals.isin(val_seasons)
        if val_mask.sum() < 10:
            split_idx = int(len(X) * 0.8)
            train_mask = pd.Series([True] * split_idx + [False] * (len(X) - split_idx), index=X.index)
            val_mask = ~train_mask
    else:
        split_idx = int(len(X) * 0.8)
        train_mask = pd.Series([True] * split_idx + [False] * (len(X) - split_idx), index=X.index)
        val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask.values]
    X_val, y_val = X[val_mask], y[val_mask.values]

    sample_weights = None
    if "sample_weight" in subset.columns:
        sample_weights = subset["sample_weight"].values[train_mask.values]

    print(f"  batting_regression: train={len(X_train)}, val={len(X_val)}, "
          f"mean_runs={y_train.mean():.1f}")

    # Train XGBoost regressor
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="rmse", early_stopping_rounds=50,
        random_state=42, verbosity=0,
    )
    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                sample_weight=sample_weights, verbose=False)

    # Train LightGBM regressor
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1,
    )
    lgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    # Compute residuals on training data (for empirical distribution)
    train_preds = (xgb_reg.predict(X_train) + lgb_reg.predict(X_train)) / 2.0
    residuals = y_train - train_preds

    model = BattingRegressionModel(xgb_reg, lgb_reg, residuals)

    # Evaluate on val set
    val_preds = model.predict_runs(X_val)
    rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
    mae = np.mean(np.abs(y_val - val_preds))

    # Evaluate threshold predictions
    threshold_metrics = {}
    for threshold in [10, 15, 20, 30, 50]:
        pred_probs = model.predict_threshold_prob(X_val, threshold)
        actual = (y_val >= threshold).astype(int)
        brier = brier_score_loss(actual, pred_probs)
        auc = roc_auc_score(actual, pred_probs) if len(np.unique(actual)) > 1 else None
        threshold_metrics[threshold] = {"brier": brier, "auc": auc}

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "val_mean_runs": y_val.mean(),
        "pred_mean_runs": val_preds.mean(),
        "threshold_metrics": threshold_metrics,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "features_used": available,
    }

    print(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    for t, m in threshold_metrics.items():
        auc_str = f"{m['auc']:.4f}" if m['auc'] else "N/A"
        print(f"    runs_{t}+: Brier={m['brier']:.4f}, AUC={auc_str}")

    # Save
    model_path = MODELS_DIR / "batting_regression.joblib"
    joblib.dump({"model": model, "features": available}, model_path)
    print(f"    Saved: {model_path}")

    return model, metrics


def load_batting_regression():
    """Load the batting regression model."""
    path = MODELS_DIR / "batting_regression.joblib"
    if path.exists():
        saved = joblib.load(path)
        return saved["model"], saved["features"]
    return None, None


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from features.engineering import build_all_features
    from data import get_processed_data

    deliveries, matches = get_processed_data()
    feature_sets = build_all_features(deliveries)
    results = train_all_models(feature_sets)
