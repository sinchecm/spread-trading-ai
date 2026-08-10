"""Time-series-aware training and comparison of several ML models across the
three candidate pair-trading targets, with a final unbiased out-of-sample
evaluation of the selected model."""
import json

import joblib
import numpy as np
import pandas as pd
from crewai.tools import tool
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pair_crew import config

TARGETS = ["target_reversion", "target_direction", "target_regime"]

# RBF-kernel SVM training cost is O(n^2)-O(n^3); on the full ~200k-row training
# window a single fit can run for hours. Cap its training rows so it stays
# tractable while still being evaluated against the other models.
SVM_MAX_TRAIN_ROWS = 15000


def _model_zoo():
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=10, random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]),
    }


def _fit(model, name: str, X: pd.DataFrame, y: pd.Series):
    if name == "svm_rbf" and len(X) > SVM_MAX_TRAIN_ROWS:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(len(X), SVM_MAX_TRAIN_ROWS, replace=False))
        X, y = X.iloc[idx], y.iloc[idx]
    model.fit(X, y)
    return model


def _score_of(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def _embargoed_splits(n_rows: int, n_splits: int, embargo: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dummy = np.zeros(n_rows)
    for train_idx, test_idx in tscv.split(dummy):
        if len(train_idx) > embargo:
            train_idx = train_idx[: -embargo]
        if len(train_idx) < 30 or len(test_idx) < 10:
            continue
        yield train_idx, test_idx


def _cv_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    results = {}
    for name, base_model in _model_zoo().items():
        fold_auc, fold_acc, fold_f1 = [], [], []
        for train_idx, test_idx in _embargoed_splits(len(X), config.N_CV_SPLITS, config.EMBARGO_BARS):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if y_tr.nunique() < 2:
                continue
            model = clone(base_model)
            _fit(model, name, X_tr, y_tr)
            scores = _score_of(model, X_te)
            preds = model.predict(X_te)
            try:
                auc = roc_auc_score(y_te, scores) if y_te.nunique() > 1 else np.nan
            except ValueError:
                auc = np.nan
            fold_auc.append(auc)
            fold_acc.append(accuracy_score(y_te, preds))
            fold_f1.append(f1_score(y_te, preds, zero_division=0))

        fold_auc_arr = np.array([a for a in fold_auc if not np.isnan(a)])
        mean_auc = float(np.mean(fold_auc_arr)) if len(fold_auc_arr) else float("nan")
        std_auc = float(np.std(fold_auc_arr)) if len(fold_auc_arr) else float("nan")
        results[name] = {
            "n_folds_used": len(fold_auc),
            "cv_auc_mean": round(mean_auc, 4) if not np.isnan(mean_auc) else None,
            "cv_auc_std": round(std_auc, 4) if not np.isnan(std_auc) else None,
            "cv_accuracy_mean": round(float(np.mean(fold_acc)), 4) if fold_acc else None,
            "cv_f1_mean": round(float(np.mean(fold_f1)), 4) if fold_f1 else None,
            "robust_score": round(mean_auc - std_auc, 4) if not np.isnan(mean_auc) else -999,
        }
    return results


@tool("train_and_compare_models")
def train_and_compare_models() -> str:
    """
    Train and compare several ML models (majority-class baseline, logistic
    regression, random forest, gradient boosting, SVM) to predict each of
    the three pair-trading targets (spread reversion, spread direction,
    spread volatility regime) engineered by engineer_pair_features, reading
    outputs/features.parquet.

    Uses time-series-aware, embargoed walk-forward cross-validation
    (expanding-window TimeSeriesSplit with a purge gap so that forward-
    looking labels never leak across the train/validation boundary) on the
    training portion of history (first ~65% chronologically). Model quality
    per fold is measured by ROC-AUC, accuracy, and F1; models are ranked by
    a robustness-adjusted score (mean AUC minus AUC std across folds) so an
    unstable model that looks good in only one fold is not preferred.

    The single best (target, model) combination overall is refit on the
    full training window and evaluated exactly once on the untouched
    out-of-sample test period (the remaining ~35% of history, never seen
    during model selection) to report unbiased OOS performance. Saves the
    fitted model to outputs/best_model.joblib, its metadata to
    outputs/best_model_meta.json, and its OOS predictions to
    outputs/predictions.parquet (for strategy construction/backtesting).

    Returns a JSON string with the full per-target model comparison tables,
    the selected best model and target, and its true out-of-sample metrics.
    """
    df = pd.read_parquet(config.FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in TARGETS + ["MHI_Close", "HHI_Close"]]

    n_train = int(len(df) * config.TRAIN_FRACTION)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    comparison = {}
    best_overall = {"robust_score": -999}
    for target in TARGETS:
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        target_results = _cv_evaluate(X_train, y_train)
        comparison[target] = target_results
        for model_name, metrics in target_results.items():
            if model_name == "baseline_majority":
                continue
            if metrics["robust_score"] is not None and metrics["robust_score"] > best_overall["robust_score"]:
                best_overall = {
                    "target": target,
                    "model_name": model_name,
                    "robust_score": metrics["robust_score"],
                    **metrics,
                }

    # refit winner on full training window, evaluate once on true OOS test set
    target = best_overall["target"]
    model_name = best_overall["model_name"]
    final_model = _model_zoo()[model_name]
    X_train_full = train_df[feature_cols]
    y_train_full = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    _fit(final_model, model_name, X_train_full, y_train_full)
    oos_scores = _score_of(final_model, X_test)
    oos_preds = final_model.predict(X_test)

    oos_metrics = {
        "n_oos_bars": len(test_df),
        "oos_auc": round(float(roc_auc_score(y_test, oos_scores)), 4) if y_test.nunique() > 1 else None,
        "oos_accuracy": round(float(accuracy_score(y_test, oos_preds)), 4),
        "oos_f1": round(float(f1_score(y_test, oos_preds, zero_division=0)), 4),
        "baseline_accuracy_majority_class": round(float(max(y_test.mean(), 1 - y_test.mean())), 4),
    }

    feature_importance = None
    fitted_clf = final_model.named_steps["clf"] if isinstance(final_model, Pipeline) else final_model
    if hasattr(fitted_clf, "feature_importances_"):
        feature_importance = {
            f: round(float(v), 4)
            for f, v in sorted(
                zip(feature_cols, fitted_clf.feature_importances_), key=lambda t: -t[1]
            )
        }
    elif hasattr(fitted_clf, "coef_"):
        feature_importance = {
            f: round(float(v), 4)
            for f, v in sorted(
                zip(feature_cols, fitted_clf.coef_[0]), key=lambda t: -abs(t[1])
            )
        }

    joblib.dump(final_model, config.BEST_MODEL_PATH)
    meta = {
        "target": target,
        "model_name": model_name,
        "feature_columns": feature_cols,
        "train_rows": len(train_df),
        "oos_metrics": oos_metrics,
        "feature_importance": feature_importance,
    }
    config.BEST_MODEL_META_PATH.write_text(json.dumps(meta, indent=2))

    preds_df = test_df[["MHI_Close", "HHI_Close", "spread", "spread_zscore"]].copy()
    preds_df["ml_score"] = oos_scores
    preds_df["ml_pred"] = oos_preds
    preds_df["target_actual"] = y_test
    preds_df.to_parquet(config.PREDICTIONS_PATH)

    result = {
        "model_comparison_by_target": comparison,
        "selected_target": target,
        "selected_model": model_name,
        "selection_rationale": (
            f"'{model_name}' predicting '{target}' had the highest robustness-adjusted "
            "CV score (mean AUC minus AUC std across embargoed walk-forward folds) among "
            "all non-baseline models across all three candidate targets."
        ),
        "cv_metrics_of_selected_model": {k: v for k, v in best_overall.items() if k not in ("target", "model_name")},
        "out_of_sample_test_metrics": oos_metrics,
        "top_feature_importance": dict(list((feature_importance or {}).items())[:8]),
        "artifacts": {
            "model_path": str(config.BEST_MODEL_PATH),
            "meta_path": str(config.BEST_MODEL_META_PATH),
            "oos_predictions_path": str(config.PREDICTIONS_PATH),
        },
    }
    config.MODEL_COMPARISON_PATH.write_text(json.dumps(result, indent=2))
    return json.dumps(result, indent=2)
