"""
Visualization helpers for the Mitgliedschafts-Voraussage model.

Creates:
1) Global feature importance (model-native if available; else permutation importance)
   - per transformed feature
   - aggregated back to original features (if ColumnTransformer used)
2) SHAP explanations: summary plot (+ single-example waterfall)
3) Confusion matrix at a sensible threshold (Youden's J by default)
4) ROC curve (with AUC)
5) Calibration (reliability) curve
6) Gains & Lift charts (by score decile)

Usage (after training and test split):
    from visualize_lead_model import make_all_viz
    make_all_viz(estimator=pipeline, X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test, figures_dir="figures")

Requirements:
    pip install matplotlib scikit-learn shap pandas numpy
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, RocCurveDisplay,
    precision_recall_curve, f1_score
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve, CalibrationDisplay

# Try SHAP import
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
    warnings.warn("SHAP not installed. SHAP plots will be skipped.")


# ----------------------------- Utilities ----------------------------- #

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _unwrap_estimator(est: Any) -> Any:
    """
    Unwraps common sklearn wrappers to get the underlying estimator:
    - Pipeline (returns final step)
    - GridSearchCV / RandomizedSearchCV (best_estimator_)
    - CalibratedClassifierCV (base_estimator)
    """
    # Grid/random search
    for attr in ("best_estimator_",):
        if hasattr(est, attr):
            return getattr(est, attr)

    # CalibratedClassifierCV
    if est.__class__.__name__ == "CalibratedClassifierCV" and hasattr(est, "base_estimator"):
        return est.base_estimator

    # Pipeline
    if isinstance(est, Pipeline):
        return _unwrap_estimator(est.steps[-1][1])

    return est


def _split_pipeline(estimator: Any) -> Tuple[Optional[Pipeline], Any]:
    """
    If `estimator` is a Pipeline, returns (preprocessor_pipeline, final_estimator).
    Otherwise returns (None, estimator).
    """
    if isinstance(estimator, Pipeline):
        if len(estimator.steps) == 1:
            return None, estimator.steps[-1][1]
        else:
            pre = Pipeline(estimator.steps[:-1])
            final = estimator.steps[-1][1]
            return pre, final
    return None, estimator


def _predict_scores(estimator: Any, X) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_score_for_metrics, y_prob_for_positive_class)
    - y_prob_for_positive_class is used when available; else None
    - y_score_for_metrics used for ROC/AUC; uses predict_proba or decision_function
    """
    # Use Pipeline uniformly
    model = estimator
    prob = None

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        # If binary classifier, take positive class column 1
        if prob.ndim == 2 and prob.shape[1] == 2:
            y_score = prob[:, 1]
            y_prob = y_score
        else:
            # Multiclass: not expected here; take max prob as a generic score
            y_score = prob.max(axis=1)
            y_prob = None
        return y_score, y_prob

    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
        # Normalize to 0..1 for display thresholds if we lack probabilities
        y_min, y_max = y_score.min(), y_score.max()
        if y_max > y_min:
            y_prob = (y_score - y_min) / (y_max - y_min)
        else:
            y_prob = None
        return y_score, y_prob

    # Fallback to predictions as scores (not ideal for ROC)
    y_pred = model.predict(X)
    y_score = y_pred.astype(float)
    y_prob = None
    return y_score, y_prob


def _auto_feature_names_from_ct(ct: ColumnTransformer, input_feature_names: Optional[List[str]]) -> List[str]:
    """
    Build transformed feature names from a ColumnTransformer (if possible).
    Falls back to generic names if unavailable.
    """
    names_out = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if trans == "drop":
            continue

        # Handle nested pipelines
        actual_trans = trans
        if isinstance(trans, Pipeline):
            actual_trans = trans.steps[-1][1]

        # Determine column list
        if isinstance(cols, slice):
            if input_feature_names is None:
                # Not enough info; make generic
                idxs = range(cols.start or 0, cols.stop or 0, cols.step or 1)
                base_cols = [f"col_{i}" for i in idxs]
            else:
                idxs = range(cols.start or 0, cols.stop or len(input_feature_names), cols.step or 1)
                base_cols = [input_feature_names[i] for i in idxs]
        elif isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            base_cols = list(cols)
        else:
            base_cols = [str(cols)]

        # Try to use estimator-provided names
        if hasattr(actual_trans, "get_feature_names_out"):
            try:
                if hasattr(actual_trans, "feature_names_in_"):
                    trans_names = list(actual_trans.get_feature_names_out())
                else:
                    trans_names = list(actual_trans.get_feature_names_out(base_cols))
                names_out.extend(trans_names)
                continue
            except Exception:
                pass

        # Fallback: just append base col names
        names_out.extend([str(c) for c in base_cols])

    return names_out


def _aggregate_importance_to_original(
    transformed_importances: pd.Series,
    preprocessor: Optional[Pipeline],
    X_sample
) -> pd.Series:
    """
    Aggregate importance from transformed features back to original features.
    Works when we can recover a ColumnTransformer and its mapping.
    """
    if preprocessor is None:
        # No preprocessor; transformed == original
        return transformed_importances

    ct = None
    # Find ColumnTransformer inside the preprocessor Pipeline if present
    if isinstance(preprocessor, Pipeline):
        for _, step in preprocessor.steps:
            if isinstance(step, ColumnTransformer):
                ct = step
                break
    elif isinstance(preprocessor, ColumnTransformer):
        ct = preprocessor

    if ct is None:
        # Nothing to aggregate
        return transformed_importances

    # Get original feature names (DataFrame columns if available)
    if hasattr(X_sample, "columns"):
        orig_names = list(X_sample.columns)
    else:
        orig_names = [f"col_{i}" for i in range(X_sample.shape[1])]

    # Build transformed names
    try:
        transformed_names = _auto_feature_names_from_ct(ct, orig_names)
    except Exception:
        # Fall back to index mapping
        transformed_names = [f"feat_{i}" for i in range(len(transformed_importances))]

    # Map transformed names to their source original column
    # Strategy: if OneHotEncoder generated "col=value" names, split at first separator.
    mapping: Dict[str, str] = {}
    for tname in transformed_names:
        base = tname
        # Common OHE patterns:
        if "=" in base:
            base = base.split("=", 1)[0]
        elif "__" in base and base.rsplit("__", 1)[-1].isdigit():
            base = base.rsplit("__", 1)[0]
        mapping[tname] = base

    # Sum importances per original feature
    agg = {}
    for tname, val in transformed_importances.items():
        base = mapping.get(tname, tname)
        agg[base] = agg.get(base, 0.0) + float(val)

    return pd.Series(agg).sort_values(ascending=False)


def _safe_get_feature_importances(
    estimator: Any,
    X_test,
    y_test,
    feature_names_transformed: Optional[List[str]],
    preprocessor: Optional[Pipeline],
    figures_dir: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (transformed_feature_importance, aggregated_original_feature_importance)
    Prefers model-native importances; falls back to permutation importance on the *pipeline*
    (so it works regardless of estimator type).
    """
    # Attempt model-native importances on final estimator w/ transformed X
    transformed_series = None
    pre, final = _split_pipeline(estimator)

    # Build transformed X for native importances / SHAP
    try:
        if pre is not None:
            X_pre = pre.transform(X_test)
        else:
            X_pre = X_test
    except Exception:
        # If pre.transform not yet fitted (rare), try fit_transform on a sample
        X_pre = X_test

    # 1) Tree-based feature_importances_
    if hasattr(_unwrap_estimator(final), "feature_importances_"):
        try:
            importances = np.asarray(_unwrap_estimator(final).feature_importances_).ravel()
            if feature_names_transformed is None:
                feature_names_transformed = [f"feat_{i}" for i in range(len(importances))]
            transformed_series = pd.Series(importances, index=feature_names_transformed).sort_values(ascending=False)
        except Exception:
            transformed_series = None

    # 2) Linear coef_
    if transformed_series is None and hasattr(_unwrap_estimator(final), "coef_"):
        try:
            coef = np.asarray(_unwrap_estimator(final).coef_)
            coef = coef[0] if coef.ndim > 1 else coef
            importance = np.abs(coef)
            if feature_names_transformed is None:
                feature_names_transformed = [f"feat_{i}" for i in range(len(importance))]
            transformed_series = pd.Series(importance, index=feature_names_transformed).sort_values(ascending=False)
        except Exception:
            transformed_series = None

    # 3) Fallback: permutation importance on full pipeline (original features)
    if transformed_series is None:
        print("Using permutation importance as a fallback (model-agnostic).")
        # Permutation on pipeline so we can pass original X_test (works with preprocessors)
        result = permutation_importance(estimator, X_test, y_test, n_repeats=7, random_state=42, scoring="roc_auc")
        if hasattr(X_test, "columns"):
            base_names = list(X_test.columns)
        else:
            base_names = [f"col_{i}" for i in range(X_test.shape[1])]
        # This is *original* feature importance (not transformed)
        orig_series = pd.Series(result.importances_mean, index=base_names).sort_values(ascending=False)
        # No transformed names to show; return orig twice
        return orig_series, orig_series

    # Aggregate transformed -> original
    agg_series = _aggregate_importance_to_original(transformed_series, pre, X_test)
    return transformed_series, agg_series


def _find_threshold(y_true: np.ndarray, y_score: np.ndarray, method: str = "youden") -> float:
    """Pick a decision threshold. Default uses Youden's J on ROC; fallback to F1 on PR; else 0.5."""
    # Try ROC-based threshold
    try:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    except Exception:
        pass

    # Try F1-based threshold
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        thresholds = np.concatenate([thresholds, [1.0]])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return float(thresholds[np.nanargmax(f1)])
    except Exception:
        pass

    return 0.5


# ----------------------------- Plotters ----------------------------- #

def plot_feature_importance(
    estimator: Any,
    X_test,
    y_test,
    feature_names: Optional[List[str]],
    figures_dir: str
) -> None:
    pre, final = _split_pipeline(estimator)

    # Transformed feature names
    feature_names_transformed = None
    if pre is not None:
        # Attempt to recover transformed names
        ct = None
        if isinstance(pre, Pipeline):
            for _, step in pre.steps:
                if isinstance(step, ColumnTransformer):
                    ct = step
                    break
        elif isinstance(pre, ColumnTransformer):
            ct = pre
        if ct is not None:
            try:
                # Use original feature names if DataFrame
                orig = list(X_test.columns) if hasattr(X_test, "columns") else None
                feature_names_transformed = _auto_feature_names_from_ct(ct, orig)
            except Exception:
                feature_names_transformed = None
    else:
        # No preprocessor
        if feature_names is not None:
            feature_names_transformed = feature_names
        elif hasattr(X_test, "columns"):
            feature_names_transformed = list(X_test.columns)

    transformed_imp, aggregated_imp = _safe_get_feature_importances(
        estimator, X_test, y_test, feature_names_transformed, pre, figures_dir
    )

    # Plot transformed feature importance (if distinct from aggregated)
    plt.figure(figsize=(10, 6))
    top_k = transformed_imp.head(30)[::-1]  # top 30 for readability
    plt.barh(top_k.index, top_k.values)
    plt.xlabel("Importance")
    plt.title("Feature Importance (transformed features)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "feature_importance_transformed.png"), dpi=180)
    plt.close()

    # Plot aggregated importance to original features
    plt.figure(figsize=(10, 6))
    top_k_agg = aggregated_imp.head(30)[::-1]
    plt.barh(top_k_agg.index, top_k_agg.values)
    plt.xlabel("Importance (aggregated)")
    plt.title("Feature Importance (aggregated to original features)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "feature_importance_aggregated.png"), dpi=180)
    plt.close()


def plot_shap(
    estimator: Any,
    X_train,
    X_test,
    figures_dir: str,
    shap_sample: int = 1000
) -> None:
    if not _HAS_SHAP:
        print("SHAP not installed; skipping SHAP plots.")
        return

    pre, final = _split_pipeline(estimator)
    base_est = _unwrap_estimator(final)

    # Build transformed matrices for SHAP on the *final* estimator
    if pre is not None:
        X_train_t = pre.transform(X_train)
        X_test_t = pre.transform(X_test)
        # Names for transformed features if possible
        feature_names_t = None
        try:
            ct = None
            if isinstance(pre, Pipeline):
                for _, step in pre.steps:
                    if isinstance(step, ColumnTransformer):
                        ct = step
                        break
            elif isinstance(pre, ColumnTransformer):
                ct = pre
            if ct is not None:
                orig = list(X_train.columns) if hasattr(X_train, "columns") else None
                feature_names_t = _auto_feature_names_from_ct(ct, orig)
        except Exception:
            feature_names_t = None
    else:
        X_train_t = X_train
        X_test_t = X_test
        feature_names_t = list(X_train.columns) if hasattr(X_train, "columns") else None

    # Subsample for speed
    if hasattr(X_test_t, "iloc"):
        idx = np.random.RandomState(42).choice(len(X_test_t), size=min(shap_sample, len(X_test_t)), replace=False)
        X_shap = X_test_t.iloc[idx]  # pandas
    else:
        idx = np.random.RandomState(42).choice(X_test_t.shape[0], size=min(shap_sample, X_test_t.shape[0]), replace=False)
        X_shap = X_test_t[idx]

    # Choose explainer
    explainer = None
    try:
        # Prefer TreeExplainer for tree models
        if base_est.__class__.__name__.lower().find("forest") >= 0 or \
           base_est.__class__.__name__.lower().find("boost") >= 0 or \
           base_est.__class__.__name__.lower().find("tree") >= 0 or \
           base_est.__class__.__name__.lower().find("xgb") >= 0 or \
           base_est.__class__.__name__.lower().find("lgbm") >= 0 or \
           base_est.__class__.__name__.lower().find("catboost") >= 0:
            explainer = shap.TreeExplainer(base_est)
        else:
            # Generic (model-agnostic) explainer
            explainer = shap.Explainer(base_est, X_train_t)
    except Exception:
        # Ultimate fallback: explain the pipeline directly (can be slow)
        explainer = shap.Explainer(estimator, X_train)

    # Compute SHAP values
    shap_values = explainer(X_shap)

    # Summary (beeswarm) plot
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=30)
    if feature_names_t is not None and hasattr(shap_values, "feature_names"):
        shap_values.feature_names = feature_names_t
    plt.title("SHAP Summary (top 30)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_summary.png"), dpi=180)
    plt.close()

    # One waterfall for the first example (class 1 if binary)
    try:
        # If binary with separate arrays, pick index 1 (positive class)
        if isinstance(shap_values.values, list) and len(shap_values.values) == 2:
            sv = shap_values[1][0]
            base = shap_values[1].base_values[0]
            data_row = X_shap.iloc[0] if hasattr(X_shap, "iloc") else X_shap[0]
            plt.figure()
            shap.plots.waterfall(shap.Explanation(values=sv, base_values=base,
                                                  data=data_row, feature_names=feature_names_t),
                                 show=False, max_display=20)
        else:
            # Unified object
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "shap_waterfall_idx0.png"), dpi=180)
        plt.close()
    except Exception as e:
        print(f"Waterfall plot skipped: {e}")


def plot_confusion_and_roc(
    estimator: Any,
    X_test,
    y_test,
    figures_dir: str
) -> None:
    y_score, y_prob = _predict_scores(estimator, X_test)

    # Choose threshold (Youden's J or fallback)
    thr = _find_threshold(y_test, y_score)
    y_pred = (y_score >= thr).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Join", "Join"])
    plt.figure(figsize=(5.5, 5))
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (threshold={thr:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=180)
    plt.close()

    # ROC Curve & AUC
    try:
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")
    fpr, tpr, _ = roc_curve(y_test, y_score)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Model").plot()
    plt.title(f"ROC Curve (AUC={auc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=180)
    plt.close()


def plot_calibration_and_gains(
    estimator: Any,
    X_test,
    y_test,
    figures_dir: str
) -> None:
    _, y_prob = _predict_scores(estimator, X_test)

    if y_prob is None:
        print("No predicted probabilities available; calibration/gains will use normalized scores.")
        y_score, _ = _predict_scores(estimator, X_test)
        # Rank-transform to pseudo-probabilities for visualization
        ranks = pd.Series(y_score).rank(method="average") / len(y_score)
        y_prob = ranks.values

    # Calibration curve
    CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=10, strategy="quantile")
    plt.title("Calibration (Reliability) Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "calibration_curve.png"), dpi=180)
    plt.close()

    # Gains & Lift charts (by deciles)
    df = pd.DataFrame({"y": y_test, "p": y_prob})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=False) + 1  # 1..10

    gains = df.groupby("decile")["y"].agg(["count", "sum"]).rename(columns={"sum": "positives"})
    gains["cum_positives"] = gains["positives"].cumsum()
    total_pos = df["y"].sum()
    total_cnt = len(df)
    gains["cum_pct_of_population"] = gains["count"].cumsum() / total_cnt
    gains["cum_pct_of_positives"] = gains["cum_positives"] / (total_pos if total_pos > 0 else 1)
    gains["lift"] = (gains["positives"] / gains["count"]) / (total_pos / total_cnt if total_pos > 0 else np.nan)

    # Plot cumulative gains
    plt.figure(figsize=(6.5, 5))
    plt.plot([0, 1], [0, 1])  # baseline
    plt.plot(gains["cum_pct_of_population"], gains["cum_pct_of_positives"])
    plt.xlabel("Cumulative % of Population (by score)")
    plt.ylabel("Cumulative % of Positives Captured")
    plt.title("Cumulative Gains Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "gains_and_lift.png"), dpi=180)
    plt.close()

    # Also print a small gains table to console
    print("\nTop-decile snapshot:")
    print(gains.head(3))
    print("\nFull gains table:\n", gains)


# ----------------------------- Orchestrator ----------------------------- #

def make_all_viz(
    estimator: ClassifierMixin,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: Optional[List[str]] = None,
    figures_dir: str = "figures",
    shap_sample: int = 1000
) -> None:
    """
    Generates all visualizations and saves them under `figures_dir`.
    """
    _ensure_dir(figures_dir)

    # 1) Feature importance (transformed + aggregated)
    print(">> Building feature importance plots …")
    plot_feature_importance(
        estimator=estimator,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        figures_dir=figures_dir
    )

    # 2) SHAP explanations (summary + 1 waterfall)
    print(">> Building SHAP plots …")
    try:
        plot_shap(
            estimator=estimator,
            X_train=X_train,
            X_test=X_test,
            figures_dir=figures_dir,
            shap_sample=shap_sample
        )
    except Exception as e:
        print(f"SHAP failed with error '{e}'. Skipping SHAP plots.")

    # 3) Confusion matrix + ROC
    print(">> Building confusion matrix and ROC …")
    plot_confusion_and_roc(
        estimator=estimator,
        X_test=X_test,
        y_test=y_test,
        figures_dir=figures_dir
    )

    # 4) Calibration + Gains/Lift
    print(">> Building calibration curve and gains/lift …")
    plot_calibration_and_gains(
        estimator=estimator,
        X_test=X_test,
        y_test=y_test,
        figures_dir=figures_dir
    )

    print(f"\nAll figures saved to: {os.path.abspath(figures_dir)}")
