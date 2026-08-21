"""Train the primary XGBoost probability-of-default model.

Run:
    python -m ml.train                    # full dataset
    CI_TRAIN_ROWS=50000 python -m ml.train  # fast demo run

Produces in `ml/artifacts/`:
    xgb_model.json          - the booster (portable, version-stable format)
    model_meta.json         - feature order, categories, training config
    metrics.json            - holdout + cross-validated metrics
    feature_importance.csv  - gain / weight / cover importance
    feature_baseline.json   - median feature vector, used for SHAP background
                              and for imputing missing API inputs
"""
from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import joblib
import xgboost as xgb

from ml import config as C
from ml.features import (
    BEHAVIOURAL_FEATURES,
    CATEGORICAL_RAW,
    MODEL_FEATURES,
    PROTECTED_ATTRIBUTES,
    build_model_frame,
    engineer_features,
)

MODEL_VERSION = "xgb-pd-1.0.0"

XGB_PARAMS: Dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",
    "max_depth": 6,
    "min_child_weight": 12,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "gamma": 0.2,
    "max_cat_to_onehot": 1,
    "random_state": C.RANDOM_STATE,
    "n_jobs": -1,
}
N_ESTIMATORS = 900
EARLY_STOPPING = 60
CV_FOLDS = 5

# Class imbalance handling.
# The policy layer thresholds a *calibrated* probability of default (approve
# below 6%, decline above 17%), so the model must emit real probabilities, not
# rank scores. `scale_pos_weight = neg/pos` would inflate every PD by ~11x and
# silently decline the whole book. We therefore keep the natural prior
# (scale_pos_weight = 1) and fix the remaining miscalibration with an isotonic
# calibrator fitted on a held-out validation split. Ranking power (ROC AUC) is
# unchanged by a monotone calibration; only the probability scale moves.
SCALE_POS_WEIGHT = 1.0


def load_dataset(nrows: int | None = None) -> pd.DataFrame:
    print(f"[load] reading {C.TRAIN_CSV}")
    df = pd.read_csv(C.TRAIN_CSV, nrows=nrows)
    print(f"[load] {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def prepare(df: pd.DataFrame, strict_fairness: bool = False
            ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    print("[features] engineering behavioural features ...")
    t0 = time.time()
    enriched = engineer_features(df)
    if strict_fairness:
        print("[features] strict fairness: dropping "
              f"{PROTECTED_ATTRIBUTES} from the model matrix")
    X = build_model_frame(enriched, strict_fairness=strict_fairness)
    y = df[C.TARGET].astype(int)
    print(f"[features] {X.shape[1]} model features in {time.time() - t0:.1f}s")
    print(f"[features] default rate = {y.mean():.4%}  |  NTC share = "
          f"{enriched['is_ntc'].mean():.2%}")
    return X, y, enriched


def _align_categories(X: pd.DataFrame, categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Force identical category dtypes so train/serve encodings never drift."""
    X = X.copy()
    for col, cats in categories.items():
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
    return X


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 5),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.linspace(0.02, 0.60, 59)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def decision_band_report(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, object]:
    """Observed default rate inside each policy band - the calibration proof.

    If the engine is honest, the realised default rate of everything it would
    auto-approve must sit at or below the approval threshold.
    """
    from ml.policy import PD_APPROVE_MAX, PD_REJECT_MIN

    bands = {
        "auto_approve": proba <= PD_APPROVE_MAX,
        "manual_review": (proba > PD_APPROVE_MAX) & (proba < PD_REJECT_MIN),
        "decline": proba >= PD_REJECT_MIN,
    }
    out = {}
    for name, mask in bands.items():
        n = int(mask.sum())
        out[name] = {
            "n": n,
            "share": round(n / len(proba), 4),
            "observed_default_rate": round(float(y_true[mask].mean()), 4) if n else None,
            "mean_predicted_pd": round(float(proba[mask].mean()), 4) if n else None,
        }
    return out


def cross_validate(X: pd.DataFrame, y: pd.Series, spw: float) -> Dict[str, object]:
    print(f"[cv] {CV_FOLDS}-fold stratified cross-validation ...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    aucs, prs = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        clf = xgb.XGBClassifier(
            **XGB_PARAMS, n_estimators=N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING,
            scale_pos_weight=spw, enable_categorical=True,
        )
        clf.fit(X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        p = clf.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y.iloc[va], p)
        pr = average_precision_score(y.iloc[va], p)
        aucs.append(auc)
        prs.append(pr)
        print(f"[cv]   fold {fold}: ROC AUC = {auc:.4f}  PR AUC = {pr:.4f}")
    return {
        "folds": CV_FOLDS,
        "roc_auc_mean": round(float(np.mean(aucs)), 4),
        "roc_auc_std": round(float(np.std(aucs)), 4),
        "pr_auc_mean": round(float(np.mean(prs)), 4),
        "per_fold_roc_auc": [round(float(a), 4) for a in aucs],
    }


def main(run_cv: bool = True, strict_fairness: bool = False) -> None:
    df = load_dataset(C.TRAIN_ROWS)
    X, y, enriched = prepare(df, strict_fairness=strict_fairness)

    categories = {c: sorted(X[c].astype(str).unique().tolist())
                  for c in CATEGORICAL_RAW if c in X.columns}
    X = _align_categories(X, categories)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=C.RANDOM_STATE
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=C.RANDOM_STATE
    )
    print(f"[split] train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}")

    spw = SCALE_POS_WEIGHT
    imbalance = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"[train] class imbalance = {imbalance:.2f}:1, "
          f"scale_pos_weight = {spw:.2f} (calibration handles the prior)")

    clf = xgb.XGBClassifier(
        **XGB_PARAMS, n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING,
        scale_pos_weight=spw, enable_categorical=True,
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
    print(f"[train] done in {time.time() - t0:.1f}s "
          f"(best_iteration={clf.best_iteration})")

    # --- probability calibration -------------------------------------------
    # Isotonic regression on the validation split maps raw scores onto observed
    # default frequencies, so `PD = 0.06` genuinely means "6 in 100 default".
    raw_val = clf.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_val, y_val.values)
    joblib.dump(calibrator, C.CALIBRATOR_PATH)

    raw_te = clf.predict_proba(X_te)[:, 1]
    proba_te = np.clip(calibrator.predict(raw_te), 1e-6, 1 - 1e-6)
    print(f"[calib] mean PD before={raw_te.mean():.4f} after={proba_te.mean():.4f} "
          f"| observed default rate={y_te.mean():.4f}")
    print(f"[calib] Brier before={brier_score_loss(y_te, raw_te):.5f} "
          f"after={brier_score_loss(y_te, proba_te):.5f}")

    thr = best_f1_threshold(y_te.values, proba_te)
    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(X.shape[1]),
        "base_default_rate": round(float(y.mean()), 5),
        "holdout_at_0.5": evaluate(y_te.values, proba_te, 0.5),
        "holdout_at_best_f1": evaluate(y_te.values, proba_te, thr),
        "holdout_at_policy_cutoff": evaluate(y_te.values, proba_te, 0.17),
        "best_iteration": int(clf.best_iteration or N_ESTIMATORS),
        "calibration": {
            "method": "isotonic",
            "mean_pd_raw": round(float(raw_te.mean()), 5),
            "mean_pd_calibrated": round(float(proba_te.mean()), 5),
            "observed_default_rate": round(float(y_te.mean()), 5),
            "brier_raw": round(float(brier_score_loss(y_te, raw_te)), 5),
            "brier_calibrated": round(float(brier_score_loss(y_te, proba_te)), 5),
        },
        "decision_bands": decision_band_report(y_te.values, proba_te),
    }
    print("\n[eval] holdout @ best-F1 threshold "
          f"{thr:.3f}: {json.dumps(metrics['holdout_at_best_f1'], indent=2)}")
    print(classification_report(y_te, (proba_te >= thr).astype(int),
                                target_names=["non-default", "default"], digits=4))

    if run_cv:
        metrics["cross_validation"] = cross_validate(X, y, spw)

    # --- feature importance -------------------------------------------------
    booster = clf.get_booster()
    imp_rows = []
    for kind in ("gain", "weight", "cover"):
        score = booster.get_score(importance_type=kind)
        for feat, val in score.items():
            imp_rows.append({"feature": feat, "kind": kind, "value": val})
    imp = (pd.DataFrame(imp_rows)
           .pivot_table(index="feature", columns="kind", values="value", fill_value=0.0)
           .reset_index())
    if "gain" in imp.columns:
        imp = imp.sort_values("gain", ascending=False)
        imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(3)
    imp.to_csv(C.FEATURE_IMPORTANCE_PATH, index=False)
    print(f"\n[importance] top 15 by gain:\n{imp.head(15).to_string(index=False)}")

    behavioural_gain = float(
        imp.loc[imp["feature"].isin(BEHAVIOURAL_FEATURES), "gain_pct"].sum()
    ) if "gain_pct" in imp.columns else 0.0
    metrics["behavioural_feature_gain_pct"] = round(behavioural_gain, 2)
    print(f"[importance] behavioural block contributes "
          f"{behavioural_gain:.1f}% of total gain")

    # --- persist ------------------------------------------------------------
    booster.save_model(str(C.MODEL_PATH))

    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_RAW]
    present_categoricals = [c for c in CATEGORICAL_RAW if c in X.columns]
    baseline = {
        "median": {c: float(pd.to_numeric(X[c], errors="coerce").median())
                   for c in numeric_cols},
        "mode": {c: str(X[c].astype(str).mode().iloc[0]) for c in present_categoricals},
    }
    C.BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    meta = {
        "model_version": MODEL_VERSION,
        "calibrated": True,
        "feature_order": list(X.columns),
        "categorical_features": present_categoricals,
        "categories": categories,
        "behavioural_features": BEHAVIOURAL_FEATURES,
        "params": {**XGB_PARAMS, "n_estimators": N_ESTIMATORS,
                   "scale_pos_weight": round(spw, 4)},
        "best_f1_threshold": round(thr, 4),
        "strict_fairness": strict_fairness,
    }
    (C.ARTIFACT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2),
                                                    encoding="utf-8")
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # A SHAP background sample and a scored slice used by the analytics endpoint.
    # Optional extras: they need pyarrow, and a missing plotting sample must never
    # invalidate a training run whose model and metrics are already on disk.
    try:
        bg = X_tr.sample(min(500, len(X_tr)), random_state=C.RANDOM_STATE)
        bg.to_parquet(C.ARTIFACT_DIR / "shap_background.parquet")

        scored = enriched.loc[X_te.index, ["SK_ID_CURR"] if "SK_ID_CURR" in enriched.columns else []]
        scored = scored.assign(pd=proba_te, target=y_te.values)
        scored.to_parquet(C.ARTIFACT_DIR / "holdout_scored.parquet")
    except Exception as exc:  # noqa: BLE001
        print(f"[save] optional parquet artefacts skipped ({exc})")

    print(f"\n[save] model      -> {C.MODEL_PATH}")
    print(f"[save] meta       -> {C.ARTIFACT_DIR / 'model_meta.json'}")
    print(f"[save] metrics    -> {C.METRICS_PATH}")
    print(f"[save] importance -> {C.FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    ap.add_argument("--strict-fairness", action="store_true",
                    help="drop protected attributes (gender, age, family status) "
                         "from the model matrix")
    args = ap.parse_args()
    main(run_cv=not args.no_cv, strict_fairness=args.strict_fairness)
