"""Model registry + scoring runtime shared by the API, batch jobs and notebooks.

Loads the trained booster once per process, guarantees that the serving feature
order and category encodings are byte-identical to training, and returns both
the probability of default and the engineered feature vector (the latter feeds
the policy layer, the SHAP layer and the LLM prompt).
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from ml import config as C
from ml.features import (
    BEHAVIOURAL_FEATURES,
    CATEGORICAL_RAW,
    applicant_to_frame,
    build_model_frame,
    engineer_features,
)

_LOCK = threading.Lock()


class ModelNotTrained(RuntimeError):
    """Raised when the API starts before `python -m ml.train` has been run."""


class CreditModel:
    """Thread-safe wrapper around the trained booster."""

    def __init__(self) -> None:
        if not C.MODEL_PATH.exists():
            raise ModelNotTrained(
                f"No model at {C.MODEL_PATH}. Run `python -m ml.train` first."
            )
        meta_path = C.ARTIFACT_DIR / "model_meta.json"
        self.meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_order: List[str] = self.meta["feature_order"]
        self.categories: Dict[str, List[str]] = self.meta["categories"]
        self.model_version: str = self.meta.get("model_version", "unknown")

        self.booster = xgb.Booster()
        self.booster.load_model(str(C.MODEL_PATH))

        # Isotonic calibrator: turns the booster's raw score into a probability
        # that matches observed default frequencies, which is what the policy
        # thresholds are defined against. Absent -> raw scores are used as-is.
        self.calibrator = None
        if C.CALIBRATOR_PATH.exists():
            try:
                import joblib

                self.calibrator = joblib.load(C.CALIBRATOR_PATH)
            except Exception:  # noqa: BLE001
                self.calibrator = None

        self.baseline: Dict[str, Any] = (
            json.loads(C.BASELINE_PATH.read_text(encoding="utf-8"))
            if C.BASELINE_PATH.exists() else {"median": {}, "mode": {}}
        )
        self.metrics: Dict[str, Any] = (
            json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
            if C.METRICS_PATH.exists() else {}
        )
        try:
            self.feature_importance = pd.read_csv(C.FEATURE_IMPORTANCE_PATH)
        except Exception:
            self.feature_importance = pd.DataFrame(columns=["feature", "gain"])

    # -- frame construction -------------------------------------------------
    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.feature_order)
        for col, cats in self.categories.items():
            X[col] = pd.Categorical(X[col].astype(str), categories=cats)
        for col in X.columns:
            if col not in CATEGORICAL_RAW:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                median = self.baseline["median"].get(col)
                if median is not None and col not in ("EXT_SOURCE_1", "EXT_SOURCE_2",
                                                      "EXT_SOURCE_3", "ext_source_mean",
                                                      "ext_source_min"):
                    # EXT_SOURCE_* stay NaN on purpose: "no bureau score" is signal,
                    # and XGBoost learns a dedicated default split direction for it.
                    X[col] = X[col].fillna(median)
        return X

    def frame_from_payload(self, payload: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (aligned model matrix, enriched raw frame with all features)."""
        raw = applicant_to_frame(payload)
        enriched = engineer_features(raw)
        X = self._align(build_model_frame(enriched))
        return X, enriched

    def frame_from_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        enriched = engineer_features(df)
        X = self._align(build_model_frame(enriched))
        return X, enriched

    # -- scoring ------------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated booster output (the space SHAP contributions live in)."""
        dm = xgb.DMatrix(X, enable_categorical=True)
        with _LOCK:
            return self.booster.predict(dm)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default."""
        raw = self.predict_raw(X)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 1e-6, 1 - 1e-6)

    def shap_contributions(self, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Exact TreeSHAP contributions in log-odds space.

        Uses XGBoost's built-in `pred_contribs`, which is the same TreeSHAP
        algorithm the `shap` package calls but without the extra dependency at
        request time.  Returns (contribs[n, f], bias) where the last column of
        the raw output is the expected value.
        """
        dm = xgb.DMatrix(X, enable_categorical=True)
        with _LOCK:
            contribs = self.booster.predict(dm, pred_contribs=True)
        return contribs[:, :-1], float(contribs[0, -1])

    def score_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One-stop scoring: PD + engineered features + behavioural snapshot."""
        X, enriched = self.frame_from_payload(payload)
        pd_value = float(self.predict_proba(X)[0])
        feat = feature_dict(enriched, X)
        return {
            "probability_of_default": pd_value,
            "features": feat,
            "behavioural": {k: float(feat.get(k, 0.0)) for k in BEHAVIOURAL_FEATURES},
            "model_version": self.model_version,
            "X": X,
        }


def feature_dict(enriched: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Flatten a one-row enriched frame into a JSON-safe dict."""
    row = enriched.iloc[0].to_dict()
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 6)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = str(v)
    return out


@lru_cache(maxsize=1)
def get_model() -> CreditModel:
    """Process-wide singleton."""
    return CreditModel()
