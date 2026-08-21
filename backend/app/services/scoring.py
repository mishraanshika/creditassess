"""Scoring orchestration: features -> model -> policy -> SHAP -> peers.

This is the single code path every decision flows through, which is what makes
the audit trail trustworthy: `/predict` and `/underwriting-report` cannot drift
apart because the report is generated from the object this service returns.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from app.services.similar import similar_service
from ml.explain import explain_prediction
from ml.inference import CreditModel, ModelNotTrained, get_model
from ml.policy import Decision, decide

logger = logging.getLogger(__name__)


class ScoringService:
    def __init__(self) -> None:
        self.model: Optional[CreditModel] = None
        self.error: str = ""

    def initialise(self) -> None:
        try:
            self.model = get_model()
            logger.info("model %s loaded (%d features)",
                        self.model.model_version, len(self.model.feature_order))
        except ModelNotTrained as exc:
            self.error = str(exc)
            logger.error(self.error)
        except Exception as exc:  # noqa: BLE001
            self.error = f"model load failed: {exc}"
            logger.exception(self.error)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def assess(
        self,
        payload: Dict[str, Any],
        top_k: int = 5,
        include_explanation: bool = True,
        include_similar: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full assessment pipeline for one applicant."""
        if self.model is None:
            raise RuntimeError(self.error or "model not loaded")

        t0 = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())

        # 1. features + probability of default
        X, enriched = self.model.frame_from_payload(payload)
        pd_value = float(self.model.predict_proba(X)[0])
        features = {k: v for k, v in enriched.iloc[0].to_dict().items()}
        features = _jsonable(features)

        # 2. peer retrieval first - the cohort's agreement feeds the confidence score
        similar: Dict[str, Any] = {"similar_borrowers": [], "cohort": {"cohort_size": 0},
                                   "query_profile": similar_service.profile_text(enriched),
                                   "backend": similar_service.backend,
                                   "encoder": getattr(similar_service.encoder, "name", "n/a")}
        if include_similar:
            similar = similar_service.query(enriched, top_k)
        agreement = (similar.get("cohort") or {}).get("agreement")

        # 3. policy decision
        decision: Decision = decide(
            pd_value,
            features,
            neighbour_agreement=agreement,
            requested_amount=float(features.get("AMT_CREDIT") or 0),
        )
        decision_dict = decision.to_dict()
        decision_dict["model_version"] = self.model.model_version

        # 4. SHAP explanation
        explanation: Optional[Dict[str, Any]] = None
        if include_explanation:
            explanation = explain_prediction(self.model, X, enriched)

        behavioural = {
            k: float(features.get(k) or 0.0)
            for k in (
                "payment_consistency_score", "spending_stability_score", "income_stability_score",
                "credit_utilization_score", "digital_trust_score", "financial_discipline_score",
                "transaction_volatility", "monthly_cashflow_consistency",
                "utility_payment_consistency", "mobile_recharge_consistency", "thin_file_score",
            )
        }

        return {
            "request_id": request_id,
            "decision": decision_dict,
            "features": features,
            "behavioural_features": behavioural,
            "explanation": explanation,
            "similar": similar,
            "model_version": self.model.model_version,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            f = float(v)
            out[k] = None if f != f else round(f, 6)  # NaN-safe
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        else:
            out[k] = str(v)
    return out


scoring_service = ScoringService()
