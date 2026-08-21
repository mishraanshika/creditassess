"""Persistence + audit trail.

Responsible-AI requirement: every decision must be reconstructible. We persist
the applicant, the decision, the frozen behavioural snapshot, the SHAP drivers,
the peer cohort and the model/policy versions - plus an append-only audit log
row for every API call.

If Postgres is not reachable the same records go to the in-memory ledger so the
demo, the dashboard and the audit view all keep working.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.db import DB_AVAILABLE, ledger, session_scope
from app.models import Applicant, AuditLog, Prediction, UnderwritingReport

logger = logging.getLogger(__name__)


def payload_hash(payload: Dict[str, Any]) -> str:
    """Stable sha256 of the request body - lets an auditor prove what was sent
    without storing raw PII in the log table."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_decision(
    request_id: str,
    raw_payload: Dict[str, Any],
    result: Dict[str, Any],
    external_ref: Optional[str] = None,
    full_name: Optional[str] = None,
) -> Optional[str]:
    """Persist applicant + prediction. Returns the prediction id when available."""
    decision = result["decision"]
    features = result["features"]
    explanation = result.get("explanation") or {}
    similar = result.get("similar") or {}
    cohort = similar.get("cohort") or {}

    prediction_id = str(uuid.uuid4())
    row = {
        "id": prediction_id,
        "request_id": request_id,
        "created_at": _now(),
        "external_ref": external_ref,
        "full_name": full_name,
        "probability_of_default": decision["probability_of_default"],
        "risk_score": decision["risk_score"],
        "risk_band": decision["risk_band"],
        "risk_tier": decision["risk_tier"],
        "recommendation": decision["recommendation"],
        "recommended_credit_limit": decision["recommended_credit_limit"],
        "max_affordable_limit": decision["max_affordable_limit"],
        "confidence_score": decision["confidence_score"],
        "requires_human_review": decision["requires_human_review"],
        "review_reasons": decision["review_reasons"],
        "fraud_flags": decision["fraud_flags"],
        "is_ntc": decision["is_ntc"],
        "behavioural_features": result["behavioural_features"],
        "cohort_repayment_rate": cohort.get("repayment_success_rate"),
        "model_version": result["model_version"],
        "policy_version": decision["policy_version"],
        "latency_ms": result["latency_ms"],
        "income_total": features.get("AMT_INCOME_TOTAL"),
        "credit_amount": features.get("AMT_CREDIT"),
    }

    if not DB_AVAILABLE:
        ledger.add_prediction(row)
        return prediction_id

    try:
        with session_scope() as session:
            if session is None:
                ledger.add_prediction(row)
                return prediction_id

            applicant = None
            if external_ref:
                applicant = (session.query(Applicant)
                             .filter(Applicant.external_ref == external_ref).one_or_none())
            if applicant is None:
                applicant = Applicant(
                    external_ref=external_ref or f"APP-{request_id[:8]}",
                    full_name=full_name,
                    contract_type=features.get("NAME_CONTRACT_TYPE"),
                    gender=features.get("CODE_GENDER"),
                    age_years=features.get("age_years"),
                    income_total=features.get("AMT_INCOME_TOTAL") or 0,
                    credit_amount=features.get("AMT_CREDIT") or 0,
                    annuity_amount=features.get("AMT_ANNUITY"),
                    goods_price=features.get("AMT_GOODS_PRICE"),
                    employment_years=features.get("employment_years"),
                    occupation_type=features.get("OCCUPATION_TYPE"),
                    organization_type=features.get("ORGANIZATION_TYPE"),
                    education_type=features.get("NAME_EDUCATION_TYPE"),
                    family_status=features.get("NAME_FAMILY_STATUS"),
                    housing_type=features.get("NAME_HOUSING_TYPE"),
                    children_count=int(features.get("CNT_CHILDREN") or 0),
                    family_members=features.get("CNT_FAM_MEMBERS"),
                    is_ntc=bool(decision["is_ntc"]),
                    raw_payload=raw_payload,
                )
                session.add(applicant)
                session.flush()

            prediction = Prediction(
                id=uuid.UUID(prediction_id),
                applicant_id=applicant.id,
                request_id=request_id,
                probability_of_default=decision["probability_of_default"],
                risk_score=decision["risk_score"],
                risk_band=decision["risk_band"],
                risk_tier=decision["risk_tier"],
                recommendation=decision["recommendation"],
                recommended_credit_limit=decision["recommended_credit_limit"],
                max_affordable_limit=decision["max_affordable_limit"],
                suggested_term_months=decision["suggested_term_months"],
                confidence_score=decision["confidence_score"],
                confidence_drivers=decision["confidence_drivers"],
                requires_human_review=decision["requires_human_review"],
                review_reasons=decision["review_reasons"],
                fraud_flags=decision["fraud_flags"],
                behavioural_features=result["behavioural_features"],
                shap_top_positive=explanation.get("top_positive_factors", []),
                shap_top_negative=explanation.get("top_negative_factors", []),
                shap_base_value=explanation.get("base_value_logodds"),
                similar_borrower_ids=[int(b["borrower_id"])
                                      for b in similar.get("similar_borrowers", [])],
                cohort_repayment_rate=cohort.get("repayment_success_rate"),
                cohort_mean_similarity=cohort.get("mean_similarity"),
                model_version=result["model_version"],
                policy_version=decision["policy_version"],
                latency_ms=result["latency_ms"],
            )
            session.add(prediction)
        return prediction_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("decision persistence failed (%s); using ledger", exc)
        ledger.add_prediction(row)
        return prediction_id


def record_report(prediction_id: Optional[str], report: Dict[str, Any],
                  decision: Dict[str, Any]) -> None:
    row = {
        "prediction_id": prediction_id,
        "created_at": _now(),
        "generator": report.get("generator"),
        "recommendation": decision["recommendation"],
        "suggested_limit": decision["recommended_credit_limit"],
        "executive_summary": report.get("executive_summary"),
    }
    if not DB_AVAILABLE:
        ledger.add_report(row)
        return
    try:
        with session_scope() as session:
            if session is None:
                ledger.add_report(row)
                return
            session.add(UnderwritingReport(
                prediction_id=uuid.UUID(prediction_id) if prediction_id else None,
                recommendation=decision["recommendation"],
                suggested_limit=decision["recommended_credit_limit"],
                strengths=report.get("strengths", []),
                risk_factors=report.get("risk_factors", []),
                conditions=report.get("conditions", []),
                explanation=report.get("detailed_explanation"),
                executive_summary=report.get("executive_summary"),
                generator=report.get("generator", "template"),
                prompt_version=report.get("prompt_version"),
                prompt_tokens=report.get("prompt_tokens"),
                completion_tokens=report.get("completion_tokens"),
            ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report persistence failed (%s); using ledger", exc)
        ledger.add_report(row)


def log_event(
    request_id: str,
    event_type: str,
    endpoint: str,
    http_status: int = 200,
    latency_ms: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    prediction_id: Optional[str] = None,
    model_version: Optional[str] = None,
    policy_version: Optional[str] = None,
    actor: str = "system",
) -> None:
    row = {
        "request_id": request_id,
        "event_type": event_type,
        "endpoint": endpoint,
        "http_status": http_status,
        "latency_ms": latency_ms,
        "actor": actor,
        "model_version": model_version,
        "policy_version": policy_version,
        "payload_hash": payload_hash(payload) if payload else None,
        "details": details or {},
        "prediction_id": prediction_id,
        "created_at": _now(),
    }
    ledger.add_audit(row)  # always keep a hot copy for the dashboard

    if not DB_AVAILABLE:
        return
    try:
        with session_scope() as session:
            if session is None:
                return
            session.add(AuditLog(
                request_id=request_id,
                event_type=event_type,
                endpoint=endpoint,
                http_status=http_status,
                latency_ms=latency_ms,
                actor=actor,
                model_version=model_version,
                policy_version=policy_version,
                payload_hash=row["payload_hash"],
                details=row["details"],
                prediction_id=uuid.UUID(prediction_id) if prediction_id else None,
            ))
    except Exception as exc:  # noqa: BLE001
        logger.debug("audit persistence failed: %s", exc)


def recent_audits(limit: int = 50) -> List[Dict[str, Any]]:
    return ledger.recent_audits(limit)


def recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    return ledger.recent_predictions(limit)
