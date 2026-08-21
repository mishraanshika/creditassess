"""POST /predict - risk score, recommendation, confidence, SHAP, peers."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.schemas import PredictRequest, PredictResponse
from app.services import audit
from app.services.scoring import scoring_service

router = APIRouter(tags=["scoring"])


@router.post("/predict", response_model=PredictResponse, summary="Score an applicant")
def predict(req: PredictRequest) -> PredictResponse:
    if not scoring_service.ready:
        raise HTTPException(status_code=503,
                            detail=f"Model not available: {scoring_service.error}")

    request_id = str(uuid.uuid4())
    payload = req.applicant.to_feature_payload()

    try:
        result = scoring_service.assess(
            payload,
            top_k=req.top_k,
            include_explanation=req.include_explanation,
            include_similar=req.include_similar,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001
        audit.log_event(request_id, "ERROR", "/predict", 500,
                        details={"error": str(exc)[:500]}, payload=payload)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    prediction_id = None
    if req.persist:
        prediction_id = audit.record_decision(
            request_id, payload, result,
            external_ref=req.applicant.external_ref,
            full_name=req.applicant.full_name,
        )

    decision = result["decision"]
    audit.log_event(
        request_id, "PREDICT", "/predict", 200, result["latency_ms"],
        details={
            "recommendation": decision["recommendation"],
            "risk_score": decision["risk_score"],
            "pd": decision["probability_of_default"],
            "confidence": decision["confidence_score"],
            "requires_human_review": decision["requires_human_review"],
            "fraud_flags": decision["fraud_flags"],
            "is_ntc": decision["is_ntc"],
        },
        payload=payload,
        prediction_id=prediction_id,
        model_version=result["model_version"],
        policy_version=decision["policy_version"],
    )

    similar = result.get("similar") or {}
    return PredictResponse(
        request_id=request_id,
        prediction_id=prediction_id,
        applicant_ref=req.applicant.external_ref,
        probability_of_default=decision["probability_of_default"],
        risk_score=decision["risk_score"],
        risk_band=decision["risk_band"],
        risk_tier=decision["risk_tier"],
        recommendation=decision["recommendation"],
        recommended_credit_limit=decision["recommended_credit_limit"],
        max_affordable_limit=decision["max_affordable_limit"],
        suggested_term_months=decision["suggested_term_months"],
        suggested_monthly_instalment=decision["suggested_monthly_instalment"],
        requested_amount=decision["requested_amount"],
        confidence_score=decision["confidence_score"],
        confidence_drivers=decision["confidence_drivers"],
        requires_human_review=decision["requires_human_review"],
        review_reasons=decision["review_reasons"],
        fraud_flags=decision["fraud_flags"],
        is_ntc=decision["is_ntc"],
        behavioural_features=result["behavioural_features"],
        explanation=result.get("explanation"),
        similar_borrowers=similar.get("similar_borrowers", []),
        cohort=similar.get("cohort"),
        model_version=result["model_version"],
        policy_version=decision["policy_version"],
        latency_ms=result["latency_ms"],
    )
