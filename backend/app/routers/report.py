"""POST /underwriting-report - LLM-generated underwriting memo."""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException

from app.schemas import PredictResponse, ReportRequest, UnderwritingReportOut
from app.services import audit
from app.services.llm import llm_underwriter
from app.services.scoring import scoring_service

router = APIRouter(tags=["underwriting"])


@router.post("/underwriting-report", response_model=UnderwritingReportOut,
             summary="Generate the AI underwriting memo for an applicant")
def underwriting_report(req: ReportRequest) -> UnderwritingReportOut:
    if not scoring_service.ready:
        raise HTTPException(status_code=503,
                            detail=f"Model not available: {scoring_service.error}")

    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    payload = req.applicant.to_feature_payload()

    # The memo is always generated from a fresh, fully explained assessment so the
    # narrative can never describe a different decision than the one on record.
    result = scoring_service.assess(payload, top_k=req.top_k, request_id=request_id)
    decision = result["decision"]

    report = llm_underwriter.generate(
        decision=decision,
        features=result["features"],
        explanation=result.get("explanation") or {},
        similar=result.get("similar") or {},
        tone=req.tone,
    )

    prediction_id = None
    if req.persist:
        prediction_id = audit.record_decision(
            request_id, payload, result,
            external_ref=req.applicant.external_ref,
            full_name=req.applicant.full_name,
        )
        audit.record_report(prediction_id, report, decision)

    latency = int((time.perf_counter() - t0) * 1000)
    audit.log_event(
        request_id, "REPORT", "/underwriting-report", 200, latency,
        details={"generator": report.get("generator"), "tone": req.tone,
                 "recommendation": decision["recommendation"],
                 "prompt_version": report.get("prompt_version")},
        payload=payload,
        prediction_id=prediction_id,
        model_version=result["model_version"],
        policy_version=decision["policy_version"],
    )

    similar = result.get("similar") or {}
    decision_out = PredictResponse(
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

    return UnderwritingReportOut(
        request_id=request_id,
        generator=report.get("generator", "template"),
        prompt_version=report.get("prompt_version", ""),
        recommendation=decision["recommendation"],
        suggested_credit_limit=decision["recommended_credit_limit"],
        confidence_score=decision["confidence_score"],
        executive_summary=report["executive_summary"],
        strengths=report.get("strengths", []),
        risk_factors=report.get("risk_factors", []),
        conditions=report.get("conditions", []),
        detailed_explanation=report.get("detailed_explanation", ""),
        similar_borrower_insight=report.get("similar_borrower_insight", ""),
        compliance_note=report.get("compliance_note", ""),
        decision=decision_out,
        latency_ms=latency,
    )
