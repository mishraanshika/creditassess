"""Analytics + governance endpoints backing the dashboard."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from app.services import audit
from app.services.scoring import scoring_service
from ml import config as C
from ml.explain import global_importance
from ml.policy import (
    MIN_AUTO_CONFIDENCE,
    PD_APPROVE_MAX,
    PD_REJECT_MIN,
    POLICY_VERSION,
    RISK_BANDS,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/model-metrics", summary="Training + holdout metrics of the live model")
def model_metrics() -> Dict[str, Any]:
    if not scoring_service.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return scoring_service.model.metrics or {"detail": "metrics.json not found"}


@router.get("/feature-importance", summary="Global gain-based feature importance")
def feature_importance(top_k: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
    if not scoring_service.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": scoring_service.model.model_version,
        "features": global_importance(scoring_service.model, top_k),
    }


@router.get("/bias", summary="Fairness audit (four-fifths rule, equal opportunity)")
def bias_report() -> Dict[str, Any]:
    if not C.BIAS_REPORT_PATH.exists():
        raise HTTPException(status_code=404,
                            detail="No bias report. Run `python -m ml.bias_check`.")
    return json.loads(C.BIAS_REPORT_PATH.read_text(encoding="utf-8"))


@router.get("/policy", summary="Active decision policy (thresholds and bands)")
def policy() -> Dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "approve_max_pd": PD_APPROVE_MAX,
        "reject_min_pd": PD_REJECT_MIN,
        "min_auto_confidence": MIN_AUTO_CONFIDENCE,
        "risk_bands": [{"max_pd": c, "band": b, "tier": t} for c, b, t in RISK_BANDS],
    }


@router.get("/portfolio", summary="Live decision statistics for the dashboard")
def portfolio(limit: int = Query(500, ge=1, le=5000)) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = audit.recent_predictions(limit)
    if not rows:
        return {"decisions": 0, "message": "No decisions recorded yet."}

    recos = Counter(r["recommendation"] for r in rows)
    bands = Counter(r["risk_band"] for r in rows)
    ntc = [r for r in rows if r.get("is_ntc")]
    approved = [r for r in rows if r["recommendation"] == "APPROVE"]

    def avg(vals: List[float]) -> float:
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "decisions": len(rows),
        "recommendation_mix": dict(recos),
        "risk_band_distribution": dict(sorted(bands.items())),
        "approval_rate": round(len(approved) / len(rows), 4),
        "review_queue": sum(1 for r in rows if r.get("requires_human_review")),
        "avg_probability_of_default": avg([r["probability_of_default"] for r in rows]),
        "avg_risk_score": avg([r["risk_score"] for r in rows]),
        "avg_confidence": avg([r["confidence_score"] for r in rows]),
        "total_limit_offered": round(sum(r["recommended_credit_limit"] for r in rows), 2),
        "ntc": {
            "applications": len(ntc),
            "share": round(len(ntc) / len(rows), 4),
            "approval_rate": round(
                sum(1 for r in ntc if r["recommendation"] == "APPROVE") / len(ntc), 4
            ) if ntc else None,
            "avg_limit": avg([r["recommended_credit_limit"] for r in ntc]),
        },
        "fraud_flag_frequency": dict(Counter(
            f.split(":")[0] for r in rows for f in (r.get("fraud_flags") or [])
        )),
        "recent": rows[:25],
    }


@router.get("/audit-log", summary="Recent audit trail entries")
def audit_log(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    return {"entries": audit.recent_audits(limit)}


@router.get("/review-queue", summary="Decisions awaiting human review")
def review_queue(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    rows = [r for r in audit.recent_predictions(500) if r.get("requires_human_review")]
    return {"count": len(rows), "items": rows[:limit]}
