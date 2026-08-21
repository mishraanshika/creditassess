"""POST /similar-borrowers - top-K peer retrieval from the vector store."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.schemas import SimilarRequest, SimilarResponse
from app.services import audit
from app.services.similar import similar_service
from ml.inference import get_model

router = APIRouter(tags=["vector-search"])


@router.post("/similar-borrowers", response_model=SimilarResponse,
             summary="Retrieve the most similar historical borrowers")
def similar_borrowers(req: SimilarRequest) -> SimilarResponse:
    if not similar_service.available:
        raise HTTPException(
            status_code=503,
            detail=("Vector index unavailable "
                    f"({similar_service.error or 'run `python -m embeddings.build_index`'})"),
        )

    request_id = str(uuid.uuid4())
    payload = req.applicant.to_feature_payload()

    try:
        model = get_model()
        _, enriched = model.frame_from_payload(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Feature pipeline unavailable: {exc}") from exc

    result = similar_service.query(enriched, req.top_k)

    audit.log_event(
        request_id, "SIMILAR", "/similar-borrowers", 200, result["latency_ms"],
        details={
            "top_k": req.top_k,
            "backend": result["backend"],
            "cohort_repayment_rate": (result.get("cohort") or {}).get("repayment_success_rate"),
        },
        payload=payload,
    )

    return SimilarResponse(
        request_id=request_id,
        query_profile=result["query_profile"],
        top_k=req.top_k,
        backend=result["backend"],
        encoder=result["encoder"],
        similar_borrowers=result["similar_borrowers"],
        cohort=result["cohort"],
        latency_ms=result["latency_ms"],
    )
