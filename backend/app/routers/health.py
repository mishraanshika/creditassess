"""GET /health - readiness of every subsystem (model, vectors, DB, LLM)."""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter

from app.core.config import settings
from app.core.db import DB_AVAILABLE, DB_ERROR
from app.schemas import HealthResponse
from app.services.llm import llm_underwriter
from app.services.scoring import scoring_service
from app.services.similar import similar_service

router = APIRouter(tags=["system"])
STARTED_AT = time.time()


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    checks: Dict[str, Any] = {
        "model": "ok" if scoring_service.ready else f"unavailable: {scoring_service.error}",
        "vector_store": "ok" if similar_service.available
                        else f"unavailable: {similar_service.error}",
        "database": "connected" if DB_AVAILABLE
                    else f"degraded (in-memory ledger): {DB_ERROR or 'not configured'}",
        "llm": llm_underwriter.status,
    }
    # The engine still scores and explains without vectors, the DB or the LLM;
    # only the model is a hard dependency.
    status = "healthy" if scoring_service.ready else "degraded"

    return HealthResponse(
        status=status,
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.ENV,
        model_loaded=scoring_service.ready,
        model_version=scoring_service.model.model_version if scoring_service.ready else None,
        vector_backend=similar_service.backend,
        vector_encoder=getattr(similar_service.encoder, "name", None),
        vector_size=similar_service.size(),
        database="connected" if DB_AVAILABLE else "in-memory",
        llm=llm_underwriter.status,
        uptime_seconds=round(time.time() - STARTED_AT, 2),
        checks=checks,
    )
