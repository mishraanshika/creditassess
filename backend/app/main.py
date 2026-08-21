"""FastAPI application entry point.

Run from the repository root:
    uvicorn backend.app.main:app --reload --port 8000
or:
    python -m backend.run

Start-up wires four independent subsystems and degrades gracefully if any of the
optional ones are missing, so a judge can run the demo with only the trained
model on disk.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Make `ml`, `embeddings` and `app` importable regardless of the launch directory.
ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.db import init_engine  # noqa: E402
from app.routers import analytics, health, predict, report, similar  # noqa: E402
from app.services.llm import llm_underwriter  # noqa: E402
from app.services.scoring import scoring_service  # noqa: E402
from app.services.similar import similar_service  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("creditassess")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting %s v%s (%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENV)
    init_engine()
    scoring_service.initialise()
    similar_service.initialise()
    llm_underwriter.initialise()
    if settings.DEMO_SEED:
        from app.services.demo_seed import seed

        logger.info("demo seed: scored %d applicant(s)", seed())
    logger.info("startup complete | model=%s vectors=%s llm=%s",
                scoring_service.ready, similar_service.backend, llm_underwriter.status)
    yield
    logger.info("shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Real-time, multi-modal underwriting engine for New-To-Credit and thin-file "
        "borrowers. XGBoost probability of default + behavioural alternative-data "
        "features + FAISS/pgvector similar-borrower retrieval + SHAP explainability "
        "+ an LLM underwriting memo, with fairness and audit controls."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a correlation id and a server-timing header to every response."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{elapsed:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal error", "error": str(exc)[:300]})


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(similar.router)
app.include_router(report.router)
app.include_router(analytics.router)


@app.get("/", tags=["system"], summary="Service banner")
def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/similar-borrowers", "/underwriting-report",
                      "/analytics/model-metrics", "/analytics/feature-importance",
                      "/analytics/bias", "/analytics/portfolio", "/analytics/policy",
                      "/analytics/audit-log", "/analytics/review-queue"],
    }
