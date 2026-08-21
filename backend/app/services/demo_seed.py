"""Seed a few representative decisions at start-up.

An empty dashboard is a bad first impression and a worse demo: the decision mix,
the review queue and the audit trail all have nothing to show until someone has
been scored. This module scores three contrasting applicants in-process the
moment the API comes up, so every panel has real data on first load.

The three are chosen to land in three different decision bands - one approve,
one review, one reject - so the decision mix is immediately legible. They are
also deliberately different people from the intake form presets, so running a
preset live during a demo does not produce a duplicate row.

Nothing is fabricated: these run through the identical `ScoringService.assess`
path as a live request, and are written to the same ledger and audit trail.
Disable with `DEMO_SEED=false`.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Tuple

from app.services import audit
from app.services.scoring import scoring_service

logger = logging.getLogger(__name__)

# (reference, display name, raw applicant payload)
SEED_APPLICANTS: List[Tuple[str, str, Dict[str, Any]]] = [
    # --- APPROVE, new-to-credit: the thesis case ---------------------------
    # No bureau record at all, but strong behavioural evidence and a modest
    # ticket against a solid income. PD ~2.5%, inside the bureau-blind
    # straight-through margin, so it clears without a human.
    (
        "NTC-2201", "Meera Iyer",
        {
            "AMT_INCOME_TOTAL": 720000, "AMT_CREDIT": 240000,
            "AMT_ANNUITY": 84000, "AMT_GOODS_PRICE": 240000,
            "DAYS_BIRTH": -12410, "DAYS_EMPLOYED": -3120,
            "DAYS_LAST_PHONE_CHANGE": -1640, "DAYS_REGISTRATION": -5200,
            "DAYS_ID_PUBLISH": -3400,
            "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
            "OCCUPATION_TYPE": "Accountants", "ORGANIZATION_TYPE": "Bank",
            "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2,
            "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": 1, "FLAG_WORK_PHONE": 1,
            "FLAG_CONT_MOBILE": 1, "FLAG_PHONE": 1, "FLAG_EMAIL": 1,
            "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_6": 1, "FLAG_DOCUMENT_8": 1,
            "FLAG_DOCUMENT_2": 1, "FLAG_DOCUMENT_5": 1,
        },
    ),
    # --- REVIEW, established file, mid risk --------------------------------
    # Has a bureau record, but weak scores, short tenure, a stretched ticket
    # and repeated recent credit enquiries. Referred rather than declined.
    (
        "APP-2202", "Sandeep Kulkarni",
        {
            "AMT_INCOME_TOTAL": 315000, "AMT_CREDIT": 780000,
            "AMT_ANNUITY": 96000, "AMT_GOODS_PRICE": 720000,
            "DAYS_BIRTH": -10980, "DAYS_EMPLOYED": -740,
            "DAYS_LAST_PHONE_CHANGE": -420, "DAYS_REGISTRATION": -2600,
            "DAYS_ID_PUBLISH": -1900,
            "NAME_INCOME_TYPE": "Commercial associate",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Civil marriage", "NAME_HOUSING_TYPE": "Rented apartment",
            "OCCUPATION_TYPE": "Drivers", "ORGANIZATION_TYPE": "Transport: type 4",
            "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "N",
            "CNT_CHILDREN": 2, "CNT_FAM_MEMBERS": 4,
            "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": 1, "FLAG_CONT_MOBILE": 1,
            "FLAG_DOCUMENT_3": 1,
            "EXT_SOURCE_2": 0.41, "EXT_SOURCE_3": 0.33,
            "AMT_REQ_CREDIT_BUREAU_QRT": 2, "AMT_REQ_CREDIT_BUREAU_YEAR": 5,
        },
    ),
    # --- REJECT, high risk with multiple fraud tells -----------------------
    # Eight times income requested, three months in the job, handset and ID
    # both brand new, conflicting addresses, heavy enquiry activity.
    (
        "APP-2203", "Vikram Shetty",
        {
            "AMT_INCOME_TOTAL": 135000, "AMT_CREDIT": 1100000,
            "AMT_ANNUITY": 132000, "AMT_GOODS_PRICE": 600000,
            "DAYS_BIRTH": -8200, "DAYS_EMPLOYED": -95,
            "DAYS_LAST_PHONE_CHANGE": -22, "DAYS_REGISTRATION": -900,
            "DAYS_ID_PUBLISH": -60,
            "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Lower secondary",
            "NAME_FAMILY_STATUS": "Single / not married", "NAME_HOUSING_TYPE": "With parents",
            "OCCUPATION_TYPE": "Low-skill Laborers", "ORGANIZATION_TYPE": "Self-employed",
            "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "N",
            "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 2,
            "FLAG_MOBIL": 1, "FLAG_CONT_MOBILE": 1,
            "REG_CITY_NOT_LIVE_CITY": 1, "REG_CITY_NOT_WORK_CITY": 1,
            "LIVE_CITY_NOT_WORK_CITY": 1,
            "EXT_SOURCE_2": 0.11, "EXT_SOURCE_3": 0.09,
            "AMT_REQ_CREDIT_BUREAU_QRT": 5, "AMT_REQ_CREDIT_BUREAU_YEAR": 14,
        },
    ),
]


def seed() -> int:
    """Score and persist the seed applicants. Returns how many succeeded."""
    if not scoring_service.ready:
        logger.warning("demo seed skipped: model not loaded")
        return 0

    seeded = 0
    for ref, name, payload in SEED_APPLICANTS:
        request_id = str(uuid.uuid4())
        try:
            result = scoring_service.assess(payload, top_k=5, request_id=request_id)
            prediction_id = audit.record_decision(
                request_id, payload, result, external_ref=ref, full_name=name)
            decision = result["decision"]
            audit.log_event(
                request_id, "PREDICT", "/predict", 200, result["latency_ms"],
                details={
                    "recommendation": decision["recommendation"],
                    "risk_score": decision["risk_score"],
                    "pd": decision["probability_of_default"],
                    "confidence": decision["confidence_score"],
                    "requires_human_review": decision["requires_human_review"],
                    "is_ntc": decision["is_ntc"],
                    "seed": True,
                },
                payload=payload, prediction_id=prediction_id,
                model_version=result["model_version"],
                policy_version=decision["policy_version"],
                actor="demo-seed",
            )
            seeded += 1
            logger.info("seeded %s (%s): %s, score %s, PD %.2f%%", ref, name,
                        decision["recommendation"], decision["risk_score"],
                        decision["probability_of_default"] * 100)
        except Exception as exc:  # noqa: BLE001 - a demo seed must never block start-up
            logger.warning("demo seed failed for %s: %s", ref, exc)
    return seeded
