"""Unit tests for the deterministic parts of the engine.

    .venv/Scripts/python -m pytest tests -q

The feature and policy layers are pure functions, so they are tested without any
model, database or network. Tests that need the trained artefacts skip cleanly
when `ml/artifacts/` is empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.features import applicant_to_frame, build_model_frame, engineer_features  # noqa: E402
from ml.policy import (  # noqa: E402
    NTC_STRAIGHT_THROUGH_MAX_PD,
    PD_APPROVE_MAX,
    PD_REJECT_MIN,
    compute_confidence,
    decide,
    detect_fraud_flags,
    pd_to_risk_score,
    risk_band,
)

STRONG = {
    "AMT_INCOME_TOTAL": 540000, "AMT_CREDIT": 300000, "AMT_ANNUITY": 100000,
    "AMT_GOODS_PRICE": 285000, "DAYS_BIRTH": -11300, "DAYS_EMPLOYED": -2000,
    "DAYS_LAST_PHONE_CHANGE": -1200, "CNT_FAM_MEMBERS": 2,
    "FLAG_EMAIL": 1, "FLAG_PHONE": 1, "FLAG_DOCUMENT_3": 1,
}
WEAK = {
    "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 900000, "AMT_ANNUITY": 78000,
    "AMT_GOODS_PRICE": 500000, "DAYS_BIRTH": -8800, "DAYS_EMPLOYED": -140,
    "DAYS_LAST_PHONE_CHANGE": -30, "CNT_FAM_MEMBERS": 3,
    "AMT_REQ_CREDIT_BUREAU_QRT": 3, "AMT_REQ_CREDIT_BUREAU_YEAR": 9,
}


def features_for(payload: dict) -> dict:
    return engineer_features(applicant_to_frame(payload)).iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def test_all_behavioural_scores_are_bounded():
    for payload in (STRONG, WEAK, {}):
        f = features_for(payload)
        for name in ("payment_consistency_score", "spending_stability_score",
                     "income_stability_score", "credit_utilization_score",
                     "digital_trust_score", "financial_discipline_score",
                     "transaction_volatility", "monthly_cashflow_consistency",
                     "utility_payment_consistency", "mobile_recharge_consistency"):
            assert 0.0 <= f[name] <= 100.0, f"{name} out of range for {payload}"


def test_strong_profile_beats_weak_profile():
    strong, weak = features_for(STRONG), features_for(WEAK)
    assert strong["financial_discipline_score"] > weak["financial_discipline_score"]
    assert strong["credit_utilization_score"] > weak["credit_utilization_score"]
    assert strong["transaction_volatility"] < weak["transaction_volatility"]


def test_term_is_expressed_in_months():
    # AMT_ANNUITY is the annual instalment: 300000 / 100000 = 3 years = 36 months.
    assert features_for(STRONG)["credit_term_months"] == pytest.approx(36.0, abs=0.5)


def test_missing_bureau_data_flags_ntc_and_stays_nan():
    f = features_for(STRONG)
    assert f["is_ntc"] == 1.0
    assert f["ext_source_missing"] == 3.0
    # EXT_SOURCE_* must NOT be imputed: "no bureau score" is signal the model uses.
    assert f["EXT_SOURCE_2"] != f["EXT_SOURCE_2"]  # NaN


def test_supplying_bureau_scores_clears_the_ntc_flag():
    f = features_for({**STRONG, "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.6})
    assert f["is_ntc"] == 0.0
    assert f["ext_source_missing"] == 1.0


def test_feature_engineering_is_stateless():
    """One row scored alone must equal the same row scored inside a batch."""
    import pandas as pd

    single = engineer_features(applicant_to_frame(STRONG)).iloc[0]
    batch = engineer_features(
        pd.concat([applicant_to_frame(WEAK), applicant_to_frame(STRONG)], ignore_index=True)
    ).iloc[1]
    for col in ("payment_consistency_score", "financial_discipline_score",
                "credit_term_months", "transaction_volatility"):
        assert single[col] == pytest.approx(batch[col])


def test_model_frame_has_a_stable_column_order():
    a = build_model_frame(engineer_features(applicant_to_frame(STRONG)))
    b = build_model_frame(engineer_features(applicant_to_frame(WEAK)))
    assert list(a.columns) == list(b.columns)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

def test_risk_score_is_monotone_and_anchored():
    assert pd_to_risk_score(0.01) > pd_to_risk_score(0.08) > pd_to_risk_score(0.30)
    assert 595 <= pd_to_risk_score(0.0807) <= 605      # base rate anchors at 600
    assert 300 <= pd_to_risk_score(0.99) <= 900


def test_risk_bands_partition_the_pd_range():
    assert risk_band(0.01)[0] == "A1"
    assert risk_band(0.99)[0] == "D2"
    assert risk_band(PD_APPROVE_MAX - 0.001)[1] == "Near Prime"


def test_low_pd_approves_and_high_pd_rejects_with_zero_limit():
    f = features_for({**STRONG, "EXT_SOURCE_2": 0.8, "EXT_SOURCE_3": 0.8, "EXT_SOURCE_1": 0.8})
    approve = decide(0.015, f, neighbour_agreement=1.0)
    assert approve.recommendation == "APPROVE"
    assert approve.recommended_credit_limit > 0

    reject = decide(PD_REJECT_MIN + 0.05, f, neighbour_agreement=1.0)
    assert reject.recommendation == "REJECT"
    assert reject.recommended_credit_limit == 0.0


def test_low_confidence_downgrades_approve_to_review():
    # Evidence-poor: no bureau score, no documents, barely contactable, no tenure,
    # and a peer cohort that disagrees with itself.
    bare = features_for({"AMT_INCOME_TOTAL": 240000, "AMT_CREDIT": 120000,
                         "AMT_ANNUITY": 40000, "DAYS_EMPLOYED": -60})
    d = decide(0.01, bare, neighbour_agreement=0.0)
    assert d.recommendation == "REVIEW"
    assert d.requires_human_review
    assert any("Confidence" in r for r in d.review_reasons)


def test_review_triggers_never_upgrade_a_rejection():
    d = decide(0.40, features_for(WEAK), neighbour_agreement=0.0)
    assert d.recommendation == "REJECT"
    assert d.requires_human_review is False


def test_limit_never_exceeds_affordable_capacity():
    f = features_for(STRONG)
    d = decide(0.03, f, neighbour_agreement=1.0)
    assert d.recommended_credit_limit <= d.max_affordable_limit
    assert d.recommended_credit_limit <= d.requested_amount


def test_limit_never_exceeds_the_requested_amount_even_for_prime():
    """A strong score must not upsell the applicant beyond what they asked for."""
    f = features_for({**STRONG, "AMT_CREDIT": 100000, "AMT_ANNUITY": 34000,
                      "EXT_SOURCE_1": 0.8, "EXT_SOURCE_2": 0.8, "EXT_SOURCE_3": 0.8})
    d = decide(0.01, f, neighbour_agreement=1.0)
    assert d.max_affordable_limit > d.requested_amount     # capacity is larger
    assert d.recommended_credit_limit <= d.requested_amount


def test_strong_ntc_applicant_can_be_approved_without_a_bureau_score():
    """Zero bureau coverage must not be an unconditional veto.

    If it were, no new-to-credit applicant could ever be auto-approved and the
    review queue would become a de-facto decline - the exact exclusion this
    engine exists to remove.
    """
    strong_ntc = features_for({
        "AMT_INCOME_TOTAL": 720000, "AMT_CREDIT": 240000, "AMT_ANNUITY": 84000,
        "AMT_GOODS_PRICE": 240000, "DAYS_EMPLOYED": -3120, "CNT_FAM_MEMBERS": 2,
        "DAYS_LAST_PHONE_CHANGE": -1640, "DAYS_ID_PUBLISH": -3400,
        "FLAG_EMAIL": 1, "FLAG_PHONE": 1, "FLAG_WORK_PHONE": 1,
        "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_6": 1, "FLAG_DOCUMENT_8": 1,
    })
    assert strong_ntc["is_ntc"] == 1.0
    assert strong_ntc["ext_source_missing"] == 3.0

    d = decide(NTC_STRAIGHT_THROUGH_MAX_PD - 0.005, strong_ntc, neighbour_agreement=1.0)
    assert d.recommendation == "APPROVE"
    assert d.recommended_credit_limit > 0


def test_weak_ntc_applicant_still_goes_to_review():
    """The margin is the whole safeguard: outside it, thin file means review."""
    thin = features_for(STRONG)
    d = decide(PD_APPROVE_MAX - 0.005, thin, neighbour_agreement=1.0)
    assert d.recommendation == "REVIEW"
    assert any("bureau" in r for r in d.review_reasons)


def test_ntc_gets_a_tighter_affordability_cap():
    ntc = decide(0.03, features_for(STRONG), neighbour_agreement=1.0)
    thick = decide(0.03, features_for({**STRONG, "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.7}),
                   neighbour_agreement=1.0)
    assert ntc.max_affordable_limit < thick.max_affordable_limit


def test_confidence_rises_with_evidence():
    thin, _ = compute_confidence(0.03, features_for(STRONG), 0.8)
    thick, _ = compute_confidence(
        0.03, features_for({**STRONG, "EXT_SOURCE_1": 0.7, "EXT_SOURCE_2": 0.7,
                            "EXT_SOURCE_3": 0.7}), 0.8)
    assert thick > thin


def test_confidence_falls_near_a_decision_boundary():
    f = features_for(STRONG)
    at_boundary, _ = compute_confidence(PD_APPROVE_MAX, f, 0.9)
    clear_of_it, _ = compute_confidence(0.005, f, 0.9)
    assert clear_of_it > at_boundary


def test_fraud_rules_fire_on_the_expected_tells():
    flags = detect_fraud_flags(features_for(WEAK))
    joined = " ".join(flags)
    assert "CASH_OUT_GAP" in joined          # 900k requested vs 500k goods
    assert "RECENT_DEVICE_CHANGE" in joined  # handset changed 30 days ago
    assert "NO_SUPPORTING_DOCUMENTS" in joined
    assert detect_fraud_flags(features_for(STRONG)) == [] or all(
        "CASH_OUT_GAP" not in f for f in detect_fraud_flags(features_for(STRONG)))


# --------------------------------------------------------------------------- #
# Trained-artefact tests (skipped when the model has not been built)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def model():
    from ml import config as C

    if not C.MODEL_PATH.exists():
        pytest.skip("model not trained; run `python -m ml.train`")
    from ml.inference import get_model

    return get_model()


def test_model_returns_a_calibrated_probability(model):
    X, _ = model.frame_from_payload(STRONG)
    p = float(model.predict_proba(X)[0])
    assert 0.0 < p < 1.0


def test_shap_contributions_reconstruct_the_prediction(model):
    from ml.explain import explain_prediction

    X, enriched = model.frame_from_payload(STRONG)
    e = explain_prediction(model, X, enriched)
    pd_value = float(model.predict_proba(X)[0])
    assert e["probability_of_default"] == pytest.approx(pd_value, abs=1e-4)
    assert e["top_positive_factors"] and e["top_negative_factors"]
    # Distinct drivers must report distinct impacts (guards the isotonic-plateau bug).
    impacts = [f["pd_impact_pp"] for f in e["top_negative_factors"][:4]]
    assert len(set(impacts)) > 1


def test_weak_profile_scores_worse_than_strong_profile(model):
    strong = float(model.predict_proba(model.frame_from_payload(STRONG)[0])[0])
    weak = float(model.predict_proba(model.frame_from_payload(WEAK)[0])[0])
    assert weak > strong
