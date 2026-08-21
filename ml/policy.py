"""Decision policy layer.

The model emits a probability of default (PD).  A PD is *not* a decision.  This
module converts PD plus behavioural context into the artefacts an underwriter
and a regulator actually need:

* `risk_score`      - 300-900 scorecard scale (higher = safer), monotonic in PD
* `risk_band`       - A1..D2 style band for portfolio slicing
* `recommendation`  - APPROVE / REVIEW / REJECT
* `credit_limit`    - affordability-capped, risk-multiplied, policy-rounded
* `confidence`      - how much the engine trusts its own answer
* `review_reasons`  - explicit human-in-the-loop triggers
* `fraud_flags`     - rule-based anomaly tells surfaced next to the score

Thresholds live in one place so they can be tuned, versioned and audited.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

POLICY_VERSION = "policy-1.2.0"

# --- decision thresholds on probability of default --------------------------
# Home Credit base default rate is ~8.07%.  Bands are set relative to it.
PD_APPROVE_MAX = 0.060      # comfortably below base rate -> straight through
PD_REJECT_MIN = 0.170       # ~2x base rate -> decline
PD_BOUNDARY_MARGIN = 0.015  # closeness to a cut-off that forces human review

# --- affordability policy ---------------------------------------------------
MAX_DTI = 0.45              # max share of monthly income going to this instalment
NTC_MAX_DTI = 0.35          # tighter for New-To-Credit / thin-file
DEFAULT_TERM_MONTHS = 36
MIN_LIMIT = 25_000.0
ABSOLUTE_MAX_LIMIT = 5_000_000.0
LIMIT_ROUNDING = 5_000.0

# --- confidence policy ------------------------------------------------------
MIN_AUTO_CONFIDENCE = 0.70  # below this, a human must look at it

# Bureau-blind straight-through limit.
# Zero bureau coverage is a reason for caution, not an automatic veto. If a
# thin-file applicant's PD sits well inside the approve band (at or below half
# the approve cut-off) with no fraud flags, the case may still be auto-decided.
# Without this, NO new-to-credit applicant could ever be approved automatically,
# and a mandatory review queue becomes a de-facto decline at any real volume -
# exactly the exclusion this engine exists to remove. Exposure is still
# controlled by the tighter NTC affordability cap and the large-first-facility
# trigger below.
NTC_STRAIGHT_THROUGH_MAX_PD = PD_APPROVE_MAX / 2

RISK_BANDS = [
    (0.020, "A1", "Prime"),
    (0.040, "A2", "Prime"),
    (0.060, "B1", "Near Prime"),
    (0.090, "B2", "Near Prime"),
    (0.130, "C1", "Sub Prime"),
    (0.170, "C2", "Sub Prime"),
    (0.250, "D1", "High Risk"),
    (1.001, "D2", "High Risk"),
]


@dataclass
class Decision:
    probability_of_default: float
    risk_score: int
    risk_band: str
    risk_tier: str
    recommendation: str
    recommended_credit_limit: float
    requested_amount: float
    max_affordable_limit: float
    suggested_term_months: int
    suggested_monthly_instalment: float
    confidence_score: float
    confidence_drivers: Dict[str, float]
    requires_human_review: bool
    review_reasons: List[str] = field(default_factory=list)
    fraud_flags: List[str] = field(default_factory=list)
    is_ntc: bool = False
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Portfolio base default rate (Home Credit application_train). The scorecard is
# anchored here so that an average applicant scores 600 - without the anchor, an
# 8% PD would print as ~730, which reads as prime to a credit analyst.
BASE_DEFAULT_RATE = 0.0807
PDO = 40.0          # points to double the odds
ANCHOR_SCORE = 600.0


def pd_to_risk_score(pd_value: float) -> int:
    """Map PD to a 300-900 scorecard.

    Industry-standard log-odds scaling:
        score = anchor + factor * (ln(odds_good) - ln(odds_good at base rate))
    with 40 points to double the odds. Linear in log-odds, which is how a credit
    analyst reasons about "20 points riskier", and anchored so 600 = portfolio
    average rather than 1:1 odds.
    """
    p = min(max(float(pd_value), 1e-6), 1 - 1e-6)
    odds_good = (1 - p) / p
    base_odds_good = (1 - BASE_DEFAULT_RATE) / BASE_DEFAULT_RATE
    factor = PDO / math.log(2)
    score = ANCHOR_SCORE + factor * (math.log(odds_good) - math.log(base_odds_good))
    return int(round(min(900.0, max(300.0, score))))


def risk_band(pd_value: float) -> tuple[str, str]:
    for ceiling, band, tier in RISK_BANDS:
        if pd_value < ceiling:
            return band, tier
    return "D2", "High Risk"


def compute_confidence(
    pd_value: float,
    features: Dict[str, float],
    neighbour_agreement: Optional[float] = None,
) -> tuple[float, Dict[str, float]]:
    """Confidence = data sufficiency x decisiveness x peer agreement.

    * data_sufficiency - how much real evidence backs the prediction (bureau
      coverage, document hygiene, contactability, employment record).
    * decisiveness     - distance of PD from the nearest decision cut-off; a PD
      sitting on a threshold is inherently a low-confidence call.
    * peer_agreement   - consistency of outcomes among retrieved similar
      borrowers (supplied by the vector layer; neutral 0.75 when unavailable).
    """
    ext_missing = float(features.get("ext_source_missing", 3.0))
    docs = float(features.get("document_completeness", 0.0))
    contact = float(features.get("contactability_flags", 1.0))
    emp_years = float(features.get("employment_years", 0.0))

    bureau_cov = 1.0 - (ext_missing / 3.0)
    doc_cov = min(docs / 3.0, 1.0)
    contact_cov = min(max(contact - 1.0, 0.0) / 4.0, 1.0)
    tenure_cov = min(emp_years / 5.0, 1.0)
    data_sufficiency = (
        0.40 * bureau_cov + 0.20 * doc_cov + 0.20 * contact_cov + 0.20 * tenure_cov
    )
    # Even a pure NTC applicant retains a floor of evidence from behavioural data.
    data_sufficiency = 0.45 + 0.55 * data_sufficiency

    dist = min(abs(pd_value - PD_APPROVE_MAX), abs(pd_value - PD_REJECT_MIN))
    decisiveness = min(dist / 0.05, 1.0)
    decisiveness = 0.55 + 0.45 * decisiveness

    peer = 0.75 if neighbour_agreement is None else float(neighbour_agreement)
    peer = 0.60 + 0.40 * min(max(peer, 0.0), 1.0)

    # Weighted GEOMETRIC mean, not a plain product.
    #
    # A plain product of three sub-scores that each sit around 0.7 lands at 0.34,
    # so with a 0.70 auto-decision floor virtually every application would be
    # routed to a human - a review queue nobody can staff, and a worse outcome
    # for thin-file applicants than a straight decline policy. The weighted
    # geometric mean keeps the same ordering and the same veto behaviour (any one
    # factor collapsing still drags the result down) while staying on a scale
    # where the floor separates evidence-poor cases from evidence-rich ones.
    confidence = (data_sufficiency ** 0.45) * (decisiveness ** 0.30) * (peer ** 0.25)
    confidence = round(min(max(confidence, 0.05), 0.99), 4)
    drivers = {
        "data_sufficiency": round(data_sufficiency, 4),
        "decisiveness": round(decisiveness, 4),
        "peer_agreement": round(peer, 4),
    }
    return confidence, drivers


def detect_fraud_flags(features: Dict[str, float]) -> List[str]:
    """Rule-based fraud / anomaly tells.

    These never auto-decline on their own - they route to human review and are
    printed verbatim in the underwriting report so the reviewer sees the tell.
    """
    flags: List[str] = []
    g = features.get

    if float(g("goods_credit_ratio", 1.0)) < 0.65:
        flags.append("CASH_OUT_GAP: credit requested far exceeds the financed goods price")
    if float(g("address_mismatch_count", 0.0)) >= 3:
        flags.append("ADDRESS_INCONSISTENCY: three or more registered/live/work address mismatches")
    if float(g("phone_stability_years", 5.0)) < 0.25:
        flags.append("RECENT_DEVICE_CHANGE: handset changed within the last 90 days")
    if float(g("bureau_enquiry_intensity", 0.0)) >= 12:
        flags.append("CREDIT_HUNGRY: unusually high recent bureau enquiry intensity")
    if float(g("id_freshness_years", 5.0)) < 0.25:
        flags.append("FRESH_IDENTITY: identity document issued within the last 90 days")
    if float(g("employment_years", 5.0)) < 0.25 and float(g("credit_income_ratio", 0.0)) > 4.0:
        flags.append("THIN_TENURE_HIGH_TICKET: large ticket against under-3-month tenure")
    if float(g("transaction_volatility", 0.0)) >= 65:
        flags.append("HIGH_VOLATILITY: behavioural cash-flow volatility above tolerance")
    if float(g("document_completeness", 0.0)) == 0:
        flags.append("NO_SUPPORTING_DOCUMENTS: no supporting document flags submitted")
    return flags


def affordable_limit(features: Dict[str, float], is_ntc: bool, term_months: int) -> float:
    """Maximum principal supportable by declared monthly cash-flow."""
    income = float(features.get("AMT_INCOME_TOTAL", 0.0))
    monthly_income = max(income / 12.0, 1.0)
    dti_cap = NTC_MAX_DTI if is_ntc else MAX_DTI

    # Existing obligation implied by the requested annuity is what we are sizing,
    # so capacity is the full DTI budget less a dependants allowance.
    fam = max(float(features.get("CNT_FAM_MEMBERS", 1.0)), 1.0)
    dependants_haircut = min(0.04 * (fam - 1.0), 0.12)
    budget = monthly_income * max(dti_cap - dependants_haircut, 0.10)
    return budget * term_months


def risk_multiplier(pd_value: float, discipline: float) -> float:
    """Shrink the offered limit as risk rises, expand it for strong behaviour."""
    if pd_value < 0.02:
        base = 1.00
    elif pd_value < 0.04:
        base = 0.90
    elif pd_value < 0.06:
        base = 0.80
    elif pd_value < 0.09:
        base = 0.62
    elif pd_value < 0.13:
        base = 0.45
    elif pd_value < 0.17:
        base = 0.30
    else:
        base = 0.15
    # +/-10% behavioural adjustment around a discipline score of 50.
    adj = 1.0 + (float(discipline) - 50.0) / 500.0
    return base * min(max(adj, 0.85), 1.15)


def _round_limit(x: float) -> float:
    if x <= 0:
        return 0.0
    return float(max(MIN_LIMIT, math.floor(x / LIMIT_ROUNDING) * LIMIT_ROUNDING))


def decide(
    pd_value: float,
    features: Dict[str, float],
    neighbour_agreement: Optional[float] = None,
    requested_amount: Optional[float] = None,
) -> Decision:
    """Turn a PD plus a feature dict into a full, auditable credit decision."""
    pd_value = float(min(max(pd_value, 0.0), 1.0))
    band, tier = risk_band(pd_value)
    score = pd_to_risk_score(pd_value)
    is_ntc = bool(features.get("is_ntc", 0.0) >= 1.0)

    requested = float(requested_amount if requested_amount is not None
                      else features.get("AMT_CREDIT", 0.0))

    term = int(features.get("credit_term_months", DEFAULT_TERM_MONTHS) or DEFAULT_TERM_MONTHS)
    term = int(min(max(term, 6), 84))

    capacity = affordable_limit(features, is_ntc, term)
    mult = risk_multiplier(pd_value, float(features.get("financial_discipline_score", 50.0)))
    # Risk and behaviour scale the *capacity*; the offer is then capped by what
    # the applicant actually asked for. A lender never offers more than the
    # requested facility on the strength of a good score alone.
    raw_limit = capacity * mult
    if requested > 0:
        raw_limit = min(raw_limit, requested)
    raw_limit = min(raw_limit, ABSOLUTE_MAX_LIMIT)

    confidence, drivers = compute_confidence(pd_value, features, neighbour_agreement)
    fraud_flags = detect_fraud_flags(features)

    # --- recommendation ----------------------------------------------------
    if pd_value <= PD_APPROVE_MAX:
        recommendation = "APPROVE"
    elif pd_value >= PD_REJECT_MIN:
        recommendation = "REJECT"
    else:
        recommendation = "REVIEW"

    review_reasons: List[str] = []
    if confidence < MIN_AUTO_CONFIDENCE:
        review_reasons.append(
            f"Confidence {confidence:.3f} is below the {MIN_AUTO_CONFIDENCE:.2f} "
            "auto-decision floor")
    if min(abs(pd_value - PD_APPROVE_MAX), abs(pd_value - PD_REJECT_MIN)) < PD_BOUNDARY_MARGIN:
        review_reasons.append("Probability of default sits within the policy boundary margin")
    if is_ntc and recommendation == "APPROVE" and raw_limit > 300_000:
        review_reasons.append("New-to-credit applicant with a large first-facility limit")
    if fraud_flags:
        review_reasons.append(f"{len(fraud_flags)} fraud/anomaly signal(s) raised")
    if requested > 0 and capacity < requested * 0.5:
        review_reasons.append("Requested amount is more than double the affordable capacity")
    if float(features.get("ext_source_missing", 0.0)) == 3.0 and (
            pd_value > NTC_STRAIGHT_THROUGH_MAX_PD or fraud_flags):
        review_reasons.append(
            "No external bureau score on any source, and the applicant does not clear "
            f"the {NTC_STRAIGHT_THROUGH_MAX_PD:.1%} PD margin required for a bureau-blind "
            "automatic decision")

    requires_review = bool(review_reasons) and recommendation != "REJECT"
    if requires_review and recommendation == "APPROVE":
        recommendation = "REVIEW"

    limit = _round_limit(raw_limit) if recommendation != "REJECT" else 0.0
    instalment = round(limit / term, 2) if limit else 0.0

    return Decision(
        probability_of_default=round(pd_value, 6),
        risk_score=score,
        risk_band=band,
        risk_tier=tier,
        recommendation=recommendation,
        recommended_credit_limit=limit,
        requested_amount=round(requested, 2),
        max_affordable_limit=_round_limit(capacity),
        suggested_term_months=term,
        suggested_monthly_instalment=instalment,
        confidence_score=confidence,
        confidence_drivers=drivers,
        requires_human_review=requires_review,
        review_reasons=review_reasons,
        fraud_flags=fraud_flags,
        is_ntc=is_ntc,
    )
