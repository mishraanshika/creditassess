"""Pydantic v2 request/response contracts.

The applicant payload is intentionally forgiving: the frontend form collects ~18
fields, everything else falls back to the documented defaults in
`ml.features.NUMERIC_DEFAULTS`.  That is what makes the engine usable for a
New-To-Credit applicant who simply has no bureau history to send.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class ApplicantIn(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    external_ref: Optional[str] = Field(None, description="SK_ID_CURR or CRM reference")
    full_name: Optional[str] = None

    # --- core financials (required) ---------------------------------------
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Declared annual income")
    AMT_CREDIT: float = Field(..., gt=0, description="Credit amount requested")
    AMT_ANNUITY: Optional[float] = Field(None, ge=0, description="Annual instalment")
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0)

    # --- demographics / tenure (accepted in human units, converted below) ---
    age_years: Optional[float] = Field(None, ge=18, le=100)
    employment_years: Optional[float] = Field(None, ge=0, le=60)
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None

    CODE_GENDER: Optional[str] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    CNT_CHILDREN: Optional[int] = Field(None, ge=0, le=20)
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=1, le=25)

    # --- alternative-data / behavioural inputs -----------------------------
    FLAG_MOBIL: Optional[int] = Field(None, ge=0, le=1)
    FLAG_EMP_PHONE: Optional[int] = Field(None, ge=0, le=1)
    FLAG_WORK_PHONE: Optional[int] = Field(None, ge=0, le=1)
    FLAG_CONT_MOBILE: Optional[int] = Field(None, ge=0, le=1)
    FLAG_PHONE: Optional[int] = Field(None, ge=0, le=1)
    FLAG_EMAIL: Optional[int] = Field(None, ge=0, le=1)
    months_on_current_handset: Optional[float] = Field(
        None, ge=0, description="Months since last handset change (telco signal)")
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    documents_submitted: Optional[int] = Field(
        None, ge=0, le=20, description="Count of supporting documents provided")

    # --- optional bureau block (absent for NTC) ----------------------------
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(None, ge=0)
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(None, ge=0)

    @field_validator("CODE_GENDER")
    @classmethod
    def _gender(cls, v: Optional[str]) -> Optional[str]:
        return v.upper()[:3] if v else v

    def to_feature_payload(self) -> Dict[str, Any]:
        """Normalise human-friendly units into raw Home Credit columns."""
        data = self.model_dump(exclude_none=True)
        data.pop("external_ref", None)
        data.pop("full_name", None)

        if "DAYS_BIRTH" not in data and "age_years" in data:
            data["DAYS_BIRTH"] = -float(data["age_years"]) * 365.25
        if "DAYS_EMPLOYED" not in data and "employment_years" in data:
            data["DAYS_EMPLOYED"] = -float(data["employment_years"]) * 365.25
        if "DAYS_LAST_PHONE_CHANGE" not in data and "months_on_current_handset" in data:
            data["DAYS_LAST_PHONE_CHANGE"] = -float(data["months_on_current_handset"]) * 30.44

        docs = data.pop("documents_submitted", None)
        if docs:
            # Home Credit's dominant document flag is DOCUMENT_3; spread the count
            # over the most commonly submitted document slots.
            for i, slot in enumerate([3, 6, 8, 2, 5, 16, 18]):
                if i < int(docs):
                    data[f"FLAG_DOCUMENT_{slot}"] = 1
        if "AMT_ANNUITY" not in data:
            # Default to a 36-month schedule when the instalment is not supplied.
            data["AMT_ANNUITY"] = float(data["AMT_CREDIT"]) / 3.0
        if "AMT_GOODS_PRICE" not in data:
            data["AMT_GOODS_PRICE"] = float(data["AMT_CREDIT"]) * 0.9

        data.pop("age_years", None)
        data.pop("employment_years", None)
        data.pop("months_on_current_handset", None)
        return data


class PredictRequest(BaseModel):
    applicant: ApplicantIn
    top_k: int = Field(5, ge=1, le=25, description="Similar borrowers to retrieve")
    include_explanation: bool = True
    include_similar: bool = True
    persist: bool = True


class SimilarRequest(BaseModel):
    applicant: ApplicantIn
    top_k: int = Field(5, ge=1, le=25)


class ReportRequest(BaseModel):
    applicant: ApplicantIn
    top_k: int = Field(5, ge=1, le=25)
    tone: str = Field("credit_committee",
                      description="credit_committee | customer_letter | risk_memo")
    persist: bool = True


class ReviewDecisionIn(BaseModel):
    prediction_id: str
    reviewer: str
    final_decision: str = Field(..., pattern="^(APPROVE|REJECT)$")
    approved_limit: Optional[float] = Field(None, ge=0)
    rationale: str = Field(..., min_length=10)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class ShapFactor(BaseModel):
    feature: str
    label: str
    value: Optional[Any] = None
    value_display: str
    shap_value: float
    direction: str
    pd_impact_pp: float


class Explanation(BaseModel):
    base_value_logodds: float
    base_probability: float
    probability_of_default: float
    top_positive_factors: List[ShapFactor]
    top_negative_factors: List[ShapFactor]
    contribution_chart: List[ShapFactor]
    narrative: str


class SimilarBorrower(BaseModel):
    model_config = ConfigDict(extra="allow")

    borrower_id: int
    similarity_score: float
    repaid: bool
    outcome: str
    profile_text: str


class CohortStats(BaseModel):
    model_config = ConfigDict(extra="allow")

    cohort_size: int
    repayment_success_rate: Optional[float] = None
    default_rate: Optional[float] = None
    mean_similarity: Optional[float] = None


class SimilarResponse(BaseModel):
    request_id: str
    query_profile: str
    top_k: int
    backend: str
    encoder: str
    similar_borrowers: List[SimilarBorrower]
    cohort: CohortStats
    latency_ms: int


class PredictResponse(BaseModel):
    request_id: str
    prediction_id: Optional[str] = None
    applicant_ref: Optional[str] = None

    probability_of_default: float
    risk_score: int
    risk_band: str
    risk_tier: str
    recommendation: str
    recommended_credit_limit: float
    max_affordable_limit: float
    suggested_term_months: int
    suggested_monthly_instalment: float
    requested_amount: float

    confidence_score: float
    confidence_drivers: Dict[str, float]
    requires_human_review: bool
    review_reasons: List[str]
    fraud_flags: List[str]
    is_ntc: bool

    behavioural_features: Dict[str, float]
    explanation: Optional[Explanation] = None
    similar_borrowers: List[SimilarBorrower] = []
    cohort: Optional[CohortStats] = None

    model_version: str
    policy_version: str
    latency_ms: int


class UnderwritingReportOut(BaseModel):
    request_id: str
    generator: str
    prompt_version: str
    recommendation: str
    suggested_credit_limit: float
    confidence_score: float
    executive_summary: str
    strengths: List[str]
    risk_factors: List[str]
    conditions: List[str]
    detailed_explanation: str
    similar_borrower_insight: str
    compliance_note: str
    decision: PredictResponse
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    env: str
    model_loaded: bool
    model_version: Optional[str] = None
    vector_backend: Optional[str] = None
    vector_encoder: Optional[str] = None
    vector_size: Optional[int] = None
    database: str
    llm: str
    uptime_seconds: float
    checks: Dict[str, Any] = {}
