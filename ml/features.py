"""Behavioural / alternative-data feature engineering.

The Home Credit `application_*.csv` files are a classical underwriting table.
For New-To-Credit (NTC) and thin-file applicants the bureau-derived columns
(`EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`) are frequently missing, which is
exactly the population this engine targets.  We therefore derive a family of
*behavioural* scores that lean on signals every applicant has - cash-flow,
tenure, device/contactability and document hygiene - and treat the bureau
columns as optional boosters rather than requirements.

Design rules
------------
1. Every transform is **stateless and deterministic**: the same formula runs on
   a 300k-row training frame and on a single applicant arriving at `/predict`.
   No fitted scalers are needed for the behavioural block, so scores are stable
   and directly comparable across time (important for audit).
2. Every score is expressed on a **0-100 "higher is better"** scale, except
   `transaction_volatility` where higher means noisier cash-flow.  This keeps
   the UI, the LLM prompt and the SHAP narrative consistent.
3. Every score is a documented blend of named sub-signals so an underwriter can
   be told *why* the number moved.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw column inventory
# ---------------------------------------------------------------------------

NUMERIC_RAW: List[str] = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
]

DOCUMENT_FLAGS: List[str] = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]

CATEGORICAL_RAW: List[str] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
]

# Behavioural scores exposed to the API / UI / LLM prompt.
BEHAVIOURAL_FEATURES: List[str] = [
    "payment_consistency_score",
    "spending_stability_score",
    "income_stability_score",
    "credit_utilization_score",
    "digital_trust_score",
    "financial_discipline_score",
    "transaction_volatility",
    "monthly_cashflow_consistency",
    "utility_payment_consistency",
    "mobile_recharge_consistency",
    "thin_file_score",
]

RATIO_FEATURES: List[str] = [
    "credit_income_ratio",
    "annuity_income_ratio",
    "credit_term_months",
    "goods_credit_ratio",
    "income_per_person",
    "employment_ratio",
    "age_years",
    "employment_years",
    "ext_source_mean",
    "ext_source_min",
    "ext_source_std",
    "ext_source_missing",
    "document_completeness",
    "social_default_ratio",
    "monthly_surplus",
    "surplus_ratio",
    "address_mismatch_count",
    "contactability_flags",
    "phone_stability_years",
    "id_freshness_years",
    "registration_age_years",
    "bureau_enquiry_intensity",
    "is_ntc",
]

# Safe defaults used when the API receives a partial applicant payload.
NUMERIC_DEFAULTS: Dict[str, float] = {
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_CREDIT": 500000.0,
    "AMT_ANNUITY": 25000.0,
    "AMT_GOODS_PRICE": 450000.0,
    "DAYS_BIRTH": -12000.0,
    "DAYS_EMPLOYED": -2000.0,
    "DAYS_REGISTRATION": -4000.0,
    "DAYS_ID_PUBLISH": -3000.0,
    "DAYS_LAST_PHONE_CHANGE": -800.0,
    "CNT_CHILDREN": 0.0,
    "CNT_FAM_MEMBERS": 2.0,
    "REGION_POPULATION_RELATIVE": 0.02,
    "REGION_RATING_CLIENT": 2.0,
    "REGION_RATING_CLIENT_W_CITY": 2.0,
    "HOUR_APPR_PROCESS_START": 12.0,
    "EXT_SOURCE_1": np.nan,
    "EXT_SOURCE_2": np.nan,
    "EXT_SOURCE_3": np.nan,
    "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
    "OBS_60_CNT_SOCIAL_CIRCLE": 0.0,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
    "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 0.0,
    "FLAG_MOBIL": 1.0,
    "FLAG_EMP_PHONE": 1.0,
    "FLAG_WORK_PHONE": 0.0,
    "FLAG_CONT_MOBILE": 1.0,
    "FLAG_PHONE": 0.0,
    "FLAG_EMAIL": 0.0,
    "REG_REGION_NOT_LIVE_REGION": 0.0,
    "REG_REGION_NOT_WORK_REGION": 0.0,
    "LIVE_REGION_NOT_WORK_REGION": 0.0,
    "REG_CITY_NOT_LIVE_CITY": 0.0,
    "REG_CITY_NOT_WORK_CITY": 0.0,
    "LIVE_CITY_NOT_WORK_CITY": 0.0,
}

CATEGORICAL_DEFAULTS: Dict[str, str] = {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "XNA",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "OCCUPATION_TYPE": "Laborers",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
}

MODEL_FEATURES: List[str] = (
    NUMERIC_RAW + DOCUMENT_FLAGS + RATIO_FEATURES + BEHAVIOURAL_FEATURES + CATEGORICAL_RAW
)

# Columns a fairness audit must never be allowed to key on directly.  They stay
# in the frame for bias *measurement* but are dropped from the model matrix when
# strict fairness mode is on.
PROTECTED_ATTRIBUTES: List[str] = ["CODE_GENDER", "age_years", "NAME_FAMILY_STATUS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Fetch a numeric column, materialising the documented default if absent."""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(np.nan, index=df.index, dtype="float64")
    default = NUMERIC_DEFAULTS.get(col, 0.0)
    if isinstance(default, float) and np.isnan(default):
        return s.astype("float64")
    return s.fillna(default).astype("float64")


def _scale(s, lo: float, hi: float, invert: bool = False) -> pd.Series:
    """Clip `s` to [lo, hi] then map linearly onto 0-100."""
    s = pd.Series(s) if not isinstance(s, pd.Series) else s
    x = (s.clip(lo, hi) - lo) / max(hi - lo, 1e-9)
    if invert:
        x = 1.0 - x
    return (x * 100.0).astype("float64")


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df_in` enriched with ratio + behavioural features.

    Works on the full training frame or on a single-row applicant frame.
    """
    df = df_in.copy()

    # --- normalise raw inputs ------------------------------------------------
    for col in CATEGORICAL_RAW:
        if col not in df.columns:
            df[col] = CATEGORICAL_DEFAULTS[col]
        df[col] = df[col].astype(object).fillna(CATEGORICAL_DEFAULTS[col]).astype(str)

    for col in DOCUMENT_FLAGS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float64")

    income = _num(df, "AMT_INCOME_TOTAL").clip(lower=1.0)
    credit = _num(df, "AMT_CREDIT").clip(lower=1.0)
    annuity = _num(df, "AMT_ANNUITY").clip(lower=1.0)
    goods = _num(df, "AMT_GOODS_PRICE").clip(lower=1.0)

    days_birth = _num(df, "DAYS_BIRTH")
    days_employed = _num(df, "DAYS_EMPLOYED")
    # 365243 is the Home Credit sentinel for "not employed / pensioner".
    days_employed = days_employed.mask(days_employed > 0, 0.0)

    # --- affordability & structure -------------------------------------------
    df["credit_income_ratio"] = _safe_div(credit, income, 0.0)
    df["annuity_income_ratio"] = _safe_div(annuity, income, 0.0)
    # AMT_ANNUITY in Home Credit is the ANNUAL instalment, so the implied number
    # of monthly payments is 12 x credit / annuity.
    df["credit_term_months"] = (12.0 * _safe_div(credit, annuity, 3.0)).clip(6, 480)
    df["goods_credit_ratio"] = _safe_div(goods, credit, 1.0).clip(0, 3)
    df["income_per_person"] = _safe_div(
        income, _num(df, "CNT_FAM_MEMBERS").clip(lower=1.0), 0.0)
    df["age_years"] = (-days_birth / 365.25).clip(18, 100)
    df["employment_years"] = (-days_employed / 365.25).clip(0, 50)
    df["employment_ratio"] = _safe_div(-days_employed, -days_birth, 0.0).clip(0, 1)
    df["phone_stability_years"] = (-_num(df, "DAYS_LAST_PHONE_CHANGE") / 365.25).clip(0, 20)
    df["id_freshness_years"] = (-_num(df, "DAYS_ID_PUBLISH") / 365.25).clip(0, 30)
    df["registration_age_years"] = (-_num(df, "DAYS_REGISTRATION") / 365.25).clip(0, 60)

    monthly_income = income / 12.0
    monthly_annuity = annuity / 12.0
    df["monthly_surplus"] = monthly_income - monthly_annuity
    df["surplus_ratio"] = _safe_div(df["monthly_surplus"], monthly_income, 0.0).clip(-2, 1)

    # --- bureau availability (thin-file detection) ---------------------------
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext = pd.DataFrame({
        c: (pd.to_numeric(df[c], errors="coerce") if c in df.columns
            else pd.Series(np.nan, index=df.index))
        for c in ext_cols
    })
    df["ext_source_mean"] = ext.mean(axis=1)
    df["ext_source_min"] = ext.min(axis=1)
    df["ext_source_std"] = ext.std(axis=1).fillna(0.0)
    df["ext_source_missing"] = ext.isna().sum(axis=1).astype("float64")
    # NTC / thin file: two or more bureau scores absent AND no bureau enquiries.
    enquiries = _num(df, "AMT_REQ_CREDIT_BUREAU_YEAR")
    df["is_ntc"] = ((df["ext_source_missing"] >= 2) & (enquiries <= 0)).astype("float64")
    df["thin_file_score"] = (_scale(df["ext_source_missing"], 0, 3) * 0.7
                             + df["is_ntc"] * 30.0).round(2)
    df["bureau_enquiry_intensity"] = (
        _num(df, "AMT_REQ_CREDIT_BUREAU_QRT") * 4.0 + enquiries
    ).clip(0, 40)

    # --- hygiene / contactability --------------------------------------------
    df["document_completeness"] = df[DOCUMENT_FLAGS].sum(axis=1).astype("float64")
    obs30 = _num(df, "OBS_30_CNT_SOCIAL_CIRCLE")
    def30 = _num(df, "DEF_30_CNT_SOCIAL_CIRCLE")
    df["social_default_ratio"] = _safe_div(def30, obs30.clip(lower=1.0), 0.0).clip(0, 1)

    contact_flags = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                     "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    df["contactability_flags"] = sum(_num(df, c) for c in contact_flags)

    mismatch_flags = ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                      "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
                      "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]
    df["address_mismatch_count"] = sum(_num(df, c) for c in mismatch_flags)

    # =======================================================================
    # Behavioural scores (0-100, higher = better unless stated otherwise)
    # =======================================================================

    # 1. payment_consistency_score
    #    Proxy for "does this person meet scheduled obligations on time".
    #    Sub-signals: affordable instalment schedule, absence of defaults in the
    #    declared social circle, a settled identity trail, document hygiene.
    term_fit = _scale(df["credit_term_months"], 6, 60)
    burden = _scale(df["annuity_income_ratio"], 0.05, 0.60, invert=True)
    peer_clean = _scale(df["social_default_ratio"], 0.0, 0.4, invert=True)
    id_settled = _scale(df["id_freshness_years"], 0, 8)
    doc_hygiene = _scale(df["document_completeness"], 0, 4)
    df["payment_consistency_score"] = (
        0.30 * burden + 0.22 * term_fit + 0.22 * peer_clean
        + 0.16 * id_settled + 0.10 * doc_hygiene
    ).round(2)

    # 2. utility_payment_consistency
    #    Utility accounts in Home Credit surface as household stability: owning
    #    or holding a registered dwelling, a long registration trail, a landline,
    #    and a residence that matches the registered address.
    owns_realty = (df["FLAG_OWN_REALTY"].str.upper() == "Y").astype(float) * 100.0
    stable_addr = _scale(df["address_mismatch_count"], 0, 4, invert=True)
    long_reg = _scale(df["registration_age_years"], 0, 20)
    landline = _num(df, "FLAG_PHONE") * 100.0
    df["utility_payment_consistency"] = (
        0.30 * owns_realty + 0.30 * stable_addr + 0.25 * long_reg + 0.15 * landline
    ).round(2)

    # 3. mobile_recharge_consistency
    #    A stable, contactable handset that has not been swapped recently is the
    #    telco-alternative-data analogue of an uninterrupted recharge history.
    handset_age = _scale(df["phone_stability_years"], 0, 4)
    reachable = _scale(df["contactability_flags"], 1, 6)
    work_phone = _num(df, "FLAG_EMP_PHONE") * 100.0
    always_on = _num(df, "FLAG_CONT_MOBILE") * 100.0
    df["mobile_recharge_consistency"] = (
        0.40 * handset_age + 0.25 * reachable + 0.20 * always_on + 0.15 * work_phone
    ).round(2)

    # 4. digital_trust_score
    #    Digital footprint breadth: email on file, reachable channels, stable
    #    handset, no address contradictions.
    df["digital_trust_score"] = (
        0.30 * (_num(df, "FLAG_EMAIL") * 100.0)
        + 0.25 * reachable
        + 0.25 * df["mobile_recharge_consistency"]
        + 0.20 * stable_addr
    ).round(2)

    # 5. spending_stability_score
    #    Requested credit that tracks the financed asset is disciplined spending;
    #    a large cash-out gap over the goods price is the classic overreach and
    #    a known fraud tell.
    asset_alignment = _scale((credit - goods).abs() / goods, 0.0, 0.5, invert=True)
    ticket_sanity = _scale(df["credit_income_ratio"], 0.5, 8.0, invert=True)
    dependants_load = _scale(_num(df, "CNT_CHILDREN"), 0, 4, invert=True)
    df["spending_stability_score"] = (
        0.45 * asset_alignment + 0.35 * ticket_sanity + 0.20 * dependants_load
    ).round(2)

    # 6. income_stability_score
    #    Tenure dominates; income level and a formally recognised employer add to
    #    it, pensioners get credit for a guaranteed inflow.
    tenure = _scale(df["employment_years"], 0, 12)
    career_share = _scale(df["employment_ratio"], 0.0, 0.35)
    income_level = _scale(np.log1p(df["income_per_person"]),
                          float(np.log1p(30000)), float(np.log1p(400000)))
    formal = (~df["ORGANIZATION_TYPE"].isin(["XNA"])).astype(float) * 100.0
    guaranteed = df["NAME_INCOME_TYPE"].isin(
        ["Pensioner", "State servant"]).astype(float) * 100.0
    df["income_stability_score"] = (
        0.35 * tenure + 0.20 * career_share + 0.20 * income_level
        + 0.15 * formal + 0.10 * guaranteed
    ).round(2)

    # 7. credit_utilization_score
    #    How much of the applicant capacity this facility consumes.
    #    High score = plenty of headroom left.
    leverage = _scale(df["credit_income_ratio"], 0.5, 10.0, invert=True)
    instalment_burden = _scale(df["annuity_income_ratio"], 0.03, 0.55, invert=True)
    enquiry_hunger = _scale(df["bureau_enquiry_intensity"], 0, 12, invert=True)
    df["credit_utilization_score"] = (
        0.45 * leverage + 0.40 * instalment_burden + 0.15 * enquiry_hunger
    ).round(2)

    # 8. monthly_cashflow_consistency
    #    Post-instalment surplus per household member, penalised for dependants.
    surplus = _scale(df["surplus_ratio"], 0.2, 0.95)
    per_head = _scale(np.log1p(df["income_per_person"]),
                      float(np.log1p(25000)), float(np.log1p(300000)))
    family_drag = _scale(_num(df, "CNT_FAM_MEMBERS"), 1, 6, invert=True)
    df["monthly_cashflow_consistency"] = (
        0.50 * surplus + 0.30 * per_head + 0.20 * family_drag
    ).round(2)

    # 9. transaction_volatility  (HIGHER = NOISIER = WORSE)
    #    Dispersion proxy: cash-out gap, credit-hunting enquiries, handset churn,
    #    address contradictions and off-hours application timing.
    cashout_gap = _scale((credit - goods).clip(lower=0) / goods, 0.0, 0.6)
    churn = _scale(df["phone_stability_years"], 0, 3, invert=True)
    hunting = _scale(df["bureau_enquiry_intensity"], 0, 10)
    contradictions = _scale(df["address_mismatch_count"], 0, 4)
    hour = _num(df, "HOUR_APPR_PROCESS_START")
    odd_hour = _scale((hour - 13).abs(), 2, 10)
    df["transaction_volatility"] = (
        0.30 * cashout_gap + 0.25 * churn + 0.20 * hunting
        + 0.15 * contradictions + 0.10 * odd_hour
    ).round(2)

    # 10. financial_discipline_score - the headline composite.
    df["financial_discipline_score"] = (
        0.28 * df["payment_consistency_score"]
        + 0.22 * df["credit_utilization_score"]
        + 0.18 * df["monthly_cashflow_consistency"]
        + 0.14 * df["spending_stability_score"]
        + 0.10 * df["utility_payment_consistency"]
        + 0.08 * (100.0 - df["transaction_volatility"])
    ).round(2)

    for col in NUMERIC_RAW:
        df[col] = _num(df, col)

    return df


def build_model_frame(df: pd.DataFrame, strict_fairness: bool = False) -> pd.DataFrame:
    """Select the model matrix columns in a stable, reproducible order."""
    cols = list(MODEL_FEATURES)
    if strict_fairness:
        cols = [c for c in cols if c not in PROTECTED_ATTRIBUTES]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = CATEGORICAL_DEFAULTS.get(c, 0.0)
    out = df[cols].copy()
    for c in CATEGORICAL_RAW:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def applicant_to_frame(payload: Dict) -> pd.DataFrame:
    """Turn a (possibly partial) API applicant payload into a 1-row raw frame."""
    row: Dict[str, object] = {}
    row.update(NUMERIC_DEFAULTS)
    row.update(CATEGORICAL_DEFAULTS)
    row.update({f: 0.0 for f in DOCUMENT_FLAGS})
    for k, v in payload.items():
        if v is None:
            continue
        row[k] = v
    return pd.DataFrame([row])
