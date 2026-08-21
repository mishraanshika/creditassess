"""Turn a numeric applicant row into a natural-language borrower profile.

Why text and not the raw feature vector?

A sentence-transformer embedding of a *described* borrower puts semantically
similar applicants next to each other even when their raw numbers differ in
scale ("stable payer, long tenure, modest ticket" clusters regardless of
currency magnitude).  It also makes the retrieved neighbours human-readable in
the UI, which matters for an explainability-first product: an underwriter can
read exactly which peers the recommendation leaned on.

The vocabulary is deliberately bucketed - continuous numbers are mapped to a
small set of underwriting adjectives so the embedding space stays dense and the
nearest-neighbour sets stay stable.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _band(value: float, cuts, labels) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def behaviour_word(score: float) -> str:
    return _band(score, [25, 45, 65, 82], ["Very Weak", "Weak", "Moderate", "Strong", "Excellent"])


def volatility_word(score: float) -> str:
    return _band(score, [20, 40, 60, 80], ["Very Low", "Low", "Moderate", "High", "Very High"])


def ticket_word(ratio: float) -> str:
    return _band(ratio, [1.5, 3.0, 5.0, 8.0], ["Small", "Moderate", "Large", "Very Large", "Extreme"])


def tenure_word(years: float) -> str:
    return _band(years, [0.5, 2, 5, 10], ["New Joiner", "Early Tenure", "Established", "Long Tenure", "Career Stable"])


def build_profile_text(row: Dict[str, Any]) -> str:
    """Render one borrower as a compact, embedding-friendly description."""
    g = lambda k, d=0.0: float(row.get(k, d) or 0.0)  # noqa: E731

    income = g("AMT_INCOME_TOTAL")
    credit = g("AMT_CREDIT")
    annuity = g("AMT_ANNUITY")
    parts = [
        f"Applicant Income: {income:,.0f}",
        f"Employment Length: {g('employment_years'):.1f} years ({tenure_word(g('employment_years'))})",
        f"Occupation: {row.get('OCCUPATION_TYPE', 'Unknown')}",
        f"Employer Type: {row.get('ORGANIZATION_TYPE', 'Unknown')}",
        f"Education: {row.get('NAME_EDUCATION_TYPE', 'Unknown')}",
        f"Family Status: {row.get('NAME_FAMILY_STATUS', 'Unknown')} with {int(g('CNT_CHILDREN'))} children",
        f"Housing: {row.get('NAME_HOUSING_TYPE', 'Unknown')}",
        f"Loan Amount: {credit:,.0f} ({ticket_word(g('credit_income_ratio'))} ticket, "
        f"{g('credit_income_ratio'):.1f}x annual income)",
        f"Annual Instalment: {annuity:,.0f} (DTI {g('annuity_income_ratio') * 100:.0f}%)",
        f"Loan Term: {g('credit_term_months'):.0f} months",
        f"Payment Behavior: {behaviour_word(g('payment_consistency_score'))}",
        f"Financial Discipline: {behaviour_word(g('financial_discipline_score'))}",
        f"Income Stability: {behaviour_word(g('income_stability_score'))}",
        f"Spending Stability: {behaviour_word(g('spending_stability_score'))}",
        f"Credit Utilisation Headroom: {behaviour_word(g('credit_utilization_score'))}",
        f"Cash-flow Consistency: {behaviour_word(g('monthly_cashflow_consistency'))}",
        f"Utility Payment Consistency: {behaviour_word(g('utility_payment_consistency'))}",
        f"Mobile Recharge Consistency: {behaviour_word(g('mobile_recharge_consistency'))}",
        f"Digital Trust: {behaviour_word(g('digital_trust_score'))}",
        f"Transaction Volatility: {volatility_word(g('transaction_volatility'))}",
        f"Credit File: {'New to credit / thin file' if g('is_ntc') >= 1 else 'Established credit file'}",
        f"Bureau Coverage: {3 - int(g('ext_source_missing'))} of 3 external scores available",
    ]
    return " | ".join(parts)


def build_profile_texts(enriched: pd.DataFrame) -> list[str]:
    """Vectorised profile rendering for the whole corpus."""
    records = enriched.to_dict(orient="records")
    return [build_profile_text(r) for r in records]


def profile_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact structured summary shown on the Similar Borrowers card."""
    g = lambda k, d=0.0: float(row.get(k, d) or 0.0)  # noqa: E731
    return {
        "income": round(g("AMT_INCOME_TOTAL"), 2),
        "loan_amount": round(g("AMT_CREDIT"), 2),
        "employment_years": round(g("employment_years"), 1),
        "age_years": round(g("age_years"), 0),
        "credit_income_ratio": round(g("credit_income_ratio"), 2),
        "dti": round(g("annuity_income_ratio"), 3),
        "payment_behaviour": behaviour_word(g("payment_consistency_score")),
        "financial_discipline": behaviour_word(g("financial_discipline_score")),
        "transaction_volatility": volatility_word(g("transaction_volatility")),
        "is_ntc": bool(g("is_ntc") >= 1),
        "occupation": str(row.get("OCCUPATION_TYPE", "Unknown")),
        "education": str(row.get("NAME_EDUCATION_TYPE", "Unknown")),
    }
