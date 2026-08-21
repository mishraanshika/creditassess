"""Prompt engineering for the LLM underwriter.

Principles applied here
-----------------------
1. **The LLM never invents the decision.** The XGBoost PD and the policy engine
   own APPROVE / REVIEW / REJECT and the credit limit.  The model's job is to
   *explain and stress-test* that decision in underwriting language.  This is
   the single most important guard-rail in a regulated credit workflow.
2. **Grounding only.** Every number the model may cite is handed to it in the
   context block; the prompt forbids inventing figures and forbids using any
   protected attribute as a justification.
3. **Structured output.** The response is constrained to a JSON schema, so the
   API contract never depends on parsing prose.
4. **Few-shot calibration.** Two worked examples fix the tone, the depth and
   the house style (a thin-file APPROVE and a high-risk REJECT).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

PROMPT_VERSION = "underwriter-prompt-2.1"

SYSTEM_PROMPT = """You are a senior credit underwriter at a regulated retail lender, writing the \
underwriting memo that accompanies an automated credit decision.

Your operating rules:

1. THE DECISION IS ALREADY MADE. A gradient-boosted risk model produced the \
probability of default, and a documented policy engine produced the \
recommendation and the credit limit. You explain, justify and stress-test that \
outcome. You never overturn it and never state a different recommendation or \
limit than the ones supplied.
2. GROUND EVERY CLAIM. Use only the figures in the CONTEXT block. Never invent \
income, balances, bureau scores, dates or peer statistics. If a figure is \
missing, say it is not available rather than estimating it.
3. NEW-TO-CREDIT IS NOT BAD CREDIT. Where the applicant is thin-file, reason \
from the behavioural and alternative-data evidence (payment consistency, \
utility and mobile-recharge consistency, cash-flow stability, digital trust) \
and say plainly that the absence of bureau history is an absence of evidence, \
not evidence of risk.
4. FAIR LENDING. Never use, cite or imply gender, marital status, family \
status, age, ethnicity, religion, disability or nationality as a reason for the \
outcome. Reason only from financial capacity, stability and behaviour.
5. SHAP IS YOUR EVIDENCE. The strengths and risk factors you cite must map to \
the supplied SHAP contributions and behavioural scores. Refer to the direction \
and rough magnitude of each driver.
6. SIMILAR BORROWERS ARE CORROBORATION, NOT PROOF. Cite the retrieved cohort's \
repayment rate and similarity, and be explicit that it is a statistical analogue \
drawn from historical borrowers, not a guarantee.
7. FRAUD SIGNALS ARE SURFACED VERBATIM. If fraud/anomaly flags are present, \
state each one and the verification step that would clear it.
8. TONE. Precise, factual, decision-useful. No marketing language, no hedging \
filler, no apologies. Write for a credit committee that has ninety seconds."""

REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {
            "type": "string",
            "description": "2-3 sentences: who the applicant is, the decision, and the single strongest reason.",
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-6 evidence-backed strengths, each citing a figure from the context.",
        },
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-6 risk factors, each citing a figure from the context.",
        },
        "conditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Conditions/verifications required before disbursal. Empty array if none.",
        },
        "similar_borrower_insight": {
            "type": "string",
            "description": "What the retrieved cohort implies, with its repayment rate and similarity.",
        },
        "detailed_explanation": {
            "type": "string",
            "description": "3-5 paragraphs of underwriting reasoning: capacity, stability, behaviour, "
                           "thin-file treatment, and what would change the decision.",
        },
        "compliance_note": {
            "type": "string",
            "description": "One paragraph on model governance: what drove the score, that no protected "
                           "attribute was used as a reason, and whether human review is required.",
        },
    },
    "required": ["executive_summary", "strengths", "risk_factors", "conditions",
                 "similar_borrower_insight", "detailed_explanation", "compliance_note"],
    "additionalProperties": False,
}

TONE_GUIDANCE = {
    "credit_committee": "Audience: internal credit committee. Dense, technical, figure-led.",
    "risk_memo": "Audience: risk management. Emphasise portfolio impact, tail risk and monitoring triggers.",
    "customer_letter": ("Audience: the applicant. Plain language, no jargon, no internal model "
                        "mechanics, but the same substantive reasons. Adverse-action style where declined."),
}


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOT: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": """CONTEXT
{
  "decision": {"recommendation": "APPROVE", "risk_score": 712, "risk_band": "A2",
               "probability_of_default": 0.031, "recommended_credit_limit": 240000,
               "confidence_score": 0.79, "requires_human_review": false, "is_ntc": true},
  "applicant": {"annual_income": 480000, "credit_requested": 250000, "employment_years": 4.2,
                "occupation": "Core staff", "education": "Higher education", "dti": 0.17},
  "behavioural_scores": {"payment_consistency_score": 78.4, "income_stability_score": 71.2,
                         "monthly_cashflow_consistency": 74.9, "credit_utilization_score": 81.0,
                         "utility_payment_consistency": 83.5, "mobile_recharge_consistency": 88.0,
                         "digital_trust_score": 76.3, "transaction_volatility": 21.5,
                         "financial_discipline_score": 79.1, "thin_file_score": 93.3},
  "shap_risk_reducing": [{"label": "Instalment vs income (DTI)", "value_display": "0.17", "pd_impact_pp": -2.1},
                         {"label": "Mobile recharge consistency", "value_display": "88.00", "pd_impact_pp": -1.4},
                         {"label": "Years in current employment", "value_display": "4.20", "pd_impact_pp": -1.1}],
  "shap_risk_increasing": [{"label": "Missing bureau scores", "value_display": "3.00", "pd_impact_pp": 1.6},
                           {"label": "Age of identity document", "value_display": "1.10", "pd_impact_pp": 0.5}],
  "similar_borrowers": {"cohort_size": 5, "repayment_success_rate": 0.8, "mean_similarity": 0.91},
  "fraud_flags": []
}
Write the underwriting memo.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "executive_summary": (
                "A new-to-credit salaried applicant requesting 250,000 against an annual income of "
                "480,000. Approved at a 240,000 limit with a modelled default probability of 3.1% "
                "(risk score 712, band A2). The decision rests on a 17% debt-to-income ratio and a "
                "consistently strong alternative-data payment record rather than on bureau history."
            ),
            "strengths": [
                "Debt-to-income of 17% leaves substantial repayment headroom; this factor alone reduces modelled PD by 2.1 percentage points.",
                "Mobile recharge consistency of 88.0 and utility payment consistency of 83.5 evidence an uninterrupted record of meeting recurring obligations.",
                "4.2 years of continuous employment with an income-stability score of 71.2 supports a durable income stream.",
                "Transaction volatility of 21.5 is low, indicating predictable cash-flow rather than episodic income.",
                "Credit utilisation headroom of 81.0 shows the requested facility consumes only a modest share of assessed capacity.",
            ],
            "risk_factors": [
                "All three external bureau scores are absent; this is the single largest upward contributor to PD at +1.6 percentage points.",
                "Identity document is 1.1 years old, offering a comparatively short verifiable identity trail (+0.5pp).",
                "As a first facility there is no observed repayment behaviour with this lender to corroborate the behavioural scores.",
                "Cohort corroboration is drawn from five analogues only, so the 80% repayment rate carries wide statistical error.",
            ],
            "conditions": [
                "Verify employment and salary credit through two months of bank statements before disbursal.",
                "Set an initial limit of 240,000 with a scheduled limit review after six on-time instalments.",
            ],
            "similar_borrower_insight": (
                "The five nearest historical borrowers, retrieved at a mean cosine similarity of 0.91, "
                "repaid in 80% of cases. Given the similarity level this cohort is a close analogue on "
                "income, tenure and behavioural profile, but it is a statistical reference class of "
                "historical outcomes and not a prediction for this individual."
            ),
            "detailed_explanation": (
                "Capacity. Annual income of 480,000 against a requested 250,000 gives a credit-to-income "
                "ratio of roughly 0.5x and a debt-to-income ratio of 17%, comfortably inside policy. The "
                "recommended limit of 240,000 sits just below the request and well inside assessed "
                "affordability.\n\n"
                "Stability. Employment of 4.2 years in a core-staff role, combined with an income-stability "
                "score of 71.2, indicates the income is structural rather than transient. Cash-flow "
                "consistency of 74.9 confirms the surplus is durable month to month.\n\n"
                "Behaviour in the absence of bureau data. The applicant is thin-file: no external score is "
                "available on any of three sources. That is an absence of evidence, not evidence of risk. "
                "The alternative-data record is strong and mutually corroborating - utility payments at "
                "83.5, mobile recharge at 88.0 and a digital trust score of 76.3 all point to an applicant "
                "who meets recurring obligations reliably and is stably contactable.\n\n"
                "What would change the decision. A DTI above 35%, a fall in payment consistency below 55, "
                "or the emergence of adverse bureau data would move this application into the review band. "
                "Conversely, six months of on-time instalments would justify a limit increase at review."
            ),
            "compliance_note": (
                "The score was produced by the XGBoost probability-of-default model, and the limit by the "
                "documented affordability policy (DTI cap and risk multiplier). The five factors cited "
                "above are the largest SHAP contributions to this prediction. No protected attribute - "
                "gender, marital or family status, age, ethnicity or nationality - was used as a reason for "
                "this outcome. Model confidence is 0.79, above the 0.70 automatic-decision floor, so no "
                "human review is mandated; the decision, its inputs and this memo are recorded in the audit log."
            ),
        }, indent=None),
    },
]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def build_context(decision: Dict[str, Any], features: Dict[str, Any],
                  explanation: Dict[str, Any], similar: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the grounded, minimal context block handed to the model.

    Deliberately narrow: only decision-relevant figures are exposed, protected
    attributes are omitted entirely so they cannot leak into the rationale.
    """
    beh_keys = ["payment_consistency_score", "spending_stability_score", "income_stability_score",
                "credit_utilization_score", "digital_trust_score", "financial_discipline_score",
                "transaction_volatility", "monthly_cashflow_consistency",
                "utility_payment_consistency", "mobile_recharge_consistency", "thin_file_score"]

    def factors(items: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
        return [{"label": i["label"], "value_display": i["value_display"],
                 "pd_impact_pp": i["pd_impact_pp"]} for i in items[:n]]

    return {
        "decision": {
            "recommendation": decision["recommendation"],
            "risk_score": decision["risk_score"],
            "risk_band": decision["risk_band"],
            "risk_tier": decision["risk_tier"],
            "probability_of_default": decision["probability_of_default"],
            "recommended_credit_limit": decision["recommended_credit_limit"],
            "max_affordable_limit": decision["max_affordable_limit"],
            "suggested_term_months": decision["suggested_term_months"],
            "confidence_score": decision["confidence_score"],
            "confidence_drivers": decision["confidence_drivers"],
            "requires_human_review": decision["requires_human_review"],
            "review_reasons": decision["review_reasons"],
            "is_ntc": decision["is_ntc"],
        },
        "applicant": {
            "annual_income": features.get("AMT_INCOME_TOTAL"),
            "credit_requested": features.get("AMT_CREDIT"),
            "annual_instalment": features.get("AMT_ANNUITY"),
            "goods_price": features.get("AMT_GOODS_PRICE"),
            "employment_years": features.get("employment_years"),
            "occupation": features.get("OCCUPATION_TYPE"),
            "employer_type": features.get("ORGANIZATION_TYPE"),
            "education": features.get("NAME_EDUCATION_TYPE"),
            "income_type": features.get("NAME_INCOME_TYPE"),
            "dti": features.get("annuity_income_ratio"),
            "credit_income_ratio": features.get("credit_income_ratio"),
            "monthly_surplus": features.get("monthly_surplus"),
            "documents_submitted": features.get("document_completeness"),
            "bureau_scores_available": 3 - int(features.get("ext_source_missing") or 0),
        },
        "behavioural_scores": {k: features.get(k) for k in beh_keys},
        "shap_risk_reducing": factors(explanation.get("top_positive_factors", [])),
        "shap_risk_increasing": factors(explanation.get("top_negative_factors", [])),
        "similar_borrowers": {
            **{k: v for k, v in (similar.get("cohort") or {}).items()},
            "examples": [
                {"similarity": b.get("similarity_score"), "outcome": b.get("outcome"),
                 "profile": (b.get("profile_text") or "")[:300]}
                for b in (similar.get("similar_borrowers") or [])[:5]
            ],
        },
        "fraud_flags": decision["fraud_flags"],
    }


def build_user_prompt(context: Dict[str, Any], tone: str = "credit_committee") -> str:
    guidance = TONE_GUIDANCE.get(tone, TONE_GUIDANCE["credit_committee"])
    return (
        "CONTEXT\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        f"{guidance}\n"
        "Write the underwriting memo. Use only figures present in CONTEXT. "
        f"State the recommendation as {context['decision']['recommendation']} and the limit as "
        f"{context['decision']['recommended_credit_limit']:,.0f}; do not propose alternatives."
    )
