"""LLM underwriter service (Google Gemini).

Calls Gemini with a constrained JSON schema so the API contract never depends on
prose parsing.  When no API key is configured - or the call fails - the service
falls back to a deterministic template report built from the same SHAP and
cohort evidence, so the product never loses its explanation layer because of a
missing credential or a network blip.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from app.core.config import settings
from ml.policy import MIN_AUTO_CONFIDENCE
from app.services.prompts import (
    FEW_SHOT,
    PROMPT_VERSION,
    REPORT_SCHEMA,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)

logger = logging.getLogger(__name__)


def _gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip JSON-Schema keywords that Gemini's OpenAPI subset rejects.

    `additionalProperties` is the one that matters: valid JSON Schema, required
    by some providers, refused by the Gemini schema dialect.
    """
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if isinstance(value, dict):
            out[key] = _gemini_schema(value)
        elif isinstance(value, list):
            out[key] = [_gemini_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


class LLMUnderwriter:
    def __init__(self) -> None:
        self.client = None
        self.enabled = False
        self.model = settings.LLM_MODEL
        self.status = "template-only (no GEMINI_API_KEY configured)"

    def initialise(self) -> None:
        if not settings.GEMINI_API_KEY:
            logger.info("LLM disabled: %s", self.status)
            return
        try:
            from google import genai

            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.enabled = True
            self.status = f"gemini:{self.model}"
            logger.info("LLM underwriter ready (%s)", self.status)
        except Exception as exc:  # noqa: BLE001
            self.status = f"template-only (init failed: {exc})"
            logger.warning(self.status)

    # -- public API ---------------------------------------------------------
    def generate(self, decision: Dict[str, Any], features: Dict[str, Any],
                 explanation: Dict[str, Any], similar: Dict[str, Any],
                 tone: str = "credit_committee") -> Dict[str, Any]:
        context = build_context(decision, features, explanation, similar)
        t0 = time.perf_counter()

        if self.enabled and self.client is not None:
            try:
                body = self._call_gemini(context, tone)
                body["generator"] = f"gemini:{self.model}"
                body["prompt_version"] = PROMPT_VERSION
                body["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                return body
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM call failed (%s); using template fallback", exc)

        body = self._template(decision, features, explanation, similar, tone)
        body["generator"] = "template"
        body["prompt_version"] = PROMPT_VERSION
        body["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return body

    # -- Gemini -------------------------------------------------------------
    def _call_gemini(self, context: Dict[str, Any], tone: str) -> Dict[str, Any]:
        from google.genai import types

        # Gemini names the assistant role "model". The few-shot pair is replayed
        # as prior turns so tone, depth and JSON shape are anchored.
        contents = []
        for msg in FEW_SHOT:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role,
                                          parts=[types.Part(text=msg["content"])]))
        contents.append(types.Content(
            role="user", parts=[types.Part(text=build_user_prompt(context, tone))]))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
                response_mime_type="application/json",
                response_schema=_gemini_schema(REPORT_SCHEMA),
                thinking_config=types.ThinkingConfig(
                    thinking_level=settings.LLM_THINKING_LEVEL),
            ),
        )

        finish = None
        if response.candidates:
            finish = getattr(response.candidates[0], "finish_reason", None)
        if finish is not None and str(finish).endswith("MAX_TOKENS"):
            raise RuntimeError("response truncated at max_output_tokens "
                               f"({settings.LLM_MAX_TOKENS}); raise LLM_MAX_TOKENS")

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("empty response from Gemini")
        data = json.loads(text)

        usage = getattr(response, "usage_metadata", None)
        data["prompt_tokens"] = getattr(usage, "prompt_token_count", None)
        data["completion_tokens"] = getattr(usage, "candidates_token_count", None)
        return data

    # -- deterministic fallback --------------------------------------------
    @staticmethod
    def _template(decision: Dict[str, Any], features: Dict[str, Any],
                  explanation: Dict[str, Any], similar: Dict[str, Any],
                  tone: str) -> Dict[str, Any]:
        """Rule-based memo assembled from the same evidence the LLM receives.

        Deterministic by design: identical inputs always produce an identical
        memo, which makes it safe to show in an audit trail.
        """
        reco = decision["recommendation"]
        pd_pct = decision["probability_of_default"] * 100
        limit = decision["recommended_credit_limit"]
        cohort = similar.get("cohort") or {}
        pos: List[Dict] = explanation.get("top_positive_factors", [])
        neg: List[Dict] = explanation.get("top_negative_factors", [])
        ntc = decision["is_ntc"]

        income = float(features.get("AMT_INCOME_TOTAL") or 0)
        credit = float(features.get("AMT_CREDIT") or 0)
        dti = float(features.get("annuity_income_ratio") or 0)
        emp = float(features.get("employment_years") or 0)

        verb = {"APPROVE": "approved", "REVIEW": "referred for manual review",
                "REJECT": "declined"}[reco]

        executive = (
            f"Applicant requesting {credit:,.0f} against declared annual income of {income:,.0f} "
            f"has been {verb}. Modelled probability of default is {pd_pct:.2f}% "
            f"(risk score {decision['risk_score']}, band {decision['risk_band']}), with a "
            f"recommended limit of {limit:,.0f} and engine confidence of "
            f"{decision['confidence_score']:.2f}."
        )
        if ntc:
            executive += (" The applicant is new-to-credit; the assessment is driven by "
                          "behavioural and alternative-data evidence rather than bureau history.")

        strengths = [
            f"{f['label']} at {f['value_display']} reduces modelled default probability by "
            f"{abs(f['pd_impact_pp']):.2f} percentage points."
            for f in pos[:5]
        ]
        if dti and dti < 0.35:
            strengths.append(f"Debt-to-income of {dti * 100:.0f}% leaves repayment headroom "
                             f"within the {35 if ntc else 45}% policy cap.")
        if emp >= 2:
            strengths.append(f"{emp:.1f} years of continuous employment supports income durability.")

        risks = [
            f"{f['label']} at {f['value_display']} increases modelled default probability by "
            f"{f['pd_impact_pp']:.2f} percentage points."
            for f in neg[:5]
        ]
        risks.extend(decision["fraud_flags"])
        if ntc:
            risks.append("No external bureau score is available, so the assessment cannot be "
                         "corroborated against a traditional credit file.")

        conditions: List[str] = []
        if decision["requires_human_review"]:
            conditions.append("Manual underwriter review is required before any offer is issued: "
                              + "; ".join(decision["review_reasons"]) + ".")
        if decision["fraud_flags"]:
            conditions.append("Complete identity and document verification to clear the "
                              f"{len(decision['fraud_flags'])} anomaly signal(s) raised.")
        if reco != "REJECT":
            conditions.append("Verify declared income against two months of bank credits before disbursal.")
            if ntc:
                conditions.append("Open at the recommended limit and review for an increase after "
                                  "six consecutive on-time instalments.")

        rate = cohort.get("repayment_success_rate")
        if rate is None:
            cohort_text = ("No comparable borrower cohort was retrieved, so no peer corroboration "
                           "is available for this assessment.")
        else:
            cohort_text = (
                f"The {cohort.get('cohort_size')} nearest historical borrowers, retrieved at a mean "
                f"similarity of {cohort.get('mean_similarity'):.2f}, repaid in {rate * 100:.0f}% of "
                f"cases (similarity-weighted {cohort.get('similarity_weighted_repayment_rate', rate) * 100:.0f}%). "
                "This is a statistical reference class of historical outcomes, not a prediction for "
                "this individual."
            )

        detail = "\n\n".join([
            f"Capacity. Declared annual income of {income:,.0f} against a requested "
            f"{credit:,.0f} gives a credit-to-income ratio of "
            f"{float(features.get('credit_income_ratio') or 0):.2f}x and a debt-to-income ratio of "
            f"{dti * 100:.0f}%. Assessed affordable capacity is "
            f"{decision['max_affordable_limit']:,.0f} over {decision['suggested_term_months']} months.",

            f"Behaviour. Financial discipline scores "
            f"{float(features.get('financial_discipline_score') or 0):.1f}/100, payment consistency "
            f"{float(features.get('payment_consistency_score') or 0):.1f}, cash-flow consistency "
            f"{float(features.get('monthly_cashflow_consistency') or 0):.1f} and transaction "
            f"volatility {float(features.get('transaction_volatility') or 0):.1f} "
            "(lower is better). Utility payment consistency of "
            f"{float(features.get('utility_payment_consistency') or 0):.1f} and mobile recharge "
            f"consistency of {float(features.get('mobile_recharge_consistency') or 0):.1f} evidence "
            "the applicant's record on recurring obligations.",

            explanation.get("narrative", ""),

            cohort_text,

            f"Sensitivity. The decision boundary sits at a default probability of 6% for automatic "
            f"approval and 17% for decline; this applicant is at {pd_pct:.2f}%. A material change in "
            "debt-to-income, employment tenure or payment consistency would move the outcome.",
        ])

        compliance = (
            f"Score produced by model {decision.get('model_version', 'xgb-pd')} under policy "
            f"{decision['policy_version']}. The factors cited are the largest SHAP contributions to "
            "this prediction; no protected attribute (gender, marital or family status, age, "
            "ethnicity, nationality) was used as a reason for this outcome. Engine confidence is "
            f"{decision['confidence_score']:.2f} against an automatic-decision floor of "
            f"{MIN_AUTO_CONFIDENCE:.2f}, and "
            f"human review is {'required' if decision['requires_human_review'] else 'not required'}. "
            "The request, decision, feature snapshot and this memo are written to the audit log."
        )

        return {
            "executive_summary": executive,
            "strengths": strengths[:6],
            "risk_factors": risks[:6],
            "conditions": conditions,
            "similar_borrower_insight": cohort_text,
            "detailed_explanation": detail,
            "compliance_note": compliance,
            "prompt_tokens": None,
            "completion_tokens": None,
        }


llm_underwriter = LLMUnderwriter()
