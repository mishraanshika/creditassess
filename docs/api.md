# API reference

Base URL `http://localhost:8000` · interactive docs at `/docs`.
Every response carries `X-Request-ID` and `X-Response-Time-ms`.

---

## `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "xgb-pd-1.0.0",
  "vector_backend": "faiss",
  "vector_encoder": "sentence-transformers/all-MiniLM-L6-v2",
  "vector_size": 20000,
  "database": "in-memory",
  "llm": "template-only (no GEMINI_API_KEY configured)",
  "uptime_seconds": 42.1,
  "checks": { "model": "ok", "vector_store": "ok", "database": "degraded …", "llm": "…" }
}
```

`status` is `healthy` when the model is loaded. Vectors, Postgres and the LLM are
optional and reported individually.

---

## `POST /predict`

### Request

```json
{
  "applicant": {
    "external_ref": "NTC-1001",
    "full_name": "Aarti Deshmukh",
    "AMT_INCOME_TOTAL": 540000,
    "AMT_CREDIT": 300000,
    "AMT_ANNUITY": 96000,
    "AMT_GOODS_PRICE": 285000,
    "age_years": 31,
    "employment_years": 5.5,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "OCCUPATION_TYPE": "Core staff",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "FLAG_OWN_REALTY": "Y",
    "CNT_FAM_MEMBERS": 2,
    "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": 1, "FLAG_PHONE": 1, "FLAG_EMAIL": 1,
    "months_on_current_handset": 42,
    "documents_submitted": 4
  },
  "top_k": 5,
  "include_explanation": true,
  "include_similar": true,
  "persist": true
}
```

Only `AMT_INCOME_TOTAL` and `AMT_CREDIT` are required. Bureau fields
(`EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`) are optional — omitting them is how
you represent a genuine New-To-Credit applicant.

Convenience conversions performed server-side: `age_years` →`DAYS_BIRTH`,
`employment_years` → `DAYS_EMPLOYED`, `months_on_current_handset` →
`DAYS_LAST_PHONE_CHANGE`, `documents_submitted` → `FLAG_DOCUMENT_*`.

### Response (abridged)

```json
{
  "request_id": "…", "prediction_id": "…",
  "probability_of_default": 0.0412,
  "risk_score": 686, "risk_band": "B1", "risk_tier": "Near Prime",
  "recommendation": "APPROVE",
  "recommended_credit_limit": 195000,
  "max_affordable_limit": 680000,
  "suggested_term_months": 37,
  "suggested_monthly_instalment": 5270.27,
  "confidence_score": 0.7241,
  "confidence_drivers": { "data_sufficiency": 0.82, "decisiveness": 0.91, "peer_agreement": 0.92 },
  "requires_human_review": false,
  "review_reasons": [], "fraud_flags": [], "is_ntc": true,
  "behavioural_features": { "payment_consistency_score": 78.4, "…": 0 },
  "explanation": {
    "base_probability": 0.0784,
    "top_positive_factors": [
      { "label": "Instalment vs income (DTI)", "value_display": "0.18",
        "shap_value": -0.31, "direction": "reduces_risk", "pd_impact_pp": -2.14 }
    ],
    "top_negative_factors": [ { "label": "Missing bureau scores", "pd_impact_pp": 1.63 } ],
    "contribution_chart": [ "…" ],
    "narrative": "Risk is reduced mainly by …"
  },
  "similar_borrowers": [ { "borrower_id": 100123, "similarity_score": 0.9312,
                           "repaid": true, "outcome": "Repaid", "profile_text": "…" } ],
  "cohort": { "cohort_size": 5, "repayment_success_rate": 0.8,
              "similarity_weighted_repayment_rate": 0.81, "mean_similarity": 0.9104,
              "agreement": 0.6 },
  "model_version": "xgb-pd-1.0.0", "policy_version": "policy-1.2.0", "latency_ms": 118
}
```

**Sign convention.** `top_positive_factors` are good for the applicant (they
*reduce* PD, negative SHAP); `top_negative_factors` increase it.

**Errors.** `503` model not trained · `422` validation · `500` scoring failure
(logged to the audit trail with the payload hash).

---

## `POST /similar-borrowers`

```json
{ "applicant": { "AMT_INCOME_TOTAL": 540000, "AMT_CREDIT": 300000 }, "top_k": 8 }
```

Returns the rendered `query_profile`, the peer list, `cohort` statistics, the
active `backend` and `encoder`, and `latency_ms`. `503` if no index has been
built.

---

## `POST /underwriting-report`

```json
{ "applicant": { "…": "…" }, "top_k": 5, "tone": "credit_committee", "persist": true }
```

`tone`: `credit_committee` (default) · `risk_memo` · `customer_letter`.

Response: `executive_summary`, `strengths[]`, `risk_factors[]`, `conditions[]`,
`similar_borrower_insight`, `detailed_explanation`, `compliance_note`, plus the
complete `decision` object (identical shape to `/predict`) and `generator`
(`gemini:gemini-3.6-flash` or `template`).

---

## Analytics

| Endpoint | Returns |
|---|---|
| `GET /analytics/model-metrics` | holdout + CV metrics, calibration, decision-band performance |
| `GET /analytics/feature-importance?top_k=20` | gain-based global importance with labels |
| `GET /analytics/bias` | fairness audit (`404` until `python -m ml.bias_check` is run) |
| `GET /analytics/policy` | thresholds and risk bands in force |
| `GET /analytics/portfolio` | live decision mix, NTC impact, fraud-flag frequency |
| `GET /analytics/audit-log?limit=50` | recent audit entries |
| `GET /analytics/review-queue` | decisions awaiting a human |

---

## curl quick start

```bash
curl -s localhost:8000/health | jq

curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "applicant": {"AMT_INCOME_TOTAL": 540000, "AMT_CREDIT": 300000,
                "employment_years": 5.5, "months_on_current_handset": 42,
                "documents_submitted": 4, "FLAG_EMAIL": 1}
}' | jq '{risk_score, recommendation, recommended_credit_limit, confidence_score, is_ntc}'
```
