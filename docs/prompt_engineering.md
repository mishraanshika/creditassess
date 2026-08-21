# Prompt engineering — the LLM underwriter

Implementation: `backend/app/services/prompts.py` (templates) and
`backend/app/services/llm.py` (invocation + fallback).
Prompt version: `underwriter-prompt-2.1`.

## 1. The governing design decision

**The LLM does not make the credit decision.** XGBoost produces the probability
of default; the policy engine produces APPROVE / REVIEW / REJECT and the limit.
The model writes the memo that explains and stress-tests that outcome.

That split is not stylistic. A model that can restate the decision in its own
words can also contradict it, and a lender cannot defend a decision whose stated
reason differs from the mechanism that produced it. So the prompt:

1. states the decision as a fact in the context block,
2. instructs the model never to propose an alternative,
3. repeats the recommendation and the limit in the final instruction line,
4. constrains the output to a JSON schema with no "recommendation" field the
   model could fill in differently.

## 2. Prompt anatomy

| Part | Content | Purpose |
|---|---|---|
| System | 8 numbered operating rules | Role, grounding, fair-lending, NTC framing, evidence discipline |
| Few-shot | 1 worked example (thin-file APPROVE) | Fixes tone, depth, house style and the exact JSON shape |
| Context | Curated JSON: decision, applicant, behavioural scores, SHAP factors, cohort, fraud flags | The only facts the model may cite |
| Instruction | Tone selector + restated decision | Locks the outcome, selects the audience |
| Output config | `json_schema` with 7 required fields | Contract stability |

### System-prompt rules (abridged)

1. **The decision is already made.** Explain, justify, stress-test. Never overturn.
2. **Ground every claim.** Only figures present in CONTEXT. Missing data is stated as missing, never estimated.
3. **New-to-credit is not bad credit.** Absence of bureau history is an absence of evidence, not evidence of risk.
4. **Fair lending.** Never use or imply gender, marital or family status, age, ethnicity, religion, disability or nationality as a reason.
5. **SHAP is the evidence.** Strengths and risks must map to supplied contributions.
6. **Similar borrowers corroborate, they do not prove.** Cite the rate and the similarity, label it a statistical reference class.
7. **Fraud signals verbatim** plus the verification that would clear each one.
8. **Tone.** Precise, factual, decision-useful; a credit committee has ninety seconds.

## 3. Context construction (`build_context`)

Deliberately narrow. Protected attributes are **omitted from the context
entirely** — the model cannot cite what it never receives. This is stronger
than instructing it not to.

```json
{
  "decision":  { "recommendation", "risk_score", "risk_band", "probability_of_default",
                 "recommended_credit_limit", "max_affordable_limit", "confidence_score",
                 "confidence_drivers", "requires_human_review", "review_reasons", "is_ntc" },
  "applicant": { "annual_income", "credit_requested", "annual_instalment", "employment_years",
                 "occupation", "employer_type", "education", "dti", "credit_income_ratio",
                 "monthly_surplus", "documents_submitted", "bureau_scores_available" },
  "behavioural_scores":   { "...11 scores, 0-100..." },
  "shap_risk_reducing":   [{ "label", "value_display", "pd_impact_pp" }],
  "shap_risk_increasing": [{ "label", "value_display", "pd_impact_pp" }],
  "similar_borrowers":    { "cohort_size", "repayment_success_rate", "mean_similarity", "examples" },
  "fraud_flags":          ["..."]
}
```

Note what is absent: `CODE_GENDER`, `NAME_FAMILY_STATUS`, `age_years`, and every
raw `DAYS_*` column from which age could be recovered.

## 4. Output schema

```json
{
  "executive_summary":       "2-3 sentences: who, what decision, strongest single reason",
  "strengths":               ["3-6 items, each citing a figure from CONTEXT"],
  "risk_factors":            ["3-6 items, each citing a figure from CONTEXT"],
  "conditions":              ["verifications required before disbursal"],
  "similar_borrower_insight":"what the cohort implies, with rate and similarity",
  "detailed_explanation":    "3-5 paragraphs: capacity, stability, behaviour, thin-file treatment, sensitivity",
  "compliance_note":         "governance: drivers, no protected attributes, review status"
}
```

Passed as `response_mime_type="application/json"` plus `response_schema` on
`GenerateContentConfig`, so the response is parseable without defensive string
handling. `additionalProperties` is stripped first - it is valid JSON Schema but
rejected by the Gemini OpenAPI subset.

## 5. Tone variants

| Tone | Audience | Shift |
|---|---|---|
| `credit_committee` (default) | Internal committee | Dense, technical, figure-led |
| `risk_memo` | Risk management | Portfolio impact, tail risk, monitoring triggers |
| `customer_letter` | The applicant | Plain language, no model mechanics, adverse-action style when declined |

## 6. Worked example (few-shot, abridged)

**Input context:** APPROVE, PD 3.1%, score 712, limit 240,000, thin file, DTI 0.17,
mobile recharge consistency 88.0, cohort 5 peers at 0.91 similarity / 80% repayment.

**Output (excerpt):**

> **Executive summary.** A new-to-credit salaried applicant requesting 250,000 against an
> annual income of 480,000. Approved at a 240,000 limit with a modelled default probability of
> 3.1% (risk score 712, band A2). The decision rests on a 17% debt-to-income ratio and a
> consistently strong alternative-data payment record rather than on bureau history.
>
> **Risk factor.** All three external bureau scores are absent; this is the single largest
> upward contributor to PD at +1.6 percentage points.
>
> **Compliance note.** …No protected attribute — gender, marital or family status, age,
> ethnicity or nationality — was used as a reason for this outcome. Model confidence is 0.79,
> above the 0.70 automatic-decision floor…

Every figure in that output appears in the context block. Nothing is invented.

## 7. Failure design

If `GEMINI_API_KEY` is unset, or the call fails or times out, `llm.py` falls
back to `_template()` — a deterministic memo assembled from the *same* SHAP
factors, behavioural scores and cohort statistics. It is:

* **deterministic** — identical inputs always produce identical text, which is
  what makes it safe to store in an audit trail;
* **complete** — the same seven sections;
* **honest** — it never claims to be model-written; `generator` is `"template"`
  and the UI displays which produced the memo.

Explainability is a compliance obligation, so it cannot depend on a third-party
credential being present.

## 8. Anti-patterns avoided

| Anti-pattern | Why it is wrong here | What we do |
|---|---|---|
| "Decide whether to approve" | LLM becomes the model of record; not validatable, not reproducible | Decision supplied as fact |
| Dumping the full feature row | Invites cherry-picking and leaks protected attributes | Curated context |
| Free-text output parsed with regex | Breaks silently on phrasing drift | JSON schema |
| "Be encouraging / be strict" | Injects a tone bias into a regulated document | Fixed tone rules per audience |
| Asking the model to compute figures | Arithmetic hallucination | Every number pre-computed and supplied |
