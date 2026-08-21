"""SHAP explainability layer.

Two levels of explanation are produced:

* **Local** - per-applicant TreeSHAP contributions in log-odds space, split into
  risk-increasing and risk-reducing factors, converted into a plain-English
  sentence and a signed contribution chart for the UI.
* **Global** - gain-based importance from training plus an optional
  `shap.summary_plot` rendered to `docs/img/` for the pitch deck.

Contributions are additive in log-odds:  logit(PD) = base_value + sum(phi_i).
We also report each factor's marginal effect on PD in percentage points, which
is what a credit officer actually reads.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.inference import CreditModel

# Human-facing labels. Anything not listed falls back to a prettified column name.
FEATURE_LABELS: Dict[str, str] = {
    "payment_consistency_score": "Payment consistency",
    "spending_stability_score": "Spending stability",
    "income_stability_score": "Income stability",
    "credit_utilization_score": "Credit utilisation headroom",
    "digital_trust_score": "Digital trust footprint",
    "financial_discipline_score": "Financial discipline",
    "transaction_volatility": "Transaction volatility",
    "monthly_cashflow_consistency": "Monthly cash-flow consistency",
    "utility_payment_consistency": "Utility payment consistency",
    "mobile_recharge_consistency": "Mobile recharge consistency",
    "thin_file_score": "Thin-file indicator",
    "is_ntc": "New-to-credit applicant",
    "credit_income_ratio": "Loan amount vs annual income",
    "annuity_income_ratio": "Instalment vs income (DTI)",
    "credit_term_months": "Implied loan term",
    "goods_credit_ratio": "Financed goods vs credit requested",
    "income_per_person": "Income per household member",
    "employment_years": "Years in current employment",
    "employment_ratio": "Share of life spent employed",
    "age_years": "Age",
    "ext_source_mean": "External bureau score (average)",
    "ext_source_min": "External bureau score (weakest)",
    "ext_source_missing": "Missing bureau scores",
    "EXT_SOURCE_1": "External bureau score 1",
    "EXT_SOURCE_2": "External bureau score 2",
    "EXT_SOURCE_3": "External bureau score 3",
    "document_completeness": "Supporting documents submitted",
    "social_default_ratio": "Defaults within social circle",
    "monthly_surplus": "Monthly surplus after instalment",
    "surplus_ratio": "Surplus as share of income",
    "address_mismatch_count": "Address inconsistencies",
    "contactability_flags": "Reachable contact channels",
    "phone_stability_years": "Years on the same handset",
    "id_freshness_years": "Age of identity document",
    "registration_age_years": "Length of registration history",
    "bureau_enquiry_intensity": "Recent credit enquiry intensity",
    "AMT_INCOME_TOTAL": "Declared annual income",
    "AMT_CREDIT": "Credit amount requested",
    "AMT_ANNUITY": "Annual instalment",
    "AMT_GOODS_PRICE": "Price of financed goods",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_INCOME_TYPE": "Income type",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer type",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing situation",
    "CODE_GENDER": "Gender",
    "REGION_RATING_CLIENT": "Region risk rating",
    "DAYS_EMPLOYED": "Employment tenure (days)",
    "DAYS_BIRTH": "Age (days)",
}


def label_for(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("FLAG_", "").replace("_", " ").strip().capitalize()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _fmt_value(feature: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not provided"
    if isinstance(value, str):
        return value
    v = float(value)
    if feature.startswith("AMT_") or feature in ("income_per_person", "monthly_surplus"):
        return f"{v:,.0f}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def explain_prediction(
    model: CreditModel,
    X: pd.DataFrame,
    enriched: pd.DataFrame,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Local TreeSHAP explanation for a single applicant."""
    contribs, base = model.shap_contributions(X)

    def calibrate(raw_p: float) -> float:
        if getattr(model, "calibrator", None) is None:
            return raw_p
        return float(np.clip(model.calibrator.predict([raw_p])[0], 1e-6, 1 - 1e-6))

    phi = contribs[0]
    features = list(X.columns)
    values = X.iloc[0]
    raw_values = enriched.iloc[0]

    total = float(base + phi.sum())
    raw_full = _sigmoid(total)
    raw_base = _sigmoid(float(base))
    pd_full = calibrate(raw_full)
    pd_base = calibrate(raw_base)

    # Per-factor impact.
    #
    # TreeSHAP is additive in the booster's raw log-odds, so marginal effects are
    # computed there. They are then rescaled onto the calibrated PD scale by the
    # single factor `k`, so that the reported percentage points sum to the actual
    # calibrated distance from the population base rate. Applying the isotonic
    # calibrator to each factor individually would be wrong: isotonic regression
    # is a step function, and small per-factor shifts collapse onto the same step,
    # reporting several distinct drivers as having identical impact.
    raw_span = raw_full - raw_base
    cal_span = pd_full - pd_base
    k = (cal_span / raw_span) if abs(raw_span) > 1e-9 else 1.0

    rows: List[Dict[str, Any]] = []
    for i, feat in enumerate(features):
        contribution = float(phi[i])
        if contribution == 0.0:
            continue
        # Marginal PD effect: PD with the factor vs PD if the factor were neutral.
        raw_without = _sigmoid(total - contribution)
        raw = raw_values.get(feat, values.get(feat))
        rows.append({
            "feature": feat,
            "label": label_for(feat),
            "value": (None if isinstance(raw, float) and np.isnan(raw)
                      else (float(raw) if isinstance(raw, (int, float, np.floating)) else str(raw))),
            "value_display": _fmt_value(feat, raw),
            "shap_value": round(contribution, 5),
            "direction": "increases_risk" if contribution > 0 else "reduces_risk",
            "pd_impact_pp": round((raw_full - raw_without) * k * 100, 3),
        })

    rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    risk_up = [r for r in rows if r["shap_value"] > 0][:top_k]
    risk_down = [r for r in rows if r["shap_value"] < 0][:top_k]

    return {
        "base_value_logodds": round(float(base), 5),
        "base_probability": round(pd_base, 6),
        "prediction_logodds": round(total, 5),
        "probability_of_default": round(pd_full, 6),
        # "negative factors" = things pushing risk UP (bad for the applicant)
        "top_negative_factors": risk_up,
        # "positive factors" = things pulling risk DOWN (good for the applicant)
        "top_positive_factors": risk_down,
        "contribution_chart": rows[: top_k * 2],
        "narrative": build_narrative(risk_down, risk_up),
    }


def build_narrative(positives: List[Dict], negatives: List[Dict]) -> str:
    """Deterministic plain-English summary - the fallback when no LLM is wired."""
    def phrase(items: List[Dict], n: int = 3) -> str:
        parts = [f"{i['label']} ({i['value_display']})" for i in items[:n]]
        if not parts:
            return "no material factors"
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    return (
        f"Risk is reduced mainly by {phrase(positives)}. "
        f"Risk is increased mainly by {phrase(negatives)}."
    )


def global_importance(model: CreditModel, top_k: int = 20) -> List[Dict[str, Any]]:
    """Gain-based global importance, labelled for the analytics dashboard."""
    imp = model.feature_importance
    if imp.empty:
        return []
    col = "gain_pct" if "gain_pct" in imp.columns else "gain"
    out = []
    for _, r in imp.sort_values(col, ascending=False).head(top_k).iterrows():
        out.append({
            "feature": r["feature"],
            "label": label_for(str(r["feature"])),
            "gain": round(float(r.get("gain", 0.0)), 3),
            "gain_pct": round(float(r.get("gain_pct", 0.0)), 3),
        })
    return out


def render_global_shap_plot(model: CreditModel, out_path: str,
                            sample: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Render a `shap.summary_plot` beeswarm for the docs/pitch deck.

    Optional: requires `shap` and `matplotlib`.  Never called from the request
    path - run it once offline via `python -m ml.explain`.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        print("[shap] shap/matplotlib not installed - skipping plot")
        return None

    from ml import config as C
    if sample is None:
        bg_path = C.ARTIFACT_DIR / "shap_background.parquet"
        if not bg_path.exists():
            print("[shap] no background sample found - run ml.train first")
            return None
        sample = pd.read_parquet(bg_path)

    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(sample)
    labels = [label_for(c) for c in sample.columns]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(values, sample, feature_names=labels, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"[shap] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    from pathlib import Path

    from ml.inference import get_model

    m = get_model()
    out = Path(__file__).resolve().parents[1] / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    render_global_shap_plot(m, str(out / "shap_summary.png"))
