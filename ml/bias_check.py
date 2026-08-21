"""Responsible-AI fairness audit.

Run after training:
    python -m ml.bias_check

Computes, for every protected/sensitive slice:

* selection rate (share recommended APPROVE) and the **disparate impact ratio**
  against the best-performing group - the classic four-fifths (80%) rule;
* **equal-opportunity gap** - difference in true-positive rate (default caught)
  between groups;
* **predictive parity** - group AUC and mean predicted PD vs observed default
  rate, i.e. is the model equally well calibrated for everyone;
* NTC vs thick-file coverage, which is the mission metric for this engine.

The report is written to `ml/artifacts/bias_report.json` and surfaced by
`GET /analytics/bias` so the fairness position is visible in the product, not
buried in a notebook.
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml import config as C
from ml.inference import get_model
from ml.policy import PD_APPROVE_MAX

FOUR_FIFTHS = 0.80
MIN_GROUP_SIZE = 100


def _age_band(age: float) -> str:
    if age < 26:
        return "18-25"
    if age < 36:
        return "26-35"
    if age < 46:
        return "36-45"
    if age < 56:
        return "46-55"
    return "56+"


def _income_band(x: float) -> str:
    if x < 100_000:
        return "<100k"
    if x < 200_000:
        return "100k-200k"
    if x < 350_000:
        return "200k-350k"
    return "350k+"


def group_metrics(df: pd.DataFrame, col: str) -> List[Dict]:
    rows: List[Dict] = []
    for value, g in df.groupby(col, observed=True):
        if len(g) < MIN_GROUP_SIZE:
            continue
        approved = (g["pd"] <= PD_APPROVE_MAX)
        defaults = g["target"] == 1
        try:
            auc = float(roc_auc_score(g["target"], g["pd"])) if g["target"].nunique() > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        tpr = float((g.loc[defaults, "pd"] > PD_APPROVE_MAX).mean()) if defaults.any() else float("nan")
        fpr = float((g.loc[~defaults, "pd"] > PD_APPROVE_MAX).mean()) if (~defaults).any() else float("nan")
        rows.append({
            "group": str(value),
            "n": int(len(g)),
            "selection_rate": round(float(approved.mean()), 4),
            "observed_default_rate": round(float(defaults.mean()), 4),
            "mean_predicted_pd": round(float(g["pd"].mean()), 4),
            "calibration_gap": round(float(g["pd"].mean() - defaults.mean()), 4),
            "roc_auc": None if np.isnan(auc) else round(auc, 4),
            "true_positive_rate": None if np.isnan(tpr) else round(tpr, 4),
            "false_positive_rate": None if np.isnan(fpr) else round(fpr, 4),
        })
    if not rows:
        return rows

    best = max(r["selection_rate"] for r in rows) or 1e-9
    tprs = [r["true_positive_rate"] for r in rows if r["true_positive_rate"] is not None]
    for r in rows:
        r["disparate_impact_ratio"] = round(r["selection_rate"] / best, 4)
        r["passes_four_fifths"] = bool(r["disparate_impact_ratio"] >= FOUR_FIFTHS)
    if tprs:
        gap = max(tprs) - min(tprs)
        for r in rows:
            r["equal_opportunity_gap"] = round(float(gap), 4)
    return rows


def run(sample_rows: int = 80_000) -> Dict:
    model = get_model()
    df = pd.read_csv(C.TRAIN_CSV, nrows=sample_rows)
    X, enriched = model.frame_from_dataframe(df)
    proba = model.predict_proba(X)

    audit = pd.DataFrame({
        "pd": proba,
        "target": df[C.TARGET].astype(int).values,
        "gender": enriched["CODE_GENDER"].values,
        "education": enriched["NAME_EDUCATION_TYPE"].values,
        "family_status": enriched["NAME_FAMILY_STATUS"].values,
        "age_band": [_age_band(a) for a in enriched["age_years"].values],
        "income_band": [_income_band(v) for v in enriched["AMT_INCOME_TOTAL"].values],
        "file_type": np.where(enriched["is_ntc"].values >= 1, "new_to_credit", "thick_file"),
    })

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "model_version": model.model_version,
        "n_audited": int(len(audit)),
        "approval_threshold_pd": PD_APPROVE_MAX,
        "four_fifths_threshold": FOUR_FIFTHS,
        "slices": {
            col: group_metrics(audit, col)
            for col in ["gender", "age_band", "education", "family_status",
                        "income_band", "file_type"]
        },
    }

    failures = [
        f"{slice_name}:{r['group']} (DI={r['disparate_impact_ratio']})"
        for slice_name, rows in report["slices"].items()
        for r in rows if not r.get("passes_four_fifths", True)
    ]
    report["four_fifths_failures"] = failures
    report["overall_pass"] = not failures

    # Mission metric: are we actually serving the thin-file population?
    ntc = audit[audit["file_type"] == "new_to_credit"]
    report["ntc_coverage"] = {
        "ntc_share_of_applicants": round(float((audit["file_type"] == "new_to_credit").mean()), 4),
        "ntc_selection_rate": round(float((ntc["pd"] <= PD_APPROVE_MAX).mean()), 4) if len(ntc) else None,
        "ntc_observed_default_rate": round(float(ntc["target"].mean()), 4) if len(ntc) else None,
    }

    C.BIAS_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("n_audited", "overall_pass", "four_fifths_failures", "ntc_coverage")},
                     indent=2))
    print(f"[bias] full report -> {C.BIAS_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
