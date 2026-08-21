"""End-to-end smoke test / demo driver.

Runs three representative applicants through the live API and prints the
decision, the SHAP drivers, the peer cohort and the underwriting memo.

    python scripts/demo.py                       # against localhost:8000
    python scripts/demo.py --base http://api:8000
    python scripts/demo.py --no-report           # skip the LLM memo
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

APPLICANTS = {
    "NTC · strong behaviour": {
        "external_ref": "NTC-1001", "full_name": "Aarti Deshmukh",
        "AMT_INCOME_TOTAL": 540000, "AMT_CREDIT": 300000,
        "AMT_ANNUITY": 96000, "AMT_GOODS_PRICE": 285000,
        "age_years": 31, "employment_years": 5.5,
        "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Higher education",
        "OCCUPATION_TYPE": "Core staff", "ORGANIZATION_TYPE": "Business Entity Type 3",
        "FLAG_OWN_REALTY": "Y", "CNT_FAM_MEMBERS": 2,
        "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": 1, "FLAG_WORK_PHONE": 1,
        "FLAG_CONT_MOBILE": 1, "FLAG_PHONE": 1, "FLAG_EMAIL": 1,
        "months_on_current_handset": 42, "documents_submitted": 4,
    },
    "Thin file · stretched": {
        "external_ref": "NTC-1002", "full_name": "Rohit Nair",
        "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 900000,
        "AMT_ANNUITY": 78000, "AMT_GOODS_PRICE": 500000,
        "age_years": 24, "employment_years": 0.4,
        "NAME_INCOME_TYPE": "Working", "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Self-employed", "NAME_HOUSING_TYPE": "Rented apartment",
        "FLAG_OWN_REALTY": "N", "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3,
        "FLAG_MOBIL": 1, "FLAG_CONT_MOBILE": 1,
        "months_on_current_handset": 1, "documents_submitted": 0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 3, "AMT_REQ_CREDIT_BUREAU_YEAR": 9,
    },
    "Established file": {
        "external_ref": "STD-2001", "full_name": "Priya Raghavan",
        "AMT_INCOME_TOTAL": 810000, "AMT_CREDIT": 640000,
        "AMT_ANNUITY": 120000, "AMT_GOODS_PRICE": 640000,
        "age_years": 42, "employment_years": 12,
        "NAME_INCOME_TYPE": "State servant", "NAME_EDUCATION_TYPE": "Higher education",
        "OCCUPATION_TYPE": "Managers", "ORGANIZATION_TYPE": "School",
        "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 2, "CNT_FAM_MEMBERS": 4,
        "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": 1, "FLAG_PHONE": 1, "FLAG_EMAIL": 1,
        "months_on_current_handset": 30, "documents_submitted": 5,
        "EXT_SOURCE_2": 0.71, "EXT_SOURCE_3": 0.66,
    },
}

RULE = "=" * 78


def call(base: str, path: str, payload=None):
    url = f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return json.loads(res.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{path} failed [{exc.code}]: {exc.read().decode()[:400]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {url} ({exc.reason}). "
                         "Start the API with `python -m backend.run`.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    health = call(args.base, "/health")
    print(RULE)
    print(f"HEALTH  {health['status']}  |  model {health.get('model_version')}  |  "
          f"vectors {health.get('vector_backend')} ({health.get('vector_size')})  |  "
          f"db {health.get('database')}")
    print(f"        llm: {health.get('llm')}")
    if not health.get("model_loaded"):
        raise SystemExit("Model not loaded. Run `python -m ml.train` first.")

    for name, applicant in APPLICANTS.items():
        print("\n" + RULE)
        print(f"APPLICANT · {name}")
        print(RULE)

        r = call(args.base, "/predict", {"applicant": applicant, "top_k": 5})
        print(f"  decision        {r['recommendation']}   "
              f"(score {r['risk_score']}, band {r['risk_band']} / {r['risk_tier']})")
        print(f"  PD              {r['probability_of_default'] * 100:.2f}%")
        print(f"  limit           {r['recommended_credit_limit']:,.0f} "
              f"(requested {r['requested_amount']:,.0f}, "
              f"capacity {r['max_affordable_limit']:,.0f})")
        print(f"  confidence      {r['confidence_score']:.2f}  {r['confidence_drivers']}")
        print(f"  new to credit   {r['is_ntc']}")
        if r["review_reasons"]:
            print("  review triggers")
            for reason in r["review_reasons"]:
                print(f"     - {reason}")
        if r["fraud_flags"]:
            print("  fraud signals")
            for flag in r["fraud_flags"]:
                print(f"     ! {flag}")

        beh = r["behavioural_features"]
        print("  behavioural     " + "  ".join(
            f"{k.replace('_score', '').replace('_', ' ')}={v:.0f}"
            for k, v in list(beh.items())[:6]))

        print("  reduces risk")
        for f in r["explanation"]["top_positive_factors"][:3]:
            print(f"     - {f['label']:<38} {f['value_display']:>14}  {f['pd_impact_pp']:+.2f} pp")
        print("  increases risk")
        for f in r["explanation"]["top_negative_factors"][:3]:
            print(f"     - {f['label']:<38} {f['value_display']:>14}  {f['pd_impact_pp']:+.2f} pp")

        cohort = r.get("cohort") or {}
        if cohort.get("cohort_size"):
            print(f"  peers           {cohort['cohort_size']} borrowers, "
                  f"repayment {cohort['repayment_success_rate'] * 100:.0f}%, "
                  f"mean similarity {cohort['mean_similarity']:.3f}")
            for b in r["similar_borrowers"][:3]:
                print(f"     #{b['borrower_id']}  sim {b['similarity_score']:.3f}  {b['outcome']}")
        print(f"  latency         {r['latency_ms']} ms")

        if not args.no_report:
            rep = call(args.base, "/underwriting-report",
                       {"applicant": applicant, "top_k": 5})
            print(f"\n  MEMO ({rep['generator']}, {rep['latency_ms']} ms)")
            print(f"  {rep['executive_summary']}")
            print("  strengths")
            for s in rep["strengths"][:3]:
                print(f"     + {s}")
            print("  risks")
            for s in rep["risk_factors"][:3]:
                print(f"     - {s}")
            if rep["conditions"]:
                print("  conditions")
                for s in rep["conditions"]:
                    print(f"     * {s}")

    print("\n" + RULE)
    portfolio = call(args.base, "/analytics/portfolio")
    print(f"PORTFOLIO  {portfolio['decisions']} decisions  |  mix {portfolio['recommendation_mix']}"
          f"  |  review queue {portfolio['review_queue']}")
    print(RULE)


if __name__ == "__main__":
    sys.exit(main())
