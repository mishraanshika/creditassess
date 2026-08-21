# Architecture

## 1. System overview

```mermaid
flowchart TB
    subgraph CLIENT["Client · React 18 + Vite"]
        UI1[Dashboard]
        UI2[Applicant Intake]
        UI3[Risk Assessment]
        UI4[Similar Borrowers]
        UI5[AI Underwriting Report]
        UI6[Analytics & Fairness]
    end

    subgraph EDGE["Edge"]
        CORS[CORS + correlation id<br/>X-Request-ID · X-Response-Time]
    end

    subgraph API["FastAPI · backend/app"]
        R1["POST /predict"]
        R2["POST /similar-borrowers"]
        R3["POST /underwriting-report"]
        R4["GET /health"]
        R5["GET /analytics/*"]
        SVC["ScoringService<br/>single decision path"]
    end

    subgraph ML["ML runtime · ml/"]
        FE["Feature engineering<br/>8 behavioural scores"]
        XGB["XGBoost booster<br/>binary:logistic"]
        CAL["Isotonic calibrator<br/>PD = real probability"]
        POL["Policy engine<br/>score · limit · confidence · review"]
        SHAP["TreeSHAP<br/>local + global attribution"]
    end

    subgraph VEC["Vector layer · embeddings/"]
        PROF["Profile renderer<br/>row to natural language"]
        ENC["all-MiniLM-L6-v2<br/>384-dim, cosine"]
        IDX[("FAISS IndexFlatIP<br/>or pgvector ivfflat")]
    end

    subgraph LLM["LLM underwriter"]
        PR["Grounded prompt<br/>+ few-shot + JSON schema"]
        GEM["Gemini"]
        TPL["Deterministic template<br/>fallback"]
    end

    subgraph DB["PostgreSQL 16 + pgvector"]
        T1[(applicants)]
        T2[(predictions)]
        T3[(borrower_embeddings)]
        T4[(underwriting_reports)]
        T5[(audit_logs)]
        T6[(review_decisions)]
    end

    CLIENT --> CORS --> API
    R1 --> SVC
    R3 --> SVC
    SVC --> FE --> XGB --> CAL --> POL
    SVC --> SHAP
    SVC --> PROF --> ENC --> IDX
    R2 --> PROF
    R3 --> PR --> GEM
    PR -. no key / API error .-> TPL
    SVC --> T1
    SVC --> T2
    R3 --> T4
    API --> T5
    IDX -. pgvector backend .-> T3
    R5 --> DB
    API --> CLIENT
```

## 2. Request flow — `POST /predict`

```mermaid
sequenceDiagram
    autonumber
    participant U as Underwriter (React)
    participant A as FastAPI
    participant F as Feature engine
    participant M as XGBoost + calibrator
    participant V as Vector store
    participant P as Policy engine
    participant S as TreeSHAP
    participant D as Postgres

    U->>A: applicant payload (18 fields, bureau optional)
    A->>F: normalise units, apply defaults
    F-->>A: 8 behavioural scores + ratios (stateless)
    A->>M: aligned feature matrix
    M-->>A: calibrated probability of default
    A->>V: embed borrower profile, cosine top-K
    V-->>A: peers + repayment rate + cohort agreement
    A->>P: PD + features + cohort agreement
    P-->>A: risk score, band, limit, confidence, review triggers, fraud flags
    A->>S: TreeSHAP contributions
    S-->>A: top positive / negative factors, PD impact in pp
    A->>D: applicants + predictions + audit_logs (append-only)
    A-->>U: one response carrying decision, evidence and explanation
```

Cohort retrieval deliberately runs **before** the policy engine: peer agreement is
an input to the confidence score, and confidence decides whether a human must
look at the case.

## 3. Decision logic

```mermaid
flowchart LR
    PD["Calibrated PD"] --> C1{"PD ≤ 6%?"}
    C1 -- yes --> APR["APPROVE"]
    C1 -- no --> C2{"PD ≥ 17%?"}
    C2 -- yes --> REJ["REJECT"]
    C2 -- no --> REV["REVIEW"]

    APR --> G1{"Confidence ≥ 0.70?"}
    G1 -- no --> REV
    G1 -- yes --> G2{"Fraud flags?"}
    G2 -- yes --> REV
    G2 -- no --> G3{"NTC and limit > 300k?"}
    G3 -- yes --> REV
    G3 -- no --> AUTO["Automatic approval"]

    AUTO --> LIM["Limit = min(requested, capacity)<br/>× risk multiplier<br/>× behaviour adjustment"]
    REV --> QUEUE["Human review queue"]
```

## 4. Why each component exists

| Layer | Choice | Rationale |
|---|---|---|
| Model | XGBoost `hist`, native categoricals | Tabular credit data with heavy missingness; learns a split direction for "no bureau score", which is exactly the NTC case |
| Calibration | Isotonic on a held-out split | The policy thresholds a probability. `scale_pos_weight` would inflate PD ~11× and decline the whole book |
| Explainability | TreeSHAP via `pred_contribs` | Exact, additive, no sampling, no extra dependency in the request path |
| Similarity | Sentence embeddings of a rendered profile | Peers are human-readable, so an underwriter can audit what the recommendation leaned on; scale-invariant vs raw feature distance |
| Vector store | FAISS `IndexFlatIP` / pgvector `ivfflat` | Exact cosine at demo scale, one env var to move to the production path |
| LLM | Gemini with a JSON schema + grounded context | The model explains a decision it cannot change; structured output means the API contract never depends on prose parsing |
| Persistence | Append-only `predictions` + `audit_logs` | Every decision reconstructible with its model and policy version — the regulatory requirement |

## 5. Failure behaviour

| Missing | Effect |
|---|---|
| Postgres | In-memory ledger; scoring, explanation and dashboard all still work |
| FAISS index | Falls back to exact numpy cosine over the saved matrix |
| MiniLM weights | Falls back to a deterministic hashing encoder (reported in `/health`) |
| `GEMINI_API_KEY` | Deterministic template memo built from the same SHAP + cohort evidence |
| Trained model | The only hard dependency: `/predict` returns `503` with the exact command to run |
