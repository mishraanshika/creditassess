-- ============================================================================
--  Next-Gen Credit Intelligence - PostgreSQL schema
--  Target: PostgreSQL 15+ with the pgvector extension (>= 0.5)
--
--  Design notes
--  ------------
--  * Every decision is immutable: `predictions` is append-only and every row
--    carries the model + policy version that produced it, so a decision can be
--    reproduced years later (a hard regulatory requirement in lending).
--  * Raw applicant payloads are stored as JSONB alongside the typed columns so
--    the exact request body can be replayed against a future model.
--  * `audit_logs` is written on every state-changing call and is the artefact an
--    auditor reads; it never contains raw PII beyond the applicant reference.
--  * `borrower_embeddings` holds the pgvector index that powers similar-borrower
--    retrieval; it is a derived table and can be rebuilt from the source data.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- 1. Applicants
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS applicants (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_ref            TEXT UNIQUE,                 -- SK_ID_CURR or CRM id
    full_name               TEXT,
    -- typed underwriting inputs (mirrors the API contract)
    contract_type           TEXT,
    gender                  TEXT,
    age_years               NUMERIC(5,2),
    income_total            NUMERIC(16,2) NOT NULL CHECK (income_total >= 0),
    credit_amount           NUMERIC(16,2) NOT NULL CHECK (credit_amount >= 0),
    annuity_amount          NUMERIC(16,2) CHECK (annuity_amount >= 0),
    goods_price             NUMERIC(16,2) CHECK (goods_price >= 0),
    employment_years        NUMERIC(6,2),
    occupation_type         TEXT,
    organization_type       TEXT,
    education_type          TEXT,
    family_status           TEXT,
    housing_type            TEXT,
    children_count          INTEGER DEFAULT 0 CHECK (children_count >= 0),
    family_members          NUMERIC(4,1) DEFAULT 1,
    is_ntc                  BOOLEAN DEFAULT FALSE,       -- new-to-credit flag
    raw_payload             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_applicants_created_at ON applicants (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_applicants_ntc        ON applicants (is_ntc);
CREATE INDEX IF NOT EXISTS idx_applicants_payload    ON applicants USING GIN (raw_payload);

-- ---------------------------------------------------------------------------
-- 2. Predictions  (append-only decision ledger)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    applicant_id              UUID NOT NULL REFERENCES applicants(id) ON DELETE CASCADE,
    request_id                TEXT NOT NULL,
    -- model output
    probability_of_default    DOUBLE PRECISION NOT NULL
                              CHECK (probability_of_default BETWEEN 0 AND 1),
    risk_score                INTEGER NOT NULL CHECK (risk_score BETWEEN 300 AND 900),
    risk_band                 TEXT NOT NULL,
    risk_tier                 TEXT,
    -- policy output
    recommendation            TEXT NOT NULL
                              CHECK (recommendation IN ('APPROVE','REVIEW','REJECT')),
    recommended_credit_limit  NUMERIC(16,2) NOT NULL DEFAULT 0,
    max_affordable_limit      NUMERIC(16,2),
    suggested_term_months     INTEGER,
    confidence_score          DOUBLE PRECISION NOT NULL
                              CHECK (confidence_score BETWEEN 0 AND 1),
    confidence_drivers        JSONB DEFAULT '{}'::jsonb,
    requires_human_review     BOOLEAN NOT NULL DEFAULT FALSE,
    review_reasons            JSONB DEFAULT '[]'::jsonb,
    fraud_flags               JSONB DEFAULT '[]'::jsonb,
    -- explainability + behavioural snapshot (frozen at decision time)
    behavioural_features      JSONB DEFAULT '{}'::jsonb,
    shap_top_positive         JSONB DEFAULT '[]'::jsonb,
    shap_top_negative         JSONB DEFAULT '[]'::jsonb,
    shap_base_value           DOUBLE PRECISION,
    -- similar-borrower evidence
    similar_borrower_ids      BIGINT[],
    cohort_repayment_rate     DOUBLE PRECISION,
    cohort_mean_similarity    DOUBLE PRECISION,
    -- provenance
    model_version             TEXT NOT NULL,
    policy_version            TEXT NOT NULL,
    feature_version           TEXT,
    latency_ms                INTEGER,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_applicant  ON predictions (applicant_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_reco       ON predictions (recommendation);
CREATE INDEX IF NOT EXISTS idx_predictions_review     ON predictions (requires_human_review)
    WHERE requires_human_review;
CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions (request_id);

-- ---------------------------------------------------------------------------
-- 3. Borrower embeddings  (vector search corpus)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS borrower_embeddings (
    borrower_id    BIGINT PRIMARY KEY,
    profile_text   TEXT NOT NULL,
    embedding      VECTOR(384) NOT NULL,          -- all-MiniLM-L6-v2
    repaid         BOOLEAN NOT NULL,
    target         SMALLINT NOT NULL CHECK (target IN (0,1)),
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IVFFLAT cosine index. Build AFTER the table is populated; `lists` ~ sqrt(rows).
-- For 20k rows, 141 lists is a good starting point.
CREATE INDEX IF NOT EXISTS idx_borrower_embeddings_cosine
    ON borrower_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 141);

CREATE INDEX IF NOT EXISTS idx_borrower_embeddings_repaid ON borrower_embeddings (repaid);
CREATE INDEX IF NOT EXISTS idx_borrower_embeddings_meta
    ON borrower_embeddings USING GIN (metadata);

-- ---------------------------------------------------------------------------
-- 4. Underwriting reports (LLM output, kept separately from the decision)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS underwriting_reports (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id     UUID REFERENCES predictions(id) ON DELETE CASCADE,
    applicant_id      UUID REFERENCES applicants(id) ON DELETE CASCADE,
    recommendation    TEXT,
    suggested_limit   NUMERIC(16,2),
    strengths         JSONB DEFAULT '[]'::jsonb,
    risk_factors      JSONB DEFAULT '[]'::jsonb,
    conditions        JSONB DEFAULT '[]'::jsonb,
    explanation       TEXT,
    executive_summary TEXT,
    generator         TEXT NOT NULL,        -- 'anthropic:claude-...' | 'template'
    prompt_version    TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reports_prediction ON underwriting_reports (prediction_id);

-- ---------------------------------------------------------------------------
-- 5. Audit logs  (responsible-AI evidence trail)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_logs (
    id             BIGSERIAL PRIMARY KEY,
    request_id     TEXT NOT NULL,
    event_type     TEXT NOT NULL,        -- PREDICT | SIMILAR | REPORT | OVERRIDE | ERROR
    applicant_id   UUID REFERENCES applicants(id) ON DELETE SET NULL,
    prediction_id  UUID REFERENCES predictions(id) ON DELETE SET NULL,
    actor          TEXT DEFAULT 'system',
    endpoint       TEXT,
    http_status    INTEGER,
    latency_ms     INTEGER,
    model_version  TEXT,
    policy_version TEXT,
    payload_hash   TEXT,                 -- sha256 of the request body (no raw PII)
    details        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_created    ON audit_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_logs (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_request    ON audit_logs (request_id);

-- ---------------------------------------------------------------------------
-- 6. Human overrides (four-eyes principle)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_decisions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id      UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    reviewer           TEXT NOT NULL,
    final_decision     TEXT NOT NULL CHECK (final_decision IN ('APPROVE','REJECT')),
    approved_limit     NUMERIC(16,2),
    override_of_model  BOOLEAN NOT NULL DEFAULT FALSE,
    rationale          TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 7. Model registry (which model was live, when)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version   TEXT UNIQUE NOT NULL,
    algorithm       TEXT NOT NULL DEFAULT 'xgboost',
    roc_auc         DOUBLE PRECISION,
    pr_auc          DOUBLE PRECISION,
    ks_statistic    DOUBLE PRECISION,
    metrics         JSONB DEFAULT '{}'::jsonb,
    feature_count   INTEGER,
    trained_at      TIMESTAMPTZ,
    promoted_at     TIMESTAMPTZ,
    is_active       BOOLEAN DEFAULT FALSE,
    notes           TEXT
);

-- ---------------------------------------------------------------------------
-- 8. Analytics views consumed by the dashboard
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_decision_summary AS
SELECT
    date_trunc('day', p.created_at)                         AS day,
    p.recommendation,
    count(*)                                                AS decisions,
    round(avg(p.probability_of_default)::numeric, 4)        AS avg_pd,
    round(avg(p.risk_score)::numeric, 1)                    AS avg_risk_score,
    round(avg(p.confidence_score)::numeric, 3)              AS avg_confidence,
    sum(p.recommended_credit_limit)                         AS total_limit_offered,
    count(*) FILTER (WHERE p.requires_human_review)         AS review_queue
FROM predictions p
GROUP BY 1, 2;

CREATE OR REPLACE VIEW v_ntc_impact AS
SELECT
    a.is_ntc,
    count(*)                                                    AS applications,
    count(*) FILTER (WHERE p.recommendation = 'APPROVE')        AS approvals,
    round(100.0 * count(*) FILTER (WHERE p.recommendation = 'APPROVE')
          / NULLIF(count(*), 0), 2)                             AS approval_rate_pct,
    round(avg(p.recommended_credit_limit)::numeric, 2)          AS avg_limit,
    round(avg(p.confidence_score)::numeric, 3)                  AS avg_confidence
FROM predictions p
JOIN applicants a ON a.id = p.applicant_id
GROUP BY 1;

CREATE OR REPLACE VIEW v_risk_band_distribution AS
SELECT risk_band,
       risk_tier,
       count(*)                                     AS decisions,
       round(avg(probability_of_default)::numeric, 4) AS avg_pd,
       round(avg(recommended_credit_limit)::numeric, 2) AS avg_limit
FROM predictions
GROUP BY 1, 2
ORDER BY 1;

CREATE OR REPLACE VIEW v_review_queue AS
SELECT p.id            AS prediction_id,
       a.external_ref,
       a.full_name,
       p.risk_score,
       p.risk_band,
       p.probability_of_default,
       p.confidence_score,
       p.review_reasons,
       p.fraud_flags,
       p.created_at
FROM predictions p
JOIN applicants a ON a.id = p.applicant_id
LEFT JOIN review_decisions r ON r.prediction_id = p.id
WHERE p.requires_human_review AND r.id IS NULL
ORDER BY p.created_at DESC;

-- ---------------------------------------------------------------------------
-- 9. updated_at maintenance
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_applicants_touch ON applicants;
CREATE TRIGGER trg_applicants_touch BEFORE UPDATE ON applicants
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS trg_embeddings_touch ON borrower_embeddings;
CREATE TRIGGER trg_embeddings_touch BEFORE UPDATE ON borrower_embeddings
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
