"""SQLAlchemy ORM mapping of `database/schema.sql`.

The SQL file remains the source of truth (it carries the pgvector column, the
IVFFLAT index and the analytics views, which ORMs model poorly).  These classes
exist so the application can read and write the same tables with type safety.
"""
from __future__ import annotations

import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Applicant(Base):
    __tablename__ = "applicants"

    id = _uuid_pk()
    external_ref = Column(Text, unique=True)
    full_name = Column(Text)
    contract_type = Column(Text)
    gender = Column(Text)
    age_years = Column(Numeric(5, 2))
    income_total = Column(Numeric(16, 2), nullable=False)
    credit_amount = Column(Numeric(16, 2), nullable=False)
    annuity_amount = Column(Numeric(16, 2))
    goods_price = Column(Numeric(16, 2))
    employment_years = Column(Numeric(6, 2))
    occupation_type = Column(Text)
    organization_type = Column(Text)
    education_type = Column(Text)
    family_status = Column(Text)
    housing_type = Column(Text)
    children_count = Column(Integer, default=0)
    family_members = Column(Numeric(4, 1), default=1)
    is_ntc = Column(Boolean, default=False)
    raw_payload = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    predictions = relationship("Prediction", back_populates="applicant",
                               cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("recommendation IN ('APPROVE','REVIEW','REJECT')",
                        name="ck_predictions_recommendation"),
    )

    id = _uuid_pk()
    applicant_id = Column(UUID(as_uuid=True), ForeignKey("applicants.id", ondelete="CASCADE"),
                          nullable=False)
    request_id = Column(Text, nullable=False)
    probability_of_default = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_band = Column(Text, nullable=False)
    risk_tier = Column(Text)
    recommendation = Column(Text, nullable=False)
    recommended_credit_limit = Column(Numeric(16, 2), nullable=False, default=0)
    max_affordable_limit = Column(Numeric(16, 2))
    suggested_term_months = Column(Integer)
    confidence_score = Column(Float, nullable=False)
    confidence_drivers = Column(JSONB, default=dict)
    requires_human_review = Column(Boolean, nullable=False, default=False)
    review_reasons = Column(JSONB, default=list)
    fraud_flags = Column(JSONB, default=list)
    behavioural_features = Column(JSONB, default=dict)
    shap_top_positive = Column(JSONB, default=list)
    shap_top_negative = Column(JSONB, default=list)
    shap_base_value = Column(Float)
    similar_borrower_ids = Column(ARRAY(BigInteger))
    cohort_repayment_rate = Column(Float)
    cohort_mean_similarity = Column(Float)
    model_version = Column(Text, nullable=False)
    policy_version = Column(Text, nullable=False)
    feature_version = Column(Text)
    latency_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    applicant = relationship("Applicant", back_populates="predictions")


class BorrowerEmbedding(Base):
    """Mapped without the vector column: pgvector KNN is issued as raw SQL."""

    __tablename__ = "borrower_embeddings"

    borrower_id = Column(BigInteger, primary_key=True)
    profile_text = Column(Text, nullable=False)
    repaid = Column(Boolean, nullable=False)
    target = Column(SmallInteger, nullable=False)
    meta_json = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())


class UnderwritingReport(Base):
    __tablename__ = "underwriting_reports"

    id = _uuid_pk()
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"))
    applicant_id = Column(UUID(as_uuid=True), ForeignKey("applicants.id", ondelete="CASCADE"))
    recommendation = Column(Text)
    suggested_limit = Column(Numeric(16, 2))
    strengths = Column(JSONB, default=list)
    risk_factors = Column(JSONB, default=list)
    conditions = Column(JSONB, default=list)
    explanation = Column(Text)
    executive_summary = Column(Text)
    generator = Column(Text, nullable=False)
    prompt_version = Column(Text)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(Text, nullable=False)
    event_type = Column(Text, nullable=False)
    applicant_id = Column(UUID(as_uuid=True), ForeignKey("applicants.id", ondelete="SET NULL"))
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="SET NULL"))
    actor = Column(Text, default="system")
    endpoint = Column(Text)
    http_status = Column(Integer)
    latency_ms = Column(Integer)
    model_version = Column(Text)
    policy_version = Column(Text)
    payload_hash = Column(String(64))
    details = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ReviewDecision(Base):
    __tablename__ = "review_decisions"

    id = _uuid_pk()
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"),
                           nullable=False)
    reviewer = Column(Text, nullable=False)
    final_decision = Column(Text, nullable=False)
    approved_limit = Column(Numeric(16, 2))
    override_of_model = Column(Boolean, nullable=False, default=False)
    rationale = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = _uuid_pk()
    model_version = Column(Text, unique=True, nullable=False)
    algorithm = Column(Text, nullable=False, default="xgboost")
    roc_auc = Column(Float)
    pr_auc = Column(Float)
    ks_statistic = Column(Float)
    metrics = Column(JSONB, default=dict)
    feature_count = Column(Integer)
    trained_at = Column(DateTime(timezone=True))
    promoted_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=False)
    notes = Column(Text)
