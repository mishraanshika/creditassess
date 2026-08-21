"""Embedding pipeline: raw applications -> profile text -> vectors -> index.

Run:
    python -m embeddings.build_index                    # FAISS (default)
    python -m embeddings.build_index --push-pgvector    # also upsert to Postgres
    CI_EMBED_ROWS=50000 python -m embeddings.build_index

Outputs:
    embeddings/artifacts/borrowers.faiss        - IndexFlatIP (exact cosine)
    embeddings/artifacts/borrowers_meta.parquet - aligned metadata + outcome
    embeddings/artifacts/borrower_vectors.npy   - raw matrix (reuse/debug)
    embeddings/artifacts/index_meta.json        - encoder + build provenance
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import numpy as np
import pandas as pd

from embeddings.profile_builder import build_profile_texts, profile_summary
from embeddings.vector_store import get_encoder
from ml import config as C
from ml.features import engineer_features

META_COLUMNS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "age_years", "employment_years",
    "credit_income_ratio", "annuity_income_ratio", "payment_consistency_score",
    "financial_discipline_score", "income_stability_score", "spending_stability_score",
    "credit_utilization_score", "digital_trust_score", "monthly_cashflow_consistency",
    "transaction_volatility", "utility_payment_consistency",
    "mobile_recharge_consistency", "is_ntc", "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
]


def build(rows: int, push_pgvector: bool = False, batch_size: int = 256) -> None:
    print(f"[embed] loading {rows:,} applications ...")
    df = pd.read_csv(C.TRAIN_CSV, nrows=rows)
    enriched = engineer_features(df)

    print("[embed] rendering borrower profile texts ...")
    texts: List[str] = build_profile_texts(enriched)
    print(f"[embed] example profile:\n  {texts[0][:400]} ...")

    encoder = get_encoder()
    print(f"[embed] encoder = {encoder.name} (dim={encoder.dim})")
    t0 = time.time()
    vectors = encoder.encode(texts, batch_size=batch_size, show_progress=True)
    print(f"[embed] encoded {vectors.shape[0]:,} profiles in {time.time() - t0:.1f}s")

    # Cosine similarity via inner product on L2-normalised vectors.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = (vectors / norms).astype("float32")

    meta = pd.DataFrame({
        "borrower_id": df[C.ID_COL].astype(int).values,
        "repaid": (df[C.TARGET].astype(int).values == 0),
        "target": df[C.TARGET].astype(int).values,
        "profile_text": texts,
    })
    for col in META_COLUMNS:
        if col in enriched.columns:
            meta[col] = enriched[col].values
    meta["summary"] = [json.dumps(profile_summary(r))
                       for r in enriched.to_dict(orient="records")]

    np.save(C.EMBED_MATRIX_PATH, vectors)
    meta.to_parquet(C.FAISS_META_PATH, index=False)

    _build_faiss(vectors)

    index_meta = {
        "encoder": encoder.name,
        "dim": int(vectors.shape[1]),
        "n_vectors": int(vectors.shape[0]),
        "metric": "cosine (inner product on normalised vectors)",
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "source": str(C.TRAIN_CSV.name),
    }
    (C.EMBED_DIR / "index_meta.json").write_text(json.dumps(index_meta, indent=2),
                                                 encoding="utf-8")
    print(f"[embed] index    -> {C.FAISS_INDEX_PATH}")
    print(f"[embed] metadata -> {C.FAISS_META_PATH}")

    if push_pgvector:
        push_to_pgvector(vectors, meta)


def _build_faiss(vectors: np.ndarray) -> None:
    try:
        import faiss
    except ImportError:
        print("[embed] faiss not installed - vectors saved, index skipped. "
              "The API will fall back to an in-memory numpy cosine search.")
        return
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(C.FAISS_INDEX_PATH))
    print(f"[embed] FAISS IndexFlatIP built with {index.ntotal:,} vectors")


def push_to_pgvector(vectors: np.ndarray, meta: pd.DataFrame,
                     dsn: str | None = None, page: int = 1000) -> None:
    """Upsert vectors + metadata into the `borrower_embeddings` pgvector table."""
    dsn = dsn or os.getenv(
        "DATABASE_URL",
        "postgresql://credit:credit@localhost:5432/credit_intelligence",
    ).replace("postgresql+psycopg2://", "postgresql://")
    try:
        import psycopg2
        from psycopg2.extras import execute_batch
    except ImportError:
        print("[embed] psycopg2 not installed - skipping pgvector push")
        return

    print(f"[embed] pushing {len(meta):,} rows to pgvector ...")
    sql = """
        INSERT INTO borrower_embeddings
            (borrower_id, profile_text, embedding, repaid, target, metadata)
        VALUES (%s, %s, %s::vector, %s, %s, %s::jsonb)
        ON CONFLICT (borrower_id) DO UPDATE SET
            profile_text = EXCLUDED.profile_text,
            embedding    = EXCLUDED.embedding,
            repaid       = EXCLUDED.repaid,
            target       = EXCLUDED.target,
            metadata     = EXCLUDED.metadata,
            updated_at   = now();
    """
    payload = []
    records = meta.to_dict(orient="records")
    for i, rec in enumerate(records):
        vec = "[" + ",".join(f"{x:.6f}" for x in vectors[i].tolist()) + "]"
        md = {k: _plain(rec.get(k)) for k in META_COLUMNS if k in rec}
        payload.append((int(rec["borrower_id"]), rec["profile_text"], vec,
                        bool(rec["repaid"]), int(rec["target"]), json.dumps(md)))

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        execute_batch(cur, sql, payload, page_size=page)
        conn.commit()
    print("[embed] pgvector upsert complete")


def _plain(v):
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if np.isnan(f) else round(f, 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=C.EMBED_ROWS)
    ap.add_argument("--push-pgvector", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    build(args.rows, args.push_pgvector, args.batch_size)
