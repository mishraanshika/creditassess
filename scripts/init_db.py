"""Apply `database/schema.sql` to the configured Postgres instance.

    python scripts/init_db.py
    python scripts/init_db.py --dsn postgresql://credit:credit@localhost:5432/credit_intelligence
    python scripts/init_db.py --push-vectors     # also load the borrower embeddings

Idempotent: the schema uses CREATE ... IF NOT EXISTS / CREATE OR REPLACE.
Docker Compose applies the same file automatically on first start, so this
script is for a Postgres you already have running.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DSN = os.getenv(
    "DATABASE_URL", "postgresql://credit:credit@localhost:5432/credit_intelligence"
).replace("postgresql+psycopg2://", "postgresql://")


def apply_schema(dsn: str) -> None:
    import psycopg2

    sql = (ROOT / "database" / "schema.sql").read_text(encoding="utf-8")
    print(f"[db] applying schema to {dsn.rsplit('@', 1)[-1]}")
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name;
        """)
        for (name,) in cur.fetchall():
            print(f"  - {name}")
    print("[db] schema applied")


def push_vectors(dsn: str) -> None:
    import numpy as np
    import pandas as pd

    from embeddings.build_index import push_to_pgvector
    from ml import config as C

    if not C.EMBED_MATRIX_PATH.exists():
        raise SystemExit("No vectors found. Run `python -m embeddings.build_index` first.")
    vectors = np.load(C.EMBED_MATRIX_PATH)
    meta = pd.read_parquet(C.FAISS_META_PATH)
    push_to_pgvector(vectors, meta, dsn=dsn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--push-vectors", action="store_true")
    args = ap.parse_args()

    try:
        apply_schema(args.dsn)
        if args.push_vectors:
            push_vectors(args.dsn)
    except ImportError:
        raise SystemExit("psycopg2 is required: pip install psycopg2-binary")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed: {exc}\nIs Postgres running? `docker compose up -d postgres`")
