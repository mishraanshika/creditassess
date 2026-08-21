"""Embedding + vector-search layer.

Encoder
-------
`all-MiniLM-L6-v2` via sentence-transformers (384-dim, cosine).  If the package
or the weights are unavailable (air-gapped judging laptop, no HF cache), the
`HashingEncoder` fallback keeps the whole feature working with a deterministic
character n-gram hashing vectoriser projected to the same 384 dimensions.  The
API therefore never hard-fails on the demo machine; the active encoder is
reported in `/health`.

Store
-----
Two interchangeable backends behind one interface:

* `FaissStore`    - `IndexFlatIP` over L2-normalised vectors (exact cosine),
                    zero infrastructure, the default for the hackathon demo.
* `PgVectorStore` - pgvector `<=>` cosine distance with an IVFFLAT index, the
                    production path; identical results, survives restarts and
                    scales past a single process.

Select with `VECTOR_BACKEND=faiss|pgvector`.
"""
from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

EMBED_DIM = 384
DEFAULT_MODEL = os.getenv("CI_EMBED_MODEL", "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class BaseEncoder(ABC):
    name: str
    dim: int = EMBED_DIM

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 256,
               show_progress: bool = False) -> np.ndarray:
        ...


class SentenceTransformerEncoder(BaseEncoder):
    name = f"sentence-transformers/{DEFAULT_MODEL}"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 256,
               show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype("float32")


class HashingEncoder(BaseEncoder):
    """Deterministic offline fallback: hashed word + bigram features, L2-normalised.

    Not as semantically rich as MiniLM, but it is stable, dependency-free and
    keeps cosine neighbourhoods meaningful because the profile text uses a small
    controlled vocabulary of underwriting adjectives.
    """

    name = "hashing-fallback-384"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> List[str]:
        words = text.lower().replace("|", " ").replace(":", " ").split()
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:12], 16)
            v[h % self.dim] += 1.0 if (h >> 12) & 1 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts: Sequence[str], batch_size: int = 256,
               show_progress: bool = False) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts]).astype("float32")


_ENCODER: Optional[BaseEncoder] = None


def get_encoder(force_fallback: bool = False) -> BaseEncoder:
    """Process-wide encoder singleton with graceful degradation."""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    if not force_fallback and os.getenv("CI_FORCE_HASH_ENCODER", "0") != "1":
        try:
            _ENCODER = SentenceTransformerEncoder()
            return _ENCODER
        except Exception as exc:  # noqa: BLE001 - degradation is intentional
            print(f"[vector] sentence-transformers unavailable ({exc}); "
                  "falling back to hashing encoder")
    _ENCODER = HashingEncoder()
    return _ENCODER


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

@dataclass
class Neighbour:
    borrower_id: int
    similarity: float
    repaid: bool
    profile_text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "borrower_id": self.borrower_id,
            "similarity_score": round(self.similarity, 4),
            "repaid": self.repaid,
            "outcome": "Repaid" if self.repaid else "Defaulted",
            "profile_text": self.profile_text,
            **self.metadata,
        }


class BaseStore(ABC):
    backend: str

    @abstractmethod
    def search(self, vector: np.ndarray, k: int = 5) -> List[Neighbour]:
        ...

    @abstractmethod
    def size(self) -> int:
        ...


class FaissStore(BaseStore):
    backend = "faiss"

    def __init__(self, index_path: str, meta_path: str) -> None:
        import faiss  # lazy import
        import pandas as pd

        self.index = faiss.read_index(index_path)
        self.meta = pd.read_parquet(meta_path)
        self._meta_records = self.meta.to_dict(orient="records")

    def size(self) -> int:
        return int(self.index.ntotal)

    def search(self, vector: np.ndarray, k: int = 5) -> List[Neighbour]:
        q = np.asarray(vector, dtype="float32").reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        sims, idxs = self.index.search(q, k)
        out: List[Neighbour] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            rec = dict(self._meta_records[int(idx)])
            out.append(Neighbour(
                borrower_id=int(rec.pop("borrower_id")),
                similarity=float(sim),
                repaid=bool(rec.pop("repaid")),
                profile_text=str(rec.pop("profile_text", "")),
                metadata={k2: _jsonable(v) for k2, v in rec.items()},
            ))
        return out


class NumpyStore(BaseStore):
    """Exact cosine search over the saved matrix - used when FAISS is missing.

    At demo scale (20k x 384 float32 = ~30 MB) a single matrix-vector product is
    sub-millisecond, so the API keeps full functionality with zero native deps.
    """

    backend = "numpy"

    def __init__(self, matrix_path: str, meta_path: str) -> None:
        import pandas as pd

        self.matrix = np.load(matrix_path).astype("float32")
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = self.matrix / norms
        self.meta = pd.read_parquet(meta_path)
        self._meta_records = self.meta.to_dict(orient="records")

    def size(self) -> int:
        return int(self.matrix.shape[0])

    def search(self, vector: np.ndarray, k: int = 5) -> List[Neighbour]:
        q = np.asarray(vector, dtype="float32").ravel()
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        sims = self.matrix @ q
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        out: List[Neighbour] = []
        for idx in top:
            rec = dict(self._meta_records[int(idx)])
            out.append(Neighbour(
                borrower_id=int(rec.pop("borrower_id")),
                similarity=float(sims[idx]),
                repaid=bool(rec.pop("repaid")),
                profile_text=str(rec.pop("profile_text", "")),
                metadata={k2: _jsonable(v) for k2, v in rec.items()},
            ))
        return out


class PgVectorStore(BaseStore):
    """Cosine KNN against `borrower_embeddings` using the pgvector extension."""

    backend = "pgvector"

    def __init__(self, dsn: str) -> None:
        import psycopg2  # lazy import

        self.dsn = dsn
        self._psycopg2 = psycopg2

    def _connect(self):
        return self._psycopg2.connect(self.dsn)

    def size(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM borrower_embeddings;")
            return int(cur.fetchone()[0])

    def search(self, vector: np.ndarray, k: int = 5) -> List[Neighbour]:
        v = np.asarray(vector, dtype="float32").ravel()
        norm = np.linalg.norm(v)
        if norm:
            v = v / norm
        literal = "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"
        sql = """
            SELECT borrower_id,
                   1 - (embedding <=> %s::vector) AS similarity,
                   repaid,
                   profile_text,
                   metadata
            FROM borrower_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (literal, literal, k))
            rows = cur.fetchall()
        return [
            Neighbour(
                borrower_id=int(r[0]),
                similarity=float(r[1]),
                repaid=bool(r[2]),
                profile_text=str(r[3] or ""),
                metadata=(r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}")),
            )
            for r in rows
        ]


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if np.isnan(f) else round(f, 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


# Numeric axes used to re-rank the embedding candidates, with the scale over
# which a difference is considered "a full unit of dissimilarity".
RERANK_AXES: Dict[str, float] = {
    "AMT_INCOME_TOTAL": 250_000.0,
    "AMT_CREDIT": 400_000.0,
    "credit_income_ratio": 3.0,
    "annuity_income_ratio": 0.25,
    "employment_years": 8.0,
    "age_years": 15.0,
    "financial_discipline_score": 25.0,
    "payment_consistency_score": 25.0,
    "transaction_volatility": 25.0,
}
EMBED_WEIGHT = 0.55       # semantic profile match
NUMERIC_WEIGHT = 0.45     # hard financial comparability


def numeric_similarity(query: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """Scaled L1 similarity over the axes an underwriter would compare on.

    The rendered profiles share a long fixed template, so raw cosine over MiniLM
    embeddings saturates near 1.0 for almost any pair - it separates *wording*,
    and the wording is mostly constant. Re-ranking the semantic candidates by a
    financial distance restores discrimination while keeping the readable,
    human-auditable profile match that made embeddings worth using.
    """
    total = 0.0
    used = 0
    for axis, scale in RERANK_AXES.items():
        qv, cv = query.get(axis), candidate.get(axis)
        if qv is None or cv is None:
            continue
        try:
            diff = abs(float(qv) - float(cv)) / scale
        except (TypeError, ValueError):
            continue
        total += min(diff, 1.5)
        used += 1
    if not used:
        return 0.5
    return float(max(0.0, 1.0 - (total / used)))


def rerank(neighbours: List[Neighbour], query: Dict[str, Any],
           top_k: int) -> List[Neighbour]:
    """Blend embedding cosine with numeric comparability and keep the best K."""
    for n in neighbours:
        num = numeric_similarity(query, n.metadata)
        n.metadata["embedding_similarity"] = round(n.similarity, 4)
        n.metadata["numeric_similarity"] = round(num, 4)
        n.similarity = EMBED_WEIGHT * n.similarity + NUMERIC_WEIGHT * num
    neighbours.sort(key=lambda n: n.similarity, reverse=True)
    return neighbours[:top_k]


def cohort_stats(neighbours: List[Neighbour]) -> Dict[str, Any]:
    """Aggregate the retrieved cohort into underwriting-grade evidence."""
    if not neighbours:
        return {
            "cohort_size": 0,
            "repayment_success_rate": None,
            "default_rate": None,
            "mean_similarity": None,
            "agreement": None,
        }
    repaid = [n.repaid for n in neighbours]
    sims = [n.similarity for n in neighbours]
    weight = np.array(sims, dtype="float64").clip(min=0)
    weighted = float((weight * np.array(repaid, dtype="float64")).sum() / max(weight.sum(), 1e-9))
    success = float(np.mean(repaid))
    # Agreement = how one-sided the cohort outcome is; feeds the confidence score.
    agreement = float(abs(success - 0.5) * 2)
    return {
        "cohort_size": len(neighbours),
        "repayment_success_rate": round(success, 4),
        "similarity_weighted_repayment_rate": round(weighted, 4),
        "default_rate": round(1 - success, 4),
        "mean_similarity": round(float(np.mean(sims)), 4),
        "max_similarity": round(float(np.max(sims)), 4),
        "agreement": round(agreement, 4),
    }
