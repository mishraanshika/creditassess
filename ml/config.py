"""Central configuration for the ML layer.

Every path used by training, explainability and the embedding pipeline is
resolved from the repository root so the code behaves identically whether it is
invoked from the repo root, from `ml/`, or from inside the FastAPI process.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("CI_DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("CI_ARTIFACT_DIR", ROOT / "ml" / "artifacts"))
EMBED_DIR = Path(os.getenv("CI_EMBED_DIR", ROOT / "embeddings" / "artifacts"))

TRAIN_CSV = DATA_DIR / "application_train.csv"
TEST_CSV = DATA_DIR / "application_test.csv"

MODEL_PATH = ARTIFACT_DIR / "xgb_model.json"
PIPELINE_PATH = ARTIFACT_DIR / "pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"
EXPLAINER_PATH = ARTIFACT_DIR / "shap_explainer.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "calibrator.joblib"
BASELINE_PATH = ARTIFACT_DIR / "feature_baseline.json"
BIAS_REPORT_PATH = ARTIFACT_DIR / "bias_report.json"

FAISS_INDEX_PATH = EMBED_DIR / "borrowers.faiss"
FAISS_META_PATH = EMBED_DIR / "borrowers_meta.parquet"
EMBED_MATRIX_PATH = EMBED_DIR / "borrower_vectors.npy"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42

# Number of rows sampled for training. Home Credit has ~307k rows; the full set
# trains in a few minutes on CPU. Override with CI_TRAIN_ROWS for a fast demo.
TRAIN_ROWS = int(os.getenv("CI_TRAIN_ROWS", "0")) or None

# Rows indexed into the vector store. Keeping this smaller than the training set
# keeps the FAISS index and the pgvector table hackathon-friendly.
EMBED_ROWS = int(os.getenv("CI_EMBED_ROWS", "20000"))

EMBED_MODEL_NAME = os.getenv("CI_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384

for _d in (DATA_DIR, ARTIFACT_DIR, EMBED_DIR):
    _d.mkdir(parents=True, exist_ok=True)
