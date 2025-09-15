# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

INDEX_PATH = MODEL_DIR / "faiss_index.bin"
EMBEDDINGS_PATH = MODEL_DIR / "jd_embeddings.npy"
JD_META_PATH = MODEL_DIR / "jd_meta.pkl"
RESUME_EMBED_PATH = MODEL_DIR / "resume_embeddings.npy"
RESUME_META_PATH = MODEL_DIR / "resume_meta.pkl"

TOP_K = 5
