# src/embedder.py
import sys
sys.path.insert(0, '../src')

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from typing import List

from config import MODEL_DIR, EMBEDDING_MODEL


MODEL_DIR.mkdir(parents=True, exist_ok=True)

class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print("Loading embedding model:", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embs

    def save_embeddings(self, arr: np.ndarray, path: Path):
        np.save(path, arr)

    def load_embeddings(self, path: Path) -> np.ndarray:
        return np.load(path)
