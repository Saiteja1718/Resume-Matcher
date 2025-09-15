import sys
from pathlib import Path

# Add current src folder to sys.path
sys.path.insert(0, '../src')

import numpy as np
from config import EMBEDDING_DIM

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

class Indexer:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = None

    def build_index(self, embeddings: np.ndarray, use_faiss: bool = True):
        if use_faiss and FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(self.dim)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            self.index = index
        else:
            self.index = {"embeddings": embeddings}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and hasattr(self.index, "ntotal"):
            faiss.write_index(self.index, str(path))
        else:
            np.save(str(path) + ".npy", self.index["embeddings"])

    def load(self, path: Path):
        if FAISS_AVAILABLE:
            try:
                import faiss
                self.index = faiss.read_index(str(path))
                return
            except Exception:
                pass
        arr = np.load(str(path) + ".npy")
        self.index = {"embeddings": arr}

    def query(self, query_emb: np.ndarray, top_k: int = 5):
        if FAISS_AVAILABLE and hasattr(self.index, "ntotal"):
            import faiss
            q = query_emb.copy()
            if q.ndim == 1:
                q = q.reshape(1, -1)
            faiss.normalize_L2(q)
            distances, indices = self.index.search(q, top_k)
            return distances[0], indices[0]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            emb = self.index["embeddings"]
            sims = cosine_similarity(query_emb, emb)[0]
            idx = sims.argsort()[::-1][:top_k]
            return sims[idx], idx
