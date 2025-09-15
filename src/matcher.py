import sys
from pathlib import Path
import os
import numpy as np
import pickle

# Add current src folder to sys.path
sys.path.insert(0, '../src')

from embedder import Embedder
from indexer import Indexer
from preprocess import load_sample_jds, load_resumes_from_folder, extract_keywords_simple
from config import DATA_DIR, MODEL_DIR, INDEX_PATH, EMBEDDINGS_PATH, JD_META_PATH, TOP_K, RESUME_META_PATH, RESUME_EMBED_PATH


class ResumeMatcher:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.indexer = Indexer(dim=self.embedder.model.get_sentence_embedding_dimension())
        self.jd_meta = None
        self.jd_embeddings = None
        self.resume_embeddings = None
        self.resume_meta = None

    def prepare_jds(self, jd_csv: Path):
        df = load_sample_jds(jd_csv)
        texts = df["text"].tolist()
        embs = self.embedder.embed_texts(texts)
        self.jd_meta = df.to_dict(orient="records")
        self.jd_embeddings = embs
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embs)
        with open(JD_META_PATH, "wb") as f:
            pickle.dump(self.jd_meta, f)
        self.indexer.build_index(embs)
        self.indexer.save(INDEX_PATH)

    def load_jds(self):
        self.jd_embeddings = np.load(EMBEDDINGS_PATH)
        with open(JD_META_PATH, "rb") as f:
            self.jd_meta = pickle.load(f)
        self.indexer.load(INDEX_PATH)

    def index_resumes(self, resumes_folder: Path):
        resumes = load_resumes_from_folder(resumes_folder)
        texts = [r["text"] for r in resumes]
        embs = self.embedder.embed_texts(texts)
        self.resume_embeddings = embs
        self.resume_meta = resumes
        np.save(RESUME_EMBED_PATH, embs)
        with open(RESUME_META_PATH, "wb") as f:
            pickle.dump(self.resume_meta, f)

    def match_resume_to_jds(self, resume_text: str, top_k: int = TOP_K):
        r_emb = self.embedder.embed_texts([resume_text])[0]
        dists, idxs = self.indexer.query(r_emb, top_k)
        results = []
        for score, i in zip(dists, idxs):
            meta = self.jd_meta[i]
            results.append({
                "score": float(score),
                "jd_id": meta.get("id", i),
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
            })
        return results

    def get_missing_keywords(self, resume_text: str, jd_text: str, top_n: int = 20):
        resume_kw = set(extract_keywords_simple(resume_text, top_n=100))
        jd_kw = set(extract_keywords_simple(jd_text, top_n=100))
        missing = list(sorted(jd_kw - resume_kw))
        return {"missing_top": missing[:top_n], "resume_top": list(resume_kw)[:top_n]}
