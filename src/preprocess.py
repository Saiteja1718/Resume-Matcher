# src/preprocess.py
import re
from pathlib import Path
import pandas as pd
from typing import List
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# STOPWORDS = set(stopwords.words("english"))

import nltk

# Download only once
nltk.download('all')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\—", "-", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\-\s]{6,}", " ", text)
    return text.strip()

def extract_keywords_simple(text: str, top_n: int = 20):
    # tokenize using standard punkt
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_items[:top_n]]


def load_sample_jds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "description" not in df.columns:
        df["description"] = df.get("requirements", "")
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df["text"] = df["text"].apply(clean_text)
    return df

def load_resumes_from_folder(folder: Path):
    resumes = []
    for f in folder.glob("*.txt"):
        text = clean_text(f.read_text(encoding="utf8"))
        resumes.append({"id": f.stem, "path": str(f), "text": text})
    return resumes

