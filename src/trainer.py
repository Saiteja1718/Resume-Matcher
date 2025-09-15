import sys
from pathlib import Path

# Add current src folder to sys.path
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import joblib

from embedder import Embedder
from config import MODEL_DIR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_classifier(pairs_csv: Path, model_out: Path = Path(MODEL_DIR)/"clf.joblib"):
    df = pd.read_csv(pairs_csv)
    e = Embedder()
    r_embs = e.embed_texts(df["resume_text"].tolist())
    jd_embs = e.embed_texts(df["jd_text"].tolist())
    X = np.concatenate([r_embs, jd_embs, np.abs(r_embs - jd_embs)], axis=1)
    y = df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    probs = clf.predict_proba(X_val)[:,1]
    print(classification_report(y_val, preds))
    print("AUC:", roc_auc_score(y_val, probs))
    joblib.dump(clf, model_out)
    return clf
