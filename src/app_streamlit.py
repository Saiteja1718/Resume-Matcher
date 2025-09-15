# src/app_streamlit.py
import sys
sys.path.insert(0, '../src')

import streamlit as st
from pathlib import Path
import pandas as pd

from embedder import Embedder
from matcher import ResumeMatcher
from config import DATA_DIR, MODEL_DIR, JD_META_PATH, RESUME_META_PATH, RESUME_EMBED_PATH, EMBEDDINGS_PATH

# Libraries for resume text extraction
import docx2txt
from PyPDF2 import PdfReader

st.set_page_config(page_title="ResumeXpert", layout="wide")
st.title("ResumeXpert — AI Resume ↔ Job Matcher")

@st.cache_resource
def get_matcher():
    e = Embedder()
    m = ResumeMatcher(e)
    if Path(EMBEDDINGS_PATH).exists() and Path(JD_META_PATH).exists():
        m.load_jds()
    else:
        m.prepare_jds("data/sample_jds.csv")
        m.load_jds()
    return m

matcher = get_matcher()

st.sidebar.header("Options")
uploaded_resume = st.sidebar.file_uploader(
    "Upload resume (txt, pdf, docx)", type=["txt", "pdf", "docx"]
)
resume_text_area = st.sidebar.text_area("Or paste resume text here", height=200)

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(uploaded_file)
    else:
        return ""

# Get resume text
if uploaded_resume:
    resume_text = extract_text_from_file(uploaded_resume)
else:
    resume_text = resume_text_area

if st.sidebar.button("Match my resume") and resume_text.strip():
    with st.spinner("Matching..."):
        res = matcher.match_resume_to_jds(resume_text, top_k=5)
        st.subheader("Top Job Matches")
        for r in res:
            st.write("---")
            st.metric(
                label=f"{r['title']}",
                value=f"Similarity score: {r['score']:.4f}"
            )
            st.write(r["description"][:600] + ("..." if len(r["description"]) > 600 else ""))
            miss = matcher.get_missing_keywords(resume_text, r["description"], top_n=12)
            st.write(
                "Missing / recommended keywords (from JD):",
                ", ".join(miss["missing_top"][:12]) or "None"
            )
            if miss["missing_top"]:
                st.info(
                    "Suggested micro-learning: take short courses on: " +
                    ", ".join(miss["missing_top"][:3])
                )

st.markdown("---")
st.header("Explore sample JDs")
df_jds = pd.read_csv(Path(DATA_DIR) / "sample_jds.csv")
for idx, row in df_jds.iterrows():
    with st.expander(f"{row['title']}"):
        st.write(row["description"])
