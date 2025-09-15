import sys
sys.path.insert(0, '../src')

from matcher import ResumeMatcher
from embedder import Embedder
from pathlib import Path


m = ResumeMatcher(Embedder())
m.prepare_jds(Path("data/sample_resumes/sample_jds.csv"))
print("JDs prepared")
