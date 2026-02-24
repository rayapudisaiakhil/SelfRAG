"""Centralized configuration"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

PDF_FILES = [
    PDF_DIR / "policies.pdf",
    PDF_DIR / "profile.pdf",
    PDF_DIR / "product_pricing.pdf",
]

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retriever ────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "4"))

# ── Models ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── Self-RAG limits ──────────────────────────────────────────────────────────
MAX_HALLUCINATION_RETRIES = int(os.getenv("MAX_HALLUCINATION_RETRIES", "5"))
MAX_QUERY_REWRITES = int(os.getenv("MAX_QUERY_REWRITES", "3"))
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "80"))

# ── Singleton instances ──────────────────────────────────────────────────────
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
