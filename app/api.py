"""FastAPI application for the Self-RAG pipeline."""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import GRAPH_RECURSION_LIMIT
from app.graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Compiled graph (loaded on startup) ───────────────────────────────────────
_app_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the graph and vectorstore on startup."""
    global _app_graph
    logger.info("Building Self-RAG graph...")
    _app_graph = build_graph()
    logger.info("Self-RAG graph ready.")
    yield


api = FastAPI(
    title="Self-RAG API",
    description="Self-Reflective RAG pipeline for NovaMind AI company information.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response Schemas ───────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., description="The question to ask.", min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    need_retrieval: Optional[bool] = None
    num_docs_retrieved: int = 0
    num_relevant_docs: int = 0
    relevant_doc_sources: List[dict] = []
    is_supported: Optional[str] = None
    evidence: List[str] = []
    retries: int = 0
    is_use: Optional[str] = None
    use_reason: Optional[str] = None
    rewrite_tries: int = 0
    elapsed_seconds: float = 0.0


# ── Endpoints ────────────────────────────────────────────────────────────────


@api.get("/health")
async def health():
    return {"status": "ok", "graph_loaded": _app_graph is not None}


@api.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if _app_graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet.")

    initial_state = {
        "question": request.question,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "is_supported": "",
        "evidence": [],
        "retries": 0,
        "is_use": "",
        "use_reason": "",
    }

    start = time.time()
    result = _app_graph.invoke(
        initial_state, config={"recursion_limit": GRAPH_RECURSION_LIMIT}
    )
    elapsed = time.time() - start

    # Extract source metadata from relevant docs
    relevant_docs = result.get("relevant_docs") or []
    sources = []
    for doc in relevant_docs:
        meta = doc.metadata or {}
        sources.append(
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
            }
        )

    return AskResponse(
        question=request.question,
        answer=result.get("answer", ""),
        need_retrieval=result.get("need_retrieval"),
        num_docs_retrieved=len(result.get("docs") or []),
        num_relevant_docs=len(relevant_docs),
        relevant_doc_sources=sources,
        is_supported=result.get("is_supported") or None,
        evidence=result.get("evidence") or [],
        retries=result.get("retries", 0),
        is_use=result.get("is_use") or None,
        use_reason=result.get("use_reason") or None,
        rewrite_tries=result.get("rewrite_tries", 0),
        elapsed_seconds=round(elapsed, 2),
    )
