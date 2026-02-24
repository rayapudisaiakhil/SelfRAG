"""State schema and Pydantic models for structured LLM outputs."""

from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


# ── Graph State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    # Hallucination evaluation
    is_supported: Literal["fully_supported", "partially_supported", "not_supported"]
    evidence: List[str]
    retries: int
    # Usefulness evaluation
    is_use: Literal["useful", "not_useful"]
    use_reason: str
    # Query rewrite
    retrieval_query: str
    rewrite_tries: int


# ── Pydantic schemas for structured output ───────────────────────────────────
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external documents are needed to answer the question, False otherwise.",
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True if the document helps answer the question, False otherwise.",
    )


class IsSupportedDecision(BaseModel):
    issupported: Literal["fully_supported", "partially_supported", "not_supported"]
    evidence: List[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal company PDFs.",
    )
