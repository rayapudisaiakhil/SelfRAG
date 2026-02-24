"""All LangGraph node functions and routing functions for the Self-RAG pipeline."""

from typing import List, Literal

from langchain_core.documents import Document

from app.config import MAX_HALLUCINATION_RETRIES, MAX_QUERY_REWRITES, llm
from app.models import (
    IsSupportedDecision,
    IsUSEDecision,
    RelevanceDecision,
    RetrieveDecision,
    RewriteDecision,
    State,
)
from app.prompts import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    issup_prompt,
    isuse_prompt,
    rag_generation_prompt,
    revise_prompt,
    rewrite_for_retrieval_prompt,
)
from app.vectorstore import get_retriever

# ── Structured-output LLM wrappers ──────────────────────────────────────────
should_retrieve_llm = llm.with_structured_output(RetrieveDecision)
relevance_llm = llm.with_structured_output(RelevanceDecision)
issup_llm = llm.with_structured_output(IsSupportedDecision)
isuse_llm = llm.with_structured_output(IsUSEDecision)
rewrite_llm = llm.with_structured_output(RewriteDecision)


# ── Node Functions ───────────────────────────────────────────────────────────


def decide_retrieval(state: State):
    """Decide whether external document retrieval is needed."""
    decision: RetrieveDecision = should_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}


def generate_direct(state: State):
    """Answer using only LLM parametric knowledge (no retrieval)."""
    out = llm.invoke(
        direct_generation_prompt.format_messages(question=state["question"])
    )
    return {"answer": out.content}


def retrieve(state: State):
    """Retrieve top-k documents from the vector store."""
    retriever = get_retriever()
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}


def is_relevant(state: State):
    """Filter retrieved documents for relevance to the question."""
    relevant_docs: List[Document] = []
    for doc in state["docs"]:
        decision: RelevanceDecision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"], document=doc.page_content
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


def generate_from_context(state: State):
    """Generate an answer grounded in the relevant documents."""
    context = "\n\n---\n\n".join(
        [doc.page_content for doc in state.get("relevant_docs", [])]
    ).strip()

    if not context:
        return {"answer": "No relevant document found.", "context": ""}

    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"], context=context
        )
    )
    return {"answer": out.content, "context": context}


def is_supported(state: State):
    """Check if the answer is grounded in the context (hallucination detection)."""
    decision: IsSupportedDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"is_supported": decision.issupported, "evidence": decision.evidence}


def accept_answer(state: State):
    """Pass-through node for accepted answers."""
    return {}


def revise_answer(state: State):
    """Revise the answer to use only direct quotes from context."""
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }


def is_use(state: State):
    """Check if the answer is useful — does it address the actual question?"""
    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"is_use": decision.isuse, "use_reason": decision.reason}


def rewrite_question(state: State):
    """Rewrite the retrieval query for better vector search results."""
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


def no_answer_found(state: State):
    """Fallback node when no relevant documents are found."""
    return {"answer": "No relevant document found.", "context": ""}


# ── Routing Functions ────────────────────────────────────────────────────────


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct"


def route_after_relevance(
    state: State,
) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


def route_after_issupported(
    state: State,
) -> Literal["accept_answer", "revise_answer"]:
    if state.get("is_supported") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_HALLUCINATION_RETRIES:
        return "accept_answer"
    return "revise_answer"


def route_after_isuse(
    state: State,
) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("is_use") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_QUERY_REWRITES:
        return "no_answer_found"
    return "rewrite_question"
