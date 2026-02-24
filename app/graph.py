"""Build and compile the Self-RAG LangGraph."""

from langgraph.graph import END, START, StateGraph

from app.models import State
from app.nodes import (
    accept_answer,
    decide_retrieval,
    generate_direct,
    generate_from_context,
    is_relevant,
    is_supported,
    is_use,
    no_answer_found,
    retrieve,
    revise_answer,
    rewrite_question,
    route_after_decide,
    route_after_issupported,
    route_after_isuse,
    route_after_relevance,
)


def build_graph():
    """Construct the Self-RAG StateGraph and return the compiled app."""
    g = StateGraph(State)

    # ── Nodes ────────────────────────────────────────────────────────────
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_supported", is_supported)
    g.add_node("revise_answer", revise_answer)
    g.add_node("is_use", is_use)
    g.add_node("rewrite_question", rewrite_question)

    # ── Edges ────────────────────────────────────────────────────────────
    g.add_edge(START, "decide_retrieval")

    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )
    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("no_answer_found", END)

    g.add_edge("generate_from_context", "is_supported")
    g.add_conditional_edges(
        "is_supported",
        route_after_issupported,
        {"accept_answer": "is_use", "revise_answer": "revise_answer"},
    )
    g.add_edge("revise_answer", "is_supported")

    g.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {
            "END": END,
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("rewrite_question", "retrieve")

    return g.compile()
