"""All ChatPromptTemplate definitions for the Self-RAG pipeline."""

from langchain_core.prompts import ChatPromptTemplate

# ── Decide Retrieval ─────────────────────────────────────────────────────────
decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that decides whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': bool}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations or info likely not in the model.\n"
            "- should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.\n"
            "- If unsure, lean towards should_retrieve=True to ensure accuracy.\n",
        ),
        ("human", "Question: {question}"),
    ]
)

# ── Direct Generation (no retrieval) ────────────────────────────────────────
direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my current knowledge.'",
        ),
        ("human", "{question}"),
    ]
)

# ── Relevance Check ─────────────────────────────────────────────────────────
is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "A document is relevant if it contains information that directly helps answer the question.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)

# ── RAG Generation from Context ─────────────────────────────────────────────
rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a startup RAG assistant.\n"
            "Answer the user's question using only the provided context.\n"
            "If the context does not contain enough information, say:\n"
            "'No relevant document found.'\n"
            "Do not use outside knowledge - rely solely on the context.\n",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}\n"),
    ]
)

# ── Is Supported (Hallucination Check) ───────────────────────────────────────
issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issupported, evidence.\n"
            "issupported must be one of: fully_supported, partially_supported, not_supported.\n\n"
            "How to decide issupported:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- not_supported:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose not_supported.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge.",
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n",
        ),
    ]
)

# ── Revise Answer ────────────────────────────────────────────────────────────
revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n",
        ),
        (
            "human",
            "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}",
        ),
    ]
)

# ── Is Useful Check ─────────────────────────────────────────────────────────
isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the SPECIFIC question asked using the EXACT concept/topic requested.\n"
            "- not_useful: The answer is generic, off-topic, uses related but DIFFERENT concepts (e.g., 'values' when asked for 'culture'), or only provides background without answering.\n"
            "- Be strict: if the question asks for X but answer provides Y (even if related), mark as not_useful.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]
)

# ── Rewrite Query for Retrieval ──────────────────────────────────────────────
rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6-16 words).\n"
            "- Preserve key entities (e.g., NovaMind AI, plan names).\n"
            "- Add 2-5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NovaMind plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NovaMind free trial duration trial period plans'}}\n\n"
            "Q: 'What is NovaMind refund policy?'\n"
            "-> {{'retrieval_query': 'NovaMind refund policy cancellation refund timeline charges'}}",
        ),
        (
            "human",
            "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer (if any):\n{answer}",
        ),
    ]
)
