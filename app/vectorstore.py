"""FAISS vector store management: build, save, load, and retriever access."""

import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_FILES,
    TOP_K,
    VECTORSTORE_DIR,
    embeddings,
)

logger = logging.getLogger(__name__)

# Module-level cache
_retriever = None


def build_index() -> FAISS:
    """Load PDFs, chunk, embed, save FAISS index to disk, and return the store."""
    logger.info("Loading PDFs from %s", [str(p) for p in PDF_FILES])
    docs = []
    for pdf_path in PDF_FILES:
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            continue
        docs.extend(PyPDFLoader(str(pdf_path)).load())

    if not docs:
        raise FileNotFoundError("No PDFs found. Check data/pdfs/ directory.")

    logger.info("Loaded %d pages. Chunking (size=%d, overlap=%d)...", len(docs), CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

    logger.info("Created %d chunks. Building FAISS index...", len(chunks))
    vector_store = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTORSTORE_DIR))
    logger.info("FAISS index saved to %s", VECTORSTORE_DIR)

    return vector_store


def load_index() -> FAISS:
    """Load a persisted FAISS index from disk."""
    index_file = VECTORSTORE_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {VECTORSTORE_DIR}. "
            "Run `python -m scripts.rebuild_index` first."
        )
    logger.info("Loading FAISS index from %s", VECTORSTORE_DIR)
    return FAISS.load_local(
        str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
    )


def get_retriever():
    """Return a cached retriever instance. Loads index from disk on first call."""
    global _retriever
    if _retriever is None:
        vector_store = load_index()
        _retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
        logger.info("Retriever ready (top_k=%d)", TOP_K)
    return _retriever
