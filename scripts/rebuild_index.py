"""Rebuild the FAISS vector store index from PDFs in data/pdfs/."""

import logging
import sys
from pathlib import Path

# Add project root to path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vectorstore import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")


def main():
    print("Rebuilding FAISS index from PDFs...")
    store = build_index()
    print(f"Done. Index contains {store.index.ntotal} vectors.")


if __name__ == "__main__":
    main()
