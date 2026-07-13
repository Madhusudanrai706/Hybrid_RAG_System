"""
Keyword (BM25) store — persists the sparse index to disk, the same way
vector_store.py persists the FAISS index, instead of keeping it only as
a raw Python list in Streamlit's session state.

For large PDFs this matters: without this, the whole tokenized corpus
only ever lives in memory and has to be rebuilt from scratch every time
the app restarts. Saving it to disk once means it can be reloaded
directly, same as the FAISS index.
"""

import os
import pickle

from config import BM25_INDEX_PATH
from embeddings import get_sparse_embeddings


def create_keyword_store(chunks):
    """Build a BM25 index from chunks and save it (with the chunks) to disk."""
    bm25 = get_sparse_embeddings(chunks)

    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return bm25


def load_keyword_store():
    """Load a previously saved BM25 index and its chunks from disk."""
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    return data["bm25"], data["chunks"]  