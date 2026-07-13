"""
Embedding generation - kept in its own file, separate from vector_store.py
and ranking.py, so both retrieval paths are swappable independently.

- Dense embeddings  -> used for Semantic Search (FAISS)
- Sparse embeddings -> used for Keyword Search (BM25)

Uses functools.lru_cache instead of st.cache_resource so this file has
no Streamlit dependency - it works the same in console mode (main.py)
and inside the Streamlit app (app.py).
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_dense_embeddings():
    """
    Dense embedding model (sentence-transformers).
    Turns each chunk into a continuous vector for semantic similarity search.
    Cached so it loads only once per process, not on every call.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_sparse_embeddings(chunks):
    """
    Sparse (BM25) representation of the chunks.
    Used for exact keyword / term-overlap matching - not cached, since
    it depends on whichever PDFs were just processed.
    """
    documents = [chunk.page_content for chunk in chunks]
    tokenized_docs = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized_docs)