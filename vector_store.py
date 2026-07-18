import os

from langchain_community.vectorstores import FAISS

from config import FAISS_INDEX_DIR
from embeddings import get_dense_embeddings


def create_vector_store(chunks):
    """Build a new FAISS index from document chunks and save it to disk."""
    embedding_model = get_dense_embeddings()

    db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    db.save_local(FAISS_INDEX_DIR)

    return db


def load_vector_store():
    """Load a previously saved FAISS index from disk."""
    embedding_model = get_dense_embeddings()

    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )


def add_to_vector_store(vector_db, new_chunks):
    """
    Add new chunks to an already-loaded FAISS index in place, then persist
    the updated index to disk. This is what makes "upload one more PDF"
    cheap - only the new chunks get embedded, the existing vectors are
    untouched.
    """
    vector_db.add_documents(new_chunks)
    vector_db.save_local(FAISS_INDEX_DIR)
    return vector_db


def vector_store_exists():
    """True if a FAISS index has already been saved to disk."""
    return os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))