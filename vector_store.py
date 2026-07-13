from langchain_community.vectorstores import FAISS

from config import FAISS_INDEX_DIR
from embeddings import get_dense_embeddings


def create_vector_store(chunks):
    """Build a FAISS index from document chunks and save it to disk."""
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
