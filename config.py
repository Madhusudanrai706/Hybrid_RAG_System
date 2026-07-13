"""
Central configuration for the Hybrid RAG QA System.
Change values here instead of hunting through every file.
"""

# ---------------------------------------------------
# Text Splitting
# ---------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------
# Embedding Model
# ---------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# ---------------------------------------------------
# Retrieval
# ---------------------------------------------------
TOP_K = 5
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
PDF_DIR = "pdfs"
FAISS_INDEX_DIR = "faiss_index"
BM25_INDEX_PATH = "bm25_index/bm25.pkl"