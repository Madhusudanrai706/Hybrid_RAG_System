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
LLM_TEMPERATURE = 0.5

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
REGISTRY_PATH = "registry/processed_files.json"

# ---------------------------------------------------
# Metadata Filtering
# ---------------------------------------------------
# FAISS can't natively filter by arbitrary metadata (page ranges, chapter,
# etc.), so when filters are active we over-fetch this many times top_k
# candidates, then filter client-side, then truncate back to top_k.
FILTER_OVERFETCH_MULTIPLIER = 5

# ---------------------------------------------------
# PDF Parser
# ---------------------------------------------------
# "pypdf"      -> fast, works for normal text-based PDFs (original loader)
# "docling"    -> use for scanned / image-based PDFs (has built-in OCR)
# "llamaparse" -> use for PDFs with complex tables (needs LLAMA_CLOUD_API_KEY)
# "hybrid"     -> runs docling AND llamaparse, merges both outputs
PARSER_MODE = "hybrid"