from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, PARSER_MODE

VALID_PARSER_MODES = {"pypdf", "docling", "llamaparse", "hybrid"}


def load_pdf(pdf_path):
    """
    Load a single PDF into LangChain Document objects.
    Parser is picked via config.PARSER_MODE:
      - "pypdf"      -> fast, default, works for normal text PDFs
      - "docling"    -> scanned / image-based PDFs (OCR)
      - "llamaparse" -> PDFs with complex tables
      - "hybrid"     -> runs docling + llamaparse, merges both outputs
    """
    if PARSER_MODE not in VALID_PARSER_MODES:
        raise ValueError(
            f"Invalid PARSER_MODE '{PARSER_MODE}' in config.py. "
            f"Check for typos - valid values are: {sorted(VALID_PARSER_MODES)}"
        )

    if PARSER_MODE == "docling":
        from document_loader import load_pdf_docling
        return load_pdf_docling(pdf_path)

    if PARSER_MODE == "llamaparse":
        from document_loader import load_pdf_llamaparse
        return load_pdf_llamaparse(pdf_path)

    if PARSER_MODE == "hybrid":
        from document_loader import load_pdf_hybrid
        return load_pdf_hybrid(pdf_path)

    return PyPDFLoader(pdf_path).load()


def split_documents(documents):
    """Split documents into overlapping chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)