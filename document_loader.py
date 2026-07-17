"""
Alternative PDF loaders, on top of the default PyPDFLoader in utils.py.

- Docling    -> best for scanned / image-based PDFs (built-in OCR)
- LlamaParse -> best for PDFs with complex tables (LLM-based reconstruction)
- Hybrid     -> runs both and merges their output into one document set,
                so retrieval has access to both representations

All parsers are imported lazily (inside the functions) so the app doesn't
fail to start if these optional packages aren't installed and PARSER_MODE
is left as "pypdf" in config.py.

All return a list of langchain_core Document objects - the same shape
PyPDFLoader.load() returns - so utils.split_documents() and everything
downstream (vector_store.py, keyword_store.py) works unchanged.
"""

from langchain_core.documents import Document


def load_pdf_docling(pdf_path):
    """Parse a PDF (including scanned pages) using Docling's OCR pipeline."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown_text = result.document.export_to_markdown()

    if not markdown_text or not markdown_text.strip():
        raise RuntimeError(
            "Docling returned no extractable text (OCR may have failed, "
            "or the PDF has no readable content on any page)."
        )

    return [Document(
        page_content=markdown_text,
        metadata={"source": pdf_path, "parser": "docling"}
    )]


def load_pdf_llamaparse(pdf_path):
    """Parse a PDF using LlamaParse - strong at reconstructing tables."""
    import os
    from llama_parse import LlamaParse

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY not found. Add it to your .env file.")

    parser = LlamaParse(api_key=api_key, result_type="markdown")
    llama_docs = parser.load_data(pdf_path)

    docs = [
        Document(
            page_content=doc.text,
            metadata={"source": pdf_path, "parser": "llamaparse"}
        )
        for doc in llama_docs
        if doc.text and doc.text.strip()
    ]

    if not docs:
        raise RuntimeError(
            "LlamaParse returned no extractable text for this PDF."
        )

    return docs


def load_pdf_hybrid(pdf_path):
    """
    Run Docling AND LlamaParse on the same PDF, then merge both outputs
    into one list of Documents (each tagged with metadata["parser"]).

    Both versions of the content end up in the vector store and BM25
    index - so a query can match either the OCR'd/Docling text or the
    LlamaParse table reconstruction, whichever is more relevant.

    If one parser fails (e.g. LlamaParse has no API key, a network
    error, or it extracts no text), we fall back to whichever one
    succeeded instead of crashing the whole pipeline.
    """
    merged_docs = []
    errors = []

    try:
        merged_docs.extend(load_pdf_docling(pdf_path))
    except Exception as e:
        errors.append(f"Docling failed: {e}")

    try:
        merged_docs.extend(load_pdf_llamaparse(pdf_path))
    except Exception as e:
        errors.append(f"LlamaParse failed: {e}")

    if not merged_docs:
        raise RuntimeError(
            f"Both parsers failed for {pdf_path}: " + " | ".join(errors)
        )

    if errors:
        print(f"[document_loader] Warning for {pdf_path}: " + " | ".join(errors))

    return merged_docs