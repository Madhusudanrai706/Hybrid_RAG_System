import os

import streamlit as st
from dotenv import load_dotenv

import file_registry
from config import PDF_DIR, TOP_K, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
from utils import load_pdf, split_documents, attach_metadata
from vector_store import (
    create_vector_store,
    load_vector_store,
    add_to_vector_store,
    vector_store_exists
)
from keyword_store import (
    load_keyword_store,
    add_to_keyword_store,
    keyword_store_exists
)
from retrieval import (
    semantic_search,
    keyword_search,
    hybrid_search,
    format_semantic_only,
    format_keyword_only
)
from qa import get_llm, build_context, generate_answer
from evaluation import calculate_bleu, calculate_rouge

load_dotenv()

st.set_page_config(page_title="Hybrid RAG QA System", page_icon="📄", layout="wide")
st.title("📄 Hybrid RAG Question Answering System")
st.markdown(
    "Upload PDFs once - they stay indexed across restarts. Ask questions using "
    "**Semantic Search**, **BM25 Keyword Search**, or **Hybrid Search**, "
    "optionally scoped to specific PDFs, pages, chapters, or subjects."
)

llm = get_llm()

# ---------------------------------------------------
# Session State + Auto-load Persisted Knowledge Base
# ---------------------------------------------------

for key in ("chunks", "bm25", "vector_db", "processed"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "processed" else False

if "load_status" not in st.session_state:
    st.session_state.load_status = None

if not st.session_state.processed:
    if vector_store_exists() and keyword_store_exists():
        with st.spinner("Loading existing knowledge base from disk..."):
            try:
                st.session_state.vector_db = load_vector_store()
                st.session_state.bm25, st.session_state.chunks = load_keyword_store()
                st.session_state.processed = True
                st.session_state.load_status = "loaded"
            except Exception as e:
                st.session_state.load_status = f"error: {e}"
    else:
        st.session_state.load_status = "empty"

if st.session_state.load_status == "loaded":
    st.success(
        f"📚 Loaded existing knowledge base - "
        f"{len(st.session_state.chunks)} chunks from "
        f"{len(file_registry.list_processed_files())} PDF(s). No re-upload needed."
    )
elif st.session_state.load_status == "empty":
    st.info("No existing knowledge base found yet. Upload PDFs below to get started.")
elif isinstance(st.session_state.load_status, str) and st.session_state.load_status.startswith("error"):
    st.warning(f"Could not load existing knowledge base: {st.session_state.load_status}")


# ---------------------------------------------------
# Sidebar - Indexed PDFs
# ---------------------------------------------------

with st.sidebar:
    st.header("📚 Indexed PDFs")
    processed_files = file_registry.list_processed_files()

    if processed_files:
        for entry in processed_files:
            st.markdown(f"**{entry['filename']}**")
            st.caption(
                f"Subject: {entry.get('subject', 'General')} | "
                f"Chunks: {entry.get('chunk_count', '-')} | "
                f"Pages: {entry.get('page_count', '-')}"
            )
            st.divider()
    else:
        st.caption("No PDFs indexed yet.")


# ---------------------------------------------------
# Upload & Process PDFs (incremental - only new files are processed)
# ---------------------------------------------------

st.subheader("Upload PDF(s)")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process PDFs"):
    os.makedirs(PDF_DIR, exist_ok=True)

    with st.spinner("Processing PDFs..."):
        new_chunks_total = []
        skipped_files = []
        failed_files = []

        for uploaded_file in uploaded_files:
            file_bytes = bytes(uploaded_file.getbuffer())
            file_hash = file_registry.compute_file_hash(file_bytes)

            if file_registry.is_processed(file_hash):
                skipped_files.append(uploaded_file.name)
                continue

            pdf_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(file_bytes)

            try:
                chunks = split_documents(load_pdf(pdf_path))
            except Exception as e:
                failed_files.append((uploaded_file.name, str(e)))
                continue

            if not chunks:
                failed_files.append((uploaded_file.name, "No text could be extracted."))
                continue

            chunks = attach_metadata(chunks, source=uploaded_file.name, subject=subject_input)

            pages = [c.metadata.get("page") for c in chunks if c.metadata.get("page") is not None]
            page_count = (max(pages) + 1) if pages else None

            file_registry.register_file(
                file_hash, uploaded_file.name,
                subject=subject_input, chunk_count=len(chunks), page_count=page_count
            )

            new_chunks_total.extend(chunks)

        for name, err in failed_files:
            st.error(f"❌ {name}: {err}")
        for name in skipped_files:
            st.info(f"⏭️ {name}: already indexed, skipped.")

        if new_chunks_total:
            if st.session_state.vector_db is not None:
                st.session_state.vector_db = add_to_vector_store(
                    st.session_state.vector_db, new_chunks_total
                )
            else:
                st.session_state.vector_db = create_vector_store(new_chunks_total)

            st.session_state.bm25, st.session_state.chunks = add_to_keyword_store(new_chunks_total)
            st.session_state.processed = True

            processed_count = len(uploaded_files) - len(skipped_files) - len(failed_files)
            st.success(f"✅ Indexed {len(new_chunks_total)} new chunks from {processed_count} file(s)!")
        elif not failed_files:
            st.info("Nothing new to process - all uploaded files were already indexed.")


# ---------------------------------------------------
# Metadata Filters
# ---------------------------------------------------
# The filtering pipeline (ranking.filter_documents, retrieval.py's
# sources/page_range/chapter/subject params) is still fully wired in and
# used by semantic_search/keyword_search/hybrid_search below. There's
# just no UI for it right now - every chunk still carries its metadata
# (source/page/chapter/subject/upload_date), so filters can be turned
# back on by setting these values (or re-adding widgets) without
# touching retrieval.py, ranking.py, or utils.attach_metadata.
filter_kwargs = {
    "sources": None,
    "page_range": None,
    "chapter": None,
    "subject": None,
}


# ---------------------------------------------------
# Ask a Question
# ---------------------------------------------------

search_mode = st.selectbox("Select Search Mode", ["Hybrid", "Semantic", "Keyword"])
question = st.text_input("Ask your question")
reference_answer = st.text_area("Reference Answer (Ground Truth) - optional, for evaluation")

if st.button("Get Answer"):
    if not st.session_state.processed:
        st.warning("Please process at least one PDF first.")
    else:
        vector_db = st.session_state.vector_db
        bm25 = st.session_state.bm25
        chunks = st.session_state.chunks

        if search_mode == "Semantic":
            ranked_docs = format_semantic_only(
                semantic_search(vector_db, question, TOP_K, **filter_kwargs)
            )
        elif search_mode == "Keyword":
            ranked_docs = format_keyword_only(
                keyword_search(bm25, chunks, question, TOP_K, **filter_kwargs)
            )
        else:
            ranked_docs = hybrid_search(
                vector_db, bm25, chunks, question,
                semantic_weight=SEMANTIC_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
                top_k=TOP_K,
                **filter_kwargs
            )

        if not ranked_docs:
            st.warning("No chunks matched your filters for this question. Try loosening them.")
        else:
            top_docs = [item["document"] for item in ranked_docs[:TOP_K]]
            context = build_context(top_docs)
            generated_answer = generate_answer(llm, context, question)

            st.subheader("Answer")
            st.success(generated_answer)

            if reference_answer.strip():
                bleu = calculate_bleu(reference_answer, generated_answer)
                rouge = calculate_rouge(reference_answer, generated_answer)

                st.subheader("Evaluation Metrics")
                st.write(f"BLEU Score : {bleu:.4f}")
                st.write(f"ROUGE-1 : {rouge['rouge1'].fmeasure:.4f}")
                st.write(f"ROUGE-2 : {rouge['rouge2'].fmeasure:.4f}")
                st.write(f"ROUGE-L : {rouge['rougeL'].fmeasure:.4f}")

            with st.expander("Retrieved Chunks"):
                for i, item in enumerate(ranked_docs[:TOP_K], start=1):
                    doc = item["document"]
                    st.markdown(f"### Rank {i}")
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.markdown(f"**Page:** {doc.metadata.get('page', '-')}")
                    st.markdown(f"**Chapter:** {doc.metadata.get('chapter', 'Unknown')}")
                    st.markdown(f"**Subject:** {doc.metadata.get('subject', 'General')}")
                    st.markdown(f"**Semantic Score:** {item['semantic_score']:.3f}")
                    st.markdown(f"**Keyword Score:** {item['keyword_score']:.3f}")
                    st.markdown(f"**Final Score:** {item['final_score']:.3f}")
                    st.write(doc.page_content)
                    st.divider()