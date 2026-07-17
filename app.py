import os

import streamlit as st
from dotenv import load_dotenv

from config import PDF_DIR, TOP_K, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
from utils import load_pdf, split_documents
from vector_store import create_vector_store
from keyword_store import create_keyword_store
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
    "Upload one or more PDFs and ask questions using "
    "**Semantic Search**, **BM25 Keyword Search**, or **Hybrid Search**."
)

llm = get_llm()

for key in ("chunks", "bm25", "vector_db", "processed"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "processed" else False


# ---------------------------------------------------
# Upload & Process PDFs
# ---------------------------------------------------

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process PDFs"):
    os.makedirs(PDF_DIR, exist_ok=True)

    with st.spinner("Processing PDFs..."):
        all_chunks = []
        failed_files = []

        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                chunks = split_documents(load_pdf(pdf_path))
            except Exception as e:
                failed_files.append((uploaded_file.name, str(e)))
                continue

            if not chunks:
                failed_files.append((uploaded_file.name, "No text could be extracted."))
                continue

            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name

            all_chunks.extend(chunks)

        for name, err in failed_files:
            st.error(f"❌ {name}: {err}")

        if not all_chunks:
            st.warning("No chunks were extracted from any file. Nothing to index.")
            st.stop()

        st.session_state.vector_db = create_vector_store(all_chunks)
        st.session_state.bm25 = create_keyword_store(all_chunks)
        st.session_state.chunks = all_chunks
        st.session_state.processed = True

    st.success(f"✅ Processed {len(all_chunks)} chunks from {len(uploaded_files) - len(failed_files)} file(s)!")


# ---------------------------------------------------
# Ask a Question
# ---------------------------------------------------

search_mode = st.selectbox("Select Search Mode", ["Hybrid", "Semantic", "Keyword"])
question = st.text_input("Ask your question")
reference_answer = st.text_area("Reference Answer (Ground Truth) — optional, for evaluation")

if st.button("Get Answer"):
    if not st.session_state.processed:
        st.warning("Please process the PDFs first.")
    else:
        vector_db = st.session_state.vector_db
        bm25 = st.session_state.bm25
        chunks = st.session_state.chunks

        if search_mode == "Semantic":
            ranked_docs = format_semantic_only(semantic_search(vector_db, question, TOP_K))
        elif search_mode == "Keyword":
            ranked_docs = format_keyword_only(keyword_search(bm25, chunks, question, TOP_K))
        else:
            ranked_docs = hybrid_search(
                vector_db, bm25, chunks, question,
                semantic_weight=SEMANTIC_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
                top_k=TOP_K
            )

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
                st.markdown(f"**Semantic Score:** {item['semantic_score']:.3f}")
                st.markdown(f"**Keyword Score:** {item['keyword_score']:.3f}")
                st.markdown(f"**Final Score:** {item['final_score']:.3f}")
                st.write(doc.page_content)
                st.divider()