import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from utils import (
    load_pdf,
    split_documents,
    create_vector_store,
    load_vector_store
)

from ranking import (
    build_bm25,
    bm25_search,
    weighted_hybrid_ranking
)
from evaluation import (
    calculate_bleu,
    calculate_rouge
)
# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()

# ---------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG QA System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Hybrid RAG Question Answering System")

st.markdown(
"""
Upload one or more PDFs and ask questions using

- Semantic Search
- BM25 Keyword Search
- Hybrid Search (Weighted Ranking)
"""
)

# ---------------------------------------------------
# Load Groq LLM
# ---------------------------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------------------------------------------
# Upload PDFs
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# Process PDFs
# ---------------------------------------------------

if uploaded_files:

    os.makedirs("pdfs", exist_ok=True)

    if st.button("Process PDFs"):

        with st.spinner("Processing PDFs..."):

            all_chunks = []

            # Save every uploaded PDF
            for uploaded_file in uploaded_files:

                pdf_path = os.path.join(
                    "pdfs",
                    uploaded_file.name
                )

                with open(pdf_path, "wb") as f:

                    f.write(uploaded_file.getbuffer())

                # Load PDF
                documents = load_pdf(pdf_path)

                # Split into chunks
                chunks = split_documents(documents)

                # Store source information
                for chunk in chunks:

                    chunk.metadata["source"] = uploaded_file.name

                all_chunks.extend(chunks)

            # -----------------------------
            # Create Vector Database
            # -----------------------------

            vector_db = create_vector_store(all_chunks)

            # -----------------------------
            # Build BM25 Index
            # -----------------------------

            bm25 = build_bm25(all_chunks)

            # -----------------------------
            # Save in Session State
            # -----------------------------

            st.session_state.vector_db = vector_db

            st.session_state.bm25 = bm25

            st.session_state.chunks = all_chunks

            st.session_state.processed = True

        st.success("✅ PDFs processed successfully!")
        # ---------------------------------------------------
# Search Mode
# ---------------------------------------------------

search_mode = st.selectbox(
    "Select Search Mode",
    [
        "Hybrid",
        "Semantic",
        "Keyword"
    ]
)

question = st.text_input("Ask your question")
reference_answer = st.text_area(
    "Reference Answer (Ground Truth)"
)

if st.button("Get Answer"):

    if not st.session_state.processed:

        st.warning("Please process the PDFs first.")

    else:

        vector_db = st.session_state.vector_db

        bm25 = st.session_state.bm25

        chunks = st.session_state.chunks

        # ----------------------------------------
        # Semantic Search
        # ----------------------------------------

        semantic_results = vector_db.similarity_search_with_relevance_scores(
            question,
            k=5
        )

        # ----------------------------------------
        # BM25 Search
        # ----------------------------------------

        keyword_results = bm25_search(
            bm25,
            chunks,
            question,
            top_k=5
        )

        # ----------------------------------------
        # Decide Retrieval Method
        # ----------------------------------------

        if search_mode == "Semantic":

            ranked_docs = []

            for doc, score in semantic_results:

                ranked_docs.append({

                    "document": doc,

                    "semantic_score": score,

                    "keyword_score": 0,

                    "final_score": score

                })

        elif search_mode == "Keyword":

            ranked_docs = []

            max_score = max(score for score, _ in keyword_results)

            for score, doc in keyword_results:

                ranked_docs.append({

                    "document": doc,

                    "semantic_score": 0,

                    "keyword_score": score,

                    "final_score": score / max_score

                })

        else:

            ranked_docs = weighted_hybrid_ranking(

                semantic_results,

                keyword_results,

                semantic_weight=0.7,

                keyword_weight=0.3

            )
                    # ----------------------------------------
        # Build Context
        # ----------------------------------------

        top_docs = [
            item["document"]
            for item in ranked_docs[:5]
        ]

        context = "\n\n".join(
            doc.page_content
            for doc in top_docs
        )

        # ----------------------------------------
        # Prompt
        # ----------------------------------------

        prompt = f"""
You are an intelligent AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
reply exactly:

"I couldn't find this information in the uploaded PDFs."

Context:

{context}

Question:

{question}
"""

        response = llm.invoke(prompt)
        generated_answer = response.content
        print("\n" + "=" * 80)
        print("HYBRID RETRIEVAL RANKING")
        print("=" * 80)
        for i, item in enumerate(ranked_docs, start=1):
            doc = item["document"]
            print(f"\nRank : {i}")
            print(f"Source : {doc.metadata.get('source', 'Unknown')}")
            print(f"Page : {doc.metadata.get('page', '-')}")
            print(f"Semantic Score : {item['semantic_score']:.4f}")
            print(f"Keyword Score  : {item['keyword_score']:.4f}")
            print(f"Final Score    : {item['final_score']:.4f}")
            print("=" * 80)

        # ----------------------------------------
        # Answer
        # ----------------------------------------

        st.subheader("Answer")

        st.success(response.content)
         # ----------------------------------------
# Evaluation Metrics
# ----------------------------------------
       
        if reference_answer.strip():
            bleu = calculate_bleu(
                reference_answer,
                generated_answer
                )
            rouge = calculate_rouge(
                reference_answer,
                generated_answer
                )
            st.subheader("Evaluation Metrics")
            st.write(
                f"BLEU Score : {bleu:.4f}"
                )
            st.write(
                f"ROUGE-1 : {rouge['rouge1'].fmeasure:.4f}"
                )
            st.write(
                f"ROUGE-2 : {rouge['rouge2'].fmeasure:.4f}"
                )
            st.write(
                f"ROUGE-L : {rouge['rougeL'].fmeasure:.4f}"
                )
            
    # ----------------------------------------
        # Ranking Table deleted 
        # ------------------------------------


        # ----------------------------------------
        # Retrieved Chunks
        # ----------------------------------------
            with st.expander(
                "Retrieved Chunks"
                ):
                for i, item in enumerate(
                    ranked_docs[:5],
                    start=1
                    ):
                    doc = item["document"]
                    st.markdown(
                        f"### Rank {i}"
                        )
                    st.markdown(
                        f"**Source:** {doc.metadata.get('source','Unknown')}"
                        )
                    st.markdown(
                        f"**Page:** {doc.metadata.get('page','-')}"
                        )
                    st.markdown(
                        f"**Final Score:** {item['final_score']:.3f}"
                        )
                    st.write(
                        doc.page_content
                        )
                    st.divider()