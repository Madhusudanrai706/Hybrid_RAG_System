"""
Console version of the Hybrid RAG QA System - same pipeline as app.py,
just without Streamlit, so it starts instantly for quick testing.

Run with:  python main.py
"""

import os

from dotenv import load_dotenv

from config import TOP_K, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
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


def process_pdfs(pdf_paths):
    all_chunks = []

    for path in pdf_paths:
        chunks = split_documents(load_pdf(path))
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(path)
        all_chunks.extend(chunks)

    print(f"\nLoaded {len(all_chunks)} chunks from {len(pdf_paths)} file(s).")

    vector_db = create_vector_store(all_chunks)
    bm25 = create_keyword_store(all_chunks)

    return vector_db, bm25, all_chunks


def ask_question(vector_db, bm25, chunks, question, mode="hybrid"):
    mode = mode.strip().lower()

    if mode == "semantic":
        ranked_docs = format_semantic_only(semantic_search(vector_db, question, TOP_K))
    elif mode == "keyword":
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

    llm = get_llm()
    answer = generate_answer(llm, context, question)

    return answer, ranked_docs


def main():
    print("=" * 60)
    print("Hybrid RAG QA System - Console Mode")
    print("=" * 60)

    pdf_input = input("\nEnter PDF path(s), comma-separated: ").strip()
    pdf_paths = [p.strip() for p in pdf_input.split(",") if p.strip()]

    if not pdf_paths:
        print("No PDF path provided. Exiting.")
        return

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

    vector_db, bm25, chunks = process_pdfs(pdf_paths)

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        mode = input("Search mode - hybrid / semantic / keyword [hybrid]: ").strip() or "hybrid"

        answer, ranked_docs = ask_question(vector_db, bm25, chunks, question, mode)

        print("\n--- Answer ---")
        print(answer)

        reference = input("\nReference answer for evaluation (Enter to skip): ").strip()
        if reference:
            bleu = calculate_bleu(reference, answer)
            rouge = calculate_rouge(reference, answer)
            print(f"\nBLEU Score : {bleu:.4f}")
            print(f"ROUGE-1    : {rouge['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2    : {rouge['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L    : {rouge['rougeL'].fmeasure:.4f}")

        print("\n--- Top Retrieved Chunks ---")
        for i, item in enumerate(ranked_docs[:TOP_K], start=1):
            doc = item["document"]
            print(
                f"Rank {i} | Source: {doc.metadata.get('source', 'Unknown')} | "
                f"Page: {doc.metadata.get('page', '-')} | "
                f"Final Score: {item['final_score']:.3f}"
            )


if __name__ == "__main__":
    main()