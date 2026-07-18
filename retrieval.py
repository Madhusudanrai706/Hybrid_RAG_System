from ranking import (
    bm25_search_filtered,
    filter_documents,
    normalize_scores,
    weighted_hybrid_ranking
)
from config import FILTER_OVERFETCH_MULTIPLIER


# ---------------------------------------------------
# Semantic Search
# ---------------------------------------------------

def semantic_search(vector_db, question, top_k=5, sources=None,
                     page_range=None, chapter=None, subject=None):
    """
    FAISS similarity search, with optional metadata filtering.

    FAISS itself has no notion of "search only within these page ranges",
    so when any filter is active we over-fetch (top_k * multiplier)
    candidates first, filter them by metadata, then truncate back to
    top_k. This is a pragmatic client-side filter that works regardless
    of langchain/FAISS version, at the cost of being an approximation:
    if the filtered subset is a small fraction of the corpus, relevant
    matches could in rare cases fall outside the over-fetched window.
    """
    has_filters = any([sources, page_range, chapter, subject])
    fetch_k = top_k * FILTER_OVERFETCH_MULTIPLIER if has_filters else top_k

    results = vector_db.similarity_search(question, k=fetch_k)

    if has_filters:
        results = filter_documents(
            results, sources=sources, page_range=page_range,
            chapter=chapter, subject=subject
        )

    return results[:top_k]


# ---------------------------------------------------
# Keyword Search
# ---------------------------------------------------

def keyword_search(bm25, chunks, question, top_k=5, sources=None,
                    page_range=None, chapter=None, subject=None):
    return bm25_search_filtered(
        bm25, chunks, question, top_k,
        sources=sources, page_range=page_range,
        chapter=chapter, subject=subject
    )


# ---------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------

def hybrid_search(
    vector_db,
    bm25,
    chunks,
    question,
    semantic_weight=0.7,
    keyword_weight=0.3,
    top_k=5,
    sources=None,
    page_range=None,
    chapter=None,
    subject=None
):
    semantic_results = semantic_search(
        vector_db, question, top_k,
        sources=sources, page_range=page_range, chapter=chapter, subject=subject
    )
    keyword_results = keyword_search(
        bm25, chunks, question, top_k,
        sources=sources, page_range=page_range, chapter=chapter, subject=subject
    )

    return weighted_hybrid_ranking(
        semantic_results,
        keyword_results,
        semantic_weight,
        keyword_weight
    )


# ---------------------------------------------------
# Formatters - convert raw results into the same
# {"document", "semantic_score", "keyword_score", "final_score"}
# shape used everywhere in the UI, for the single-mode cases too.
# ---------------------------------------------------

def format_semantic_only(semantic_results):
    total = len(semantic_results)

    ranked = []
    for rank, doc in enumerate(semantic_results, start=1):
        score = (total - rank + 1) / total if total else 0
        ranked.append({
            "document": doc,
            "semantic_score": score,
            "keyword_score": 0,
            "final_score": score
        })
    return ranked


def format_keyword_only(keyword_results):
    if not keyword_results:
        return []

    scores = normalize_scores([score for score, _ in keyword_results])

    return [
        {
            "document": doc,
            "semantic_score": 0,
            "keyword_score": score,
            "final_score": score
        }
        for score, (_, doc) in zip(scores, keyword_results)
    ]